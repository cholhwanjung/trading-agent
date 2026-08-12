"""Risk Engine — LLM 비개입 결정론 가드레일.

정책(LLM 포함)이 무엇을 출력하든 이 레이어를 통과한 배분만 어댑터로 간다.
LLM 이 이 레이어를 우회/완화하는 경로는 존재하지 않는다 — 입력은 배분 벡터뿐.

적용 순서 (앞의 룰이 절대적):
1. MDD 서킷브레이커 — 임계 초과 시 직전 배분 동결(신규 진입 정지) + 알림 플래그.
2. Forbidden 하드 veto — 고신뢰 실패 패턴 종목은 배분 0 (APV).
3. 배분 정의역 밖 종목 제거 — 계좌가 주문을 낼 수 없는 종목의 비중은 0 (CASH 로).
4. 종목당 최대 배분 클램프 — 초과분은 CASH 로.
5. 최소 현금 비중 — 부족 시 비현금 자산 비례 축소.
6. 일일 turnover 상한 — 직전 배분에서의 이동량(L1/2)을 상한으로 스케일.
   (veto·정의역 밖 종목은 blend 대상에서 제외 — 배분할 수 없다는 판정이
    turnover 완화로 되살아나지 않게)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adapters.allocation import CASH


@dataclass(frozen=True)
class RiskLimits:
    """시장별 캘리브레이션 대상. 기본값은 설계 문서의 예시 초기값."""

    max_weight_per_asset: float = 0.20
    min_cash: float = 0.10
    max_daily_turnover: float = 0.50  # 0.5 * Σ|Δw| 상한
    mdd_circuit: float = 0.15  # 초과 시 서킷브레이커
    # 심볼별 상한 예외. 종목당 상한은 단일 종목 고유위험을 막는 장치인데, 지수 ETF 는
    # 그 위험이 이미 구성 비중만큼 희석돼 있어 개별주와 같은 값을 쓸 근거가 없다.
    asset_caps: dict[str, float] = field(default_factory=dict)
    # 배분 벡터의 정의역 — 계좌가 실제로 주문을 낼 수 있는 종목. 관측 유니버스는 이보다
    # 넓을 수 있다. 빈 집합은 "제약 없음"이며, 정의역을 모르는 호출부의 동작을 바꾸지 않는다.
    tradable: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RiskDecision:
    weights: dict[str, float]
    violations: list[str] = field(default_factory=list)  # key=value 로그용
    circuit_open: bool = False


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def enforce(
        self,
        weights: dict[str, float],
        prev_weights: dict[str, float] | None = None,
        mdd: float = 0.0,
        forbidden: frozenset[str] = frozenset(),
    ) -> RiskDecision:
        lim = self.limits
        violations: list[str] = []
        w = dict(weights)
        w.setdefault(CASH, 0.0)

        # 1. MDD 서킷브레이커 — 직전 배분 동결 (직전이 없으면 전액 현금)
        if mdd >= lim.mdd_circuit:
            frozen = dict(prev_weights) if prev_weights else {CASH: 1.0}
            return RiskDecision(
                weights=frozen,
                violations=[f"mdd_circuit mdd={mdd:.4f} limit={lim.mdd_circuit}"],
                circuit_open=True,
            )

        # 2. Forbidden 하드 veto
        for symbol in sorted(forbidden):
            if w.get(symbol, 0.0) > 0:
                violations.append(f"forbidden_veto symbol={symbol} weight={w[symbol]:.4f}")
                w[CASH] += w.pop(symbol)

        # 3. 배분 정의역 밖 종목 제거 — 정의역은 결정 단계(스키마 검증)에서도 강제되지만,
        #    직전 배분에는 정의역이 좁아지기 전의 종목이 남아 있고 아래 turnover blend 가
        #    그것을 다시 써넣는다. 결정론 계층이 되돌릴 수 있는 제약은 제약이 아니다.
        if lim.tradable:
            for symbol in sorted(w):
                if symbol != CASH and symbol not in lim.tradable:
                    violations.append(
                        f"not_tradable symbol={symbol} weight={w[symbol]:.4f}"
                    )
                    w[CASH] += w.pop(symbol)

        # 4. 종목당 최대 배분 클램프
        for symbol in sorted(w):
            if symbol == CASH:
                continue
            cap = lim.asset_caps.get(symbol, lim.max_weight_per_asset)
            if w[symbol] > cap:
                violations.append(
                    f"max_weight symbol={symbol} weight={w[symbol]:.4f} limit={cap}"
                )
                w[CASH] += w[symbol] - cap
                w[symbol] = cap

        # 5. 최소 현금 비중 — 비현금 자산 비례 축소
        if w[CASH] < lim.min_cash:
            violations.append(f"min_cash cash={w[CASH]:.4f} limit={lim.min_cash}")
            noncash = 1.0 - w[CASH]
            if noncash > 0:
                factor = (1.0 - lim.min_cash) / noncash
                for symbol in w:
                    if symbol != CASH:
                        w[symbol] *= factor
            w[CASH] = lim.min_cash

        # 6. 일일 turnover 상한 — 직전 배분 방향으로 blend (veto·정의역 밖 종목 제외)
        if prev_weights is not None:
            # CASH 포함 전체 L1/2 — 매수(현금→자산)도 이동량이다. 정의역 밖 종목의
            # 청산도 계좌가 실제로 내는 주문이므로 이동량에는 그대로 센다.
            turnover = 0.5 * sum(
                abs(w.get(s, 0.0) - prev_weights.get(s, 0.0))
                for s in set(w) | set(prev_weights)
            )
            symbols = (set(w) | set(prev_weights)) - {CASH}
            if turnover > lim.max_daily_turnover:
                violations.append(
                    f"turnover value={turnover:.4f} limit={lim.max_daily_turnover}"
                )
                alpha = lim.max_daily_turnover / turnover
                for s in symbols:
                    # turnover 상한은 **정책이 마음을 바꾸는 속도**를 제한하는 장치이지,
                    # 배분할 수 없다고 판정된 종목을 직전 배분에서 되살리는 근거가 아니다.
                    if s in forbidden or (lim.tradable and s not in lim.tradable):
                        continue
                    prev = prev_weights.get(s, 0.0)
                    w[s] = prev + (w.get(s, 0.0) - prev) * alpha
                w[CASH] = 1.0 - sum(v for s, v in w.items() if s != CASH)

        return RiskDecision(weights=w, violations=violations, circuit_open=False)
