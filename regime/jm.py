"""Jump model 피처 — 하방 리스크·리스크조정수익의 지수가중 요약 (결정론·순수).

분산일·FTD 룰 기반 FSM(`regime/pulse.py`)과 **같은 지수 프록시 봉**을 받아 나란히
계산되는 별개 관측 채널. 적합(fitting)이 없는 순수 함수이며, 세 값 모두 수익률
시계열 하나에서만 나온다. 통계적 jump model(Shu·Yu·Mulvey 2024)이 쓰는 피처셋을
이식했고 모델 자체는 아직 붙이지 않는다 — 피처를 먼저 쌓아 두 접근을 동일 기간·
동일 입력으로 비교하기 위한 사전 단계다.

    | EWM 하방편차 | 반감기 10일 | 리스크
    | EWM Sortino  | 반감기 20일 | 수익
    | EWM Sortino  | 반감기 60일 | 수익

분산이 아니라 하방편차를 쓰는 이유: 투자자는 상방 불확실성이 아니라 하방 손실을
걱정한다(Roy 1952 안전제일 · Markowitz 1959 semi-variance). 리스크 측정치는 완만히
움직이므로 짧은 반감기(10일)로 충분하고, 수익 측정치는 노이즈가 커서 반감기가 다른
둘(20·60일)을 넣어 빠른 값 변화가 상태 판정을 흔드는 것을 줄인다. 리스크 1 : 수익 2
배분은 변동성만 보는 접근과 달리 리스크·수익 양쪽에 균형을 두려는 것 — 실현변동성
게이지(`regime/vol.py`)가 부호 없는 진폭만 보는 사각을 메운다.

무위험수익률을 빼지 않고 단순수익률을 쓴다 — 일간 무위험수익률은 일간 변동성 대비
무시할 수준인데, 시장마다 무료 원천을 붙이면 실패 경로만 늘어난다.

반감기 60일 피처는 완전 워밍업에 120봉 이상이 필요하다. 브로커 시세 API 가 1회 응답
행 수를 제한해 시장에 따라 그보다 짧은 창이 올 수 있으므로 `n_bars` 를 함께 남긴다 —
창이 짧았던 날을 사후에 골라낼 수 있어야 비교가 정직해진다.
"""

from __future__ import annotations

from dataclasses import dataclass

DD_HALFLIFE = 10  # 하방편차 반감기 (거래일)
SORTINO_HALFLIVES = (20, 60)  # Sortino 반감기 2종 (노이즈 완화용 장·단)
MIN_BARS = 70  # 최소 수익률 표본 — 짧은 반감기 2종이 안정되는 하한
WARMUP_BARS = 120  # 학습용 시계열에서 버릴 앞부분 (반감기 60 이 안정되기 전 구간)
_ANNUALIZE = 252**0.5


@dataclass(frozen=True)
class JumpFeatures:
    downside_dev: float  # 연율화 %, 반감기 10일 (실현변동성과 같은 스케일)
    sortino_20: float  # 연율화 Sortino, 반감기 20일
    sortino_60: float  # 연율화 Sortino, 반감기 60일
    n_bars: int  # 실제 사용한 수익률 표본 수 (창 충분성 감사용)


def _ewm_series(values: list[float], halflife: float) -> list[float]:
    """각 시점까지의 지수가중평균 시계열 (재귀식, O(n)).

    t 시점 값은 t 까지의 데이터만 쓴다 — 학습용 시계열을 뽑아도 미래가 새지 않는다.
    """
    decay = 0.5 ** (1.0 / halflife)
    out, num, den = [], 0.0, 0.0
    for v in values:
        num = v + decay * num
        den = 1.0 + decay * den
        out.append(num / den)
    return out


def _ewm_mean(values: list[float], halflife: float) -> float:
    return _ewm_series(values, halflife)[-1]


def _downside_series(rets: list[float], halflife: float) -> list[float]:
    """√(EWM[r²·1{r<0}]) 시계열 — 음수 수익률만 제곱해 가중평균 후 제곱근. 상방은 벌하지 않는다."""
    squared = [r * r if r < 0 else 0.0 for r in rets]
    return [v**0.5 for v in _ewm_series(squared, halflife)]


def _downside_dev(rets: list[float], halflife: float) -> float:
    return _downside_series(rets, halflife)[-1]


def _returns(closes: list[float]) -> list[float]:
    return [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1] > 0]


def compute_jump_features(closes: list[float]) -> JumpFeatures | None:
    """종가 시퀀스(오름차순, 상한 t−1) → JumpFeatures. 표본 부족 시 None(판정 보류).

    창 안에 하락일이 하나도 없으면 하방편차가 0 이 되어 Sortino 가 정의되지 않는다.
    이때도 None 을 돌려준다 — 0 으로 채우면 '중립'으로 읽혀 실제(=하방 없음)와
    정반대 신호가 된다.
    """
    rets = _returns(closes)
    if len(rets) < MIN_BARS:
        return None

    sortinos = []
    for halflife in SORTINO_HALFLIVES:
        dd = _downside_dev(rets, halflife)
        if dd <= 0:
            return None
        sortinos.append(_ewm_mean(rets, halflife) / dd * _ANNUALIZE)

    return JumpFeatures(
        downside_dev=round(_downside_dev(rets, DD_HALFLIFE) * _ANNUALIZE * 100, 2),
        sortino_20=round(sortinos[0], 3),
        sortino_60=round(sortinos[1], 3),
        n_bars=len(rets),
    )


def feature_matrix(closes: list[float]) -> tuple[list[list[float]], list[float]]:
    """종가 시퀀스 → (피처 행렬, 같은 길이의 수익률). 적합·추론이 공유하는 입력.

    행 t 는 t 까지의 데이터만으로 계산된다(재귀 EWM). 앞 WARMUP_BARS 행은 반감기 60
    피처가 안정되기 전이라 버린다. 하락일이 아직 없어 하방편차가 0 인 행도 버린다 —
    Sortino 가 정의되지 않는데 0 으로 채우면 '중립'으로 잘못 읽힌다.

    반환되는 수익률은 각 행과 같은 날의 것이다. 상태별 누적수익으로 bull/bear 라벨을
    정할 때 쓰인다 — 라벨이 없으면 군집 번호가 재적합마다 뒤집힌다.
    """
    rets = _returns(closes)
    if len(rets) <= WARMUP_BARS:
        return [], []

    dd10 = _downside_series(rets, DD_HALFLIFE)
    means = {hl: _ewm_series(rets, hl) for hl in SORTINO_HALFLIVES}
    devs = {hl: _downside_series(rets, hl) for hl in SORTINO_HALFLIVES}

    rows, aligned = [], []
    for t in range(WARMUP_BARS, len(rets)):
        if dd10[t] <= 0 or any(devs[hl][t] <= 0 for hl in SORTINO_HALFLIVES):
            continue
        rows.append(
            [dd10[t] * _ANNUALIZE]
            + [means[hl][t] / devs[hl][t] * _ANNUALIZE for hl in SORTINO_HALFLIVES]
        )
        aligned.append(rets[t])
    return rows, aligned
