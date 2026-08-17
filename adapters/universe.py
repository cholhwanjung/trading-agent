"""유니버스 자산 분류 · 지수 ETF 참조 정보 · ETF 종목당 상한 도출.

**관측 유니버스와 집행 유니버스는 다를 수 있다.** 소액 계좌에서는 1주 값이 종목당
상한을 넘는 종목이 있고, 그런 종목에는 어떤 비중도 배정할 수 없다. 관측은 그대로
두고 배분 벡터의 정의역만 좁힌다 — 개별 종목의 재무·뉴스를 읽어 시장을 판단하고,
그 판단을 담을 수 있는 지수 ETF 의 비중으로 표현하는 구조다.

ETF 정적 정보는 관측이 아니라 **참조**다. 수집 시각이 없고 분기 이하로 바뀌므로
어댑터 호출을 늘리지 않고 상수로 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass

EQUITY = "equity"
ETF = "etf"


@dataclass(frozen=True)
class EtfRef:
    """지수 ETF 의 구조 정보.

    top_weight 는 최대 구성종목 비중의 **보수적 상한**이다 — 상한 도출의 입력이라
    실제보다 높게 잡는 쪽이 안전하다(실제가 더 낮으면 검사는 여전히 통과한다).

    name 은 상품명이고 index 는 추종 지수다 — 둘은 다르다(같은 지수를 여러 운용사가
    낸다). 표시 계층 전용이라 관측·결정 경로에는 넘기지 않는다.
    """

    name: str
    index: str
    replication: str  # physical(실물 복제) | synthetic(선물·스왑)
    expense_ratio: float  # 연 총보수
    holdings: int
    top_weight: float


# 편입 ETF — 광범위 시장지수만. 섹터·테마는 분산 효과가 없어 개별주 집중과 같아지고,
# 레버리지·인버스는 일간 리밸런싱 감쇠로 장기 보유 전제와 충돌해 넣지 않는다.
ETF_REF: dict[str, EtfRef] = {
    "278530": EtfRef("KODEX 200TR", "KOSPI 200", "physical", 0.0012, 200, 0.35),
    "SCHX": EtfRef("Schwab U.S. Large-Cap ETF", "Dow Jones U.S. Large-Cap",
                   "physical", 0.0003, 750, 0.10),
}


def asset_class(symbol: str) -> str:
    return ETF if symbol in ETF_REF else EQUITY


def resolve_asset_caps(
    symbols: list[str], equity_cap: float, min_cash: float
) -> dict[str, float]:
    """지수 ETF 의 종목당 상한. ETF 가 없으면 빈 dict.

    종목당 상한은 원래 **단일 종목 고유위험**을 막는 장치다. 지수 ETF 에는 그 위험이
    구성 비중만큼 희석돼 있으므로, ETF 를 상한까지 담아도 최대 구성종목 노출이 개별주
    상한을 넘지 않는다면 그 장치의 목적은 이미 지켜진 것이다. 그래서 ETF 상한은
    개별주와 같은 값이 아니라 최소 현금이 허용하는 최대치로 잡는다.

    부등식이 깨지면 로드 시점에 실패시킨다 — 다른 ETF 로 교체했을 때 상한이 조용히
    틀린 채로 돌아가는 것을 막는다.
    """

    cap = round(1.0 - min_cash, 4)
    caps: dict[str, float] = {}
    for symbol in symbols:
        ref = ETF_REF.get(symbol)
        if ref is None:
            continue
        lookthrough = cap * ref.top_weight
        if lookthrough > equity_cap:
            raise ValueError(
                f"etf_cap_violation symbol={symbol} cap={cap} top_weight={ref.top_weight} "
                f"lookthrough={lookthrough:.4f} equity_cap={equity_cap}"
            )
        caps[symbol] = cap
    return caps


def universe_meta(universe: list[str], tradable: list[str]) -> dict[str, dict]:
    """종목별 자산군·집행 가능 여부·ETF 구조 정보 — 프롬프트 주입용."""

    tradable_set = set(tradable)
    out: dict[str, dict] = {}
    for symbol in universe:
        block: dict = {
            "asset_class": asset_class(symbol),
            "tradable": symbol in tradable_set,
        }
        ref = ETF_REF.get(symbol)
        if ref is not None:
            block |= {
                "index": ref.index,
                "replication": ref.replication,
                "expense_ratio": ref.expense_ratio,
                "holdings": ref.holdings,
                "top_weight": ref.top_weight,
            }
        out[symbol] = block
    return out
