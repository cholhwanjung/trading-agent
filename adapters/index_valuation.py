"""시장 밸류에이션 관측 — 지수 ETF 의 재무 대체 채널.

지수 ETF 에는 PER·ROE 가 없다. 수집에 실패한 것이 아니라 적용 대상이 아니다. 그런데
결정 원칙은 안전마진을 요구하고, 소액 계좌에서 담을 수 있는 것이 ETF 뿐이 되면
밸류에이션 근거가 통째로 비어버린다. 구성종목 재무에서 유추하라는 지시만으로는
관측에 grounded 되지 않는다 — 그 유추의 기준선이 관측에 없기 때문이다.

**대리(proxy) 관측이다.** 보유 ETF 자신의 지수(KOSPI 200 · Dow Jones U.S. Large-Cap)의
PER·PBR 을 무료·키 없이 일간으로 주는 곳을 찾지 못했다(한국거래소 정보데이터시스템은
브라우저 세션을 요구하고, 운용사 페이지는 값을 자바스크립트로 그린다). 대신 같은 시장의
대표 지수를 담은 미국 상장 펀드가 서버에서 그려 내보내는 포트폴리오 특성값을 쓴다.
지수가 다르므로 값은 **수준과 방향**을 읽는 용도이지 그 ETF 의 정확한 배수가 아니며,
payload 에 proxy 티커와 지수명을 함께 실어 그 사실을 감추지 않는다.

누출 통제: 펀드 페이지가 함께 게시하는 **종가 기준일**을 이 값의 기준일로 쓴다. 여기의
PER 은 "최근 종가 / 최근 회계연도 EPS" 라 가격 기준일이 곧 값의 기준일이다. 기준일을
읽지 못하면 비율이 멀쩡해도 값을 버린다 — 얼마나 묵었는지 모르는 밸류에이션은
없는 것보다 나쁘다.

같은 관측일에 여러 번 조회하지 않는다. 페이지가 수 MB 라 15분 간격 워처가 그대로
받으면 하루 수십 MB 를 같은 값 때문에 다시 받게 된다. 관측일 단위 파일 캐시로 막는다.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import httpx

TIMEOUT = 20.0

#: 캐시 위치. 잡은 리포지토리 루트에서 실행되므로 상대 경로로 충분하다.
#: 테스트·다른 실행 위치에서는 이 값을 바꿔 넣는다(호출 시점에 읽는다).
CACHE_DIR = Path("data/state")


@dataclass(frozen=True)
class IndexValuation:
    """한 시장의 밸류에이션 수준. proxy·index 는 값의 출처를 드러내는 필수 라벨이다."""

    proxy: str          # 값을 공개한 상품 티커
    index: str          # 그 상품이 추종하는 지수
    asof: date          # 값의 기준일(=종가 기준일)
    pe: float | None
    pb: float | None
    dividend_yield: float | None  # 12개월 추적 분배율, 비율(0.0109 = 1.09%)


@dataclass(frozen=True)
class _Proxy:
    ticker: str
    index: str
    url: str


#: 시장 → 대리 상품. 대응물이 없는 시장(크립토)은 여기 없고 채널이 통째로 빠진다.
#: 지수가 보유 ETF 의 것과 다르다는 점은 payload 라벨로 노출된다.
MARKET_PROXY: dict[str, _Proxy] = {
    "KR": _Proxy(
        "EWY", "MSCI Korea",
        "https://www.ishares.com/us/products/239681/ishares-msci-south-korea-capped-etf",
    ),
    "US": _Proxy(
        "IVV", "S&P 500",
        "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf",
    ),
}

#: 페이지에서 값을 들고 있는 속성. 라벨 엘리먼트는 다른 속성(data-id)을 쓰므로
#: 이 속성으로 앵커하면 라벨 텍스트를 값으로 잘못 집는 일이 없다.
_PE = "fundamentalsAndRisk-priceEarnings"
_PB = "fundamentalsAndRisk-priceBook"
_YIELD = "fundamentalsAndRisk-twelveMonTrlYld"
_ASOF = "keyFundFacts-closingPrice-asOf"


def _datapoint(html: str, key: str) -> str | None:
    m = re.search(r'webqc-datapoint="%s"[^>]*>([^<]{0,40})<' % re.escape(key), html)
    return m.group(1).strip() if m else None


def _number(raw: str | None) -> float | None:
    """'30.74' · '1.09%' · '1,695' → float. 퍼센트는 비율로 환산한다."""
    if not raw:
        return None
    percent = raw.endswith("%")
    try:
        value = float(raw.rstrip("%").replace(",", "").replace("$", ""))
    except ValueError:
        return None
    return round(value / 100, 4) if percent else value


def _asof_date(raw: str | None) -> date | None:
    """'as of Aug 11, 2026' → date. 형식이 바뀌면 None(=값 폐기)."""
    if not raw:
        return None
    m = re.search(r"([A-Z][a-z]{2}) (\d{1,2}), (\d{4})", raw)
    if not m:
        return None
    try:
        return datetime.strptime(" ".join(m.groups()), "%b %d %Y").date()
    except ValueError:
        return None


def parse_index_valuation(html: str, proxy: _Proxy) -> IndexValuation | None:
    """펀드 페이지 → 밸류에이션. 기준일이 없거나 비율이 전부 비면 None.

    비율 하나가 빠지는 것은 허용한다(그 항목만 비운다). 기준일 부재는 허용하지 않는다.
    """
    asof = _asof_date(_datapoint(html, _ASOF))
    if asof is None:
        return None
    pe, pb, dy = (_number(_datapoint(html, k)) for k in (_PE, _PB, _YIELD))
    if pe is None and pb is None and dy is None:
        return None
    return IndexValuation(proxy.ticker, proxy.index, asof, pe, pb, dy)


def _cache_path(market: str) -> Path:
    return CACHE_DIR / f"index_valuation_{market}.json"


def _load_cache(market: str, asof_day: date) -> IndexValuation | None:
    """관측일이 일치하는 캐시만 유효. 하루가 지나면 그대로 무효가 된다."""
    try:
        data = json.loads(_cache_path(market).read_text())
    except (OSError, ValueError):
        return None
    if data.get("asof_day") != str(asof_day):
        return None
    value = data.get("value") or {}
    try:
        return IndexValuation(
            value["proxy"], value["index"], date.fromisoformat(value["asof"]),
            value["pe"], value["pb"], value["dividend_yield"],
        )
    except (KeyError, TypeError, ValueError):
        return None


def _store_cache(market: str, asof_day: date, value: IndexValuation) -> None:
    payload = {**asdict(value), "asof": str(value.asof)}
    path = _cache_path(market)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"asof_day": str(asof_day), "value": payload}, ensure_ascii=False)
        )
    except OSError:
        pass  # 캐시는 대역폭 절약 수단일 뿐 — 못 써도 관측 자체는 성립한다


async def fetch_index_valuation(
    market: str, asof_day: date, *, client: httpx.AsyncClient | None = None
) -> IndexValuation | None:
    """시장 밸류에이션 1건. 대리 상품이 없는 시장·조회 실패·기준일이 t 이상이면 None.

    조회 실패를 예외로 올리지 않는다 — 관측 보조 채널이라 원천 하나 때문에 결정
    경로를 막지 않는다. 대신 결정 로그에 블록 유무가 남아 채널이 죽은 것을 알 수 있다.
    """
    proxy = MARKET_PROXY.get(market)
    if proxy is None:
        return None
    cached = _load_cache(market, asof_day)
    if cached is not None:
        return cached

    own = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True)
    try:
        resp = await client.get(proxy.url)
        resp.raise_for_status()
        value = parse_index_valuation(resp.text, proxy)
    except (httpx.HTTPError, ValueError):
        return None
    finally:
        if own:
            await client.aclose()

    # 상한 t-1. 페이지는 최신 종가 기준이라 정상적으로는 항상 만족하지만, 장중에
    # 당일 종가가 먼저 반영되는 경우를 관측이 조용히 받아들이면 안 된다.
    if value is None or value.asof >= asof_day:
        return None
    _store_cache(market, asof_day, value)
    return value
