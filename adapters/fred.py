"""FRED 매크로 데이터 — 무료 일간 시계열 (regime 보조 입력, 알파 예측 아님).

각 시리즈의 최근 관측을 상한 t−1 로 잘라 최신 유효값만 가져온다(same-day 차단). 용도는
regime 분류·리스크 게이트·FX 컨텍스트 — 설계 데이터 정책상 지표는 알파 원천이 아니라 보조.
개별 시리즈 조회 실패는 비치명(None) — 관측 보조라 fail-open.
"""

from __future__ import annotations

from datetime import date, timedelta

import httpx

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# 라벨 → FRED series_id. 전부 일간·무료.
MACRO_SERIES = {
    "vix": "VIXCLS",           # CBOE 변동성지수 (주 스트레스 게이지)
    "yield_spread": "T10Y2Y",  # 10년-2년 국채 스프레드 (음수=침체 경고)
    "fed_funds": "DFF",        # 실효 연방기금금리
    "dollar": "DTWEXBGS",      # 무역가중 달러지수 (광의)
    "usdkrw": "DEXKOUS",       # 원/달러 환율
}


def _parse_observations(observations: list[dict]) -> list[tuple[date, float]]:
    """관측 리스트 → (날짜, 값) 오름차순. 결측('.')·파싱 불가 행은 버린다."""
    out: list[tuple[date, float]] = []
    for obs in observations:
        v = obs.get("value")
        if not v or v == ".":
            continue
        try:
            out.append((date.fromisoformat(obs["date"]), float(v)))
        except (KeyError, ValueError):
            continue
    return sorted(out)


def _latest_valid(observations: list[dict]) -> float | None:
    """관측 리스트(내림차순)에서 최신 유효값. FRED 결측은 '.' 문자로 온다."""
    for obs in observations:
        v = obs.get("value")
        if v and v != ".":
            try:
                return float(v)
            except ValueError:
                continue
    return None


async def fetch_fred_latest(
    api_key: str,
    asof_day: date,
    series: dict[str, str] = MACRO_SERIES,
    lookback_days: int = 30,
) -> dict[str, float | None]:
    """각 시리즈의 t−1 상한 최신값. 상한 = asof_day−1(당일 데이터 차단)."""
    end = asof_day - timedelta(days=1)
    start = asof_day - timedelta(days=lookback_days)
    out: dict[str, float | None] = {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        for label, sid in series.items():
            try:
                resp = await client.get(
                    FRED_BASE,
                    params={
                        "series_id": sid,
                        "api_key": api_key,
                        "file_type": "json",
                        "observation_start": start.isoformat(),
                        "observation_end": end.isoformat(),
                        "sort_order": "desc",
                        "limit": 10,
                    },
                )
                resp.raise_for_status()
                out[label] = _latest_valid(resp.json().get("observations") or [])
            except Exception:
                out[label] = None
    return out


async def fetch_fred_history(
    api_key: str, series_id: str, start: date, end: date
) -> list[tuple[date, float]]:
    """한 시리즈의 [start, end] 일간 관측 (오름차순). 조회 실패 → [] (fail-open).

    `fetch_fred_latest` 가 최신 한 점만 보는 것과 달리 구간 전체를 돌려준다 — 곡선을
    그리려면 점이 아니라 시계열이 필요하다. 상한은 호출부가 t−1 로 잘라 넘긴다.
    """
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get(
                FRED_BASE,
                params={
                    "series_id": series_id,
                    "api_key": api_key,
                    "file_type": "json",
                    "observation_start": start.isoformat(),
                    "observation_end": end.isoformat(),
                    "sort_order": "asc",
                },
            )
            resp.raise_for_status()
            return _parse_observations(resp.json().get("observations") or [])
        except Exception:
            return []
