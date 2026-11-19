"""개장 기준 실행 게이트 — '몇 시에'가 아니라 '개장 뒤 얼마 만에'.

일간 스텝이 겨냥하는 것은 벽시계 시각이 아니라 개장 직후의 변동 구간을 지난 시점이다.
스케줄러는 벽시계밖에 모르므로, 잡은 평소 시각에 뜨고 이 게이트가 그날의 실제 개장에
맞춰 필요한 만큼만 기다린다. 평소에는 기다림이 0 이다.

기준은 실측(그날 지수의 첫 체결)이 먼저고, 실측이 아직 없을 때만 예고(브로커가 알려 준
운영 시각)를 쓴다. 예고는 장이 열리기 전에 '얼마나 기다려야 하는지'를 알려 주는 용도다.

게이트는 주문을 막지 않는다 — 늦출 뿐이다. 조회가 전부 실패하면 기다리지 않고 종전대로
진행하고, 그 사실을 기록에 남긴다. 게이트 고장으로 그날의 결정이 사라지는 쪽이 개장
직후에 한 번 매매하는 쪽보다 비싸다.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from adapters.session import (
    TradingSession,
    default_session,
    load_session,
    local_day,
    save_session,
)

# 개장 뒤 이만큼 지나서 실행한다. 평소 09:00 개장 · 10:00 실행과 같은 간격이다.
OPEN_OFFSET = timedelta(minutes=60)
# 예고된 개장 시각이 지났는데 체결이 없을 때 '오늘은 안 열린다'고 보기까지의 여유.
NO_PRINT_GRACE = timedelta(minutes=5)
# 이보다 오래 기다려야 하면 기다리지 않는다 — 새벽의 수동 실행을 장 시작까지 붙잡아 두지
# 않기 위한 상한. 실제 순연(1시간)은 넉넉히 들어온다.
MAX_WAIT = timedelta(hours=2)
POLL_S = 60.0


def session_path(state_dir: Path, market: str) -> Path:
    return Path(state_dir) / f"session_{market}.json"


async def resolve_session(
    adapter: object, market: str, state_dir: Path, now: datetime
) -> TradingSession | None:
    """오늘의 정규장 — 그날 조회해 둔 것 → 브로커 조회 → 고정 시각표 순.

    정규장이 없는 시장은 None. 조회에 성공한 것만 저장한다 — 실패를 저장하면 아침의
    일시 장애 하나가 그날 종일 고정 시각표를 강제한다.
    """
    day = local_day(market, now)
    if day is None:
        return None
    cached = load_session(session_path(state_dir, market), market, day)
    if cached is not None:
        return cached
    fetch = getattr(adapter, "trading_session", None)
    if fetch is not None:
        try:
            session = await fetch(day)
        except Exception as e:  # 조회 실패는 고정 시각표로 흡수한다(게이트는 늦출 뿐)
            print(
                f"market={market} status=warn event=session_lookup_failed"
                f" error={type(e).__name__}: {e}"
            )
        else:
            if session is not None:
                save_session(session_path(state_dir, market), session)
                return session
    return default_session(market, day)


async def _first_print(adapter: object, session: TradingSession) -> datetime | None:
    fetch = getattr(adapter, "first_trade_time", None)
    if fetch is None:
        return None
    try:
        first = await fetch(session.day)
    except Exception as e:
        print(
            f"market={session.market} status=warn event=first_print_failed"
            f" error={type(e).__name__}: {e}"
        )
        return None
    if first is None:
        return None
    return datetime.combine(session.day, first, tzinfo=ZoneInfo(session.tz))


async def wait_for_open_offset(
    adapter: object,
    session: TradingSession,
    *,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict:
    """개장 + OPEN_OFFSET 까지 기다리고, 무엇을 근거로 얼마나 기다렸는지 돌려준다.

    status: open(기다림 없음) · waited · after_close · too_early(상한 초과로 대기 안 함).
    basis: first_print(실측) · announced(실측 없음 — 휴장일이거나 조회 실패).

    남은 시간을 한 번에 자지 않고 짧게 끊어 벽시계를 다시 본다 — 기기가 잠들면 단조
    시계가 멈춰서, 한 번에 잔 프로세스는 깨어난 뒤에도 잠든 시간만큼 더 잔다.
    """
    started = now_fn()
    record = {
        **session.to_record(),
        "shifted": session.shifted,
        "offset_min": OPEN_OFFSET // timedelta(minutes=1),
    }
    slept = False

    async def sleep(seconds: float) -> None:
        nonlocal slept
        slept = True
        await sleep_fn(seconds)

    def done(status: str, **extra: object) -> dict:
        waited_s = round((now_fn() - started).total_seconds())
        return {**record, "status": status, "waited_s": waited_s, **extra}

    if started > session.close_at:
        return done("after_close")
    if session.open_at + OPEN_OFFSET - started > MAX_WAIT:
        return done("too_early")

    # 실측. 개장 전이면 예고된 개장 시각을 조금 지날 때까지만 첫 체결을 기다린다.
    no_print_deadline = max(session.open_at, started) + NO_PRINT_GRACE
    first = await _first_print(adapter, session)
    while first is None and now_fn() < no_print_deadline:
        await sleep(POLL_S)
        first = await _first_print(adapter, session)

    basis = "first_print" if first is not None else "announced"
    target = (first or session.open_at) + OPEN_OFFSET
    extra = {
        "basis": basis,
        "first_print": first.strftime("%H:%M") if first else None,
        "target": target.strftime("%H:%M"),
    }
    if target - now_fn() > MAX_WAIT:
        return done("too_early", **extra)
    while (remaining := (target - now_fn()).total_seconds()) > 0:
        await sleep(min(POLL_S, remaining))
    return done("waited" if slept else "open", **extra)
