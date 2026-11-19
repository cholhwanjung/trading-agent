"""오늘의 정규장 — 고정 시각표가 아니라 그날 실제로 운영되는 시각.

거래소는 정규장 시각을 드물게 옮긴다(대학수학능력시험일의 1시간 순연, 연초 첫 거래일의
지연 개장). 실행 스케줄은 벽시계 고정이라, '개장 1시간 뒤'를 겨냥한 잡이 그날은 개장
직후에 주문을 내고 장중 감시는 마지막 1시간을 비운다. 브로커는 두 경우 모두 주문을
받아 주므로 어긋남이 어디에도 드러나지 않는다.

그래서 시각을 두 군데서 얻는다.
- **예고**: 브로커가 그날 아침 알려 주는 운영 시각. 장이 열리기 전에 알 수 있고 폐장
  시각까지 준다. 다만 해석이 한 단계 끼어 있다(아래 파생상품시장 시각 → 주식).
- **실측**: 지수 분봉의 그날 첫 봉. 해석이 없어 예고가 틀려도 맞지만, 장이 열린 뒤에만
  알 수 있고 폐장 시각은 주지 않는다.

달력 파일은 두지 않는다 — 사람이 해마다 채워야 하는 상태가 되고, 전날 저녁에 바뀌는
일정은 어차피 담지 못한다.

이 값들은 **실행 시각을 정하는 데만** 쓴다. 당일 데이터이므로 결정의 관측에는 넣지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from adapters.base import REGULAR_SESSIONS

DEFAULT_SOURCE = "default"

# 예고 시각이 고정 시각표에서 이보다 멀면 해석을 믿지 않고 고정 시각표로 돌아간다.
# 실제 순연은 1시간 안팎이라, 그보다 훨씬 큰 값은 응답 형식이 바뀌었다는 뜻이다.
MAX_SESSION_SHIFT = timedelta(hours=3)

# KIS 영업일조회가 돌려주는 것은 주식이 아니라 파생상품시장의 운영 시각이다(평소
# 08:45~15:45). 주식 정규장은 이 표준 시각에서 밀린 만큼 같이 밀린다고 해석한다.
# 수능일(09:45~16:45 → 주식 10:00~16:30)에 정확히 맞는 것을 실제 응답으로 확인했다.
# 연초 첫 거래일처럼 두 시장이 같은 시각에 여는 날은 개장을 15분 늦게 읽는다 —
# 이른 쪽이 아니라 늦은 쪽으로 틀리고, 그날도 실측이 바로잡는다.
KR_DERIVATIVES_REGULAR = (time(8, 45), time(15, 45))


@dataclass(frozen=True)
class TradingSession:
    market: str
    day: date  # 시장 현지 날짜
    tz: str
    open: time
    close: time
    source: str  # DEFAULT_SOURCE = 고정 시각표, 그 외 = 조회 출처

    @property
    def open_at(self) -> datetime:
        return datetime.combine(self.day, self.open, tzinfo=ZoneInfo(self.tz))

    @property
    def close_at(self) -> datetime:
        return datetime.combine(self.day, self.close, tzinfo=ZoneInfo(self.tz))

    @property
    def shifted(self) -> bool:
        """고정 시각표와 다른가 — 사람이 알아야 하는 날."""
        _, open_t, close_t = REGULAR_SESSIONS[self.market]
        return (self.open, self.close) != (open_t, close_t)

    def contains(self, now: datetime) -> bool:
        """now 가 이 날의 정규장 안인가. 주말은 항상 밖이다."""
        local = _aware(now).astimezone(ZoneInfo(self.tz))
        return (
            local.date() == self.day
            and local.weekday() < 5
            and self.open <= local.time() <= self.close
        )

    def to_record(self) -> dict:
        return {
            "market": self.market,
            "day": self.day.isoformat(),
            "tz": self.tz,
            "open": self.open.strftime("%H:%M"),
            "close": self.close.strftime("%H:%M"),
            "source": self.source,
        }


def _aware(now: datetime) -> datetime:
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def local_day(market: str, now: datetime) -> date | None:
    """그 시장의 현지 날짜. 정규장이 없는 시장(24/7)은 None."""
    session = REGULAR_SESSIONS.get(market)
    if session is None:
        return None
    return _aware(now).astimezone(ZoneInfo(session[0])).date()


def default_session(market: str, day: date) -> TradingSession | None:
    """고정 시각표의 정규장. 정규장이 없는 시장은 None."""
    session = REGULAR_SESSIONS.get(market)
    if session is None:
        return None
    tz, open_t, close_t = session
    return TradingSession(market, day, tz, open_t, close_t, DEFAULT_SOURCE)


def _hhmmss(value: object) -> time | None:
    text = str(value or "")
    if len(text) != 6 or not text.isdigit():
        return None
    hour, minute, second = int(text[:2]), int(text[2:4]), int(text[4:])
    if hour > 23 or minute > 59 or second > 59:
        return None
    return time(hour, minute, second)


def _shift(actual: time, regular: time) -> timedelta:
    base = date(2000, 1, 1)
    return datetime.combine(base, actual) - datetime.combine(base, regular)


def kr_session_from_market_time(output: dict, day: date) -> TradingSession | None:
    """KIS 영업일조회 응답 → 그날의 주식 정규장. 해석할 수 없으면 None.

    None 은 '고정 시각표를 쓰라'는 뜻이다. 응답의 날짜가 다르거나(자정 전후·휴장일),
    시각이 읽히지 않거나, 표준보다 이르거나 터무니없이 늦으면 추측하지 않는다.
    """
    if str(output.get("today") or "") != day.strftime("%Y%m%d"):
        return None
    started, ended = _hhmmss(output.get("s_time")), _hhmmss(output.get("e_time"))
    if started is None or ended is None:
        return None
    open_shift = _shift(started, KR_DERIVATIVES_REGULAR[0])
    close_shift = _shift(ended, KR_DERIVATIVES_REGULAR[1])
    zero = timedelta(0)
    if not (zero <= open_shift <= MAX_SESSION_SHIFT and zero <= close_shift <= MAX_SESSION_SHIFT):
        return None
    tz, open_t, close_t = REGULAR_SESSIONS["KR"]
    base = date(2000, 1, 1)
    return TradingSession(
        "KR",
        day,
        tz,
        (datetime.combine(base, open_t) + open_shift).time(),
        (datetime.combine(base, close_t) + close_shift).time(),
        "kis_market_time",
    )


def first_print_time(rows: list[dict], day: date) -> time | None:
    """지수 분봉 행에서 그날 첫 봉의 시작 시각. 아직 체결이 없으면 None.

    봉의 시각 칸에는 실제 시각이 아닌 표식(장 마감·시간외 합계 행)이 섞여 온다 —
    시각으로 읽히지 않는 값은 버린다.
    """
    stamp = day.strftime("%Y%m%d")
    times = [
        t
        for row in rows
        if str(row.get("stck_bsop_date") or "") == stamp
        and (t := _hhmmss(row.get("stck_cntg_hour"))) is not None
    ]
    return min(times) if times else None


def load_session(path: Path, market: str, day: date) -> TradingSession | None:
    """그날 이미 조회해 둔 정규장. 없거나 다른 날·다른 시장 것이면 None."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw["market"] != market or raw["day"] != day.isoformat():
            return None
        return TradingSession(
            market,
            day,
            raw["tz"],
            time.fromisoformat(raw["open"]),
            time.fromisoformat(raw["close"]),
            raw["source"],
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_session(path: Path, session: TradingSession) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session.to_record(), ensure_ascii=False), encoding="utf-8")
