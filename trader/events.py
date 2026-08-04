"""거시 이벤트 캘린더 — 사전 공지된 정책 일정을 결정 컨텍스트로 제공.

미래 "일정"은 공개 정보라 관측 상한(전일까지)과 충돌하지 않는다 — 결과·가격이 아니라
"언제 무엇이 예정돼 있는가"만 담아, 이벤트 직전 노출 조절을 에이전트가 스스로 추론하게 한다.
연 1회 공표 원천: Fed FOMC 연간 캘린더 · 한국은행 금통위 정기회의 일정. 연말에 이듬해분 갱신.
"""

from __future__ import annotations

from datetime import date, timedelta

_ALL = frozenset({"KR", "US", "CRYPTO"})
_KR = frozenset({"KR"})

# (일자, 라벨, 영향 시장) — 이틀 회의는 결과 발표일(둘째 날) 기준. 금리는 전 시장 영향.
SCHEDULED_EVENTS: tuple[tuple[date, str, frozenset[str]], ...] = (
    (date(2026, 1, 15), "한국은행 금통위 기준금리 결정", _KR),
    (date(2026, 1, 28), "FOMC 금리 결정", _ALL),
    (date(2026, 2, 26), "한국은행 금통위 기준금리 결정", _KR),
    (date(2026, 3, 18), "FOMC 금리 결정", _ALL),
    (date(2026, 4, 10), "한국은행 금통위 기준금리 결정", _KR),
    (date(2026, 4, 29), "FOMC 금리 결정", _ALL),
    (date(2026, 5, 28), "한국은행 금통위 기준금리 결정", _KR),
    (date(2026, 6, 17), "FOMC 금리 결정", _ALL),
    (date(2026, 7, 16), "한국은행 금통위 기준금리 결정", _KR),
    (date(2026, 7, 29), "FOMC 금리 결정", _ALL),
    (date(2026, 8, 27), "한국은행 금통위 기준금리 결정", _KR),
    (date(2026, 8, 27), "잭슨홀 심포지엄 개막(~08-29, 연준 의장 연설)", _ALL),
    (date(2026, 9, 16), "FOMC 금리 결정", _ALL),
    (date(2026, 10, 22), "한국은행 금통위 기준금리 결정", _KR),
    (date(2026, 10, 28), "FOMC 금리 결정", _ALL),
    (date(2026, 11, 26), "한국은행 금통위 기준금리 결정", _KR),
    (date(2026, 12, 9), "FOMC 금리 결정", _ALL),
)


def upcoming_events(market: str, asof_day: date, horizon_days: int = 3) -> list[dict]:
    """[asof_day, asof_day+horizon] 예정 이벤트 — 날짜순 [{day, days_until, event}].

    days_until 0 = 오늘(결과 발표 전 결정 시점일 수 있음). 지난 이벤트는 반환하지
    않는다 — 결과 해석은 뉴스·봉 채널의 몫. 빈 리스트 = 창 내 예정 없음.
    """
    end = asof_day + timedelta(days=horizon_days)
    out = [
        {"day": d.isoformat(), "days_until": (d - asof_day).days, "event": label}
        for d, label, markets in SCHEDULED_EVENTS
        if market in markets and asof_day <= d <= end
    ]
    return sorted(out, key=lambda e: e["day"])
