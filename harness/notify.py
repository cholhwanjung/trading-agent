"""실패 알림 — 무인 운용 중 사람이 개입해야 하는 사건을 외부 채널로 push.

ALERT_WEBHOOK_URL(.env) 로 JSON POST. Slack/Discord incoming webhook 을 함께 만족하도록
text·content 두 키를 모두 실어 보낸다(각 서비스는 자기 키만 읽고 나머지는 무시).
미설정이면 조용히 생략(사건은 구조화 로그에 이미 남는다). 알림 전송 실패가 매매 런을
죽이면 안 되므로 전부 best-effort — 예외를 삼키고 False 를 돌려준다.
"""

from __future__ import annotations

import httpx


async def notify(env: dict[str, str], title: str, body: str) -> bool:
    """웹훅으로 알림 전송. 전송 성공 True, 미설정/실패 False."""
    url = env.get("ALERT_WEBHOOK_URL")
    if not url:
        return False
    msg = f"[{title}] {body}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json={"text": msg, "content": msg})
            return resp.status_code < 400
    except Exception:
        return False
