"""네트워크 준비 게이트 — 스케줄 잡의 wake/부팅 직후 조기 실행 방지.

launchd 는 전원이 꺼져/슬립이라 놓친 캘린더 잡을 깨어난 즉시 발동한다. 이때 네트워크
스택(Wi-Fi 재연결·DHCP·DNS)이 아직 안 올라온 상태면 브로커/거래소 호출이 DNS 실패로
즉시 죽는다. 이 게이트는 DNS 해석 + TCP 연결이 될 때까지 대기해 그 창을 흡수한다.

특정 API 도달성은 확인하지 않는다(그건 각 호출의 재시도가 담당) — DNS+TCP 가 되는지만
본다. 같은 거래일 안의 지연이므로 관측 윈도우·누출과 무관하다.
"""

from __future__ import annotations

import asyncio

# DNS 해석 + TCP 443 을 함께 시험하는 안정적 호스트(내용 무관, 도달성만 확인).
DEFAULT_HOSTS = ("one.one.one.one", "dns.google")


async def _reachable(host: str, port: int, timeout: float) -> bool:
    """host:port 로 TCP 연결이 서면 True. DNS 실패·연결 거부·타임아웃은 False."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return True


async def wait_for_network(
    hosts: tuple[str, ...] = DEFAULT_HOSTS,
    *,
    timeout_s: float = 600.0,
    interval_s: float = 15.0,
    connect_timeout_s: float = 5.0,
    port: int = 443,
) -> bool:
    """네트워크(DNS+TCP)가 될 때까지 최대 timeout_s 동안 interval_s 간격으로 폴링.

    hosts 중 하나라도 도달하면 즉시 True. deadline 까지 못 서면 False(호출자가 판단).
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while True:
        for host in hosts:
            if await _reachable(host, port, connect_timeout_s):
                return True
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(interval_s, remaining))
