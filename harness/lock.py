"""단일 인스턴스 락 — 같은 잡의 동시 실행 차단 (실계좌 이중 주문 방지).

launchd 는 wake 시 놓친 잡을 catch-up 실행하고, 인터벌 잡은 이전 런이 느리면 다음
틱과 겹칠 수 있다. 두 프로세스가 같은 계좌에 붙으면 주문이 중복되거나 상태 파일
(risk_*·live_notional_*)의 read-modify-write 가 레이스로 유실된다. POSIX 파일 락으로
한 번에 하나만 돌게 강제한다 — 논블로킹이라 이미 잡혀 있으면 즉시 실패(대기 X).

락은 프로세스 종료 시 커널이 자동 해제하므로 크래시·kill 후에도 stale 락이 남지 않는다
(pidfile 방식의 고질적 문제 회피). darwin/Linux 공용(fcntl).
"""

from __future__ import annotations

import fcntl
from pathlib import Path
from typing import IO


def single_instance(lock_path: Path) -> IO | None:
    """배타적 논블로킹 파일 락 획득.

    성공 시 열린 파일 객체를 반환한다 — 락은 이 객체가 열려 있는 동안 유지되므로
    프로세스 생존 동안 참조를 잡아둘 것(close() 하면 즉시 해제). 이미 다른 프로세스가
    잡고 있으면 None.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh
