"""단일 인스턴스 락 — 같은 잡의 동시 실행 차단 (실계좌 이중 주문 방지) + 실행중 표식.

launchd 는 wake 시 놓친 잡을 catch-up 실행하고, 인터벌 잡은 이전 런이 느리면 다음
틱과 겹칠 수 있다. 두 프로세스가 같은 계좌에 붙으면 주문이 중복되거나 상태 파일
(risk_*·live_notional_*)의 read-modify-write 가 레이스로 유실된다. POSIX 파일 락으로
한 번에 하나만 돌게 강제한다 — 논블로킹이라 이미 잡혀 있으면 즉시 실패(대기 X).

락은 프로세스 종료 시 커널이 자동 해제하므로 크래시·kill 후에도 stale 락이 남지 않는다
(pidfile 방식의 고질적 문제 회피). darwin/Linux 공용(fcntl).

락 파일 본문에는 실행 표식(pid·시작시각·라벨)을 남긴다. 상호배제의 근거는 어디까지나
flock 이고 이 내용은 **읽기 전용 정보**다 — 대시보드가 "지금 무엇이 돌고 있는지"를
락을 건드리지 않고 알기 위한 것. flock 을 비파괴적으로 떠보는 방법이 없어서
(테스트 삼아 잡는 순간 진짜 런의 획득이 실패한다) 별도 표식이 필요하다.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import IO


def single_instance(lock_path: Path, label: str | None = None) -> IO | None:
    """배타적 논블로킹 파일 락 획득.

    성공 시 열린 파일 객체를 반환한다 — 락은 이 객체가 열려 있는 동안 유지되므로
    프로세스 생존 동안 참조를 잡아둘 것(close() 하면 즉시 해제). 이미 다른 프로세스가
    잡고 있으면 None. label 은 실행 표식에 남는 사람이 읽을 이름이다.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "label": label or lock_path.stem,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    )
    fh.flush()
    return fh


def market_locks(state_dir: Path, markets: list[str], label: str) -> list[IO] | None:
    """시장(계좌) 단위 락 — 같은 계좌를 건드리는 **서로 다른 잡** 사이의 상호 배제.

    잡 이름으로 락을 잡으면 자기 중첩만 막고 교차 중첩은 못 막는다. 실제로 위험한 것은
    후자다 — 15분 워처와 일일 스텝은 다른 잡이지만 같은 시장의 같은 계좌에 주문하고
    같은 `risk_{market}.json`·`live_notional_{market}.json` 을 read-modify-write 한다.
    락 키를 잡이 아니라 시장으로 두면 누가 먼저 잡든 그 계좌에는 한 번에 하나만 붙는다.

    all-or-nothing: 하나라도 이미 잡혀 있으면 취득분을 전부 반납하고 None 을 돌려준다.
    부분 취득으로 진행하면 요청한 시장 중 일부만 매매하는 절반짜리 런이 된다.
    """
    acquired: list[IO] = []
    for market in sorted(markets):
        fh = single_instance(state_dir / f"account_{market}.lock", label=f"{label} {market}")
        if fh is None:
            for held in acquired:
                held.close()
            return None
        acquired.append(fh)
    return acquired


def read_run_marker(lock_path: Path) -> dict | None:
    """락 파일의 실행 표식을 읽어 **지금 살아 있는** 런만 반환. 락은 건드리지 않는다.

    본문은 프로세스가 죽어도 파일에 남으므로(커널이 푸는 것은 flock 뿐) 판정 기준은
    pid 생존이다. 종료된 pid 가 재사용돼 살아 있는 것처럼 보일 수 있으나, 그 경우의
    결과는 표시가 한 줄 더 뜨는 것뿐이라 started_at 을 함께 노출해 분간할 수 있게 한다.
    """
    try:
        marker = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(marker["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # 다른 사용자 소유 = 살아 있음
    return marker
