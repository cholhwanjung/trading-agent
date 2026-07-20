"""컨테이너용 통합 스케줄러 — 일일/주간/월간 잡을 KST 기준으로 실행.

    uv run python scripts/scheduler.py

호스트에서는 launchd 가 같은 역할을 한다 — **둘을 동시에 켜지 말 것**(이중 실행).
잡 실패는 다음 주기에 재시도될 뿐 스케줄러를 죽이지 않는다. 로그는 key=value.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True)
class Job:
    name: str
    script: str
    hour: int
    minute: int = 0
    weekday: int | None = None  # 0=월 .. 6=일 (None = 매일)
    monthday: int | None = None  # 1..31 (None = 매일)
    args: tuple[str, ...] = ()


JOBS = [
    # KR 은 장중(10:00 KST) 시장가 주문이 필요해 별도 잡 — 23:00 잡은 KR 제외
    Job("paper_step_kr", "scripts/run_paper_step.py", hour=10, args=("--markets", "KR")),
    Job("paper_step", "scripts/run_paper_step.py", hour=23, args=("--markets", "CRYPTO,US")),
    Job("alpha_lab", "scripts/run_alpha_lab.py", hour=22, weekday=6),  # 일요일
    Job("monthly_proposal", "scripts/propose_improvements.py", hour=21, monthday=1),
]


def next_run_at(job: Job, now: datetime) -> datetime:
    """now(tz-aware) 이후 첫 실행 시각. 순수 함수 — 테스트 대상."""
    candidate = now.replace(hour=job.hour, minute=job.minute, second=0, microsecond=0)
    for _ in range(370):  # 최악(월 1일 잡)도 1년 내 반드시 존재
        ok = candidate > now
        if job.weekday is not None:
            ok = ok and candidate.weekday() == job.weekday
        if job.monthday is not None:
            ok = ok and candidate.day == job.monthday
        if ok:
            return candidate
        candidate += timedelta(days=1)
    raise AssertionError("unreachable")


def run_job(job: Job) -> int:
    started = datetime.now(KST).isoformat()
    proc = subprocess.run(
        [sys.executable, str(ROOT / job.script), *job.args],
        cwd=ROOT, capture_output=True, text=True,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    print(f"job={job.name} started={started} exit={proc.returncode} tail={tail}", flush=True)
    return proc.returncode


def main() -> None:
    print(f"scheduler_start jobs={[j.name for j in JOBS]} tz=Asia/Seoul", flush=True)
    while True:
        now = datetime.now(KST)
        upcoming = sorted((next_run_at(j, now), j) for j in JOBS)
        when, job = upcoming[0]
        wait = (when - now).total_seconds()
        print(f"next job={job.name} at={when.isoformat()} wait_s={int(wait)}", flush=True)
        time.sleep(max(1.0, wait))
        try:
            run_job(job)
        except Exception as e:  # 잡 실패가 스케줄러를 죽이면 안 된다
            print(f"job={job.name} error={type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
