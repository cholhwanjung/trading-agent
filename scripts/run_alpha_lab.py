"""Alpha Lab 1사이클 — writer→judge→백테스트→admission.

사용법:
    uv run python scripts/run_alpha_lab.py [--n 5]

- 시장: CRYPTO 만 (선봉 — 연구 유니버스 10종). US 는 유니버스 확장 후.
- 라이브러리: data/state/alpha_library_CRYPTO.json (경험 메모리 포함)
- 백테스트는 스크리닝 전용 — 라이브 기여는 Trader 인용→reflection 으로 계측.
- 개선 곡선 포화(QuantaAlpha ~11-12 iteration decay) 모니터링을 위해 사이클 이벤트를
  data/logs/ALPHA/ 에 구조화 기록한다.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alpha_lab import FactorLibrary, generate_candidates  # noqa: E402
from alpha_lab.data import fetch_crypto_panel  # noqa: E402
from harness import JsonlLogger, load_env  # noqa: E402
from llm import LLMRouter  # noqa: E402

LIBRARY_PATH = ROOT / "data" / "state" / "alpha_library_CRYPTO.json"


async def main() -> int:
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 5
    env = load_env(ROOT / ".env")
    router = LLMRouter(env)
    logger = JsonlLogger(ROOT / "data" / "logs")
    library = FactorLibrary(LIBRARY_PATH)
    today = datetime.now(timezone.utc).date()

    try:
        print(f"library: active={len(library.active())} total={len(library.factors)}")
        panel, symbols, dates = await fetch_crypto_panel()
        print(f"panel: {len(dates)}일 × {len(symbols)}종 ({dates[0]} ~ {dates[-1]})")

        # 라이브 감쇠 퇴출 먼저 — writer 가 살아있는 라이브러리만 보게
        decayed = library.review_decay(panel, dates, today)
        for e in decayed:
            name, live_ic, oos_ic = e["name"], e["live_ic"], e["admission_oos_ic"]
            print(f"  DECAYED name={name} live_ic={live_ic} (admission oos_ic={oos_ic})")
            logger.log("ALPHA", e.pop("event"), e | {"cycle_day": str(today)})
        print(f"decay_review retired={len(decayed)} active_remaining={len(library.active())}")

        candidates = await generate_candidates(router, library, n=n)
        print(f"writer: {len(candidates)}개 생성, judge/dsl 기각 "
              f"{sum(1 for c in candidates if c.rejected)}개")
        for c in candidates:
            if c.rejected:
                print(f"  pre-reject name={c.name} reason={c.rejected[:80]}")

        events = library.admit(candidates, panel, today)
        admitted = [e for e in events if e["event"] == "factor_admitted"]
        for e in events:
            logger.log("ALPHA", e.pop("event"), e | {"cycle_day": str(today)})
        for e in admitted:
            print(f"  ADMITTED name={e['name']} train_ic={e['train_ic']}"
                  f" oos_ic={e['oos_ic']} oos_icir={e['oos_icir']}")
        for e in events:
            if "reason" in e:
                print(f"  rejected name={e['name']} reason={e['reason'][:80]}")

        print(f"cycle_done admitted={len(admitted)} library_active={len(library.active())}")
        return 0
    finally:
        await router.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
