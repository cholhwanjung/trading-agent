"""능력 갭 요구 생성 — 에이전트→사용자 능력(데이터/도구) 요구. **자동 획득 없음**.

측정된 갭 신호(뉴스 공백·debate 빈도·alpha 기근)를 **근거로만** 요구를 생성하고,
각 요구를 데이터 정책 대비 분류한다(정책 외=의도적으로 회피하는 게임 라벨). 어떤
데이터/도구도 설치·구독하지 않는다 — 사용자가 조달 후 코드/문서로 배선할 때만
반영된다. 결정 경로·메모리 admission 과 격리(라이브 측정 무오염).

사용법:
    uv run python scripts/request_capabilities.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import iter_events, load_env, make_usage_sink, wait_for_network  # noqa: E402
from llm import LLMRouter, extract_json  # noqa: E402

REQUEST_DIR = ROOT / "data" / "requests"
MARKETS = ("CRYPTO", "US", "KR")
RECENT = 10  # 최근 결정 창 (갭 신호 집계 범위)
DEBATE_RATE_THRESHOLD = 0.5  # 이 이상이면 관측 모호성 신호

# 데이터 정책이 "들어가지 않는 게임" — 요구 텍스트에 이 용어가 있으면 정책 외로 강제.
# 무료·일간 데이터로 이기고 자원 군비경쟁은 회피한다는 전략 포지셔닝의 코드측 집행.
OUT_OF_POLICY_TERMS = (
    "틱", "호가", "고빈도", "초단타", "미시구조", "옵션 플로우", "옵션플로우",
    "다크풀", "대체데이터", "유료", "구독", "프리미엄",
    "tick", "order book", "orderbook", "level 2", "level2", " l2",
    "microstructure", "high frequency", "high-frequency", "hft", "intraday",
    "options flow", "option flow", "dark pool", "darkpool",
    "alternative data", "alt data", "paid", "subscription", "premium",
)


def classify_policy(text: str) -> str:
    """요구 텍스트를 데이터 정책 대비 분류. 회피 게임 용어 매칭 시 'out', 아니면 'in'."""

    low = text.lower()
    for term in OUT_OF_POLICY_TERMS:
        if term in text or term.lower() in low:
            return "out"
    return "in"


def _read_steps(log_dir: Path, market: str, limit: int = RECENT) -> list[dict]:
    """daily_step → asof_day 오름차순. n_news(top-level)·debate·influence 만 추린다."""

    rows: dict[str, dict] = {}
    for rec in iter_events(log_dir, market, "daily_step"):
        day = str(rec.get("asof_day", ""))[:10]
        if not day:
            continue
        d = rec.get("decision") or {}
        rows[day] = {"day": day, "n_news": rec.get("n_news"),
                     "debate": d.get("debate"), "influence": d.get("influence") or {}}
    return [rows[k] for k in sorted(rows)][-limit:]


def gather_gap_signals(root: Path | str) -> dict:
    """측정된 능력 갭 신호(시장별) + 인용 ID. 결정론 — LLM 은 이 신호만 근거로 쓴다."""

    root = Path(root)
    log_dir, state = root / "data" / "logs", root / "data" / "state"
    out: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "markets": {}}
    for market in MARKETS:
        steps = _read_steps(log_dir, market)
        if not steps:
            continue
        zero_news = [s["day"] for s in steps if s.get("n_news") == 0]
        debate_days = [s["day"] for s in steps if s.get("debate")]
        active, retired, ics, names = 0, 0, [], []
        lib = state / f"alpha_library_{market}.json"
        if lib.exists():
            for f in json.loads(lib.read_text(encoding="utf-8")).get("factors", []):
                if f.get("status") == "active":
                    active += 1
                    names.append(f["name"])
                elif f.get("status") == "retired":
                    retired += 1
                if isinstance(f.get("oos_ic"), int | float):
                    ics.append(f["oos_ic"])
        out["markets"][market] = {
            "n_decisions": len(steps),
            "news": {"zero_news_days": zero_news,
                     "citations": [f"decision:{market}:{d}" for d in zero_news]},
            "debate": {"rate": round(len(debate_days) / len(steps), 3),
                       "citations": [f"decision:{market}:{d}" for d in debate_days]},
            "alpha": {"active": active, "retired": retired, "lib_exists": lib.exists(),
                      "min_oos_ic": min(ics) if ics else None,
                      "citations": [f"alpha:{n}" for n in names]},
        }
    return out


def has_signal(signals: dict) -> bool:
    """요구를 낼 만한 측정 갭이 하나라도 있는가. 없으면 LLM 호출·요구 생성을 스킵."""

    for s in signals.get("markets", {}).values():
        if s["news"]["zero_news_days"]:
            return True
        if s["debate"]["rate"] >= DEBATE_RATE_THRESHOLD:
            return True
    return False


def enforce_policy(requests: list[dict]) -> list[dict]:
    """LLM 이 매긴 policy_class 를 코드측에서 재검증 — 회피 게임 용어는 'out' 강제.

    라벨을 LLM 이 누락/오분류해도 정책 경계가 뚫리지 않도록 하는 결정론 게이트.
    """

    for r in requests:
        forced = classify_policy(f"{r.get('proposed_capability', '')} {r.get('gap', '')}")
        if forced == "out":
            r["policy_class"] = "out"
            r["policy_note"] = "의도적으로 회피하는 게임 (무료·일간 데이터 정책 밖)"
        else:
            r.setdefault("policy_class", "in")
    return requests


def parse_requests(text: str) -> list[dict]:
    """LLM 응답에서 JSON 배열 추출(코드펜스 허용). 실패 시 []."""

    data = extract_json(text)
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


SYSTEM = (
    "너는 트레이딩 에이전트의 self-improve 리뷰어다. 아래 **측정된 갭 신호**만 근거로 "
    "'사용자에게 요구할 능력(데이터/도구)'을 도출한다. 규칙: "
    "① 신호가 뒷받침하지 않는 요구 금지 — 신호가 약하면 빈 배열 []. "
    "② 각 요구는 신호의 인용 ID(decision:*/alpha:*)를 evidence_ids 로 명시. "
    "③ policy_class: 무료·일간·공개 데이터(예: 추가 뉴스 소스·FRED·공시·더 긴 일봉)는 "
    "'in', 유료·틱·호가·옵션플로우·고빈도·대체데이터 등 자원 군비경쟁은 'out'. "
    "출력은 JSON 배열만. 각 원소: "
    '{"gap","evidence_ids","measured_impact","proposed_capability","policy_class","est_cost"}.'
)


async def main() -> int:
    env = load_env(ROOT / ".env")
    signals = gather_gap_signals(ROOT)
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    out_path = REQUEST_DIR / f"{month}.json"

    if not has_signal(signals):
        print("status=skip detail=측정된 능력 갭 신호 없음 — 요구할 근거가 없다")
        return 0

    # wake/부팅 직후(launchd 캘린더 catch-up) LLM 호출이 DNS 실패로 죽지 않게 대기
    if not await wait_for_network():
        print("status=fail event=network_unavailable detail=네트워크 게이트 타임아웃(10분)")
        return 1

    router = LLMRouter(env, usage_sink=make_usage_sink(ROOT))
    try:
        resp = await router.complete(
            "smart", purpose="capability", system=SYSTEM,
            messages=[{"role": "user", "content": "측정된 갭 신호:\n"
                       + json.dumps(signals, ensure_ascii=False, indent=1)}],
            max_tokens=4096,
        )
        requests = enforce_policy(parse_requests(resp.text))
        REQUEST_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"generated_at": signals["generated_at"], "month": month,
                        "signals": signals, "requests": requests}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        n_out = sum(1 for r in requests if r.get("policy_class") == "out")
        print(f"status=ok requests={out_path} n={len(requests)} out_of_policy={n_out}")
        return 0
    finally:
        await router.close()


if __name__ == "__main__":
    from harness import with_deadline

    sys.exit(asyncio.run(with_deadline(main(), label="request_capabilities")))
