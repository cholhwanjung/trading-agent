"""뉴스 서사 다이제스트 — 일간 관측에 쌓인 헤드라인을 시장당 몇 줄로 압축한다.

새 수집기는 없다. 일간 스텝이 저장한 관측(`data/state/observations/{market}/{day}.json`)의
뉴스는 이미 t−1 상한·품질 필터·수집 타임스탬프를 거쳤다. 여기서는 최근 N 캘린더일을
합쳐 중복을 걷고, 출처별 라운드로빈으로 상한을 건 뒤, fast tier 가 줄마다 헤드라인 id 를
인용하는 서사로 압축한다. 3시장 × 7일 헤드라인을 그대로 넣으면 절삭으로 채널이 죽는다.

공시(DART·SEC)는 뺀다 — 개별 기업 단위라 시장 전체 서사의 재료가 아니다.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from adapters.base import DISCLOSURE_SOURCES
from llm import LLMRouter, extract_json
from trader.prompt_store import PromptSpec

NEWS_DAYS = 7
CAP_PER_MARKET = 12
MAX_LINES_PER_MARKET = 5


def load_recent_news(
    obs_dir: Path | str, market: str, asof_day: date, days: int = NEWS_DAYS
) -> list[dict]:
    """[asof−days, asof−1] 관측 파일의 뉴스를 합쳐 중복(url → 헤드라인)을 걷고 최신순 정렬."""
    seen: set[str] = set()
    out: list[dict] = []
    for i in range(days, 0, -1):
        path = Path(obs_dir) / market / f"{(asof_day - timedelta(days=i)).isoformat()}.json"
        if not path.exists():
            continue
        try:
            items = json.loads(path.read_text(encoding="utf-8")).get("news") or []
        except (json.JSONDecodeError, AttributeError):
            continue
        for it in items:
            if (it.get("source") or "") in DISCLOSURE_SOURCES:
                continue
            key = it.get("url") or it.get("headline") or ""
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "published_at": it.get("published_at", ""),
                    "headline": it.get("headline", ""),
                    "source": it.get("source", ""),
                    "url": it.get("url"),
                }
            )
    out.sort(key=lambda it: it["published_at"], reverse=True)
    return out


def select_news(items: list[dict], cap: int = CAP_PER_MARKET) -> list[dict]:
    """출처별 라운드로빈으로 cap 개 — 기사가 많은 출처가 창을 독식하지 않게."""
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(it.get("source") or "", []).append(it)
    queues = [list(g) for g in groups.values()]
    out: list[dict] = []
    while queues and len(out) < cap:
        for q in list(queues):
            if len(out) >= cap:
                break
            out.append(q.pop(0))
            if not q:
                queues.remove(q)
    return out


def assign_ids(by_market: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """헤드라인에 `news:<market>:<nn>` id 를 붙인다 — 제안·감사가 같은 id 를 본다."""
    return {
        m: [{"id": f"news:{m}:{k:02d}", **it} for k, it in enumerate(items, start=1)]
        for m, items in by_market.items()
    }


def digest_prompt(by_market: dict[str, list[dict]]) -> str:
    lines = []
    for m, items in by_market.items():
        lines.append(f"## {m}")
        if not items:
            lines.append("(헤드라인 없음)")
        for it in items:
            lines.append(f"{it['id']} [{it['published_at'][:10]}] {it['headline']}")
    return "\n".join(lines)


def parse_digest(text: str, valid_ids: set[str], markets: tuple[str, ...]) -> dict[str, list[dict]]:
    """출력 JSON → 시장별 서사 줄. 입력에 없는 id 를 인용한 줄은 버린다(날조 차단)."""
    data = extract_json(text)
    out: dict[str, list[dict]] = {m: [] for m in markets}
    if not isinstance(data, dict):
        return out
    for m in markets:
        for line in (data.get(m) or [])[:MAX_LINES_PER_MARKET]:
            if not isinstance(line, dict):
                continue
            ids = [str(x) for x in (line.get("ids") or []) if str(x) in valid_ids]
            text_ = str(line.get("text", "")).strip()
            if not ids or not text_:
                continue
            direction = line.get("direction")
            if direction not in ("up", "down", "neutral"):
                direction = "neutral"
            out[m].append({"ids": ids, "text": text_, "direction": direction})
    return out


async def digest_news(
    router: LLMRouter,
    spec: PromptSpec,
    by_market: dict[str, list[dict]],
    purpose: str = "meta_digest",
) -> dict[str, list[dict]]:
    """시장별 id 부착 헤드라인 → 서사 줄. 헤드라인이 전혀 없으면 호출 없이 빈 결과."""
    markets = tuple(by_market)
    if not any(by_market.values()):
        return {m: [] for m in markets}
    resp = await router.complete(
        spec.tier,
        purpose=purpose,
        system=spec.text,
        messages=[{"role": "user", "content": digest_prompt(by_market)}],
        json_mode=True,
        temperature=0.0,
    )
    valid = {it["id"] for items in by_market.values() for it in items}
    return parse_digest(resp.text, valid, markets)
