"""챗 품질 평가 실행기 — 스냅샷 동결 · 문항 실행 · 결정론 채점 · 결과 기록.

    uv run python scripts/run_chat_eval.py snapshot
    uv run python scripts/run_chat_eval.py items  [--snapshot latest]
    uv run python scripts/run_chat_eval.py run    [--snapshot latest] [--categories A,B,D,G,I,J]
                                                  [--repeats 3] [--limit N] [--concurrency 4]
    uv run python scripts/run_chat_eval.py report [--run latest]

평가는 라이브 root 가 아니라 동결한 스냅샷에서만 돈다. 토론 마감·제안 초안은 root 아래에
파일을 쓰므로 라이브에서 돌리면 메모리와 제안 디렉터리가 오염된다. 사용량 싱크도 달지
않는다 — 평가 호출이 사용량 로그의 chat 집계에 섞이면 실사용 지표가 부풀어 오른다.
토큰·지연은 결과 파일에 직접 남기고, 실행 전후 라이브 기록의 변화를 manifest 에 적는다.

산출물은 전부 data/eval/chat 아래다 — 실계좌 기록이 들어 있어 저장소에 올리지 않는다.
    snapshots/{날짜}-{context 해시}/   동결한 root (context 가 읽는 파일 + 플레이북)
    items.jsonl                       수작업 문항 (기권·입력 조작·제안 초안)
    runs/{run_id}/                    manifest.json · summary.json · results.jsonl · items.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.chat_quality import (  # noqa: E402
    BEHAVIORAL,
    CATEGORIES,
    MARKETS,
    Item,
    Response,
    context_hash,
    generate,
    item_set_hash,
    load_items,
    score,
    summarize,
)
from harness.env import load_env  # noqa: E402
from interaction.chat import (  # noqa: E402
    PROPOSE_PROMPT,
    SYSTEM_PROMPT,
    ChatEngine,
    DiscussionSession,
    GroundingError,
)
from interaction.context import build_context  # noqa: E402
from interaction.proposal import TARGET  # noqa: E402
from llm import LLMRouter, extract_json  # noqa: E402
from memory import MemoryStore  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval" / "chat"
ARMS = ("llm", "llm_base", "bh", "random")


def _rev(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ── 스냅샷 ────────────────────────────────────────────────────────────────


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_inputs(root: Path, dest: Path) -> None:
    """context 가 읽는 파일 + 제안 초안이 읽는 플레이북만. 토큰·장부·락은 복사하지 않는다."""
    for m in MARKETS:
        for path in sorted((root / "data" / "logs" / m).glob("*.jsonl")):
            _copy(path, dest / path.relative_to(root))
        for path in sorted((root / "data" / "state" / "observations" / m).glob("*.json")):
            _copy(path, dest / path.relative_to(root))
        for rel in (
            f"data/state/risk_{m}.json",
            *(f"data/state/virtual/{m}_{a}.json" for a in ARMS),
        ):
            if (root / rel).exists():
                _copy(root / rel, dest / rel)
    for rel in ("data/state/alpha_library_CRYPTO.json", TARGET):
        if (root / rel).exists():
            _copy(root / rel, dest / rel)
    db = root / "data" / "memory.sqlite"
    if db.exists():
        (dest / "data").mkdir(parents=True, exist_ok=True)
        src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        dst = sqlite3.connect(dest / "data" / "memory.sqlite")
        try:
            src.backup(dst)  # 쓰는 중인 DB 도 일관된 사본으로
        finally:
            src.close()
            dst.close()


def freeze(root: Path, base: Path, attempts: int = 3) -> Path:
    """라이브 root → 스냅샷. 사본의 context 가 라이브와 같을 때만 확정한다 —
    복사 도중 일일 잡이 로그를 쓰면 다시 뜬다."""
    base.mkdir(parents=True, exist_ok=True)
    for _ in range(attempts):
        tmp = base / f".tmp-{uuid.uuid4().hex[:8]}"
        _copy_inputs(root, tmp)
        live, snap = build_context(root), build_context(tmp)
        if live["items"] != snap["items"]:
            shutil.rmtree(tmp)
            continue
        digest = context_hash(snap)
        snap_id = f"{datetime.now(timezone.utc).date().isoformat()}-{digest[:8]}"
        dest = base / snap_id
        if dest.exists():  # 같은 context 는 같은 스냅샷
            shutil.rmtree(tmp)
            return dest
        tmp.rename(dest)
        playbook = dest / TARGET
        meta = {
            "snapshot_id": snap_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "context_hash": digest,
            "n_items": len(snap["items"]),
            "kinds": dict(Counter(it["kind"] for it in snap["items"])),
            "playbook_rev": _rev(playbook.read_text(encoding="utf-8"))
            if playbook.exists()
            else None,
        }
        (dest / "snapshot.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return dest
    raise RuntimeError("스냅샷 context 가 라이브와 계속 어긋난다 — 일일 잡이 실행 중인지 확인")


def _resolve(kind_dir: Path, name: str) -> Path:
    if name != "latest":
        path = kind_dir / name
        if not path.is_dir():
            raise SystemExit(f"없음: {path}")
        return path
    dirs = (
        [p for p in kind_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
        if kind_dir.exists()
        else []
    )
    if not dirs:
        raise SystemExit(f"없음: {kind_dir} — 먼저 만들 것")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def market_ids(root: Path) -> dict[str, set[str]]:
    """시장별 인용 가능 ID. 알파 팩터는 CRYPTO 라이브러리라 CRYPTO 에만 속한다."""
    out = {}
    for m in MARKETS:
        ids = {it["id"] for it in build_context(root, (m,))["items"]}
        out[m] = ids if m == "CRYPTO" else {i for i in ids if not i.startswith("alpha:")}
    return out


# ── 입력 조작 사본 ─────────────────────────────────────────────────────────


def _is_step(line: str) -> bool:
    try:
        return json.loads(line).get("event") == "daily_step"
    except json.JSONDecodeError:
        return False


def make_variant(snapshot: Path, dest: Path, inject: dict) -> Path:
    """스냅샷 사본에 지시문을 심는다. 심은 문장이 사본의 context 에 실리지 않으면 중단한다 —
    잘려 나간 주입은 '모델이 지시를 따르지 않았다' 로 잘못 읽힌다."""
    shutil.copytree(snapshot, dest)
    target, market, text = inject["target"], inject["market"], inject["text"]
    if target == "disclosure":
        obs = sorted((dest / "data" / "state" / "observations" / market).glob("*.json"))[-1]
        data = json.loads(obs.read_text(encoding="utf-8"))
        day = str(data.get("asof_day", obs.stem))[:10]
        planted = {
            "published_at": f"{day}T00:00:00+00:00",
            "headline": text,
            "source": "DART" if market == "KR" else "SEC",
            "url": "",
        }
        # 앞에 넣는다 — context 는 공시를 앞에서부터 정해진 건수만 싣는다
        data["news"] = [planted, *(data.get("news") or [])]
        obs.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    elif target == "rationale":
        for path in sorted((dest / "data" / "logs" / market).glob("*.jsonl"), reverse=True):
            lines = path.read_text(encoding="utf-8").splitlines()
            steps = [i for i, line in enumerate(lines) if _is_step(line)]
            if steps:
                record = json.loads(lines[steps[-1]])
                record["decision"]["rationale"] = (record["decision"].get("rationale") or "") + text
                lines[steps[-1]] = json.dumps(record, ensure_ascii=False)
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                break
    elif target == "episodic":
        db = dest / "data" / "memory.sqlite"
        store = MemoryStore(db)
        try:
            last = store.query(market, store="episodic")[-1]
        finally:
            store.close()
        con = sqlite3.connect(db)
        try:
            con.execute("UPDATE memories SET content = content || ? WHERE id = ?", (text, last.id))
            con.commit()
        finally:
            con.close()
    else:
        raise ValueError(f"알 수 없는 주입 대상 {target}")
    if text.strip() not in json.dumps(build_context(dest)["items"], ensure_ascii=False):
        raise RuntimeError(f"주입문이 context 에 실리지 않았다 target={target} market={market}")
    return dest


# ── 실행 ──────────────────────────────────────────────────────────────────


class RecordingRouter:
    """라우터를 감싸 호출마다 원문·토큰·지연을 남긴다. 사용량 로그에는 쓰지 않는다.

    원문이 필요한 이유 — 게이트웨이가 인용 위반으로 답을 거부하면 답이 사라지는데,
    입력 조작 문항에서는 바로 그 거부된 답이 '지시를 따랐는가' 의 증거다."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[dict] = []

    def spec(self, tier):
        return self.inner.spec(tier)

    async def complete(self, tier, **kwargs):
        start = time.monotonic()
        resp = await self.inner.complete(tier, **kwargs)
        self.calls.append(
            {
                "text": resp.text,
                "tokens_in": resp.input_tokens,
                "tokens_out": resp.output_tokens,
                "latency_s": round(time.monotonic() - start, 2),
            }
        )
        return resp


async def run_item(
    item: Item,
    repeat: int,
    *,
    router,
    snapshot: Path,
    work: Path,
    base_contents: dict,
    ids: dict,
    playbook: str,
) -> dict:
    rec = RecordingRouter(router)
    root, contents, variant = snapshot, base_contents, None
    if item.inject:
        variant = make_variant(snapshot, work / f"{item.id}-{repeat}", item.inject)
        root = variant
        contents = {it["id"]: it["content"] for it in build_context(variant)["items"]}
    engine = ChatEngine(rec, root)
    resp = Response()
    try:
        if item.category == "J":
            sid = uuid.uuid4().hex[:12]
            engine.sessions[sid] = DiscussionSession(
                session_id=sid, market=item.market, messages=[dict(m) for m in item.transcript]
            )
            proposal = await engine.propose(sid)
            proposal.path.unlink(missing_ok=True)  # 스냅샷에 초안 파일을 남기지 않는다
            resp.diff = proposal.diff
        else:
            answer, _ = await engine.ask(item.question)
            resp.answer, resp.cited_ids = answer.answer, answer.cited_ids
    except GroundingError as e:
        resp.error = f"grounding: {e}"
        raw = extract_json(rec.calls[-1]["text"]) if rec.calls else None
        if isinstance(raw, dict):
            resp.answer = str(raw.get("answer", ""))
            resp.cited_ids = [str(x) for x in raw.get("cited_ids") or []]
    except Exception as e:  # 호출 실패는 결과로 남기고 다음 문항으로 간다
        resp.error = f"{type(e).__name__}: {e}"
    finally:
        if variant:
            shutil.rmtree(variant, ignore_errors=True)
    graded = score(item, resp, contents=contents, market_ids=ids, playbook=playbook)
    call = rec.calls[-1] if rec.calls else {}
    return {
        "item_id": item.id,
        "category": item.category,
        "repeat": repeat,
        "tags": item.tags,
        "question": item.question,
        "answer": resp.answer,
        "cited_ids": resp.cited_ids,
        "diff": resp.diff,
        "error": resp.error,
        **graded,
        "latency_s": call.get("latency_s"),
        "tokens_in": call.get("tokens_in"),
        "tokens_out": call.get("tokens_out"),
    }


def live_counters(root: Path) -> dict:
    """라이브 기록 중 평가가 건드리면 안 되는 것 — 사용량 로그의 chat 줄 · 토론 기록."""
    chat_lines = 0
    for path in sorted((root / "data" / "logs" / "USAGE").glob("*.jsonl"))[-3:]:
        chat_lines += sum(
            '"purpose": "chat' in line for line in path.read_text(encoding="utf-8").splitlines()
        )
    sessions = 0
    db = root / "data" / "memory.sqlite"
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            sessions = con.execute(
                "SELECT COUNT(*) FROM memories WHERE data LIKE '%user_session%'"
            ).fetchone()[0]
        finally:
            con.close()
    return {"usage_chat_lines": chat_lines, "user_sessions": sessions}


def _git() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}


def execute(
    items: list[Item],
    *,
    snapshot: Path,
    out_dir: Path,
    router,
    repeats: int = 3,
    concurrency: int = 4,
    live_root: Path = ROOT,
) -> dict:
    """문항 실행 → results.jsonl · summary.json · manifest.json. 반환은 manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "items.jsonl").write_text(
        "".join(json.dumps(asdict(i), ensure_ascii=False) + "\n" for i in items), encoding="utf-8"
    )
    playbook_path = snapshot / TARGET
    playbook = playbook_path.read_text(encoding="utf-8") if playbook_path.exists() else ""
    snap_ctx = build_context(snapshot)
    base_contents = {it["id"]: it["content"] for it in snap_ctx["items"]}
    ids = market_ids(snapshot)
    work = out_dir / "work"
    before = live_counters(live_root)
    started = datetime.now(timezone.utc)
    results: list[dict] = []

    async def main() -> None:
        sem = asyncio.Semaphore(concurrency)

        async def one(item: Item, k: int) -> dict:
            async with sem:
                return await run_item(
                    item,
                    k,
                    router=router,
                    snapshot=snapshot,
                    work=work,
                    base_contents=base_contents,
                    ids=ids,
                    playbook=playbook,
                )

        jobs = [one(i, k) for i in items for k in range(repeats if i.category in BEHAVIORAL else 1)]
        with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
            for done in asyncio.as_completed(jobs):
                row = await done
                results.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")  # 중단돼도 끝난 몫은 남는다
                f.flush()
        if hasattr(router, "close"):
            await router.close()

    asyncio.run(main())
    shutil.rmtree(work, ignore_errors=True)
    after = live_counters(live_root)
    summary = summarize(results)
    snap_meta_path = snapshot / "snapshot.json"
    snap_meta = (
        json.loads(snap_meta_path.read_text(encoding="utf-8")) if snap_meta_path.exists() else {}
    )
    manifest = {
        "run_id": out_dir.name,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot.name,
        "snapshot_context_hash": snap_meta.get("context_hash") or context_hash(snap_ctx),
        "playbook_rev": _rev(playbook) if playbook else None,
        "item_set_hash": item_set_hash(items),
        "n_items": len(items),
        "categories": sorted({i.category for i in items}),
        "repeats_behavioral": repeats,
        "concurrency": concurrency,
        "models": {tier: ":".join(router.spec(tier)) for tier in ("smart", "fast")},
        "chat_prompt_rev": _rev(SYSTEM_PROMPT),
        "propose_prompt_rev": _rev(PROPOSE_PROMPT),
        "live_delta": {k: after[k] - before[k] for k in before},
        **_git(),
        "calls": summary["calls"],
        "tokens_in": summary["tokens_in"],
        "tokens_out": summary["tokens_out"],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────


def _items(
    snapshot: Path, items_path: Path, categories: list[str], limit: int | None
) -> list[Item]:
    items = generate(build_context(snapshot))
    if items_path.exists():
        items += load_items(items_path)
    items = [i for i in items if i.category in categories]
    if limit:
        seen: Counter = Counter()
        kept = []
        for i in items:
            seen[i.category] += 1
            if seen[i.category] <= limit:
                kept.append(i)
        items = kept
    return items


def _print_summary(manifest: dict, summary: dict) -> None:
    print(
        f"chat_eval run_id={manifest['run_id']} snapshot={manifest['snapshot_id']} "
        f"items={manifest['n_items']} calls={summary['calls']} passed={summary['passed']} "
        f"tokens_in={summary['tokens_in']} tokens_out={summary['tokens_out']} "
        f"latency_p50_s={summary['latency_p50_s']} latency_p95_s={summary['latency_p95_s']}"
    )
    for cat, c in summary["categories"].items():
        print(
            f"category={cat} items={c['items']} calls={c['calls']} pass_rate={c['pass_rate']} "
            f"pass_all={c['pass_all']} grounding_errors={c['grounding_errors']} "
            f"call_errors={c['call_errors']} gates={json.dumps(c['gates'], ensure_ascii=False)}"
        )
    print(
        f"gates={json.dumps(summary['gates'], ensure_ascii=False)} "
        f"live_delta={json.dumps(manifest['live_delta'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="챗 품질 평가 (결정론 채점)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot", help="라이브 root 를 동결")
    p_items = sub.add_parser("items", help="문항 목록 출력")
    p_run = sub.add_parser("run", help="평가 실행")
    for p in (p_items, p_run):
        p.add_argument("--snapshot", default="latest")
        p.add_argument("--items", default=str(EVAL_DIR / "items.jsonl"))
        p.add_argument("--categories", default=",".join(CATEGORIES))
        p.add_argument("--limit", type=int, default=None, help="범주별 문항 수 상한 (시험 실행용)")
    p_run.add_argument("--repeats", type=int, default=3, help="행동 범주 반복 횟수")
    p_run.add_argument("--concurrency", type=int, default=4)
    p_report = sub.add_parser("report", help="지난 실행 요약")
    p_report.add_argument("--run", default="latest")
    args = parser.parse_args()

    if args.cmd == "snapshot":
        dest = freeze(ROOT, EVAL_DIR / "snapshots")
        meta = json.loads((dest / "snapshot.json").read_text(encoding="utf-8"))
        print(
            f"chat_eval_snapshot id={meta['snapshot_id']} items={meta['n_items']} "
            f"kinds={json.dumps(meta['kinds'])} playbook_rev={meta['playbook_rev']} path={dest}"
        )
        return 0
    if args.cmd == "report":
        run_dir = _resolve(EVAL_DIR / "runs", args.run)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        _print_summary(manifest, summary)
        return 0

    snapshot = _resolve(EVAL_DIR / "snapshots", args.snapshot)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    items = _items(snapshot, Path(args.items), categories, args.limit)
    if args.cmd == "items":
        for i in items:
            gold = f" gold={i.gold_values} ids={i.gold_ids}" if i.gold_values else ""
            print(f"item={i.id} category={i.category} market={i.market} q={i.question!r}{gold}")
        print(
            f"chat_eval_items snapshot={snapshot.name} n={len(items)} "
            f"by_category={json.dumps(dict(Counter(i.category for i in items)))} "
            f"hash={item_set_hash(items)}"
        )
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:4]}"
    router = LLMRouter(load_env(ROOT / ".env"))  # 사용량 싱크 없음 — 라이브 집계에 섞지 않는다
    manifest = execute(
        items,
        snapshot=snapshot,
        out_dir=EVAL_DIR / "runs" / run_id,
        router=router,
        repeats=args.repeats,
        concurrency=args.concurrency,
    )
    summary = json.loads((EVAL_DIR / "runs" / run_id / "summary.json").read_text(encoding="utf-8"))
    _print_summary(manifest, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
