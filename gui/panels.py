"""대시보드 데이터 로더 — 순수 stdlib(pandas/streamlit 미의존, 테스트 가능).

일일 루프가 남긴 관측 스냅샷(`data/state/observations/`)과 결정 로그
(`data/logs/`)를 읽어 UI 가 바로 쓰는 평범한 dict/list 로 정규화한다.
브로커 API 를 치지 않고, 파일만 읽는다(결정론·감사 가능).
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.meta import load_arm_history
from harness.jsonlog import iter_events


def list_observation_days(obs_dir: Path, market: str) -> list[str]:
    """해당 시장의 관측 스냅샷 날짜(YYYY-MM-DD)를 최신순으로 반환."""

    market_dir = obs_dir / market
    if not market_dir.exists():
        return []
    return sorted((p.stem for p in market_dir.glob("*.json")), reverse=True)


def load_observation(obs_dir: Path, market: str, day: str) -> dict | None:
    """{obs_dir}/{market}/{day}.json 스냅샷을 읽는다. 없으면 None."""

    path = obs_dir / market / f"{day}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_recent_decisions(log_dir: Path, market: str, limit: int = 30) -> list[dict]:
    """daily_step 로그 → asof_day 오름차순 정규화 결정 행.

    로그 파일명은 실행일이지만 여기서는 레코드의 asof_day 로 키를 잡는다
    (관측 스냅샷과 같은 축). 같은 asof 재실행은 최신 레코드로 덮어쓴다.
    """

    rows: dict[str, dict] = {}
    for rec in iter_events(log_dir, market, "daily_step"):
        day = str(rec.get("asof_day", ""))[:10]
        if not day:
            continue
        d = rec.get("decision") or {}
        rows[day] = {
            "day": day,
            "policy": rec.get("policy"),
            "weights": rec.get("weights") or {},
            "accepted": rec.get("accepted"),
            "features": d.get("features") or {},
            "rationale": d.get("rationale"),
            "debate": d.get("debate"),
            "risk_violations": d.get("risk_violations") or [],
            "weights_pre_risk": d.get("weights_pre_risk"),
            "mdd": d.get("mdd"),
            "cited_signal_ids": d.get("cited_signal_ids") or [],
            "cited_memory_ids": d.get("cited_memory_ids") or [],
        }
    return [rows[k] for k in sorted(rows)][-limit:]


def decision_for_day(rows: list[dict], day: str) -> dict | None:
    """정규화된 결정 행 목록에서 특정 날짜의 결정을 찾는다."""

    return next((r for r in rows if r["day"] == day), None)


def veto_rows(rows: list[dict]) -> list[dict]:
    """risk 위반이 있었던 날만 추려 타임라인 표로."""

    return [
        {"day": r["day"], "violations": "; ".join(r["risk_violations"]), "mdd": r.get("mdd")}
        for r in rows
        if r["risk_violations"]
    ]


def load_latest_requests(requests_dir: Path) -> dict | None:
    """가장 최근 달의 능력 갭 요구 파일({YYYY-MM}.json)을 읽는다. 없으면 None."""

    if not requests_dir.exists():
        return None
    files = sorted(requests_dir.glob("*.json"), reverse=True)
    if not files:
        return None
    return json.loads(files[0].read_text(encoding="utf-8"))


# ── 자본 배분 (파이 데이터) ──


def load_market_allocation(virtual_dir: Path, meta_ledger: Path, arm: str = "llm") -> dict:
    """마켓별 자본 배분 — 현재(가상 equity 비중) vs 목표(meta_shadow 최신 제안).

    현재 비중은 arm equity 를 시장별로 정규화(전 시장 동일 nominal base 라 공통 단위).
    목표는 meta_shadow 원장의 최신 제안 weights. 각각 없으면 빈 dict.
    """
    suffix = f"_{arm}.json"
    equities: dict[str, float] = {}
    if virtual_dir.exists():
        for path in sorted(virtual_dir.glob(f"*{suffix}")):
            market = path.name[: -len(suffix)]
            history = load_arm_history(virtual_dir, market, arm)
            eq = history[-1]["equity"] if history else None
            if eq and eq > 0:
                equities[market] = eq
    total = sum(equities.values())
    current = {m: eq / total for m, eq in equities.items()} if total > 0 else {}

    target: dict[str, float] = {}
    if meta_ledger.exists():
        history = json.loads(meta_ledger.read_text(encoding="utf-8")).get("history") or []
        if history:
            target = history[-1].get("weights") or {}
    return {"current": current, "target": target, "arm": arm}


def load_intramarket_weights(state_dir: Path, market: str) -> dict[str, float]:
    """마켓 내 목표 배분 벡터(CASH 포함) — risk_{market}.json 의 prev_weights. 없으면 {}."""
    path = state_dir / f"risk_{market}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("prev_weights") or {}


# ── 스케줄 잡 상태 (launchd 로그 기반) ──

_OK_MARKERS = ("briefing=", "status=ok")
_FAIL_MARKERS = ("status=fail", "status=error")


def _tail(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]


def load_launchd_jobs(log_dir: Path, tail_lines: int = 20) -> list[dict]:
    """launchd*.log(out/err)를 잡별로 묶어 상태 요약. 파일만 읽는다(launchctl 미호출).

    상태는 **out 로그 tail 마커** 기반 추정(최신 런의 완료/실패 신호). 정확한 종료코드가
    아니므로 raw tail 을 함께 노출한다. err 로그는 누적되므로(과거 트레이스백이 남음)
    stderr 존재는 실패가 아니라 '확인 힌트'로만 표시한다. last_run 은 로그 mtime(로컬).
    """
    from datetime import datetime

    if not log_dir.exists():
        return []
    groups: dict[str, dict[str, Path]] = {}
    for p in sorted(log_dir.glob("launchd*.log")):
        stem = p.name[len("launchd"):].removesuffix(".log")  # 예: ".kr.out"
        kind = "err" if stem.endswith(".err") else "out"
        job = stem.removesuffix(".err").removesuffix(".out").strip(".") or "main"
        groups.setdefault(job, {})[kind] = p

    jobs: list[dict] = []
    for job, files in sorted(groups.items()):
        out_p, err_p = files.get("out"), files.get("err")
        out_tail = _tail(out_p, tail_lines) if out_p else []
        err_tail = _tail(err_p, tail_lines) if err_p else []
        mtimes = [p.stat().st_mtime for p in (out_p, err_p) if p and p.exists()]
        last_run = datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds") if mtimes else None
        joined = "\n".join(out_tail)
        if any(m in joined for m in _FAIL_MARKERS):
            status = "error"
        elif any(m in joined for m in _OK_MARKERS):
            status = "ok"
        else:
            status = "unknown"
        jobs.append({
            "job": job, "status": status, "last_run": last_run,
            "out_tail": out_tail, "err_tail": err_tail,
            "has_stderr": bool(err_tail and any(ln.strip() for ln in err_tail)),
        })
    return jobs
