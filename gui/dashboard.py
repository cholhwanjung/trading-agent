"""GUI 대시보드 v1 — Streamlit (옵션 A · [ADR-019]).

실행:
    uv run --group gui streamlit run gui/dashboard.py

원칙 (GUI 계획 검토에서 확정):
- **읽기 전용 + 대화만** — 리스크 한도·프롬프트·메모리 수정 UI 를 두지 않는다 (하드룰 5·8).
- 브로커 API 를 직접 치지 않는다 — 일일 루프가 갱신한 로그·상태 파일만 읽는다.
- 챗은 게이트웨이 /chat 프록시 — R15 grounding 집행 지점을 게이트웨이 하나로 유지.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.meta import combined_index  # noqa: E402
from eval.rolling import ROLLING_K, rolling_report  # noqa: E402
from harness.env import load_env  # noqa: E402
from interaction.briefing import build_briefing  # noqa: E402
from interaction.context import build_context  # noqa: E402

STATE = ROOT / "data" / "state"
VIRTUAL = STATE / "virtual"
MARKETS = ("CRYPTO", "US", "KR")
ARMS = ("llm", "llm_base", "bh", "random")

st.set_page_config(page_title="trading-agent", page_icon="📈", layout="wide")


def load_equity_frame(market: str) -> pd.DataFrame | None:
    """가상 arm equity 곡선 → wide DataFrame (index=day, columns=arm)."""
    series = {}
    for arm in ARMS:
        path = VIRTUAL / f"{market}_{arm}.json"
        if not path.exists():
            continue
        history = json.loads(path.read_text(encoding="utf-8")).get("history") or []
        if history:
            series[arm] = pd.Series(
                [h["equity"] for h in history], index=[h["day"] for h in history]
            )
    return pd.DataFrame(series) if series else None


@st.cache_data(ttl=60)
def load_context() -> dict:
    return build_context(ROOT)


tab_dash, tab_chat, tab_ops = st.tabs(["📊 대시보드", "💬 챗", "🔧 운영"])


# ── 대시보드 ──

with tab_dash:
    context = load_context()
    by_kind: dict[str, list[dict]] = {}
    for item in context["items"]:
        by_kind.setdefault(item["kind"], []).append(item)

    # META 결합 지수
    meta = combined_index(VIRTUAL, "llm")
    cols = st.columns(4)
    if meta:
        cols[0].metric("META 결합 지수 (llm)", f"{meta['index']:.4f}", f"{meta['ret_pct']:+.3f}%")
        cols[1].metric("META MDD", f"{meta['mdd_pct']:.2f}%")
        bh_meta = combined_index(VIRTUAL, "bh")
        if bh_meta:
            cols[2].metric("META α vs B&H", f"{meta['ret_pct'] - bh_meta['ret_pct']:+.3f}%p")
        base_meta = combined_index(VIRTUAL, "llm_base")
        if base_meta:
            cols[3].metric("메모리 델타 (R9)", f"{meta['ret_pct'] - base_meta['ret_pct']:+.3f}%p")

    for market in MARKETS:
        frame = load_equity_frame(market)
        if frame is None:
            continue
        st.subheader(f"{market} — 가상 4-arm equity")
        st.line_chart(frame)
        risk = next((i for i in by_kind.get("risk_state", []) if i["id"] == f"risk:{market}"), None)
        if risk:
            st.caption(f"현재 목표 배분: `{risk['content']['target_weights']}` · equity 고점: {risk['content']['peak_equity']}")

    st.subheader("최근 결정 (근거·인용)")
    rows = [
        {
            "id": d["id"],
            "day": d["content"]["day"],
            "weights": json.dumps(d["content"]["weights"], ensure_ascii=False),
            "risk 위반": ", ".join(d["content"]["risk_violations"] or []),
            "근거": d["content"]["rationale"],
        }
        for d in by_kind.get("decision", [])
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("오늘 브리핑")
    st.markdown(build_briefing(ROOT))


# ── 챗 (게이트웨이 프록시) ──

with tab_chat:
    st.caption(
        "게이트웨이 `/chat` 프록시 — 답변은 R15 grounding(근거 ID 인용) 강제. "
        "게이트웨이 실행: `uv run uvicorn interaction.api:app --port 8721`"
    )
    gateway = st.text_input("게이트웨이 URL", value="http://localhost:8721")
    env = load_env(ROOT / ".env")
    token = env.get("INTERACTION_API_TOKEN", "")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # [{"role", "content", "cited_ids"?}]
        st.session_state.chat_session_id = None

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("cited_ids"):
                st.caption("인용: " + " · ".join(f"`{c}`" for c in msg["cited_ids"]))

    if question := st.chat_input("에이전트에게 질문 (예: 지금 KR 포지션의 근거는?)"):
        st.session_state.chat_history.append({"role": "user", "content": question})
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            resp = httpx.post(
                f"{gateway}/chat",
                json={"question": question, "session_id": st.session_state.chat_session_id},
                headers=headers,
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.chat_session_id = data["session_id"]
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": data["answer"], "cited_ids": data["cited_ids"]}
                )
            else:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"⚠️ 게이트웨이 오류 {resp.status_code}: {resp.text[:200]}"}
                )
        except httpx.HTTPError as e:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"⚠️ 게이트웨이 연결 실패: {e} — 게이트웨이가 떠 있는지 확인"}
            )
        st.rerun()


# ── 운영 (읽기 전용) ──

with tab_ops:
    st.subheader("rolling-k delta (승격 판정 입력)")
    for market in MARKETS:
        if not (VIRTUAL / f"{market}_llm.json").exists():
            continue
        rolled = rolling_report(VIRTUAL, market)
        line = f"**{market}** — "
        for name in ("memory", "alpha"):
            r = rolled[name]
            if r is None:
                line += f"{name}: 데이터 {ROLLING_K + 1}일 미만 · "
            else:
                p = f"p={r['p_value']:.3f}" if r["p_value"] is not None else "p=n/a"
                line += f"{name}: 승률 {r['win_rate']:.0%} ({p}) · "
        st.markdown(line.rstrip(" ·"))

    st.subheader("메모리 (교훈 상태)")
    context = load_context()
    mem_rows = [
        {"id": i["id"], "kind": i["kind"], "status": i["content"].get("status", ""),
         "importance": i["content"].get("importance", ""), "outcome": i["content"].get("outcome", ""),
         "content": i["content"].get("text", "")}
        for i in context["items"]
        if i["kind"].startswith("memory_")
    ]
    if mem_rows:
        st.dataframe(pd.DataFrame(mem_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("승격된 교훈 없음 — admission 게이트 통과분이 생기면 여기 표시된다.")

    st.subheader("Alpha 팩터 라이브러리")
    lib_path = STATE / "alpha_library_CRYPTO.json"
    if lib_path.exists():
        factors = json.loads(lib_path.read_text(encoding="utf-8"))["factors"]
        st.dataframe(pd.DataFrame(factors), use_container_width=True, hide_index=True)

    st.subheader("월간 self-improve 제안서 (승인은 코드/문서 경로로만 — R12)")
    proposals = sorted((ROOT / "data" / "proposals").glob("*.md")) if (ROOT / "data" / "proposals").exists() else []
    if proposals:
        pick = st.selectbox("제안서", [p.name for p in proposals])
        st.markdown((ROOT / "data" / "proposals" / pick).read_text(encoding="utf-8"))
    else:
        st.caption("아직 없음 — 매월 1일 21:00 자동 생성.")
