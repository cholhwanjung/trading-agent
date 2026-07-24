"""GUI 대시보드 v1 — Streamlit (옵션 A).

실행:
    uv run --group gui streamlit run gui/dashboard.py

원칙 (GUI 계획 검토에서 확정):
- **읽기 전용 + 대화만** — 리스크 한도·프롬프트·메모리 수정 UI 를 두지 않는다.
- 브로커 API 를 직접 치지 않는다 — 일일 루프가 갱신한 로그·상태 파일만 읽는다.
- 챗은 게이트웨이 /chat 프록시 — grounding 집행 지점을 게이트웨이 하나로 유지.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import altair as alt
import httpx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.meta import combined_index, load_arm_history  # noqa: E402
from eval.perf import drawdown_series, perf_stats  # noqa: E402
from eval.rolling import ROLLING_K, rolling_report  # noqa: E402
from gui.panels import (  # noqa: E402
    decision_for_day,
    kill_switch_active,
    list_observation_days,
    load_intramarket_weights,
    load_latest_requests,
    load_launchd_jobs,
    load_market_allocation,
    load_observation,
    load_pricing,
    load_regime,
    market_health,
    read_recent_decisions,
    usage_cost_report,
    veto_rows,
)
from harness.env import load_env  # noqa: E402
from risk.engine import RiskLimits  # noqa: E402
from interaction.briefing import build_briefing  # noqa: E402
from interaction.context import build_context  # noqa: E402

STATE = ROOT / "data" / "state"
VIRTUAL = STATE / "virtual"
OBS_DIR = STATE / "observations"
LOG_DIR = ROOT / "data" / "logs"
REQUESTS_DIR = ROOT / "data" / "requests"
MARKETS = ("CRYPTO", "US", "KR")
ARMS = ("llm", "llm_base", "bh", "random")
PERIODS_PER_YEAR = {"CRYPTO": 365.0, "US": 252.0, "KR": 252.0}  # 연율화 계수 (연 거래일 수)
REGIME_BADGE = {"UPTREND": "🟢", "UNDER_PRESSURE": "🟡", "CORRECTION": "🔴"}
STALE_DAYS = 2  # 마지막 결정이 이 일수 초과로 오래되면 staleness 경고

st.set_page_config(page_title="trading-agent", page_icon="📈", layout="wide")


def load_equity_frame(market: str) -> pd.DataFrame | None:
    """가상 arm equity 곡선 → wide DataFrame (index=day, columns=arm)."""
    series = {}
    for arm in ARMS:
        history = load_arm_history(VIRTUAL, market, arm)
        if history:
            series[arm] = pd.Series(
                [h["equity"] for h in history], index=[h["day"] for h in history]
            )
    return pd.DataFrame(series) if series else None


@st.cache_data(ttl=60)
def load_context() -> dict:
    return build_context(ROOT)


def pie(data: dict[str, float], title: str) -> None:
    """비중 dict → 도넛 파이. 0/음수 비중은 제외. 데이터 없으면 캡션."""
    rows = [{"label": k, "value": v} for k, v in data.items() if v and v > 0]
    if not rows:
        st.caption(f"{title}: 데이터 없음")
        return
    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_arc(innerRadius=45)
        .encode(
            theta=alt.Theta("value:Q", stack=True),
            color=alt.Color("label:N", legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["label:N", alt.Tooltip("value:Q", format=".1%")],
        )
        .properties(title=title, height=240)
    )
    st.altair_chart(chart, use_container_width=True)


tab_dash, tab_obs, tab_chat, tab_ops = st.tabs(["📊 대시보드", "🔭 관측", "💬 챗", "🔧 운영"])


# ── 대시보드 ──

with tab_dash:
    context = load_context()
    by_kind: dict[str, list[dict]] = {}
    for item in context["items"]:
        by_kind.setdefault(item["kind"], []).append(item)

    # ── 안전·헬스 배너 (자율 운용 최상단 요소) ──
    if kill_switch_active(STATE):
        st.error("🛑 KILL_SWITCH 활성 — 실주문 정지 중. 해제: `rm data/state/KILL_SWITCH`")
    mdd_circuit = RiskLimits().mdd_circuit
    today = date.today()
    health_cols = st.columns(len(MARKETS))
    for col, market in zip(health_cols, MARKETS):
        h = market_health(LOG_DIR, market, today)
        with col:
            if h["last_day"] is None:
                st.caption(f"**{market}** · 결정 로그 없음")
                continue
            stale = h["days_stale"] is not None and h["days_stale"] > STALE_DAYS
            tripped = h["mdd"] is not None and h["mdd"] >= mdd_circuit
            age = f"{h['days_stale']}일 전" if h["days_stale"] is not None else h["last_day"]
            line = f"**{market}** · 최근 결정 {age}"
            if h["mdd"]:
                line += f" · MDD {h['mdd']:.1%}"
            if tripped:
                st.error(line + " · 🛑 서킷")
            elif stale:
                st.warning(line + " · ⏳ stale")
            else:
                st.success(line)
            if h["violation_days"]:
                st.caption("최근 risk 클램프: " + ", ".join(h["violation_days"]))

    # ── 국면(regime) 배너 — shadow(결정·리스크 미개입, 관측만) ──
    regime = load_regime(STATE)
    if regime:
        chips = []
        for market in MARKETS:
            r = regime.get(market)
            if not r:
                continue
            badge = REGIME_BADGE.get(r["state"], "⚪")
            chips.append(f"{badge} **{market}** {r['state']} (낙폭 {r.get('drawdown', 0):.1%})")
        if chips:
            st.caption("시장 국면 (shadow — 결정·리스크 미개입) · " + " · ".join(chips))
    st.divider()

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
            cols[3].metric("메모리 델타", f"{meta['ret_pct'] - base_meta['ret_pct']:+.3f}%p")

    st.divider()
    st.subheader("자본 배분")
    alloc = load_market_allocation(VIRTUAL, STATE / "meta_shadow.json")
    ca, cb = st.columns(2)
    with ca:
        pie(alloc["current"], "마켓별 — 현재 (가상 equity 비중)")
    with cb:
        if alloc["target"]:
            pie(alloc["target"], "마켓별 — 메타 목표 (shadow 제안)")
        else:
            st.caption("마켓별 메타 목표: shadow 제안 없음 (paper_step 이 쌓으면 표시)")

    st.caption("마켓 내 포트폴리오 구성 — 목표 배분 벡터 (CASH 포함)")
    for col, market in zip(st.columns(len(MARKETS)), MARKETS):
        with col:
            pie(load_intramarket_weights(STATE, market), market)

    for market in MARKETS:
        frame = load_equity_frame(market)
        if frame is None:
            continue
        st.subheader(f"{market} — 가상 4-arm equity")
        st.line_chart(frame)
        risk = next((i for i in by_kind.get("risk_state", []) if i["id"] == f"risk:{market}"), None)
        if risk:
            st.caption(f"현재 목표 배분: `{risk['content']['target_weights']}` · equity 고점: {risk['content']['peak_equity']}")

        # 위험조정 성과 지표 (arm × 지표) + 드로다운(언더워터)
        ppy = PERIODS_PER_YEAR.get(market, 252.0)
        stat_rows: dict[str, dict] = {}
        dd_frame: dict[str, pd.Series] = {}
        for arm in frame.columns:
            series = frame[arm].dropna()
            eq = [float(v) for v in series.tolist()]
            s = perf_stats(eq, ppy)
            if s:
                stat_rows[arm] = {
                    "n": s["n"], "수익률": s["total_return"], "연변동성": s["ann_vol"],
                    "Sharpe": s["sharpe"], "Sortino": s["sortino"], "Calmar": s["calmar"],
                    "MDD": s["mdd"], "승률": s["win_rate"], "avg win": s["avg_win"],
                    "avg loss": s["avg_loss"], "best": s["best"], "worst": s["worst"],
                }
            if arm in ("llm", "bh") and len(eq) > 1:
                dd_frame[arm] = pd.Series(drawdown_series(eq), index=series.index)
        if stat_rows:
            st.caption(f"위험조정 성과 (일간 · rf=0 · 연율화 √{int(ppy)} · n=관측일, 짧으면 노이즈 큼)")
            perf_df = pd.DataFrame(stat_rows).T
            pct = ("수익률", "연변동성", "MDD", "승률", "avg win", "avg loss", "best", "worst")
            fmt = {c: "{:.2%}" for c in pct}
            fmt.update({c: "{:.2f}" for c in ("Sharpe", "Sortino", "Calmar")})
            fmt["n"] = "{:.0f}"
            st.dataframe(perf_df.style.format(fmt, na_rep="—"), use_container_width=True)
        if dd_frame:
            st.caption("드로다운 (언더워터) — llm vs bh")
            st.line_chart(pd.DataFrame(dd_frame))

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


# ── 관측 (에이전트가 그때 본 것 + 그 관측이 낳은 결정) ──

with tab_obs:
    st.caption(
        "에이전트가 그때 본 관측(OHLC·뉴스·feature)과 그 관측이 낳은 결정을 나란히. "
        "스냅샷은 일일 루프가 결정 시점에 기록 — 브로커 API 미호출, 읽기 전용."
    )
    obs_market = st.selectbox("시장", MARKETS, key="obs_market")
    days = list_observation_days(OBS_DIR, obs_market)
    decisions = read_recent_decisions(LOG_DIR, obs_market)

    if not days:
        st.info(f"{obs_market} 관측 스냅샷이 아직 없음 — 다음 스케줄 런부터 기록됩니다.")
    else:
        day = st.selectbox("관측일 (asof)", days, key="obs_day")
        snap = load_observation(OBS_DIR, obs_market, day)
        decision = decision_for_day(decisions, day)

        col_obs, col_dec = st.columns(2)
        with col_obs:
            st.markdown(f"#### 🔭 관측 — window `{snap['window'][0]} ~ {snap['window'][1]}`")
            closes = {
                sym: {b["day"]: b["close"] for b in bars}
                for sym, bars in snap["bars"].items() if bars
            }
            if closes:
                st.caption("종가 (관측 윈도우)")
                st.line_chart(pd.DataFrame(closes))
            for sym, bars in snap["bars"].items():
                if bars:
                    st.caption(f"`{sym}` OHLCV")
                    st.dataframe(pd.DataFrame(bars).set_index("day"), use_container_width=True)
            st.markdown("**뉴스**")
            if snap["news"]:
                for n in snap["news"]:
                    pub = n["published_at"][:10]
                    if n.get("url"):
                        st.markdown(f"- `{pub}` [{n['headline']}]({n['url']}) · _{n['source']}_")
                    else:
                        st.markdown(f"- `{pub}` {n['headline']} · _{n['source']}_")
            else:
                st.caption("이 윈도우에 수집된 뉴스 없음")

        with col_dec:
            st.markdown("#### 🎯 이 관측이 낳은 결정")
            if decision is None:
                st.caption("이 날짜의 결정 로그 없음 (관측만 기록되었거나 결정 실패).")
            else:
                feats = decision["features"]
                if feats:
                    st.caption("관측 feature (심볼 × 정예 지표)")
                    # features 는 {심볼: {지표: 값}} 중첩 — 심볼을 행, 지표를 열로
                    if all(isinstance(v, dict) for v in feats.values()):
                        st.dataframe(pd.DataFrame(feats).T.round(4), use_container_width=True)
                    else:
                        st.dataframe(pd.DataFrame([feats]), use_container_width=True, hide_index=True)
                st.caption("목표 배분 (risk 통과 후)")
                st.json(decision["weights"])
                if decision.get("weights_pre_risk"):
                    st.caption(f"risk 전 제안 배분: `{decision['weights_pre_risk']}`")
                if decision["rationale"]:
                    st.markdown(f"**근거**: {decision['rationale']}")
                cites = decision["cited_signal_ids"] + decision["cited_memory_ids"]
                if cites:
                    st.caption("인용: " + " · ".join(f"`{c}`" for c in cites))
                if decision["risk_violations"]:
                    st.warning("risk 위반 → 클램프: " + "; ".join(decision["risk_violations"]))
                if decision["debate"]:
                    with st.expander(f"🗣 debate ({decision['debate'].get('trigger')})"):
                        st.json(decision["debate"])

        st.divider()
        st.subheader("배분 변화 타임라인")
        if decisions:
            wide: dict[str, dict[str, float]] = {}
            for r in decisions:
                for sym, w in r["weights"].items():
                    wide.setdefault(sym, {})[r["day"]] = w
            st.area_chart(pd.DataFrame(wide).fillna(0.0))

        st.subheader("risk veto/클램프 타임라인")
        vr = veto_rows(decisions)
        if vr:
            st.dataframe(pd.DataFrame(vr), use_container_width=True, hide_index=True)
        else:
            st.caption("최근 창에 risk 위반 없음")


# ── 챗 (게이트웨이 프록시) ──

with tab_chat:
    st.caption(
        "게이트웨이 `/chat` 프록시 — 답변은 grounding(근거 ID 인용) 강제. "
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

    @st.fragment(run_every="15s")
    def launchd_panel() -> None:
        from datetime import datetime

        st.subheader("스케줄 잡 상태 (launchd) · 15초 자동 갱신")
        st.caption(
            "out/err 로그 tail 기반 추정 — 파일만 읽음(launchctl 미호출). stderr 는 과거 로그가 "
            "누적되므로 실패가 아니라 확인 힌트. 정확한 종료코드는 tail 을 직접 확인."
        )
        jobs = load_launchd_jobs(LOG_DIR)
        if not jobs:
            st.caption("launchd 로그 없음 — 스케줄 잡이 로그를 남기면 여기 표시된다.")
            return
        names = {"main": "paper_step (CRYPTO/US · 23:00)", "kr": "paper_step (KR · 10:00)",
                 "alpha": "alpha_lab (일요일)", "requests": "request_capabilities (매월 1일 20:30)"}
        badge = {"ok": "✅ 성공", "error": "⚠️ 오류", "unknown": "❓ 불명"}
        for j in jobs:
            title = names.get(j["job"], j["job"])
            last = j["last_run"].replace("T", " ") if j["last_run"] else "?"
            hint = " · 🟡 stderr 있음" if j["has_stderr"] else ""
            with st.expander(f"{badge.get(j['status'], '❓')} · {title} · 최근 {last}{hint}"):
                if j["out_tail"]:
                    st.caption("stdout (tail)")
                    st.code("\n".join(j["out_tail"]))
                if j["err_tail"]:
                    st.caption("stderr (tail)")
                    st.code("\n".join(j["err_tail"]))
        st.caption(f"조회: {datetime.now().isoformat(timespec='seconds')}")

    launchd_panel()
    st.divider()

    st.subheader("LLM 비용·토큰 (라우터 usage 로그)")
    st.caption(
        "모든 프로바이더 호출을 라우터 초크포인트에서 기록 — 결정·debate·챗·alpha·"
        "reflection·self-improve·capability 전부 포함. 토큰은 사실로 저장하고 비용(USD)은 "
        "단가표로 파생(임베딩 제외). 단가는 `data/state/llm_pricing.json` 으로 갱신."
    )
    usage = usage_cost_report(LOG_DIR, load_pricing(STATE))
    if usage["total_in"] or usage["total_out"]:
        uc = st.columns(3)
        uc[0].metric("누적 입력 토큰", f"{usage['total_in']:,}")
        uc[1].metric("누적 출력 토큰", f"{usage['total_out']:,}")
        uc[2].metric("누적 비용 (USD, 추정)", f"${usage['total_cost']:,.2f}")
        if usage["daily"]:
            df = pd.DataFrame(usage["daily"]).set_index("day")
            st.caption("일별 비용 (USD, 추정)")
            st.bar_chart(df["cost"])
            st.caption("일별 토큰 (in/out)")
            st.line_chart(df[["in", "out"]])
        if usage["by_purpose"]:
            st.caption("목적별 누적 (비용 내림차순)")
            st.dataframe(pd.DataFrame(usage["by_purpose"]), use_container_width=True, hide_index=True)
        if usage["unpriced"]:
            st.warning(
                "단가 미등록 모델 → 비용 0 처리. `data/state/llm_pricing.json` 에 추가: "
                + ", ".join(f"`{m}`" for m in usage["unpriced"])
            )
    else:
        st.caption("아직 usage 로그 없음 — 다음 LLM 호출부터 `data/logs/USAGE/` 에 쌓인다.")

    st.divider()

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

    st.subheader("월간 self-improve 제안서 (승인은 코드/문서 경로로만)")
    proposals = sorted((ROOT / "data" / "proposals").glob("*.md")) if (ROOT / "data" / "proposals").exists() else []
    if proposals:
        pick = st.selectbox("제안서", [p.name for p in proposals])
        st.markdown((ROOT / "data" / "proposals" / pick).read_text(encoding="utf-8"))
    else:
        st.caption("아직 없음 — 매월 1일 21:00 자동 생성.")

    st.subheader("에이전트 능력 갭 요구 (조달·배선은 사용자 경로로만)")
    st.caption(
        "에이전트가 측정된 갭을 근거로 요구하는 데이터/도구. 정책 외(유료·틱·고빈도)는 "
        "'의도적으로 회피하는 게임'으로 라벨링 — 자동 획득 없음, 읽기 전용."
    )
    reqs = load_latest_requests(REQUESTS_DIR)
    if not reqs or not reqs.get("requests"):
        st.caption("아직 없음 — 측정된 갭 신호가 쌓이면 생성된다 (근거 없으면 미생성).")
    else:
        st.caption(f"생성월: {reqs.get('month', '?')}")
        for r in reqs["requests"]:
            out = r.get("policy_class") == "out"
            badge = "⚠️ 회피 게임 (정책 외)" if out else "✅ 정책 내 (무료·일간)"
            with st.expander(f"{badge} · {r.get('proposed_capability', '(제안 미상)')}"):
                if out and r.get("policy_note"):
                    st.warning(r["policy_note"])
                st.markdown(f"**갭**: {r.get('gap', '')}")
                st.markdown(f"**측정 영향**: {r.get('measured_impact', '')}")
                st.markdown(f"**예상 비용**: {r.get('est_cost', '')}")
                cites = r.get("evidence_ids") or []
                if cites:
                    st.caption("근거: " + " · ".join(f"`{c}`" for c in cites))
