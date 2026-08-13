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

from eval.meta import combined_index, load_arm_history, load_meta_shadow  # noqa: E402
from eval.perf import drawdown_series, perf_stats  # noqa: E402
from eval.rolling import ROLLING_K, meta_shadow_delta, rolling_report  # noqa: E402
from gui.panels import (  # noqa: E402
    admission_progress,
    counterfactual_ledger,
    decision_for_day,
    episodic_ledger,
    exposure_turnover,
    kill_switch_active,
    latest_meta_event,
    list_observation_days,
    load_intramarket_weights,
    load_latest_requests,
    load_launchd_jobs,
    load_market_allocation,
    load_observation,
    load_regime,
    market_health,
    monthly_proposals,
    promoted_memories,
    read_recent_decisions,
    running_jobs,
    scenario_outcomes,
    session_draft_diff,
    session_proposals,
    treasury_dryrun_report,
    usage_report,
    veto_rows,
    weekly_reflections,
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
    st.altair_chart(chart, width="stretch")


@st.fragment(run_every=10)
def running_banner() -> None:
    """실행 중인 잡을 최상단에 계속 표시 — 탭과 무관하게 보이도록 tabs 앞에 둔다.

    로그의 `status=ok` 는 그 시장의 주문 왕복이 끝났다는 뜻일 뿐 런 전체의 완료가 아니다.
    그걸 완료로 읽고 기기를 재우면 뒤따르는 단계가 통째로 유실되므로(실제로 주간 회고를
    그렇게 잃었다), 살아 있는 프로세스를 직접 확인해 별도로 알린다. 10초마다 자동 갱신.
    """
    jobs = running_jobs(STATE)
    if not jobs:
        return
    for job in jobs:
        seconds = job["elapsed_s"]
        elapsed = f"{seconds // 60}분 {seconds % 60}초 경과" if seconds is not None else "경과 시간 미상"
        st.warning(f"🔄 **실행 중** — {job['label']} · {elapsed} (pid {job['pid']})")
    st.caption(
        "이 표시가 떠 있는 동안 기기를 재우면 남은 단계(메모리 승격·주간 회고·shadow 채널)가 "
        "유실됩니다. 잡 로그의 `status=ok` 는 해당 시장의 주문 왕복이 끝났다는 뜻일 뿐 "
        "런 전체의 완료가 아닙니다."
    )


running_banner()

tab_dash, tab_obs, tab_mem, tab_chat, tab_ops = st.tabs(
    ["📊 대시보드", "🔭 관측", "📓 메모리", "💬 챗", "🔧 운영"]
)


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
            st.dataframe(perf_df.style.format(fmt, na_rep="—"))
        if dd_frame:
            st.caption("드로다운 (언더워터) — llm vs bh")
            st.line_chart(pd.DataFrame(dd_frame))

    st.subheader("노출·회전율 & 기간 수익률")
    st.caption("결정 배분에서 파생한 현금/투자 비중·turnover(리스크 캡 0.5 대비)와 arm 수익률 리샘플.")
    for market in MARKETS:
        decisions = read_recent_decisions(LOG_DIR, market)
        et = exposure_turnover(decisions)
        history = load_arm_history(VIRTUAL, market, "llm")
        if not et and not history:
            continue
        st.markdown(f"**{market}**")
        ce, cr = st.columns(2)
        with ce:
            if et:
                et_df = pd.DataFrame(et).set_index("day")
                st.caption("현금 vs 투자 비중")
                st.area_chart(et_df[["invested", "cash"]])
                if et_df["turnover"].notna().any():
                    st.caption("일일 turnover (0.5·Σ|Δw|)")
                    st.bar_chart(et_df["turnover"])
        with cr:
            if history:
                eq = pd.Series(
                    [h["equity"] for h in history],
                    index=pd.to_datetime([h["day"] for h in history]),
                )
                for label, rule in (("주간", "W"), ("월간", "ME")):
                    per = eq.resample(rule).last().pct_change().dropna()
                    if not per.empty:
                        st.caption(f"{label} 수익률 (llm)")
                        st.bar_chart(per)
                        break  # 표본이 짧으면 주간만, 쌓이면 주간 우선

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
        st.dataframe(pd.DataFrame(rows), hide_index=True)

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
                    st.dataframe(pd.DataFrame(bars).set_index("day"))
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
                        st.dataframe(pd.DataFrame(feats).T.round(4))
                    else:
                        st.dataframe(pd.DataFrame([feats]), hide_index=True)
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
                if decision.get("scenario_expected") or decision.get("scenario_invalidation"):
                    st.caption("시나리오 (사후 검증 기준)")
                    if decision.get("scenario_expected"):
                        st.markdown(f"- **예상**: {decision['scenario_expected']}")
                    if decision.get("scenario_invalidation"):
                        st.markdown(f"- **무효화 조건**: {decision['scenario_invalidation']}")

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
            st.dataframe(pd.DataFrame(vr), hide_index=True)
        else:
            st.caption("최근 창에 risk 위반 없음")

        st.subheader("시나리오 vs 실제 (익일 수익률)")
        st.caption(
            "결정 시점 예상·무효화 조건과 익일 arm 실현 수익률을 나란히 — 무효화는 자연어라 "
            "자동 판정하지 않고, 사람이 조건 충족 여부를 읽는다."
        )
        so = scenario_outcomes(decisions, load_arm_history(VIRTUAL, obs_market, "llm"))
        if so:
            so_df = pd.DataFrame([
                {"day": r["day"], "예상": r["expected"], "무효화 조건": r["invalidation"],
                 "익일 수익률": None if r["next_day_return"] is None else round(r["next_day_return"] * 100, 3)}
                for r in so
            ])
            st.dataframe(so_df, hide_index=True)
        else:
            st.caption("이 시장에 시나리오 기록이 아직 없음.")


# ── 메모리 (회고 · 승격 파이프라인 · 원장) ──

with tab_mem:
    st.caption(
        "메모리 store 를 그대로 읽는다 — 읽기 전용. 승격·retire·importance 갱신은 자동 "
        "게이트의 단독 권한이라 여기에 조작 UI 를 두지 않는다."
    )
    mem_market = st.selectbox("시장", MARKETS, key="mem_market")

    st.subheader("주간 회고")
    st.caption(
        "일요일 스텝에서 생성 — 최근 7일 결정(결과 기입분)의 승률·평균 성과와 인용된 "
        "신호/메모리별 기여. 요약문은 통계 리포트를 LLM 이 옮겨 쓴 것으로, 판정 근거는 "
        "어디까지나 아래 수치다."
    )
    reflections = weekly_reflections(ROOT / "data" / "memory.sqlite", mem_market, date.today())
    if not reflections:
        st.caption("아직 회고 없음 — 결정이 쌓이고 첫 일요일 스텝이 돌면 생성된다.")
    newest_ok = next((r["day"] for r in reflections if r["status"] == "ok"), None)
    for r in reflections:
        rep = r["report"]
        if r["status"] == "ok":
            # 승률은 동점 제외 분모라 전부 동점인 주에는 None — 그때는 승률 자리를 비운다
            wr = rep.get("win_rate")
            ties = rep.get("n_ties") or 0
            title = (
                f"✅ {r['day']} · {rep.get('n_decisions', 0)}건"
                + (f" (무승부 {ties})" if ties else "")
                + f" · 승률 {'—' if wr is None else format(wr, '.0%')}"
                + f" · 평균 {rep.get('mean_outcome', 0):+.3%}"
            )
        elif r["status"] == "missing":
            title = f"⚠️ {r['day']} · 미생성 (대상 결정 {r['eligible']}건은 있었음)"
        else:
            title = f"⚪ {r['day']} · 표본 부족 ({r['eligible']}건, 최소 미달)"
        # 가장 최근 '생성된' 회고만 펼쳐 둔다 — 최신 주가 미생성이어도 읽을 게 보이도록
        with st.expander(title, expanded=r["day"] == newest_ok):
            if r["status"] == "missing":
                st.warning(
                    "이 주의 일요일 스텝이 돌지 않아 회고가 유실됐다. 회고는 그 주에만 "
                    "생성되므로 이후 스텝이 소급하지 않는다."
                )
                continue
            if r["status"] == "insufficient":
                st.caption("결과가 기입된 결정이 최소 건수에 못 미쳐 생성 대상이 아니었다(정상).")
                continue
            st.markdown(r["summary"])
            st.caption(f"창: `{rep.get('window', ['?', '?'])[0]} ~ {rep.get('window', ['?', '?'])[1]}`")
            # 기여 항목은 많아야 두어 건이고 ID 가 길다 — 표로 만들면 열이 잘려 오히려 안 읽힌다
            for label, credit in (("신호 기여", rep.get("signal_credit")),
                                  ("메모리 기여", rep.get("memory_credit"))):
                st.caption(label)
                if credit:
                    for cid, v in credit.items():
                        st.markdown(f"- `{cid}` — {v['n']}건, 평균 {v['mean_outcome']:+.3%}")
                else:
                    st.caption("— 인용 없음")

    st.divider()
    st.subheader("승격 파이프라인 (episodic → semantic/procedural)")
    st.caption(
        "패턴별 반복 표본과 부호검정 p 값 — 승격 게이트가 실제로 보는 숫자를 그대로 표시. "
        "단일 매매로는 승격되지 않으며, 반복 n 과 유의성이 게이트다. `pending` 은 아직 "
        "다음 관측이 없어 결과가 안 채워진 건수, `ties` 는 행동이 결과를 바꾸지 못한 "
        "건수(휴장·배분 무변동)로 부호검정 표본에서 빠진다."
    )
    prog = admission_progress(ROOT / "data" / "memory.sqlite", mem_market)
    ledger = episodic_ledger(ROOT / "data" / "memory.sqlite", mem_market, limit=200)
    promoted = promoted_memories(ROOT / "data" / "memory.sqlite", mem_market)

    mc = st.columns(4)
    mc[0].metric("episodic 기록", len(ledger))
    mc[1].metric("후보 패턴", len(prog))
    mc[2].metric("게이트 통과", sum(1 for r in prog if r["stage"].startswith("게이트")))
    mc[3].metric("승격됨 (semantic+procedural)", len(promoted))

    if prog:
        prog_df = pd.DataFrame(prog).set_index("pattern")
        st.dataframe(
            prog_df.style.format(
                {"mean_outcome": "{:+.3%}", "p_value": "{:.3f}"}, na_rep="—"
            )
        )
    else:
        st.caption("아직 결과가 기입된 패턴 표본이 없다.")

    st.caption("승격 이후 (probation → active/retired)")
    if promoted:
        st.dataframe(pd.DataFrame(promoted), hide_index=True)
    else:
        st.caption("승격된 교훈 없음 — 위 표에서 표본·유의성 진행도를 확인.")

    st.divider()
    st.subheader("episodic 원장 (일간 결정 기록)")
    st.caption(
        "결정 1건 = 기록 1건. `outcome` 은 행동 수익 − 무행동(직전 배분 유지) 수익으로 "
        "다음 관측에서 소급 기입된다 — '안 사도 올랐다'를 '잘한 매수'와 구분하는 값."
    )
    if ledger:
        st.dataframe(
            pd.DataFrame(ledger).style.format(
                {"outcome": "{:+.3%}", "importance": "{:.2f}"}, na_rep="— (대기)"
            ),
            hide_index=True,
        )
    else:
        st.caption("아직 기록 없음.")

    st.divider()
    st.subheader("veto 원장 (집행되지 않은 원안)")
    st.caption(
        "Forbidden 패턴에 걸려 동결된 결정의 '원안대로 갔다면' 결과. 집행은 막되 가상 "
        "성과는 계속 계측한다 — veto 는 배분을 직전 값으로 얼려 행동을 hold 로 만들기 "
        "때문에 막힌 패턴의 표본이 그대로 끊긴다. 이 기록이 없으면 한 번 선 제약을 "
        "무효화할 증거가 영영 모이지 않는다. `outcome` 이 양수로 쌓이면 퇴출 심사가 "
        "제약을 푼다. 이 표본은 제약을 해제할 때만 쓰이고 승격에는 쓰이지 않는다."
    )
    vetoed = counterfactual_ledger(ROOT / "data" / "memory.sqlite", mem_market, limit=200)
    if vetoed:
        vc = st.columns(3)
        vc[0].metric("veto 건수", len(vetoed))
        vc[1].metric("veto 적중", sum(1 for r in vetoed if (r["outcome"] or 0) < 0))
        vc[2].metric("veto 손해", sum(1 for r in vetoed if (r["outcome"] or 0) > 0))
        st.dataframe(
            pd.DataFrame(vetoed).style.format(
                {"outcome": "{:+.3%}", "executed_outcome": "{:+.3%}"}, na_rep="— (대기)"
            ),
            hide_index=True,
        )
    else:
        st.caption("하드 veto 발동 이력 없음 — Forbidden 패턴이 아직 승격되지 않았다.")


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
        st.session_state.chat_draft = None  # 마지막 제안초안 응답 (세션 종료 시 비운다)

    # 토론 결론은 시장 네임스페이스로 기록된다 — 고르지 않으면 KR 토론이 다른 시장의
    # 저널에 앉는다. 세션 시작 시점의 값만 쓰이므로 진행 중에는 바꿀 수 없다.
    chat_market = st.selectbox(
        "시장 (토론 결론이 기록될 네임스페이스)",
        MARKETS,
        key="chat_market",
        disabled=bool(st.session_state.chat_session_id),
    )

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
                json={
                    "question": question,
                    "session_id": st.session_state.chat_session_id,
                    "market": chat_market,
                },
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

    # 토론 마감 — 결론 요약을 episodic 저널에 남긴다. 결정에 개입하지 않는다:
    # 승격 통계는 outcome·pattern_key 가 있는 엔트리만 세고, 결정 프롬프트로 회수되는
    # 것은 승격을 통과한 semantic·procedural 뿐이다. 저널은 사후 감사용 기록이다.
    if st.session_state.chat_session_id:
        st.divider()
        col_draft, col_close = st.columns(2)
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        # 제안초안 — 마감과 달리 세션을 유지한다(초안을 보고 토론을 이어갈 수 있어야 한다).
        if col_draft.button("제안초안 요청 — 플레이북 diff"):
            try:
                resp = httpx.post(
                    f"{gateway}/discuss/propose",
                    json={"session_id": st.session_state.chat_session_id},
                    headers=headers,
                    timeout=240,
                )
                st.session_state.chat_draft = (
                    resp.json() if resp.status_code == 200
                    else {"error": f"게이트웨이 오류 {resp.status_code}: {resp.text[:200]}"}
                )
            except httpx.HTTPError as e:
                st.session_state.chat_draft = {"error": f"게이트웨이 연결 실패: {e}"}

        if col_close.button("토론 마감 — 결론을 기록", type="primary"):
            try:
                resp = httpx.post(
                    f"{gateway}/discuss/conclude",
                    json={"session_id": st.session_state.chat_session_id},
                    headers=headers,
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"기록됨 — `{data['memory_id']}` ({chat_market})")
                    st.caption(data.get("note", ""))
                    # 게이트웨이가 세션을 지웠다 — 남겨두면 다음 질문이 죽은 id 를 보낸다.
                    st.session_state.chat_session_id = None
                    st.session_state.chat_draft = None  # 다음 세션에 남의 초안이 뜨지 않게
                else:
                    st.error(f"게이트웨이 오류 {resp.status_code}: {resp.text[:200]}")
            except httpx.HTTPError as e:
                st.error(f"게이트웨이 연결 실패: {e}")

        draft = st.session_state.get("chat_draft")
        if draft:
            if draft.get("error"):
                st.error(draft["error"])
            elif not (draft.get("diff") or "").strip():
                # 설계된 결과다 — 억지로 diff 를 만들지 않는다. 실패와 구분해 보여준다.
                st.info(f"변경 없음 — 이 토론은 플레이북 수정으로 이어지지 않았다. ({draft['path']})")
            else:
                if draft["applies"]:
                    st.success(f"초안 생성 — `{draft['path']}`")
                else:
                    st.warning(f"초안 생성 — `{draft['path']}` · ⚠️ 현재 원문에 붙지 않는다: {draft['reason']}")
                st.caption(draft.get("note", ""))
                st.code(draft["diff"], language="diff")

        st.caption(
            "**초안**은 플레이북 변경안을 파일로 쓸 뿐 적용하지 않는다 — 대상 파일을 직접 "
            "고칠 때만 반영된다. **마감**하면 결론 요약이 이 시장의 episodic 저널에 남는다. "
            "둘 다 검증 교훈이 아니며 결정에는 개입하지 않는다 — 승격은 admission 게이트를 "
            "따로 통과해야 한다."
        )


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
        badge = {"ok": "✅ 성공", "partial": "🟠 부분 실패", "error": "⚠️ 오류", "unknown": "❓ 불명"}
        for j in jobs:
            title = names.get(j["job"], j["job"])
            last = j["last_run"].replace("T", " ") if j["last_run"] else "?"
            hint = " · 🟡 stderr 있음" if j["has_stderr"] else ""
            chips = " ".join(
                f"{'✅' if s == 'ok' else '⚠️'}{m}" for m, s in sorted(j.get("markets", {}).items())
            )
            chip_str = f" · {chips}" if chips else ""
            with st.expander(f"{badge.get(j['status'], '❓')} · {title}{chip_str} · 최근 {last}{hint}"):
                if j.get("markets"):
                    st.caption("시장별 최신 상태 — 한 잡이 여러 시장을 묶어 돌므로 한 시장 실패가 다른 시장 실패를 뜻하지 않음.")
                if j["out_tail"]:
                    st.caption("stdout (tail)")
                    st.code("\n".join(j["out_tail"]))
                if j["err_tail"]:
                    st.caption("stderr (tail)")
                    st.code("\n".join(j["err_tail"]))
        st.caption(f"조회: {datetime.now().isoformat(timespec='seconds')}")

    launchd_panel()
    st.divider()

    st.subheader("LLM 토큰 사용량 (라우터 usage 로그)")
    st.caption(
        "모든 프로바이더 호출을 라우터 초크포인트에서 기록 — 결정·debate·챗·alpha·"
        "reflection·self-improve·capability 전부 포함. 토큰(사실)만 집계하고 비용(USD)은 "
        "추정하지 않는다 — 실제 청구는 각 프로바이더 콘솔에서 확인. 임베딩 호출은 기록되지 않음."
    )
    usage = usage_report(LOG_DIR)
    if usage["total_in"] or usage["total_out"]:
        uc = st.columns(2)
        uc[0].metric("누적 입력 토큰", f"{usage['total_in']:,}")
        uc[1].metric("누적 출력 토큰", f"{usage['total_out']:,}")
        if usage["daily"]:
            df = pd.DataFrame(usage["daily"]).set_index("day")
            st.caption("일별 토큰 (in/out)")
            st.line_chart(df[["in", "out"]])
        if usage["by_model"]:
            st.caption("모델별 누적 토큰 (in+out 내림차순)")
            st.dataframe(pd.DataFrame(usage["by_model"]), hide_index=True)
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

    st.subheader("메타 shadow — 동적 배분 vs 고정균등 (집행 전 검증)")
    st.caption(
        "시장 간 동적 틸트(regime 기반)가 고정 1/3 을 이기는지 rolling delta. shadow — "
        "실자본 재배분 없음. 집행 승격은 델타>0·유의 + 실계좌 전환 후."
    )
    wbd = load_meta_shadow(STATE / "meta_shadow.json")
    if not wbd:
        st.caption("메타 제안 없음 — paper_step 이 쌓으면 표시.")
    else:
        line = f"제안 {len(wbd)}일 — "
        for arm in ("llm", "bh"):
            r = meta_shadow_delta(VIRTUAL, arm, wbd)
            if r is None:
                line += f"{arm}: 데이터 {ROLLING_K + 1}일 미만 · "
            else:
                p = f"p={r['p_value']:.3f}" if r["p_value"] is not None else "p=n/a"
                line += f"{arm}: 승률 {r['win_rate']:.0%} ({p}) · "
        st.markdown(line.rstrip(" ·"))
        meta_evt = latest_meta_event(LOG_DIR)
        if meta_evt:
            st.caption(
                f"최신 제안 ({meta_evt.get('day')}): `{meta_evt.get('weights')}` · "
                f"편차 L1 {meta_evt.get('deviation_l1')} · 근거 {meta_evt.get('cited')}"
            )

    st.subheader("Treasury 이체 dry-run (집행 전 · 결정론 가드)")
    st.caption("메타 제안 → 버킷 이체 계획을 dry-run 으로만 로깅(실집행·잔고변경 없음).")
    tr = treasury_dryrun_report(LOG_DIR)
    if tr is None:
        st.caption("Treasury dry-run 로그 없음 — run_treasury_step 이 쌓으면 표시.")
    else:
        plan = tr["plan"]
        pc = st.columns(3)
        pc[0].metric("버킷 목표", str(plan.get("bucket_target")))
        pc[1].metric("현재 split", str(plan.get("current_split")))
        pc[2].metric("이체 의도 수", plan.get("n_intents", 0))
        st.caption(f"한도: {plan.get('limits')}")
        if tr["intents"]:
            st.dataframe(
                pd.DataFrame([
                    {"from": i.get("from"), "to": i.get("to"), "금액": i.get("amount"),
                     "사유": i.get("reason"), "허용": i.get("would_allow"),
                     "위반": ", ".join(i.get("violations") or []),
                     "자동레그": i.get("auto_leg"), "집행": i.get("executed")}
                    for i in tr["intents"]
                ]),
                hide_index=True,
            )

    st.subheader("Alpha 팩터 라이브러리")
    lib_path = STATE / "alpha_library_CRYPTO.json"
    if lib_path.exists():
        factors = json.loads(lib_path.read_text(encoding="utf-8"))["factors"]
        st.dataframe(pd.DataFrame(factors), hide_index=True)

    PROPOSALS = ROOT / "data" / "proposals"

    st.subheader("월간 self-improve 제안서 (승인은 코드/문서 경로로만)")
    monthly = monthly_proposals(PROPOSALS)
    if monthly:
        pick = st.selectbox("제안서", monthly)
        st.markdown((PROPOSALS / pick).read_text(encoding="utf-8"))
    else:
        st.caption("아직 없음 — 매월 1일 21:00 자동 생성.")

    st.subheader("세션 제안초안 (토론에서 나온 플레이북 diff)")
    st.caption(
        "챗 토론에서 뽑은 초안이다. 승격 통계가 아니라 **사람과의 대화**에서 나왔으므로 "
        "위 월간 제안서와 출처가 다르다. 적용 경로는 없다 — 대상 파일을 직접 고칠 때만 "
        "반영된다. grounding=none 은 대화가 인용한 근거 없이 나온 초안이라는 뜻이고, "
        "applies=false 는 diff 가 현재 원문에 붙지 않는다는 뜻이다."
    )
    drafts = session_proposals(PROPOSALS)
    if not drafts:
        st.caption("아직 없음 — 챗 탭에서 토론 후 초안을 요청하면 생성된다.")
    else:
        st.dataframe(
            pd.DataFrame([
                {"파일": d["name"], "생성": d["created"][:16], "시장": d["market"],
                 "근거": f"{d['grounding']}({d['n_cited']})",
                 "적용가능": "✅" if d["applies"] else f"❌ {d['applies_reason'][:60]}",
                 "적용됨": "✅" if d["applied"] else "—"}
                for d in drafts
            ]),
            hide_index=True,
            width="stretch",
        )
        draft = st.selectbox("초안", [d["name"] for d in drafts], key="session_draft")
        row = next(d for d in drafts if d["name"] == draft)
        st.caption(
            f"대상 `{row['target']}` · 시장 {row['market'] or '-'} · 세션 `{row['session_id']}` · "
            f"근거 {row['grounding']}({row['n_cited']}건)"
            + ("" if row["applies"] else f" · ⚠️ 붙지 않음: {row['applies_reason']}")
        )
        st.code(session_draft_diff(PROPOSALS / draft) or "(diff 없음)", language="diff")

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
