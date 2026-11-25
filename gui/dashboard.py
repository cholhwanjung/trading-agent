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
from datetime import date, datetime, timezone
from pathlib import Path

import altair as alt
import httpx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adapters.news_kr import KR_STOCK_NAMES  # noqa: E402
from adapters.universe import ETF, ETF_REF, universe_meta  # noqa: E402
from eval.attribution import STAGES, stage_attribution  # noqa: E402
from eval.exposure import meta_alpha_decomposition, meta_exposure  # noqa: E402
from eval.index_bench import index_hist, load_index_series, normalized  # noqa: E402
from eval.meta import combined_index, load_arm_history, load_meta_shadow  # noqa: E402
from eval.perf import drawdown_series, perf_stats  # noqa: E402
from eval.rolling import (  # noqa: E402
    ROLLING_K,
    meta_rolling_report,
    meta_shadow_delta,
    rolling_report,
)
from gui.panels import (  # noqa: E402
    CHANNELS,
    FRESH_MAX_AGE,
    RUN_BAD,
    RUN_WARN,
    admission_progress,
    change_markers,
    channel_grid,
    citation_check,
    counterfactual_ledger,
    dead_channels,
    decision_for_day,
    episodic_ledger,
    eval_runs,
    execution_chain,
    exposure_turnover,
    freshness,
    kill_switch_active,
    latest_budget,
    latest_meta_event,
    list_observation_days,
    load_holding_weights,
    load_latest_requests,
    load_launchd_jobs,
    load_market_allocation,
    load_observation,
    load_regime,
    load_stall,
    market_health,
    memory_ladder,
    monthly_proposals,
    promoted_memories,
    read_recent_decisions,
    revision_spans,
    risk_interventions,
    risk_summary,
    run_ledger,
    run_summary,
    running_jobs,
    scenario_outcomes,
    scorecard,
    session_draft_diff,
    session_proposals,
    stage_moves,
    universe_rows,
    usage_breakdown,
    usage_report,
    veto_rows,
    weekly_reflections,
)
from harness.env import load_env  # noqa: E402
from risk.limits_config import load_limits  # noqa: E402
from scripts.run_paper_step import (  # noqa: E402
    CRYPTO_UNIVERSE,
    KR_UNIVERSE,
    TRADABLE,
    US_UNIVERSE,
)
from interaction.briefing import build_briefing  # noqa: E402
from interaction.context import build_context  # noqa: E402

STATE = ROOT / "data" / "state"
VIRTUAL = STATE / "virtual"
OBS_DIR = STATE / "observations"
LOG_DIR = ROOT / "data" / "logs"
REQUESTS_DIR = ROOT / "data" / "requests"
MARKETS = ("CRYPTO", "US", "KR")
# 관측 유니버스 — 일일 루프가 실제로 쓰는 정의를 그대로 읽는다(대시보드에 사본을 두지 않는다).
OBSERVED_UNIVERSE = {"CRYPTO": CRYPTO_UNIVERSE, "US": US_UNIVERSE, "KR": KR_UNIVERSE}
ARMS = ("llm", "llm_base", "bh", "random")
PERIODS_PER_YEAR = {"CRYPTO": 365.0, "US": 252.0, "KR": 252.0}  # 연율화 계수 (연 거래일 수)
REGIME_BADGE = {"UPTREND": "🟢", "UNDER_PRESSURE": "🟡", "CORRECTION": "🔴"}
STALE_DAYS = 2  # 마지막 결정이 이 일수 초과로 오래되면 staleness 경고

st.set_page_config(page_title="trading-agent", page_icon="📈", layout="wide")


def load_equity_frame(market: str) -> pd.DataFrame | None:
    """가상 arm equity 곡선 + 시장 지수 → wide DataFrame (index=day, columns=arm).

    지수는 수준(KOSPI 6900 · NASDAQ 26000)이 arm equity 와 스케일이 달라, arm 첫날의
    equity 에 맞춰 정규화해 같은 축에 올린다. 지수 원천이 없는 시장(CRYPTO)은 열이 없다.
    """
    series = {}
    for arm in ARMS:
        history = load_arm_history(VIRTUAL, market, arm)
        if history:
            series[arm] = pd.Series(
                [h["equity"] for h in history], index=[h["day"] for h in history]
            )
    if not series:
        return None
    idx, state = index_hist(STATE, market), load_index_series(STATE, market)
    if idx and state:
        first = min(s.index.min() for s in series.values())
        base = next(iter(series.values())).iloc[0]
        curve = normalized(idx, first)
        if curve:
            series[f"지수({state['index']})"] = pd.Series(
                [p["equity"] * base for p in curve], index=[p["day"] for p in curve]
            )
    return pd.DataFrame(series)


@st.cache_data(ttl=60)
def load_context() -> dict:
    return build_context(ROOT)


def latest_closes(market: str) -> dict[str, float]:
    """가장 최근 관측 스냅샷의 마지막 종가 — 트레이더가 예산을 비중으로 환산할 때 쓴 값과 같다."""
    days = list_observation_days(OBS_DIR, market)
    if not days:
        return {}
    snap = load_observation(OBS_DIR, market, days[0]) or {}
    return {s: bars[-1]["close"] for s, bars in (snap.get("bars") or {}).items() if bars}


def display_name(symbol: str) -> str:
    """화면 표기용 이름. 숫자 종목코드는 사람이 못 읽으므로 이름으로 바꾼다.

    바꾸는 것은 **숫자 코드뿐**이다 — `SCHX`·`BTC/USDT` 같은 티커는 그 자체가 통용되는
    이름이라 정식 상품명으로 늘리면 오히려 알아보기 어려워진다. 표시 계층 전용이며,
    관측·주문·로그가 쓰는 심볼은 브로커가 아는 코드 그대로 둔다.
    """
    if not symbol.isdigit():
        return symbol
    ref = ETF_REF.get(symbol)
    return ref.name if ref is not None else KR_STOCK_NAMES.get(symbol, symbol)


def cell(value: float | None, spec: str) -> str:
    """표 셀 문자열. 값을 미리 굳혀 넣는다 — 열마다 결측이 다른 뜻이라(미기록 vs 해당 없음)
    포매터의 결측 처리에 맡기면 그 구분이 빈칸 하나로 뭉개진다."""
    return format(value, spec) if value is not None else "—"


def pie(data: dict[str, float], title: str, domain: list[str] | None = None) -> None:
    """비중 dict → 도넛 파이. 0/음수 비중은 제외. 데이터 없으면 캡션.

    domain 을 주면 색을 그 순서로 고정한다 — 나란히 둔 두 도넛의 조각 수가 다르면
    같은 종목이 서로 다른 색을 받아 비교가 어긋난다. 범례는 색 고정과 무관하게 **실제
    조각만** 싣는다 — 없는 종목이 범례에 뜨면 그것도 담긴 것처럼 읽힌다.
    """
    rows = [{"label": k, "value": v} for k, v in data.items() if v and v > 0]
    if not rows:
        st.caption(f"{title}: 데이터 없음")
        return
    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_arc(innerRadius=45)
        .encode(
            theta=alt.Theta("value:Q", stack=True),
            color=alt.Color(
                "label:N",
                legend=alt.Legend(
                    title=None,
                    orient="bottom",
                    values=[r["label"] for r in rows] if domain else alt.Undefined,
                ),
                scale=alt.Scale(domain=domain) if domain else alt.Undefined,
            ),
            tooltip=["label:N", alt.Tooltip("value:Q", format=".1%")],
        )
        .properties(title=title, height=240)
    )
    st.altair_chart(chart, width="stretch")


# ── 에이전트 건강 (성과와 별개로 시스템이 제대로 돌았는가) ──

HEALTH_LEDGER_DAYS = 30  # 런 원장 창(달력일)
HEALTH_WINDOW = 60  # 개입·체결·인용 요약 창(최근 결정 수)
HEALTH_GRID_RUNS = 14  # 채널 격자 창(최근 결정 수)
DRIFT_NOTE = 0.05  # 체결 괴리 참고선 — 판정 임계가 아니라 눈금
SERIES_COLOR = "#2a78d6"
MUTED_INK = "#898781"
# 상태 → (색, 칸 글리프, 표기). 색만으로 뜻을 싣지 않는다 — 실패 계열 셋은 같은 색이고
# 글리프·툴팁·아래 표가 구분을 맡는다.
RUN_STATUS_STYLE = {
    "ok": ("#0ca30c", "", "정상"),
    "closed": ("#c3c2b7", "", "휴장"),
    "stalled": ("#fab219", "S", "집행 정지"),
    "late": ("#fab219", "L", "지연 실행"),
    "degraded": ("#ec835a", "D", "계좌 조회 실패"),
    "rejected": ("#d03b3b", "R", "주문 거부"),
    "error": ("#d03b3b", "E", "스텝 예외"),
    "missing": ("#d03b3b", "✕", "기록 없음"),
    "pending": ("#e1e0d9", "", "오늘(미판정)"),
}
# 단계 → (색, 표기). 순서 고정 — 결정이 실제로 지나는 순서다.
STAGE_STYLE = {
    "memory": ("#2a78d6", "교훈 블렌딩"),
    "debate": ("#eb6834", "토론 재결정"),
    "risk": ("#1baf7a", "리스크 적용"),
}
CHANNEL_LABELS = {
    "bars": "봉(종목 수)", "features": "feature(종목 수)", "news": "뉴스 수집",
    "disclosures": "공시 수집", "news_injected": "뉴스 → 프롬프트",
    "disclosures_injected": "공시 → 프롬프트", "fundamentals": "재무(종목 수)",
    "index_valuation": "지수 밸류에이션", "etf_premium": "ETF 괴리(종목 수)",
    "alpha_signals": "알파 신호", "budget": "계좌 예산", "lessons": "교훈(회수)",
    "upcoming_events": "예정 일정",
}


@st.cache_data(ttl=60)
def load_health(market: str, today_iso: str) -> dict:
    """한 시장의 건강 지표 묶음. 로그를 여러 번 훑으므로 60초 캐시한다."""
    today = date.fromisoformat(today_iso)
    history = read_recent_decisions(LOG_DIR, market, limit=10**6)
    recent = history[-HEALTH_WINDOW:]
    ledger = run_ledger(LOG_DIR, market, today, days=HEALTH_LEDGER_DAYS)
    risk_rows = risk_interventions(recent)
    grid = channel_grid(LOG_DIR, OBS_DIR, market, days=HEALTH_GRID_RUNS)
    spans = revision_spans(history)
    citations = citation_check(recent)
    dead = dead_channels(grid, market)
    store: dict[str, int] = {}
    for mem in promoted_memories(ROOT / "data" / "memory.sqlite", market):
        store[mem["status"]] = store.get(mem["status"], 0) + 1
    return {
        "score": scorecard(ledger, recent, risk_rows, grid, dead, citations),
        "moves": stage_moves(recent),
        "ladder": memory_ladder(recent),
        "store": store,
        "ledger": ledger,
        "run": run_summary(ledger, today),
        "risk_rows": risk_rows,
        "risk": risk_summary(risk_rows),
        "mdd": recent[-1].get("mdd") if recent else None,
        "chain": execution_chain(LOG_DIR, market)[-HEALTH_WINDOW:],
        "stall": load_stall(STATE, market),
        "citations": citations,
        "spans": spans,
        "marks": change_markers(spans),
        "grid": grid,
        "dead": dead,
        "fresh": freshness(STATE, OBS_DIR, LOG_DIR, market, today),
    }


def run_strip(rows: list[dict]) -> None:
    """런 원장 띠 — 하루 한 칸(시장 × 날짜), 상태를 색 + 글리프로."""
    df = pd.DataFrame(rows)
    df["label"] = df["status"].map(lambda s: RUN_STATUS_STYLE[s][2])
    df["glyph"] = df["status"].map(lambda s: RUN_STATUS_STYLE[s][1])
    shown = [s for s in RUN_STATUS_STYLE if s in set(df["status"])]
    base = alt.Chart(df).encode(
        x=alt.X(
            "day:O",
            title=None,
            scale=alt.Scale(paddingInner=0.12),
            axis=alt.Axis(labelExpr="slice(datum.value, 5)", labelAngle=-90),
        ),
        y=alt.Y(
            "market:N", title=None, sort=list(MARKETS),
            scale=alt.Scale(paddingInner=0.2), axis=alt.Axis(labelOverlap=False),
        ),
    )
    cells = base.mark_rect(cornerRadius=2).encode(
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=[RUN_STATUS_STYLE[s][2] for s in shown],
                range=[RUN_STATUS_STYLE[s][0] for s in shown],
            ),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("market:N", title="시장"), alt.Tooltip("day:O", title="날짜(UTC)"),
            alt.Tooltip("label:N", title="상태"), alt.Tooltip("detail:N", title="상세"),
        ],
    )
    glyphs = base.mark_text(fontWeight="bold").encode(
        text="glyph:N",
        color=alt.condition(
            "datum.status == 'stalled' || datum.status == 'late'",
            alt.value("#0b0b0b"), alt.value("#ffffff"),
        ),
    )
    # 높이는 축 라벨·범례를 포함한 전체다 — 세 줄이 눌리지 않게 넉넉히 잡는다.
    st.altair_chart((cells + glyphs).properties(height=240), width="stretch")


def marks_within(marks: list[dict], df: pd.DataFrame) -> list[dict]:
    """차트의 날짜 범위 안에 드는 변경 표식만."""
    return [m for m in marks if df["day"].min() <= m["day"] <= df["day"].max()]


def mark_rules(inside: list[dict]) -> alt.Chart:
    """변경 표식 층 — 세로 점선 + 툴팁."""
    return alt.Chart(pd.DataFrame(inside)).mark_rule(color=MUTED_INK, strokeDash=[4, 3]).encode(
        x="day:T",
        tooltip=[alt.Tooltip("day:T", title="변경일"), alt.Tooltip("label:N", title="바뀐 것")],
    )


def stage_chart(rows: list[dict], marks: list[dict]) -> None:
    """단계별 이동량 — 결정마다 교훈·토론·리스크가 배분을 옮긴 크기를 쌓아 그린다."""
    long = [
        {"day": r["day"], "stage": STAGE_STYLE[k][1], "order": i, "l1": r[k]}
        for r in rows for i, k in enumerate(STAGE_STYLE) if r[k]
    ]
    if not long:
        st.caption("단계별 이동: 이 창에서 배분을 옮긴 단계가 없다.")
        return
    df = pd.DataFrame(long)
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("day:T", title=None),
        y=alt.Y("l1:Q", title="단계별 이동 L1"),
        color=alt.Color(
            "stage:N",
            scale=alt.Scale(
                domain=[v[1] for v in STAGE_STYLE.values()],
                range=[v[0] for v in STAGE_STYLE.values()],
            ),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        order=alt.Order("order:Q"),
        tooltip=[
            alt.Tooltip("day:T", title="날짜"), alt.Tooltip("stage:N", title="단계"),
            alt.Tooltip("l1:Q", title="이동 L1", format=".3f"),
        ],
    )
    inside = marks_within(marks, df)
    layers = [bars, mark_rules(inside)] if inside else [bars]
    st.altair_chart(alt.layer(*layers).properties(height=260), width="stretch")


def marked_chart(
    rows: list[dict], field: str, title: str, marks: list[dict],
    kind: str = "bar", ref: float | None = None,
) -> None:
    """한 계열 시계열 + 변경 표식(세로 점선). 판본·설정·서킷 경계를 같은 축에 얹는다."""
    df = pd.DataFrame([r for r in rows if r.get(field) is not None])
    if df.empty:
        st.caption(f"{title}: 기록 없음")
        return
    base = alt.Chart(df).encode(
        x=alt.X("day:T", title=None),
        y=alt.Y(f"{field}:Q", title=title),
        tooltip=[alt.Tooltip("day:T", title="날짜"), alt.Tooltip(f"{field}:Q", title=title, format=".3f")],
    )
    layers = [
        base.mark_bar(color=SERIES_COLOR, cornerRadiusEnd=3) if kind == "bar"
        else base.mark_line(color=SERIES_COLOR, strokeWidth=2, point=alt.OverlayMarkDef(size=40))
    ]
    inside = marks_within(marks, df)
    if inside:
        layers.append(mark_rules(inside))
    if ref is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"ref": [ref]})).mark_rule(color=MUTED_INK, strokeWidth=1).encode(y="ref:Q")
        )
    st.altair_chart(alt.layer(*layers).properties(height=220), width="stretch")
    if inside:
        st.caption("변경 표식(점선): " + " · ".join(f"{m['day']} {m['label']}" for m in inside))


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

tab_dash, tab_health, tab_obs, tab_mem, tab_chat, tab_ops = st.tabs(
    ["📊 대시보드", "🩺 건강", "🔭 관측", "📓 메모리", "💬 챗", "🔧 운영"]
)
# 런 원장의 날짜 축은 기록 시각(UTC)의 날짜다 — '오늘'도 같은 축으로 잡아야 아직 돌 차례가
# 아닌 런을 누락으로 세지 않는다.
UTC_TODAY = datetime.now(timezone.utc).date()


# ── 대시보드 ──

with tab_dash:
    context = load_context()
    by_kind: dict[str, list[dict]] = {}
    for item in context["items"]:
        by_kind.setdefault(item["kind"], []).append(item)

    # ── 안전·헬스 배너 (자율 운용 최상단 요소) ──
    if kill_switch_active(STATE):
        st.error("🛑 KILL_SWITCH 활성 — 실주문 정지 중. 해제: `rm data/state/KILL_SWITCH`")
    # 서킷 임계는 시장마다 다르다 — 화면을 그릴 때마다 운용 중인 값을 읽는다.
    banner_limits = load_limits(ROOT, TRADABLE).limits
    today = date.today()
    health_cols = st.columns(len(MARKETS))
    for col, market in zip(health_cols, MARKETS):
        h = market_health(LOG_DIR, market, today)
        with col:
            if h["last_day"] is None:
                st.caption(f"**{market}** · 결정 로그 없음")
                continue
            # 경고는 잡이 마지막으로 돈 날 기준 — 주말에는 돌아도 결정을 내리지 않는다
            stale = h["days_since_run"] is not None and h["days_since_run"] > STALE_DAYS
            mdd_circuit = banner_limits[market].mdd_circuit
            tripped = h["mdd"] is not None and h["mdd"] >= mdd_circuit
            age = f"{h['days_stale']}일 전" if h["days_stale"] is not None else h["last_day"]
            line = f"**{market}** · 최근 결정 {age}"
            if h["skipped_since"]:
                line += " · 주말 스킵"
            if h["mdd"]:
                line += f" · MDD {h['mdd']:.1%} / 한도 {mdd_circuit:.0%}"
            # 마지막으로 판정된 런 — 결정이 최근이어도 그 런이 실패·정지로 끝났을 수 있다.
            last_run = load_health(market, UTC_TODAY.isoformat())["run"]["last"]
            run_state = last_run["status"] if last_run else None
            if run_state in RUN_BAD or run_state in RUN_WARN:
                line += f" · 마지막 런 {RUN_STATUS_STYLE[run_state][2]}({last_run['day']})"
            if tripped:
                st.error(line + " · 🛑 서킷")
            elif run_state in RUN_BAD:
                st.error(line + " · ⚠️ 런 실패")
            elif stale:
                st.warning(line + " · ⏳ stale")
            elif run_state in RUN_WARN:
                st.warning(line)
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
    cols = st.columns(5)
    if meta:
        cols[0].metric("META 결합 지수 (llm)", f"{meta['index']:.4f}", f"{meta['ret_pct']:+.3f}%")
        cols[1].metric("META MDD", f"{meta['mdd_pct']:.2f}%")
        bh_meta = combined_index(VIRTUAL, "bh")
        if bh_meta:
            cols[2].metric("META α vs B&H", f"{meta['ret_pct'] - bh_meta['ret_pct']:+.3f}%p")
        base_meta = combined_index(VIRTUAL, "llm_base")
        if base_meta:
            cols[3].metric(
                "후처리 델타", f"{meta['ret_pct'] - base_meta['ret_pct']:+.3f}%p",
                help="llm − llm_base. 1차 결정 뒤의 교훈·토론·리스크 세 단계를 합친 효과다 — "
                "메모리만의 몫은 아래 단계별 기여에서 본다.",
            )
        expo = meta_exposure(LOG_DIR)
        if expo:
            cols[4].metric(
                "평균 노출 (1−현금)", f"{expo['mean']:.1%}", f"최근 {expo['last']:.1%}",
                delta_color="off",
            )

    # α 를 만든 것이 노출인지 선별인지 — 두 항의 합이 위 α 와 같다.
    decomp = meta_alpha_decomposition(VIRTUAL, LOG_DIR)
    if decomp:
        per_market = " · ".join(
            f"{m} {d['cash_drag_pct']:+.2f}/{d['selection_pct']:+.2f}"
            for m, d in sorted(decomp["by_market"].items())
        )
        st.caption(
            f"**α 분해** — 현금 드래그 `{decomp['cash_drag_pct']:+.2f}%p` + "
            f"종목선택 `{decomp['selection_pct']:+.2f}%p` = α `{decomp['alpha_pct']:+.2f}%p`. "
            "매일 llm 이 실제로 가졌던 노출만큼만 B&H 바스켓을 든 합성 포트폴리오와 대조한 것 — "
            "**드래그는 안 들어가서 잃은 몫, 선택은 같은 노출에서 무엇을 골랐나**다. "
            f"시장별 드래그/선택(%p): {per_market}"
        )

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

    # 목표가 아니라 **체결**을 그린다 — 목표는 계좌가 그것을 표현하지 못한 날에도 갱신되므로
    # 자금 없는 시장이 담고 있는 것처럼 보인다. 목표 배분은 아래 시장별 절에 따로 있다.
    st.caption("마켓 내 포트폴리오 구성 — **실보유** (마지막 체결 배분, CASH 포함)")
    for col, market in zip(st.columns(len(MARKETS)), MARKETS):
        with col:
            held = load_holding_weights(LOG_DIR, market)
            day, weights = held if held else (None, {})
            pie(weights, f"{market} · 체결 {day}" if day else market)

    st.subheader("유니버스 — 관측 vs 집행")
    st.caption(
        "에이전트는 개별 종목까지 **관측**하지만(재무·뉴스·공시), 배분 벡터를 채울 수 있는 "
        "**집행** 대상은 그보다 좁을 수 있다. 위 파이에 종목이 몇 개 없다면 전략 판단이 아니라 "
        "계좌 크기 때문이다 — 1주 비중이 종목당 상한을 넘는 종목에는 어떤 비중도 표현할 수 "
        "없어서, 배정해도 주문 투영이 매일 통째로 현금으로 되돌린다. 그래서 낼 수 있는 주문만 "
        "정의역에 두고 나머지는 판단 근거로만 읽는다. 금액 열은 그 시장의 관측 통화 그대로다."
    )
    # 화면을 그릴 때마다 읽는다 — 이 프로세스는 오래 떠 있어서, 뜰 때 읽은 값을 쓰면 운용 중
    # 바뀐 한도가 재시작 전까지 보이지 않는다.
    current_limits = load_limits(ROOT, TRADABLE).limits
    for market in MARKETS:
        limits = current_limits[market]
        budget_rec = latest_budget(read_recent_decisions(LOG_DIR, market))
        rows = universe_rows(
            universe_meta(OBSERVED_UNIVERSE[market], TRADABLE[market]),
            limits.max_weight_per_asset,
            limits.asset_caps,
            budget_rec[1] if budget_rec else None,
            latest_closes(market),
        )
        n_tradable = sum(1 for r in rows if r["tradable"])
        n_exec = sum(1 for r in rows if r["executable"])
        head = (f"**{market}** · 집행 {n_exec}종(실계좌 기준)"
                f" / 설정 {n_tradable}종 / 관측 {len(rows)}종")
        if budget_rec:
            day, budget = budget_rec
            head += f" · 예산 기준 `{day}` (순자산 {budget['equity']:,.2f} {budget['currency']})"
        st.markdown(head)

        # 유니버스는 집합이라 조각 크기로 쓸 값이 없다 — 종목마다 같은 크기를 주고
        # 두 도넛의 조각 수 차이로 '관측 ⊋ 집행' 을 보인다. 색은 관측 쪽 순서로 고정해
        # 같은 종목이 두 도넛에서 같은 색을 받게 한다.
        labels = [display_name(r["symbol"]) for r in rows]
        col_obs, col_exec = st.columns(2)
        with col_obs:
            pie({name: 1.0 for name in labels}, "관측 유니버스", domain=labels)
        with col_exec:
            pie(
                {display_name(r["symbol"]): 1.0 for r in rows if r["executable"]},
                "집행 유니버스 (실계좌가 담을 수 있는 것)",
                domain=labels,
            )

        if not budget_rec:
            # 판정 열이 통째로 비는 이유를 여기서 밝힌다 — 빈 칸만 보면 채널이 죽은 것으로 읽힌다.
            st.caption(
                "예산 스냅샷 없음 — 순자산이 0 이거나(입금 전) 계좌 조회가 실패한 날은 예산이 "
                "기록되지 않는다. 계좌가 무엇을 담을 수 있는지 판정할 근거가 없어 **집행 "
                "유니버스는 비어 있게 표시**된다. 설정상의 집행 종목은 아래 표의 '역할' 열에 "
                "남아 있고, 1주 비중 판정은 입금 후 첫 결정부터 채워진다."
            )
        st.dataframe(
            pd.DataFrame([
                {
                    "종목": display_name(r["symbol"]),
                    "자산군": "ETF" if r["asset_class"] == ETF else "개별",
                    "추종 지수": r["index"] or "—",
                    "역할": "집행" if r["tradable"] else "관측 전용",
                    "종목당 상한": cell(r["cap"], ".0%"),
                    "1주 비중": cell(r["min_step_weight"], ".1%"),
                    "t−1 종가": cell(r["last_close"], ",.2f"),
                    "편입 최소 순자산": cell(r["min_equity_for_entry"], ",.0f"),
                    "판정": r["verdict"],
                }
                for r in rows
            ]),
            hide_index=True,
            width="stretch",
        )

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


# ── 건강 (성과와 별개로 시스템이 제대로 돌았는가 · 읽기 전용) ──

with tab_health:
    st.caption(
        "성과가 아니라 **시스템 자체**를 본다 — 런이 돌았는가, 리스크 엔진이 배분을 얼마나 "
        "바꿨는가, 목표가 계좌에 닿았는가, 입력이 비지 않았는가, 근거가 실재하는가. 하루 한 번 "
        "결정하는 시스템이라 비율·백분위 대신 **있어야 할 것이 있는지**를 센다. 전부 이미 남은 "
        "로그·상태 파일에서 계산하며(60초 캐시), 날짜는 기록 시각(UTC) 기준이다."
    )
    health = {m: load_health(m, UTC_TODAY.isoformat()) for m in MARKETS}

    st.subheader("점검표")
    st.caption(
        "규칙만으로 판정하는 일곱 점검 — 칸은 `위반 / 판정 대상`, 위반이 있으면 가장 최근 날짜. "
        "비율이 아니라 건수다: 하루 한 건짜리 표본에서 비율은 한 번의 실패로 크게 흔들린다. "
        f"창은 점검마다 다르다(런 {HEALTH_LEDGER_DAYS}일 · 채널 {HEALTH_GRID_RUNS}결정 · "
        f"나머지 {HEALTH_WINDOW}결정). 서킷이 열린 날은 원안의 한도 점검에서 뺀다."
    )

    def score_cell(row: dict) -> str:
        if not row["total"]:
            return "— 대상 없음"
        if not row["bad"]:
            return f"✓ 0 / {row['total']}"
        return f"⚠ {row['bad']} / {row['total']} · 최근 {row['last_bad'][5:]}"

    checks = [r["check"] for r in health[MARKETS[0]]["score"]]
    st.dataframe(
        pd.DataFrame(
            {m: [score_cell(r) for r in health[m]["score"]] for m in MARKETS}, index=checks
        )
    )

    st.subheader(f"런 원장 — 최근 {HEALTH_LEDGER_DAYS}일")
    st.caption(
        "하루 한 칸. 주말·휴장일에도 잡은 돌아서 건너뛴 사실을 남기므로, **한 줄도 없는 날은 "
        "잡이 돌지 않은 날**이다. 집행 정지는 결정은 수리됐으나 주문을 한 건도 못 낸 날(예산·"
        "거래단위), 지연 실행은 그날 몫의 결정이 날짜를 넘겨 뒤늦게 난 날. 오늘 칸은 그 날짜(UTC)가 "
        "끝나기 전까지 판정하지 않는다."
    )
    strip = [{"market": m, **r} for m in MARKETS for r in health[m]["ledger"]]
    if strip:
        run_strip(strip)
        run_table = []
        for m in MARKETS:
            run, counts = health[m]["run"], health[m]["run"]["counts"]
            last = run["last"]
            run_table.append({
                "시장": m,
                "마지막 런": f"{last['day']} {RUN_STATUS_STYLE[last['status']][2]}" if last else "—",
                "마지막 정상 런": run["last_ok_day"] or f"{HEALTH_LEDGER_DAYS}일 내 없음",
                "정상": counts.get("ok", 0),
                "집행 정지": counts.get("stalled", 0),
                "지연 실행": counts.get("late", 0),
                "휴장": counts.get("closed", 0),
                "실패": sum(counts.get(s, 0) for s in ("degraded", "rejected", "error")),
                "기록 없음": counts.get("missing", 0),
            })
        st.dataframe(pd.DataFrame(run_table), hide_index=True)
        trouble = [
            {"날짜": r["day"], "시장": r["market"], "상태": RUN_STATUS_STYLE[r["status"]][2],
             "상세": r["detail"]}
            for r in strip if r["status"] in RUN_BAD or r["status"] in RUN_WARN
        ]
        if trouble:
            with st.expander(f"정상·휴장이 아닌 날 {len(trouble)}건"):
                st.dataframe(
                    pd.DataFrame(sorted(trouble, key=lambda r: r["날짜"], reverse=True)),
                    hide_index=True,
                )
    else:
        st.caption("런 기록 없음.")

    st.divider()
    st.subheader(f"리스크 엔진 개입 — 최근 {HEALTH_WINDOW}결정")
    st.caption(
        "LLM 원안과 리스크 엔진을 지난 최종 배분의 거리. L1 은 비중 절대차의 합이다(나누지 "
        "않음 — 전액을 옮기면 2.0). 개입이 잦다는 것은 LLM 이 한도 밖을 자주 낸다는 뜻이고, "
        "서킷이 열린 날은 엔진이 배분을 대신 정한 날이다."
    )
    limits_now = load_limits(ROOT, TRADABLE).limits
    risk_table = []
    for m in MARKETS:
        rs, mdd, limit = health[m]["risk"], health[m]["mdd"], limits_now[m].mdd_circuit
        risk_table.append({
            "시장": m,
            "원안 기록 결정": rs["n"],
            "개입 일수": rs["n_changed"],
            "평균 L1": cell(rs["mean_l1"], ".3f"),
            "최대 L1": cell(rs["max_l1"], ".3f"),
            "사유별 일수": ", ".join(f"{k} {v}" for k, v in sorted(rs["by_rule"].items())) or "—",
            "MDD(최근 결정)": cell(mdd, ".1%"),
            "서킷 한도": f"{limit:.0%}",
            "한도까지": cell(limit - mdd if mdd is not None else None, ".1%"),
        })
    st.dataframe(pd.DataFrame(risk_table), hide_index=True)

    st.subheader(f"목표 → 체결 — 최근 {HEALTH_WINDOW}런")
    st.caption(
        "최종 목표 배분과 체결 뒤 실보유의 거리(같은 L1). 소액 계좌에서는 1주 단위 때문에 "
        f"생기는 괴리가 대부분이라 사유를 함께 본다. {DRIFT_NOTE} 는 참고 눈금이지 판정 임계가 아니다."
    )
    chain_table = []
    for m in MARKETS:
        chain = health[m]["chain"]
        drifts = [r["drift_l1"] for r in chain if r["drift_l1"] is not None]
        reasons: dict[str, int] = {}
        for r in chain:
            for reason in r["dropped"].values():
                reasons[reason] = reasons.get(reason, 0) + 1
        stall = health[m]["stall"] or {}
        chain_table.append({
            "시장": m,
            "대사 기록 런": len(drifts),
            "평균 괴리": cell(sum(drifts) / len(drifts) if drifts else None, ".3f"),
            f"{DRIFT_NOTE} 초과 런": sum(1 for d in drifts if d > DRIFT_NOTE),
            "집행 정지 런": sum(1 for r in chain if r["stall"]),
            "지금 정지": f"{stall['reason']} ({stall.get('since')}~)" if stall.get("reason") else "아님",
            "목표 미달 사유(종목·런)": ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) or "—",
        })
    st.dataframe(pd.DataFrame(chain_table), hide_index=True)

    st.subheader(f"인용 실재 — 최근 {HEALTH_WINDOW}결정")
    st.caption(
        "근거로 든 신호·교훈 ID 가 그날 실제로 제공된 목록 안에 있는가. 인용률은 **인용할 "
        "재료가 있던 날** 기준이다 — 신호 라이브러리가 없는 시장은 인용할 것이 없어서 0 이다."
    )
    cite_table = []
    for m in MARKETS:
        tally: dict[str, int] = {}
        for r in health[m]["citations"]:
            tally[r["status"]] = tally.get(r["status"], 0) + 1
        with_material = sum(tally.get(s, 0) for s in ("cited", "unknown", "uncited"))
        cite_table.append({
            "시장": m,
            "인용 정상": tally.get("cited", 0),
            "목록 밖 인용": tally.get("unknown", 0),
            "재료 있으나 미인용": tally.get("uncited", 0),
            "재료 없음": tally.get("no_material", 0),
            "대조 불가(기록 전)": tally.get("unrecorded", 0),
            "인용률(재료 있던 날)": cell(
                tally.get("cited", 0) / with_material if with_material else None, ".0%"
            ),
        })
    st.dataframe(pd.DataFrame(cite_table), hide_index=True)
    outside = [
        f"{m} {r['day']}: {', '.join(r['outside'])}"
        for m in MARKETS for r in health[m]["citations"] if r["outside"]
    ]
    if outside:
        st.warning("제공 목록 밖 ID 를 인용한 결정 — " + " / ".join(outside))

    st.subheader(f"교훈 사다리 — 최근 {HEALTH_WINDOW}결정")
    st.caption(
        "교훈이 결정에 닿는 단계별 결정 수: 회수됨 → 배분에 반영됨 → 근거로 인용됨. 어느 "
        "단에서 끊기는지를 본다 — 회수가 0 이면 쓸 교훈이 없는 것이고, 회수는 되는데 반영이 "
        "0 이면 신뢰도 게이트가 막은 것이다. **개입의 빈도이지 효과가 아니다** — 교훈이 성과를 "
        "개선했는지는 여기서 판정하지 않는다."
    )
    st.dataframe(
        pd.DataFrame([
            {
                "시장": m,
                "결정 수": health[m]["ladder"]["n"],
                "교훈 회수": health[m]["ladder"]["retrieved"],
                "배분 반영": health[m]["ladder"]["applied"],
                "교훈 인용": health[m]["ladder"]["cited"],
                "지금 active": health[m]["store"].get("active", 0),
                "probation": health[m]["store"].get("probation", 0),
                "retired": health[m]["store"].get("retired", 0),
            }
            for m in MARKETS
        ]),
        hide_index=True,
    )

    st.divider()
    focus = st.radio("시장 상세", MARKETS, horizontal=True, key="health_market")
    detail = health[focus]

    hc = st.columns(2)
    with hc[0]:
        marked_chart(detail["risk_rows"], "l1", "리스크 개입 L1", detail["marks"], kind="bar")
    with hc[1]:
        marked_chart(
            detail["chain"], "drift_l1", "체결 괴리 L1", detail["marks"], kind="line", ref=DRIFT_NOTE
        )

    sc = st.columns(2)
    with sc[0]:
        stage_chart(detail["moves"], detail["marks"])
        debated = sum(1 for r in detail["moves"] if r["debated"])
        st.caption(
            f"토론 소집 {debated} / {len(detail['moves'])}결정. 쌓인 높이는 세 단계가 각각 옮긴 "
            "크기의 합이지 순이동이 아니다(서로 되돌릴 수 있다)."
        )
    with sc[1]:
        marked_chart(detail["moves"], "tokens_in", "결정당 입력 토큰", detail["marks"], kind="line")
        st.caption(
            "한 결정에 든 입력 토큰 합(1차 + 교훈 재결정 + 토론 재결정). 계단식 변화는 "
            "프롬프트에 들어가는 것이 달라졌다는 뜻이다."
        )

    latest = next((r for r in reversed(detail["chain"]) if r["intended"] and r["executed"]), None)
    if latest:
        st.caption(f"가장 최근 대사 — {latest['day']} (괴리 L1 {latest['drift_l1']:.3f})")
        symbols = sorted(set(latest["intended"]) | set(latest["executed"]))
        st.dataframe(
            pd.DataFrame([
                {
                    "종목": display_name(s),
                    "목표": f"{latest['intended'].get(s, 0.0):.1%}",
                    "체결 후": f"{latest['executed'].get(s, 0.0):.1%}",
                    "차이": f"{latest['executed'].get(s, 0.0) - latest['intended'].get(s, 0.0):+.1%}",
                    "사유": latest["dropped"].get(s, ""),
                }
                for s in symbols
            ]),
            hide_index=True,
        )

    st.subheader(f"입력 채널 도달 — 최근 {HEALTH_GRID_RUNS}결정")
    st.caption(
        "결정마다 각 채널이 몇 건 들어갔는가. `∅` = 0건, `·` = 그 결정에는 기록이 없다. "
        "**수집과 프롬프트 주입은 다른 칸이다** — 프롬프트는 건수 상한에서 자르므로 수집됐다고 "
        "실린 것이 아니다. `⚠` 는 매 런 채워져 있어야 하는 채널이 빈 칸."
    )
    if detail["grid"]:
        dead = {(d["day"], d["channel"]) for d in detail["dead"]}

        def grid_cell(row: dict, channel: str) -> str:
            value = row.get(channel)
            if value is None:
                return "·"
            if value == 0:
                return "∅ ⚠" if (row["day"], channel) in dead else "∅"
            return str(value)

        st.dataframe(
            pd.DataFrame(
                {row["day"][5:]: [grid_cell(row, ch) for ch in CHANNELS] for row in detail["grid"]},
                index=[CHANNEL_LABELS[ch] for ch in CHANNELS],
            )
        )
        if detail["dead"]:
            st.warning(
                "채워져 있어야 하는데 빈 채널 — "
                + ", ".join(f"{d['day']} {CHANNEL_LABELS[d['channel']]}" for d in detail["dead"])
            )
        if all(row.get("news_injected") is None for row in detail["grid"]):
            st.caption(
                "뉴스·공시의 프롬프트 주입 건수는 아직 기록되지 않는다 — 수집 건수로 대신 "
                "채우지 않고 비워 둔다."
            )
    else:
        st.caption("결정 기록 없음.")

    fc = st.columns(2)
    with fc[0]:
        st.subheader("채널 신선도")
        st.caption(
            f"최신 관측이 며칠 묵었는가. 매 런 갱신되는 채널만 판정한다({focus}: "
            f"{FRESH_MAX_AGE[focus]}일 초과 시 표시) — 나머지는 주기가 달라 나이만 적는다."
        )
        if detail["fresh"]:
            st.dataframe(
                pd.DataFrame([
                    {
                        "채널": r["channel"],
                        "기준일": r["asof"] or "—",
                        "나이(일)": cell(r["age_days"], "d"),
                        "판정": (
                            "—" if not r["daily"] or r["age_days"] is None
                            else "⚠ 묵음" if r["age_days"] > FRESH_MAX_AGE[focus] else "정상"
                        ),
                    }
                    for r in detail["fresh"]
                ]),
                hide_index=True,
            )
        else:
            st.caption("상태 파일 없음.")
    with fc[1]:
        st.subheader("판본·모델·설정 구간")
        st.caption(
            "결정에 새겨진 프롬프트 판본·설정 판본·모델이 유지된 구간, 그리고 서킷이 열렸거나 "
            "교훈이 실린 구간. 구간이 다르면 같은 조건의 표본이 아니다 — 위 차트의 점선이 이 경계다."
        )
        if detail["spans"]:
            st.dataframe(
                pd.DataFrame([
                    {
                        "항목": s["field"],
                        "값": "기록 전" if s["value"] is None else str(s["value"]),
                        "시작": s["first_day"],
                        "끝": s["last_day"],
                        "결정 수": s["n"],
                    }
                    for s in detail["spans"]
                ]),
                hide_index=True,
            )
        else:
            st.caption("결정 기록 없음.")

    st.divider()
    st.subheader("오프라인 평가 실행 (챗 경로)")
    runs = eval_runs(ROOT / "data" / "eval" / "chat" / "runs")
    if runs:
        newest = runs[-1]["started_at"][:10]
        idle = (UTC_TODAY - date.fromisoformat(newest)).days if newest else None
        st.caption(
            "동결 스냅샷 위에서 돌린 챗 품질 평가의 실행 기록 — 통과 수·게이트 위반·범주별 "
            f"통과율. 마지막 실행 {newest}"
            + (f"({idle}일 전)" if idle is not None else "")
            + ". 그 뒤 프롬프트·모델이 바뀌었다면 아래 숫자는 지금의 시스템을 말하지 않는다. "
            "**일간 결정 경로에는 이런 평가셋이 없다.**"
        )
        st.dataframe(
            pd.DataFrame([
                {
                    "시작(UTC)": r["started_at"],
                    "호출": r["calls"],
                    "통과": r["passed"],
                    "게이트 위반": ", ".join(f"{k} {v}" for k, v in sorted(r["gates"].items())) or "—",
                    "지연 p50(s)": cell(r["latency_p50_s"], ".2f"),
                    "범주별 통과율": " ".join(
                        f"{k} {v:.0%}" for k, v in sorted(r["categories"].items()) if v is not None
                    ),
                    "프롬프트 판본": r["prompt_rev"] or "—",
                    "미커밋 상태": "예" if r["git_dirty"] else "아니오",
                }
                for r in reversed(runs)
            ]),
            hide_index=True,
        )
    else:
        st.caption("평가 실행 기록 없음.")


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
                rev = decision.get("prompt_rev")
                st.caption(
                    f"정책 판본: `{rev}`" if rev
                    else "정책 판본: 기록 이전 결정 (판본 표기 도입 전)"
                )
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
        breakdown = usage_breakdown(LOG_DIR)
        bc = st.columns(2)
        for col, title, key, rows in (
            (bc[0], "목적별 누적 (호출 수·토큰)", "목적", breakdown["by_purpose"]),
            (bc[1], "시장별 누적 (호출 수·토큰)", "시장", breakdown["by_market"]),
        ):
            with col:
                st.caption(title)
                st.dataframe(
                    pd.DataFrame([
                        {key: r["key"], "호출": r["calls"], "in": r["in"], "out": r["out"]}
                        for r in rows
                    ]),
                    hide_index=True,
                )
    else:
        st.caption("아직 usage 로그 없음 — 다음 LLM 호출부터 `data/logs/USAGE/` 에 쌓인다.")

    st.divider()

    st.subheader("rolling-k delta (승격 판정 입력)")
    st.caption(
        f"{ROLLING_K}거래일 창을 하루씩 굴린 상대 성과의 **승률**. pipeline=llm−llm_base"
        "(교훈·토론·리스크 세 단계의 합) · alpha=llm−bh(매매 가능 종목 매수 후 보유) · "
        "index=llm−시장 지수. **승격 판정은 pipeline·alpha 로만** 한다 — index 는 벤치마크가 유니버스가 아니라 시장 전체라 "
        "\"그 기간이 어떤 장이었나\"를 답하는 맥락이다. p 는 겹치지 않는 청크에만 적용하며 "
        "청크 5개(=100거래일) 미만이면 n/a."
    )

    def rolling_line(label: str, rolled: dict, names: tuple[str, ...]) -> str:
        line = f"**{label}** — "
        for name in names:
            r = rolled.get(name)
            if r is None:
                line += f"{name}: —  · "
                continue
            p = f"p={r['p_value']:.3f}" if r["p_value"] is not None else "p=n/a"
            line += f"{name}: 승률 {r['win_rate']:.0%} (Δ{r['mean_delta_pct']:+.2f}%p, {p}) · "
        return line.rstrip(" ·")

    meta_rolled = meta_rolling_report(VIRTUAL)
    if meta_rolled["alpha"] or meta_rolled["pipeline"]:
        st.markdown(rolling_line("META (결합 지수)", meta_rolled, ("pipeline", "alpha")))
    for market in MARKETS:
        if not (VIRTUAL / f"{market}_llm.json").exists():
            continue
        st.markdown(
            rolling_line(market, rolling_report(VIRTUAL, market), ("pipeline", "alpha", "index"))
        )
    if any(load_index_series(STATE, m) is None for m in MARKETS):
        missing = [m for m in MARKETS if load_index_series(STATE, m) is None]
        st.caption(
            f"지수 열 `—` ({', '.join(missing)}): 무료 실지수 원천이 없는 시장. 대장 코인은 "
            "지수가 아니라 B&H 바스켓의 구성종목이라 벤치마크로 쓰지 않는다."
        )

    st.subheader("단계별 기여 — 후처리 델타의 분해")
    st.caption(
        "llm − llm_base 를 1차 결정 뒤의 세 단계로 가른다: memory(교훈 블렌딩) · debate(토론 "
        "재결정) · risk(리스크 엔진). 배분 차이 × 그 배분이 유효했던 구간(체결 시가 → 다음 시가)의 "
        "수익률이며 비용·복리를 뺀 1차 근사다. 일수는 그 단계가 배분을 실제로 바꾼 날."
    )
    staged = [r for m in MARKETS if (r := stage_attribution(LOG_DIR, OBS_DIR, m))]
    if staged:
        st.dataframe(
            pd.DataFrame([
                {"시장": r["market"], "결정": r["n"],
                 **{f"{k} (%p)": round(r["stages"][k]["pct"], 2) for k in STAGES},
                 **{f"{k} 일수": r["stages"][k]["days"] for k in STAGES},
                 "합계 (%p)": round(r["total_pct"], 2)}
                for r in staged
            ]),
            hide_index=True,
        )
    else:
        st.caption("결정 로그·관측 스냅샷이 쌓이면 표시.")

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
