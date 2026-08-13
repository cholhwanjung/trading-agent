"""라이브 국면 계산 — 어댑터로 지수 프록시 봉을 조회해 classify_regime 에 넘긴다.

별도 지수 API 없이 기존 어댑터의 심볼 조회만 사용(`get_ohlcv_history`, 상한 t−1 로
당일 데이터 차단). 지수 프록시:
    CRYPTO = BTC/USDT (대장), US = SPY, KR = 069500 (KODEX 200).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

from regime.jm import JumpFeatures, compute_jump_features, feature_matrix
from regime.jump_model import fit, infer, load_model, needs_refit, save_model
from regime.macro import MacroRegime, classify_macro
from regime.pulse import RegimeResult, classify_regime
from regime.vol import VolResult, classify_vol

# US 프록시는 QQQ — 유니버스(나스닥 메가캡)와 정합하고, KIS 해외 시세계가 나스닥
# 코드(NAS)로 고정돼 있어 NYSE Arca 상장 SPY 는 빈 응답이 온다(조용한 실패 방지).
INDEX_PROXY = {"CRYPTO": "BTC/USDT", "US": "QQQ", "KR": "069500"}
LOOKBACK_DAYS = 300  # DD 25세션 윈도우 + CORRECTION 추적 여유
VOL_LOOKBACK_DAYS = 60  # 실현변동성 20거래일 창 + 휴장 여유
# jump model 학습·추론 창. 브로커가 주는 만큼만 온다(크립토 3000봉 · KIS 는 1200행 상한).
JM_TRAIN_DAYS = 3000


async def compute_regime(adapter, market: str, asof_day: date) -> RegimeResult | None:
    """지수 프록시 국면. 프록시 없음/조회 실패 → None (fail-open — 관측 보조일 뿐)."""
    proxy = INDEX_PROXY.get(market)
    if not proxy:
        return None
    try:
        bars = await adapter.get_ohlcv_history([proxy], asof_day, lookback_days=LOOKBACK_DAYS)
        return classify_regime(bars.get(proxy, []))
    except Exception:
        return None


async def compute_market_vol(adapter, market: str, asof_day: date) -> VolResult | None:
    """지수 프록시 실현변동성 국면. 프록시 없음/조회 실패 → None (fail-open)."""
    proxy = INDEX_PROXY.get(market)
    if not proxy:
        return None
    try:
        bars = await adapter.get_ohlcv_history([proxy], asof_day, lookback_days=VOL_LOOKBACK_DAYS)
        return classify_vol([b.close for b in bars.get(proxy, [])])
    except Exception:
        return None


async def compute_jm_features(adapter, market: str, asof_day: date) -> JumpFeatures | None:
    """지수 프록시 jump model 피처. 프록시 없음/조회 실패 → None (fail-open).

    FSM(`compute_regime`)과 **같은 LOOKBACK_DAYS·같은 프록시**를 쓴다 — 두 접근의 차이가
    입력 차이로 오염되면 나중에 무엇이 나은지 판정할 수 없다.
    """
    proxy = INDEX_PROXY.get(market)
    if not proxy:
        return None
    try:
        bars = await adapter.get_ohlcv_history([proxy], asof_day, lookback_days=LOOKBACK_DAYS)
        return compute_jump_features([b.close for b in bars.get(proxy, [])])
    except Exception:
        return None


async def compute_jm_regime(
    adapter, market: str, asof_day: date, model_path: Path | str
) -> dict | None:
    """jump model 국면 상태. 필요 시 재적합 후 추론. 프록시 없음/조회 실패/표본 부족 → None.

    적합과 추론이 **같은 창**을 쓴다 — 논문의 온라인 추론도 훈련창 길이의 룩백에 DP 를
    돌린다. 창을 따로 두면 추론 상태열의 끈적함이 훈련과 달라진다.

    재적합은 모델이 없거나 REFIT_DAYS 가 지났을 때만 — 매일 다시 맞추면 중심점이 흔들려
    같은 시장 상태가 날마다 다른 라벨을 받는다.
    """
    proxy = INDEX_PROXY.get(market)
    if not proxy:
        return None
    try:
        bars = await adapter.get_ohlcv_history([proxy], asof_day, lookback_days=JM_TRAIN_DAYS)
        rows, returns = feature_matrix([b.close for b in bars.get(proxy, [])])
    except Exception:
        return None
    if not rows:
        return None

    model = load_model(model_path)
    refit = needs_refit(model, asof_day)
    if refit:
        fitted = fit(rows, returns, trained_through=asof_day)
        if fitted is None:  # 학습 표본 부족 — 기존 모델이 있으면 그것으로 계속 간다
            if model is None:
                return None
        else:
            model = fitted
            save_model(model_path, model)

    state = infer(rows, model)
    if state is None:
        return None
    return {
        "state": state,
        "n_infer": len(rows),
        "n_train": model.n_train,
        "trained_through": model.trained_through,
        "refit": refit,
        "penalty": model.penalty,
        "proxy": proxy,
    }


async def compute_macro_regime(env: Mapping[str, str], asof_day: date) -> MacroRegime | None:
    """FRED 매크로 국면 (전역 shadow). 키 없음/조회 실패 → None (fail-open — 보조 신호)."""
    api_key = env.get("FRED_API_KEY")
    if not api_key:
        return None
    from adapters.fred import fetch_fred_latest

    values = await fetch_fred_latest(api_key, asof_day)
    return classify_macro(values)
