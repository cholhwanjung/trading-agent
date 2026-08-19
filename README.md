# trading-agent

An LLM trading agent that learns from its own decision history. Verified memory mechanisms — statistical admission gates, residual and confidence-controlled influence — are ported into live trading decisions, with a deterministic risk engine underneath.

> Orders go to a live brokerage account at deliberately small size; crypto stays on an exchange testnet until keys are supplied. Four virtual portfolios run alongside every decision, and those, not the account, are the measurement. This is not investment advice.

[한국어 README](README.ko.md)

## Ground rules

- **"Time travel is cheating."** Past markets are baked into LLM weights, so backtest performance cannot justify promoting anything. Promotion is judged on live data after the model cutoff; backtests only screen signals.
- **Verifier before agent.** Adapters, the daily loop and structured logging came first. Without measurement, self-improvement is self-deception.
- **Win on process, lose on resources.** Free daily-resolution data only: OHLCV, RSS, public filings, FRED. No high-frequency trading, no paid alternative data, no factor arms race. The target is α > 0 against buy-and-hold, not institutional alpha.
- **Memory is a calibrated residual.** A lesson is a correction term on the base decision. It intervenes only after n ≥ 5 repetitions, a sign test at p ≤ 0.05, and probation on live out-of-sample data. Influence scales with confidence and is bounded. Uncited memory has zero influence.
- **The risk engine never consults the LLM.** Drawdown circuit → failure-pattern veto → allocation domain → per-asset cap → minimum cash → turnover limit.
- **The action space is allocation weights.** Not discrete buy/sell/hold: weights sum to 1 including cash, long-only. Cash is a position.

## Daily loop

```
observe (bars · news · filings ≤ t−1) ─ leakage guard
  → 7 price features · 3 fundamentals · index valuation · ETF premium
  → top-3 alpha signals · scheduled macro events · account budget
  → LLM decision (second pass only when memory is retrieved, then residual blend)
  → risk engine → tradability gate → live order caps
  → order
  → 4 virtual arms marked at t−1 closes (llm / llm_base / B&H / random)
  → memory record → outcome scoring → admission / probation / retention
  → shadow channels → daily briefing
```

## Universe and execution

| Market | Observed | Allocated | Venue |
|---|---|---|---|
| KR | 4 large caps + KODEX 200TR | `278530` | KIS live (simulated account without live keys) |
| US | 5 megacaps + SCHX | `SCHX` | KIS overseas live (Alpaca paper as fallback) |
| Crypto | BTC · ETH · SOL | same | Upbit live (Binance testnet as fallback) |

The allocated set is narrower than the observed set. On a small account one share of an individual name already exceeds the per-asset cap, so no weight can be expressed in it and any allocation there is projected back to cash. The allocation vector is therefore defined over broad index ETFs, while individual names remain in the observation as evidence for judgment. As the account grows, names move into the allocated set.

### Live-order guards

- **Per-order buy cap** — `min(absolute ceiling, equity × ratio)`. Equity is read from the broker each step, so deposits, withdrawals and P&L are reflected without reconfiguring anything.
- **Daily cumulative buy cap** — reset by date.
- **Sells are never capped.** Under long-only a sell is bounded by the position itself and its result is cash. A cap on sells would slow defense exactly when defense is urgent.
- **Tradability gate** — halted, zero-volume and limit-locked names are dropped before submission. Names in delisting liquidation block buys but not sells.

## Intraday trigger

A stateless worker polls every 15 minutes and fires only on a sharp move against a rolling reference price. On fire it re-enters the same decision path with the intraday move injected as a **separate labeled channel**, so today's information never contaminates the observation window.

| Market | Threshold | Session gating |
|---|---|---|
| Crypto | 8% | none (24/7) |
| KR | 5% | 09:00–15:30 KST, weekdays |

US has no watcher yet. Trigger decisions are excluded from the learning pipeline — admitting same-day information into memory would poison the admission statistics.

## Memory

```
decision ──▶ episodic (pattern key + outcome: action return − hold return)
               │  admission: n≥5 + sign test p≤0.05 + embedding dedup
               ▼
            probation (7-day live OOS, promotion samples excluded)
               │  pass
               ▼
  semantic (success → soft prior) / procedural (failure → hard veto)
               │  retention: sign-flip invalidation + merge, diversity first
               ▼
            retired
```

A single trade promotes nothing; individual P&L is noise. Vetoed decisions are kept as counterfactuals, so a blocked pattern still accumulates the samples needed to re-test it.

## Observation channels

Channels marked *shadow* are computed and logged but do not touch decisions or the risk engine. They are promoted only after live comparison shows they help.

| Channel | Source | State |
|---|---|---|
| OHLCV bars, 7 features | broker APIs | injected |
| News | Google News RSS (KR), Alpaca (US), CoinDesk / Cointelegraph | injected |
| Filings | DART (KR), SEC EDGAR 8-K/10-Q (US) | injected |
| Fundamentals (PER · ROE · debt/equity, TTM) | DART statements, EDGAR XBRL | injected |
| Index valuation | market-level multiples via proxy funds | injected |
| ETF premium to NAV | KIS — KR only, no free US source | injected |
| Alpha signals | Alpha Lab admitted factors | injected |
| Scheduled macro events | published FOMC / BOK calendars | injected |
| Account budget | broker cash vs share price and target notional | injected |
| Intraday move | broker quote, separate labeled channel | injected on trigger |
| Market regime | index proxy bars, distribution-day state machine | shadow |
| Market regime (jump model) | same proxy, statistical state classifier | shadow |
| Macro regime | FRED daily series | shadow |
| Realized volatility | index proxy, 20-day annualized | shadow |
| Investor flows (KR) | KIS foreign / institutional / retail net buying | shadow |
| Allocation concentration | own weights + 60-day correlations | shadow |
| Cross-market budget | regime tilt against a fixed equal split | shadow |
| Quote status · execution gap · buying power | broker responses | shadow |

News is filtered deterministically: a publisher blocklist drops blogs and personal content, and republished articles are deduplicated by normalized headline. A market-level query runs alongside per-name queries so index moves, circuit breakers and regulatory news are visible at all. Channels are merged round-robin rather than concatenated, since anything appended last is what prompt truncation removes first.

Fundamentals are a separate channel from the price feature vector, not extra features. A value that changes once a quarter would sit inside the pattern key and split every repeat observation the admission gate depends on.

The concentration metric exists because a per-asset cap enforces only **nominal** diversification. Two names correlated at 0.9 can each sit at the cap while the portfolio is effectively a single bet, so the metric reports correlation-adjusted effective positions, `1/(wᵀΡw)`, which collapses toward 1 in exactly that case.

## Alpha Lab (weekly)

An LLM writer proposes factors in a restricted DSL (AST whitelist, no forward references). A lightweight rank IC/ICIR backtester screens them, then a four-stage admission gate applies: |IC| ≥ 0.02 → correlation < 0.70 against the library → batch dedup → out-of-sample sign consistency. Admitted factors feed the next daily decision as top-3 signals. Research and trading universes are separate: 10 crypto pairs, 12 US names.

Each cycle begins by retiring factors whose realized live IC edge has decayed. Alpha decays once discovered, so the system tracks that actively instead of assuming permanence. Currently active: one factor in crypto, one in US; two retired.

## Cross-market capital (shadow)

Each market's regime is classified from free daily index bars, and the budget is tilted deterministically off an equal split by the regime difference. No relative difference means no transfer — downside defense belongs to each market's own cash weight, which keeps the markets isolated.

Actual capital movement passes LLM-free guards only: a hardcoded destination allowlist (never sourced from runtime, observations or the LLM), per-transfer and daily caps, a drift gate, cooldown and balance reconciliation. Exchange KRW withdrawal is automatic to a registered account of the same owner. The brokerage-to-bank leg has no transfer API, so a human moves the funds and the system verifies completion by balance query rather than trusting a confirmation.

Preview without execution: `uv run python scripts/run_treasury_step.py`

## Evaluation

Four arms run virtually against the same observations, marked at t−1 closes with transaction costs.

| Arm | Content |
|---|---|
| `llm` | the allocation actually submitted, after the risk engine |
| `llm_base` | raw LLM allocation — no memory, no risk engine |
| `bh` | buy and hold |
| `random` | random allocation |

Significance uses a sign test over **non-overlapping** 20-day chunks. Rolling windows are reported as descriptive statistics only, since overlapping windows inflate the sample through autocorrelation.

Report: `uv run python scripts/report_ablation.py`

## Results (2026-08-19)

Virtual arms since July 2026, starting from an index of 100,000.

| Market | Steps | Agent | B&H | Agent MDD | B&H MDD |
|---|---|---|---|---|---|
| Crypto | 32 | +1.8% | +4.9% | 3.8% | 5.2% |
| US | 26 | +3.4% | +4.8% | 3.6% | 5.3% |
| KR | 24 | +5.2% | +4.6% | 7.2% | 21.4% |

The first sign-test p-value needs roughly 100 aligned trading days, so **no arm is yet statistically better or worse than its baseline**. Directionally, drawdown is the only consistent edge: it is below buy-and-hold in all three markets while excess return is mixed. Read that as the risk engine working, not as alpha.

Nothing has been promoted out of memory yet — 102 episodic entries, 0 promotions — which is the admission gate behaving as designed on a thin sample. Until a promotion happens, the `llm − llm_base` gap measures the risk engine and turnover limits rather than memory.

Backtest numbers are deliberately not published here.

## Layout

```
adapters/     market adapters (KIS domestic/overseas · Alpaca · Upbit · Binance testnet)
              + leakage guard, news/RSS, DART & EDGAR filings, FRED, per-market cash ledger
harness/      daily loop + baselines + structured JSON logging + run deadline + usage metering
trader/       LLM decision (2-pass residual) · decision schema · features · fundamentals · events
memory/       episodic/semantic/procedural + admission/retention gates + influence control
risk/         deterministic guardrails · failure veto · live-order caps · concentration metric
regime/       state machine + jump model + macro + realized volatility + cross-market proposal
watcher/      intraday trigger evaluation (pure functions)
treasury/     cross-market capital transfer guards, automatic and manual legs
alpha_lab/    factor DSL → IC backtester → 4-stage admission (writer/judge LLM loop)
reflection/   weekly performance and contribution re-scoring
interaction/  chat gateway, answers must cite evidence IDs (FastAPI + MCP)
eval/         4-arm ablation + rolling significance + cross-market combined index
gui/          read-only dashboard
llm/          multi-provider backbone + token/cost accounting
scripts/      operations: daily step · alpha cycle · watcher · treasury · reports · scheduler
```

## Getting started

```bash
uv sync
cp .env.example .env   # broker and LLM keys
uv run python scripts/check_credentials.py
uv run python scripts/run_paper_step.py --dry-run
uv run python scripts/run_paper_step.py --markets CRYPTO
```

Adapters are chosen by which keys are present. With no live keys the system runs entirely on simulated and testnet accounts.

| Variable | Purpose |
|---|---|
| `KIS_REAL_APP_KEY/SECRET/ACCOUNT` | live KR + US through one KIS margin account |
| `KIS_PAPER_APP_KEY/SECRET/ACCOUNT` | KR simulated account (used when live keys are absent) |
| `UPBIT_API_KEY/SECRET` | live crypto (KRW markets) — grant balance and order rights only, never withdrawal |
| `BINANCE_TESTNET_API_KEY/SECRET` | crypto testnet execution; quotes and research data come from the public API |
| `ALPACA_PAPER_API_KEY/SECRET` | US news · Alpha Lab US panel bars · US paper execution when live keys are absent |
| `LIVE_MAX_ORDER_*` / `LIVE_MAX_DAILY_*` | absolute buy ceilings per venue currency (USD / KRW) |
| `LIVE_MAX_ORDER_PCT_{KR,US,CRYPTO}` | per-order buy cap as a fraction of equity; the smaller of the two applies |
| `LLM_SMART` / `LLM_FAST` | `provider:model` + that provider's API key |
| `DART_API_KEY` / `FRED_API_KEY` / `SEC_USER_AGENT` | KR filings, US macro, EDGAR contact identifier |
| `OBSERVATION_TRADING_DAYS` / `OBSERVATION_NEWS_DAYS` | window lengths (defaults 3 / 7) |
| `ALERT_WEBHOOK_URL` | Slack or Discord webhook for order rejections and circuit trips |
| `INTERACTION_API_TOKEN` | chat gateway bearer token, required if exposed |

## Operations

Markets run as separate jobs because orders only fill during their own session.

| Command | Schedule | Purpose |
|---|---|---|
| `scripts/run_paper_step.py --markets KR` | daily 10:00 KST | observe → decide → risk → order → memory |
| `scripts/run_paper_step.py --markets US` | daily 00:30 KST | same, inside US regular hours year-round |
| `scripts/run_paper_step.py --markets CRYPTO` | daily 23:00 KST | same, 24/7 venue |
| `scripts/run_watcher.py --market CRYPTO\|KR` | every 15 min | intraday trigger check |
| `scripts/run_alpha_lab.py` | Sunday 22:00 | factor generation → backtest → admission |
| `scripts/request_capabilities.py` | monthly, 1st 20:30 | agent-authored capability requests from measured gaps |
| `scripts/propose_improvements.py` | monthly, 1st 21:00 | self-improvement proposals, never auto-applied |
| `scripts/run_treasury_step.py` | manual, dry-run | transfer plan and guard verdict, no execution |
| `scripts/report_ablation.py` | manual | memory delta, α vs B&H, regime scoring |
| `streamlit run gui/dashboard.py` | manual | read-only dashboard |
| `uvicorn interaction.api:app --port 8721` | continuous | chat gateway |
| `scripts/run_mcp_server.py` | on demand | MCP server for Claude |

Schedule with either launchd (`scripts/launchd/`) or the Docker Compose scheduler, never both.

```bash
docker compose up -d   # gateway (:8721) + scheduler
```

State and logs persist in the `./data` volume.

## Lineage

Verified mechanisms adapted to individual-scale operation.

| Source | What was adapted | Reference |
|---|---|---|
| FactorMiner | factor admission gates | [arXiv:2602.14670](https://arxiv.org/abs/2602.14670) |
| AlphaMemo | residual memory, asymmetric veto | [arXiv:2606.20625](https://arxiv.org/abs/2606.20625) |
| LiveTradeBench | live evaluation protocol | [arXiv:2511.03628](https://arxiv.org/abs/2511.03628) |
| FinMem | layered agent memory | [arXiv:2311.13743](https://arxiv.org/abs/2311.13743) |
| Alpha Arena | evidence for keeping the feature set small | [nof1.ai](https://nof1.ai/) |

## Disclaimer

Live trading is limited to small size and expands only against explicit promotion criteria measured on live data. This software is not investment advice, and losses from its use are the user's responsibility.
