# trading-agent

A self-improving trading agent built around agent memory. It ports verified memory mechanisms — statistical admission gates, residual/confidence-controlled influence — into live paper-trading decisions.

> Paper and simulated trading only. This is not investment advice.

[한국어 README](README.ko.md)

## Why

- **"Time travel is cheating."** Past markets are already baked into LLM weights, so backtest performance cannot justify promotion. Success is judged only on live (paper) data after the model cutoff. Backtests are for signal screening.
- **Verifier before agent.** Without a measurement harness — adapters, paper loop, structured logging — self-improvement is self-deception.
- **Win on process, lose on resources.** Free daily-resolution data only (OHLCV, RSS, public filings). No high-frequency trading, no market microstructure, no paid alternative data, no large-scale factor arms race. The target is α > 0 against buy-and-hold, not institutional alpha.

## Design principles

- **Memory is a calibrated residual.** A lesson is a correction term on the base decision. It only intervenes after repetition n≥5, sign test p≤0.05, and probation on live out-of-sample data. Influence scales with confidence and is capped at a bounded deviation. Uncited memory has zero influence.
- **Failures hard-veto, successes soft-prior.** Only forbidden (failure) patterns block deterministically, and only at high confidence.
- **The Risk Engine never consults the LLM.** Drawdown circuit breaker → forbidden veto → per-asset cap → minimum cash → turnover limit.
- **The action space is allocation weights**, not discrete buy/sell/hold. Weights sum to 1 including cash, long-only. Cash is a position.
- **Markets are isolated.** Strategies and memory are namespaced per market. Cross-market generalization is not assumed.
- **Leakage control.** The observation upper bound of `t−1` is immutable, blocking same-day data. Window *lengths* are tunable experiment variables: bars use the last N trading days (default 3, so a Monday still sees N bars), news uses the last N calendar days (default 7, decoupled from the bar window since events do not respect trading sessions). Every observation records a collection timestamp.
- **No automatic prompt mutation.** Self-improvement produces monthly proposals that require explicit approval.

## Daily loop

```
observe (bars ≤ t−1, news ≤ t−1) ─ leakage guard
  → 7 features (RSI · MACD hist · SMA gap · return · ATR · volume ratio · drawdown)
  → top-3 alpha signals (Alpha Lab admitted factors)
  → scheduled macro events (FOMC / BOK, public calendar)
  → LLM 2-pass decision (base → residual blend of cited memory only)
  → Risk Engine (deterministic guardrails)
  → order (paper)
  → 4 virtual arms in parallel (llm / llm_base / B&H / random)
  → memory record → outcome scoring → admission / probation / retention
  → shadow channels (regime, volatility, flows, concentration, cross-market budget)
  → daily briefing
```

Every decision log is structured JSON and includes the memory and signal IDs it cited — the source of credit assignment.

## Intraday trigger

A stateless worker polls every 15 minutes and fires only on a sharp move against a rolling reference price. On fire it re-enters the same decision path with the intraday move injected as a separate labeled channel, so today's information never contaminates the observation window.

| Market | Threshold | Session gating |
|---|---|---|
| Crypto | 8% | none (24/7) |
| KR | 5% | 09:00–15:30 KST, weekdays |

Trigger decisions are excluded from the learning pipeline — admitting same-day information into memory would poison the admission statistics.

## Layout

```
adapters/     market adapters (Binance testnet · Alpaca · KIS domestic/overseas · Upbit)
              + observation leakage guard, news/RSS, DART filings, FRED macro
harness/      daily paper loop + baselines + structured JSON logging + run deadline
trader/       LLM decision (2-pass residual) + decision schema + features + event calendar
memory/       episodic/semantic/procedural + admission/retention gates + influence control
risk/         deterministic guardrails + forbidden veto + allocation concentration
regime/       market state machine + macro + realized volatility + cross-market proposal
watcher/      intraday trigger evaluation (pure functions)
treasury/     cross-market capital transfer guards, automatic and manual legs
alpha_lab/    factor DSL → IC backtester → 4-stage admission (writer/judge LLM loop)
reflection/   weekly performance and contribution re-scoring
interaction/  chat gateway, answers must cite evidence IDs (FastAPI + MCP)
eval/         4-arm ablation + rolling significance + cross-market combined index
gui/          read-only dashboard
llm/          multi-provider backbone + token/cost accounting
scripts/      operations: daily step, alpha cycle, watcher, treasury, briefing, scheduler
```

## Memory pipeline

```
decision ──▶ episodic (pattern_key + outcome: action return − hold return)
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

A single trade never promotes anything. Individual P&L is noise.

## Observation channels

Channels marked *shadow* are computed and logged but do not touch decisions or the Risk Engine. They are promoted only after live comparison shows they help.

| Channel | Source | State |
|---|---|---|
| OHLCV bars, 7 features | broker APIs | injected |
| News headlines | Google News RSS (per-name + market-level), CoinDesk/Cointelegraph | injected |
| Corporate filings (KR) | DART OpenAPI, merged into the news channel | injected |
| Alpha signals | Alpha Lab admitted factors | injected |
| Scheduled macro events | published FOMC / BOK calendars | injected |
| Intraday move | broker quote, separate labeled channel | injected on trigger |
| Market regime | index proxy daily bars, distribution-day state machine | shadow |
| Macro regime | FRED daily series | shadow |
| Realized volatility | index proxy, 20-day annualized | shadow |
| Investor flows (KR) | KIS daily foreign/institutional/retail net buying | shadow |
| Allocation concentration | own weights + 60-day correlations | shadow |
| Cross-market budget | regime tilt against fixed equal split | shadow |

News is filtered deterministically: a publisher blocklist drops blogs and personal content, and republished articles are deduplicated by normalized headline. A market-level query runs alongside per-name queries so index moves, circuit breakers, and regulatory news are visible at all.

The concentration metric exists because a per-asset cap only enforces *nominal* diversification. Two names correlated at 0.9 can each sit at the cap without violating anything, while the portfolio is effectively a single bet. The metric reports correlation-adjusted effective positions, `1/(wᵀΡw)`, which collapses toward 1 in exactly that case.

## Alpha Lab (weekly)

An LLM writer proposes factors in a restricted DSL (AST whitelist, no forward references). A lightweight rank IC/ICIR backtester screens them, then a 4-stage admission gate applies: |IC| ≥ 0.02 → correlation < 0.7 against the library → batch dedup → out-of-sample sign consistency. Admitted factors feed the next daily decision as top-3 signals. Research and trading universes are separate.

Each cycle begins by retiring factors whose realized live IC edge has decayed — alpha decays once discovered, so the system tracks it actively rather than assuming permanence.

Markets: crypto (10-pair research universe) and US equities (12 names).

## Cross-market capital allocation (shadow)

The three markets share one budget. Each market's regime is classified from free daily index bars, and the budget is tilted deterministically off a fixed equal split by the regime difference. No relative difference means no transfer — downside defense is handled by each market's own cash weight, preserving isolation.

Actual capital movement passes LLM-free guards only: a hardcoded destination allowlist (never sourced from runtime, observations, or the LLM), per-transfer and daily caps, a drift gate, cooldown, and balance reconciliation. Exchange KRW withdrawal is automatic, with the destination fixed to a registered account. The brokerage-to-bank leg has no transfer API, so a human moves the funds and the system verifies completion by balance query rather than trusting confirmation.

Preview without execution: `uv run python scripts/run_treasury_step.py`

## Evaluation

Four arms run virtually against the same observations to accumulate an ablation every day.

| Arm | Content |
|---|---|
| `llm` | full pipeline with memory |
| `llm_base` | memory removed — memory delta = llm − llm_base |
| `bh` | buy and hold |
| `random` | random allocation |

Significance uses a sign test over **non-overlapping** 20-day chunks. Rolling windows are reported as descriptive statistics only, since overlapping windows inflate the sample through autocorrelation.

Report: `uv run python scripts/report_ablation.py`

## Getting started

```bash
uv sync
cp .env.example .env   # broker and LLM keys
uv run python scripts/check_credentials.py
uv run python scripts/run_paper_step.py --dry-run
uv run python scripts/run_paper_step.py
```

### Environment variables

| Variable | Purpose |
|---|---|
| `BINANCE_TESTNET_API_KEY/SECRET` | crypto paper — orders on testnet, quotes from mainnet public API |
| `ALPACA_PAPER_API_KEY/SECRET` | US equities paper |
| `KIS_PAPER_APP_KEY/SECRET/ACCOUNT` | KR equities simulated account |
| `KIS_REAL_APP_KEY/SECRET/ACCOUNT` | optional — live small-size US via KIS overseas |
| `LIVE_MAX_ORDER_USD` / `LIVE_MAX_DAILY_USD` | absolute notional caps on the live path |
| `LLM_SMART` / `LLM_FAST` / `LLM_EMBED` | `provider:model`, e.g. `openai:gpt-5.5` |
| `DART_API_KEY` / `FRED_API_KEY` | KR filings, US macro series |
| `OBSERVATION_TRADING_DAYS` / `OBSERVATION_NEWS_DAYS` | window lengths (defaults 3 / 7) |
| `INTERACTION_API_TOKEN` | chat gateway bearer token, required if exposed |

## Operations

| Command | Schedule | Purpose |
|---|---|---|
| `scripts/run_paper_step.py --markets CRYPTO,US` | daily 23:00 KST | observe → decide → risk → order → memory |
| `scripts/run_paper_step.py --markets KR` | daily 10:00 KST | same, during KR market hours |
| `scripts/run_watcher.py --market CRYPTO\|KR` | every 15 min | intraday trigger check |
| `scripts/run_alpha_lab.py` | Sunday 22:00 | factor generation → backtest → admission |
| `scripts/request_capabilities.py` | monthly, 1st 20:30 | agent-authored capability requests from measured gaps |
| `scripts/propose_improvements.py` | monthly, 1st 21:00 | self-improvement proposals, never auto-applied |
| `scripts/run_treasury_step.py` | weekly/monthly, dry-run | transfer plan and guard verdict, no execution |
| `scripts/report_ablation.py` | manual | memory delta, α vs B&H, cross-market delta |
| `streamlit run gui/dashboard.py` | manual | read-only dashboard |
| `uvicorn interaction.api:app --port 8721` | continuous | chat gateway |
| `scripts/run_mcp_server.py` | on demand | MCP server for Claude |

Use either launchd (`scripts/launchd/`) or the Docker Compose scheduler, never both.

## Docker

```bash
docker compose up -d   # gateway (:8721) + scheduler
```

Do not run alongside host launchd jobs. State and logs persist in the `./data` volume.

## Status

| Area | State |
|---|---|
| Measurement harness (adapters, paper loop, logging) | done — crypto, US, KR all live |
| LLM trader + Risk Engine | done |
| Memory stack | done, accumulating data |
| Alpha Lab + weekly reflection | done, cycling |
| Intraday trigger | deployed — crypto and KR |
| Cross-market allocation | shadow and dry-run, execution pending verification |
| Interaction gateway | done |
| Live small-size transition | after paper verification |

## Results

Paper trading started in July 2026. Current samples run 14–21 aligned trading days per market, well short of the ~100 needed for the first sign-test p-value, so **no arm can yet be called statistically better or worse than its baseline**.

Directionally, drawdown is the only consistent edge so far: the agent's max drawdown is below buy-and-hold in every market, while excess return is roughly a wash. Read that as the risk engine working, not as alpha.

Backtest numbers are deliberately not published here.

## Lineage

Verified mechanisms adapted to individual-scale operation: FactorMiner (factor admission gates), AlphaMemo (residual memory, asymmetric valuation), Alpha Arena (evidence for keeping the feature set small), LiveTradeBench (live evaluation protocol), FinMem (layered agent memory).

## Disclaimer

Live transition happens only after sustained paper verification and explicit promotion criteria, starting at small size. This software is not investment advice, and losses from its use are the user's responsibility.
