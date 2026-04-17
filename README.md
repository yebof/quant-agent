# quant-agent

LLM multi-agent quantitative trading system for US equities. 8 specialized AI agents analyze markets from different angles (technical, macro, news intelligence, SEC earnings), synthesize into trading decisions with explicit chain-of-thought reasoning, and execute via Alpaca with multi-layer risk controls.

## Architecture

```
Morning (pre-market)
     │
     ├─ Cancel stale orders
     ├─ Check market calendar (skip holidays)
     │
     ├─── Parallel Data Fetch ─────────────────────────────┐
     │  Macro Analyst    News Intelligence    Tech Analyst  │  Earnings
     │  (FRED: VIX,      (3-layer: narrative  (pre-filtered │  (SEC EDGAR,
     │   yields, fed)     + state changes      actionable    │   background)
     │                    + stock alerts)       signals only) │
     │                                                       │
     ├─ Load yesterday's evening insights (cross-session memory)
     │
     ▼
  Portfolio Manager (7-step CoT; R/R-weighted sizing; drawdown-aware)
     │
     ├─ Symbol Guard (universe + analysis check)
     ├─ Hard Risk Engine (per-BUY, leverage-adjusted)
     │     includes advisory violations:
     │     • macro_exposure_deviation  (>15pp from Macro target)
     │     • correlation_cluster       (>50% of book in corr>0.7 peers)
     │     • data_degraded             (>=2 upstream sources failed)
     ├─ Risk Manager LLM (6-step CoT; R/R veto; scale_all_buys; tech fidelity)
     │
     ▼
  Execute: SELLs first → refresh cash → BUYs
           (limit orders, auto-raised to market price, tick-size quantized)
           (OTO bracket: stop-loss only, no hard take-profit)

Midday (intraday)
     │
     ├─ Sync positions (closed symbols purged)
     ├─ Load morning trade context (stop/target/reasoning)
     ▼
  Midday Reviewer (real trailing stop + profit management)
     │  < 3%  profit → keep original stop
     │  3-8%  profit → trail to breakeven
     │  8-15% profit → trail to halfway
     │  > 15% profit → trail to 70% of move
     │
     ├─ Daily loss check → emergency sell-all if > 3%
     └─ Execute SELL / REDUCE / TRAIL_STOP recommendations
        (TRAIL_STOP actually cancels broker stop + submits new one)

Evening (post-market)
     │
     ├─ Record daily PnL
     ▼
  Evening Analyst → save insights for next morning
     (lessons, outlook, suggested actions, risk rating)
```

## Agents

| Agent | Role | Key Feature |
|-------|------|-------------|
| **Tech Analyst** | Batch technical analysis | 5-step CoT (trend / momentum / volatility / volume / S&R). ATR-based default stop (`entry − 2*ATR`). Output rating + `conviction` (high/medium/low) + `reference_target` + `thesis_invalid_if` (soft exit condition). **Auto-computed `risk_reward`** (Python-calculated, not LLM-trusted) flows into PM sizing and RM veto logic. **Signal-age memory** (`data/tech/last_ratings.json`): prior rating surfaced to LLM as context; `signal_age_days` counted to spot stale setups — PM cuts allocation on 8+ day stale BUYs. **Valuation context** (yfinance trailing PE / forward PE / P/S) surfaced per symbol — LLM flags >40x forward PE or >15x P/S as stretched in `reasoning_chain.support_resistance`. Pre-filter thresholds normalized by ATR. Auto-chunks batch > 30 symbols. Cross-field validator: BUY stop must be below entry, SELL above. |
| **News Intelligence** | 3-layer news analysis | Layer 1: Persistent macro narrative. Layer 2: State change detection. Layer 3: Per-symbol alerts with conviction. Daily storage in `data/news/` |
| **Macro Analyst** | Regime assessment & sector guidance | 6-step CoT (vol / curve / monetary / inflation+labor+credit / cross-signal / sector). Inputs: VIX, 2Y/10Y yields, **DFF** (daily fed funds), **core & headline CPI**, **UNRATE**, **HY OAS**. Persists yesterday's regime → detects `regime_shift`. Cross-references News narrative via `alignment_with_news`. Emits bull/bear view-change triggers. |
| **Earnings Analyst** | SEC 10-Q/10-K analysis | Revenue, margins, cash flow, strategic direction, competitive positioning, strategic vs operational risks, strategy consistency across filings. `investment_implications` carries a 5-step `reasoning_chain` (fundamental_quality / growth_trajectory / strategic_risks / management_execution / valuation_context) — sentiment call is derivable from the numbers, not a vibe check. |
| **Portfolio Manager** | Central decision maker | Mandatory 7-step reasoning chain (+ continuity check) across 4 memory layers: L1 today's signals, L2 per-position entry context + Tech rating 7-day trajectory, L3a rolling Portfolio Narrative (last 7 evenings), L3b Macro Regime Trajectory (7 days), L3c Active HIGH-conviction state_changes (14 days). Sizing scales by TechAnalyst's `risk_reward`: R/R ≥ 3 boost, R/R < 1.5 requires catalyst or shrinks. **Drawdown-aware**: halves new BUYs when `in_drawdown` flagged. **Holding discipline** (tiered by days_held): <5d → default HOLD unless thesis_invalid_if or today's macro flip; 5-15d → standard; >15d profitable + trend intact → let it run. Early exits via `thesis_invalid_if` save 3-5% vs stop-triggered. |
| **Risk Manager** | Trade review with veto power | Mandatory 6-step `reasoning_chain` (rr_audit / signal_fidelity / correlation_check / event_risk / sizing_sanity / overall) — vague approvals rejected. Enforces R/R discipline: BUYs with R/R < 1.5 must be downsized via modifications or rejected unless PM named a catalyst. Sees raw Tech ratings + R/R + full macro context. Can modify per-symbol fields OR apply portfolio-wide `scale_all_buys` (0.0-1.0). |
| **Midday Reviewer** | Profit management & trailing-stop execution | Trailing-stop logic is **real**, not cosmetic — `TRAIL_STOP` action actually cancels the broker's old stop and submits a new one at the specified price via `AlpacaBroker.replace_stop_loss`. Sees VIX + HY OAS + core CPI to gauge whether to tighten stops broadly. Output is Pydantic `MiddayReview` — action enum enforced (typos like `TRIAL_STOP` rejected); `TRAIL_STOP` requires `new_stop_price > 0`. |
| **Evening Analyst** | Daily P&L review & learning | Pydantic `EveningReport` with enum `risk_rating`. **Outlook retrospective**: reads yesterday's `tomorrow_outlook` and grades it honestly against today's reality via `previous_outlook_assessment` — builds calibration over time. Outputs feed into next morning's PM prompt (cross-session memory). |

## Risk Management

### Hard Risk Engine (non-negotiable, per-BUY)
- Single position: max 20% (gross exposure — SQQQ 3x, SDS 2x counted at full magnitude)
- Total **net** exposure: max 90% (hedges cancel — e.g. long SPY + short SH ≈ zero net)
- Daily loss: max 3% of prior-close equity (`equity − last_equity`; includes realized fills from broker-triggered OTO stops, not just marks)
- Sector concentration: max 40% (gross, cumulative including pending same-sector buys)
- Stop loss required
- Inverse ETFs (SH, SDS, PSQ, SQQQ) carry signed multipliers for net exposure and gross magnitude for sizing/sector caps
- **Advisory**: if projected net exposure deviates > 15pp from Macro's `target_invested_pct`, emits a non-blocking `macro_exposure_deviation` violation — RiskManager sees it and can respond with `scale_all_buys`
- **Correlation cluster** (advisory): a proposed BUY plus already-held positions correlated > 0.7 with it (120-day daily returns) must not together exceed 50% of book. Catches AI / mega-cap-growth concentration that sector caps miss when yfinance tags NVDA (Technology) and GOOGL (Communication Services) separately.

### Execution Safety
- Stale orders cancelled before each session
- SELLs execute before BUYs (free cash first)
- SELL uses limit price (0.5% below market for slippage protection); midday emergency sells use a wider 1% buffer to ensure fill during cascades
- BUY limit price auto-raised to market if below (prevents unfilled orders)
- BUY attaches OTO stop-loss via Alpaca (broker-enforced)
- No hard take-profit — profit managed by midday reviewer's trailing stop logic
- Partial sell via `allocation_pct` (1–99 = partial, 100 = full exit; 0 is treated as a no-op)

### LLM Risk Manager
- Mandatory 6-step `reasoning_chain`: rr_audit → signal_fidelity → correlation_check → event_risk → sizing_sanity → overall
- Enforces R/R ≥ 1.5 discipline; downsizes or rejects BUYs that miss without an explicit catalyst
- Receives raw TechAnalyst signals + computed R/R to verify PM's fidelity (no silent contradictions)
- Sees hard-engine advisories inline: `correlation_cluster`, `macro_exposure_deviation`, `data_degraded` — must address each in `reasoning_chain`
- Reviews risk/reward, correlation concentration, and imminent event risk (earnings / FOMC within 3 days)
- Can modify per-symbol fields (aliases: `target`→`take_profit`, `stop`→`stop_loss`) OR set portfolio-level `scale_all_buys` (0.0-1.0) to shrink all BUYs uniformly
- Modifications and scaling re-validated through the risk engine

## Setup

### Prerequisites
- Python 3.11+
- [Alpaca](https://alpaca.markets/) account (paper trading supported)
- [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) API key
- OpenAI or Anthropic API key

### Install

```bash
git clone <repo-url> && cd quant-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure

1. Create `.env`:
```bash
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
FRED_API_KEY=...
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
EOF
```

2. Edit `config/settings.yaml` — models per agent, risk parameters, trading universe, schedule

## Usage

```bash
source .env

python main.py --mode morning    # Analyze + trade
python main.py --mode midday     # Position review + trailing stops
python main.py --mode evening    # PnL report + insights for tomorrow
python main.py --mode live       # Scheduler (all three on cron, weekdays)
```

Automated via macOS launchd — plist files in `~/Library/LaunchAgents/com.quant-agent.*.plist`.

**Timezone-resilient scheduling**: plists fire every 30 minutes (`StartInterval=1800`); a bash wrapper `scripts/run_if_et_window.sh` checks whether the current **US/Eastern** time is inside the target window and whether the mode already ran in the last hour. Runs the right session at the right ET moment regardless of the host's timezone (handy when traveling). Windows: morning 09:30-12:00 ET, midday 15:00-16:30 ET, evening 20:00-22:00 ET, Mon-Fri.

## Trading Universe

77 symbols across 10 sectors:
- **Index ETFs**: SPY, QQQ, IWM, DIA
- **Sector ETFs**: XLF, XLE, XLV, XLI, XLP, XLY, XLU, XLRE, XLB, SMH, DRAM
- **Inverse ETFs**: SH, SDS, PSQ, SQQQ (leverage-corrected in risk engine)
- **Individual stocks**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, AVGO, JPM, CAT, etc.

## Project Structure

```
quant-agent/
├── main.py                        # CLI entry point
├── config/
│   ├── settings.yaml              # Models, risk params, universe, schedule
│   └── prompts/                   # System prompts for each agent
├── src/
│   ├── pipeline.py                # Orchestrator (morning/midday/evening)
│   ├── scheduler.py               # APScheduler + launchd
│   ├── config.py                  # Pydantic config with API key validation
│   ├── models.py                  # Data models (ReasoningChain, MacroNarrative, etc.)
│   ├── agents/                    # 8 LLM agents
│   ├── data/
│   │   ├── market.py              # yfinance OHLCV
│   │   ├── macro.py               # FRED API (VIX, yields, DFF, CPI, UNRATE, HY OAS)
│   │   ├── macro_store.py         # Persists yesterday's regime for shift detection
│   │   ├── news.py                # RSS feeds + symbol mention tagging
│   │   ├── news_store.py          # Dated news storage + narrative persistence
│   │   ├── earnings.py            # SEC EDGAR provider
│   │   ├── technical.py           # TA indicators (MA, RSI, MACD, BB, ATR)
│   │   ├── correlation.py         # 120d pairwise return correlations + cluster detection
│   │   └── tech_store.py          # Per-symbol rating memory + signal-age computation
│   ├── execution/
│   │   └── broker.py              # Alpaca (OTO brackets, calendar, live prices)
│   ├── risk/
│   │   └── rules.py               # Hard risk engine (leverage-adjusted)
│   └── storage/
│       └── db.py                  # SQLite (trades, positions, logs, PnL, insights)
├── tests/                         # 204 tests
├── data/
│   ├── quant_agent.db             # SQLite audit trail
│   ├── earnings/                  # Cached SEC filing analyses
│   └── news/                      # Daily reports + persistent macro narrative
└── logs/                          # Timestamped run logs
```

## Tests

```bash
pytest tests/ -v    # 204 tests
```

## Data Sources

| Source | Data | Provider |
|--------|------|----------|
| Market data | OHLCV, sector performance, valuation (trailing PE / forward PE / P/S) | yfinance |
| Macro | VIX, 2Y/10Y yields, DFF (daily fed funds), headline & core CPI, PCE, UNRATE, HY OAS spread | FRED API |
| News | Real-time headlines (9 RSS feeds) | Reuters, CNBC, MarketWatch, AP, BBC, NPR, Fed |
| Earnings | 10-Q/10-K filings + strategic analysis | SEC EDGAR |
| Trading | Orders, positions, account, calendar, live quotes | Alpaca API |

## Storage

**SQLite** (`data/quant_agent.db`, WAL mode):
- Trades (with stop/target, reasoning, actual submitted fill price)
- Position snapshots (synced each midday — rows for closed symbols are purged)
- Agent logs for all 8 LLM agents including earnings (full input/output, tokens, model — auto-pruned after 2 years for quarter-over-quarter learning)
- Daily P&L records
- Evening insights (cross-session memory)

**File-based** (`data/news/`):
- `macro_narrative.json` — persistent grand backdrop, evolves daily
- `YYYY-MM-DD/full_report.json` — daily news intelligence report
- `YYYY-MM-DD/stock_alerts/` — per-symbol news alerts
- `YYYY-MM-DD/raw_headlines.json` — raw headlines for audit

**File-based** (`data/macro/`):
- `last_state.json` — yesterday's regime/confidence/outlook snapshot for shift detection

**File-based** (`data/tech/`):
- `last_ratings.json` — per-symbol prior rating + first_seen_date for signal-age tracking; TechAnalyst reads on next morning, PM uses age to cut allocations on stale setups

**File-based** (`data/earnings/`):
- `{SYMBOL}/analysis_{10-Q}_{date}.md` — cached SEC filing analyses
