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
  Portfolio Manager (7-step CoT reasoning chain)
     │
     ├─ Symbol Guard (universe + analysis check)
     ├─ Hard Risk Engine (per-BUY, leverage-adjusted)
     ├─ Risk Manager LLM (audits PM reasoning chain)
     │
     ▼
  Execute: SELLs first → refresh cash → BUYs
           (limit orders, auto-raised to market price)
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
| **Tech Analyst** | Batch technical analysis | 5-step CoT (trend / momentum / volatility / volume / S&R). ATR-based default stop (`entry − 2*ATR`). Output rating + `conviction` (high/medium/low) + `reference_target` (soft, not hard TP). Pre-filter thresholds normalized by ATR so leveraged ETFs use proportional bars. Auto-chunks batch > 30 symbols to stay under LLM context. Cross-field validator: BUY stop must be below entry, SELL above. |
| **News Intelligence** | 3-layer news analysis | Layer 1: Persistent macro narrative. Layer 2: State change detection. Layer 3: Per-symbol alerts with conviction. Daily storage in `data/news/` |
| **Macro Analyst** | Regime assessment & sector guidance | 6-step CoT (vol / curve / monetary / inflation+labor+credit / cross-signal / sector). Inputs: VIX, 2Y/10Y yields, **DFF** (daily fed funds), **core & headline CPI**, **UNRATE**, **HY OAS**. Persists yesterday's regime → detects `regime_shift`. Cross-references News narrative via `alignment_with_news`. Emits bull/bear view-change triggers. |
| **Earnings Analyst** | SEC 10-Q/10-K analysis | Revenue, margins, strategic direction, competitive positioning, strategic vs operational risks, strategy consistency across filings |
| **Portfolio Manager** | Central decision maker | Mandatory 7-step reasoning chain (macro → news → earnings → signal conflicts → sizing → balance → cash). Each decision traces which signals aligned/conflicted |
| **Risk Manager** | Trade review with veto power | Audits PM's reasoning chain for logic errors. Also receives the **raw Tech Analyst ratings** to audit PM's fidelity to underlying signals. Sees full macro context (VIX + yields + spread + fed funds). Can modify per-symbol fields OR apply portfolio-wide `scale_all_buys` (0.0-1.0) to pull all BUY sizes down uniformly. |
| **Midday Reviewer** | Profit management & trailing-stop execution | Trailing-stop logic is **real**, not cosmetic — `TRAIL_STOP` action actually cancels the broker's old stop and submits a new one at the specified price via `AlpacaBroker.replace_stop_loss`. Sees VIX + HY OAS + core CPI to gauge whether to tighten stops broadly. |
| **Evening Analyst** | Daily P&L review & learning | Outputs feed into next morning's PM prompt (cross-session memory) |

## Risk Management

### Hard Risk Engine (non-negotiable, per-BUY)
- Single position: max 20% (gross exposure — SQQQ 3x, SDS 2x counted at full magnitude)
- Total **net** exposure: max 90% (hedges cancel — e.g. long SPY + short SH ≈ zero net)
- Daily loss: max 3% of prior-close equity (`equity − last_equity`; includes realized fills from broker-triggered OTO stops, not just marks)
- Sector concentration: max 40% (gross, cumulative including pending same-sector buys)
- Stop loss required
- Inverse ETFs (SH, SDS, PSQ, SQQQ) carry signed multipliers for net exposure and gross magnitude for sizing/sector caps
- **Advisory**: if projected net exposure deviates > 15pp from Macro's `target_invested_pct`, emits a non-blocking `macro_exposure_deviation` violation — RiskManager sees it and can respond with `scale_all_buys`

### Execution Safety
- Stale orders cancelled before each session
- SELLs execute before BUYs (free cash first)
- SELL uses limit price (0.5% below market for slippage protection); midday emergency sells use a wider 1% buffer to ensure fill during cascades
- BUY limit price auto-raised to market if below (prevents unfilled orders)
- BUY attaches OTO stop-loss via Alpaca (broker-enforced)
- No hard take-profit — profit managed by midday reviewer's trailing stop logic
- Partial sell via `allocation_pct` (1–99 = partial, 100 = full exit; 0 is treated as a no-op)

### LLM Risk Manager
- Audits PM's 7-step reasoning chain for internal contradictions
- Receives TechAnalyst signals to verify PM translated them faithfully
- Reviews risk/reward ratios (min 1:2 preferred)
- Checks correlation and concentration risk
- Can modify per-symbol fields (aliases: `target`→`take_profit`, `stop`→`stop_loss`) OR set portfolio-level `scale_all_buys` (0.0-1.0) to shrink all BUYs uniformly
- Modifications and scaling re-validated through risk engine

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

Automated via macOS launchd — plist files in `~/Library/LaunchAgents/com.quant-agent.*.plist`

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
│   │   └── technical.py           # TA indicators (MA, RSI, MACD, BB, ATR)
│   ├── execution/
│   │   └── broker.py              # Alpaca (OTO brackets, calendar, live prices)
│   ├── risk/
│   │   └── rules.py               # Hard risk engine (leverage-adjusted)
│   └── storage/
│       └── db.py                  # SQLite (trades, positions, logs, PnL, insights)
├── tests/                         # 157 tests
├── data/
│   ├── quant_agent.db             # SQLite audit trail
│   ├── earnings/                  # Cached SEC filing analyses
│   └── news/                      # Daily reports + persistent macro narrative
└── logs/                          # Timestamped run logs
```

## Tests

```bash
pytest tests/ -v    # 157 tests
```

## Data Sources

| Source | Data | Provider |
|--------|------|----------|
| Market data | OHLCV, sector performance | yfinance |
| Macro | VIX, 2Y/10Y yields, DFF (daily fed funds), headline & core CPI, PCE, UNRATE, HY OAS spread | FRED API |
| News | Real-time headlines (9 RSS feeds) | Reuters, CNBC, MarketWatch, AP, BBC, NPR, Fed |
| Earnings | 10-Q/10-K filings + strategic analysis | SEC EDGAR |
| Trading | Orders, positions, account, calendar, live quotes | Alpaca API |

## Storage

**SQLite** (`data/quant_agent.db`, WAL mode):
- Trades (with stop/target, reasoning, actual submitted fill price)
- Position snapshots (synced each midday — rows for closed symbols are purged)
- Agent logs (full input/output, tokens, model — auto-pruned after 30 days)
- Daily P&L records
- Evening insights (cross-session memory)

**File-based** (`data/news/`):
- `macro_narrative.json` — persistent grand backdrop, evolves daily
- `YYYY-MM-DD/full_report.json` — daily news intelligence report
- `YYYY-MM-DD/stock_alerts/` — per-symbol news alerts
- `YYYY-MM-DD/raw_headlines.json` — raw headlines for audit

**File-based** (`data/macro/`):
- `last_state.json` — yesterday's regime/confidence/outlook snapshot for shift detection

**File-based** (`data/earnings/`):
- `{SYMBOL}/analysis_{10-Q}_{date}.md` — cached SEC filing analyses
