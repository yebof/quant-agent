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
     ├─ Sync positions
     ├─ Load morning trade context (stop/target/reasoning)
     ▼
  Midday Reviewer (trailing stop profit management)
     │  < 3%  profit → keep original stop
     │  3-8%  profit → trail to breakeven
     │  8-15% profit → trail to halfway
     │  > 15% profit → trail to 70% of move
     │
     ├─ Daily loss check → emergency sell-all if > 3%
     └─ Execute SELL/REDUCE recommendations

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
| **Tech Analyst** | Batch technical analysis | Pre-filtered: only actionable signals sent to LLM (RSI extremes, BB proximity, MACD crossover, volume spike) |
| **News Intelligence** | 3-layer news analysis | Layer 1: Persistent macro narrative. Layer 2: State change detection. Layer 3: Per-symbol alerts with conviction. Daily storage in `data/news/` |
| **Macro Analyst** | Regime assessment & sector guidance | VIX, Treasury yields, Fed funds rate via FRED API |
| **Earnings Analyst** | SEC 10-Q/10-K analysis | Revenue, margins, strategic direction, competitive positioning, strategic vs operational risks, strategy consistency across filings |
| **Portfolio Manager** | Central decision maker | Mandatory 7-step reasoning chain (macro → news → earnings → signal conflicts → sizing → balance → cash). Each decision traces which signals aligned/conflicted |
| **Risk Manager** | Trade review with veto power | Audits PM's reasoning chain for logic errors. Sees full macro context (VIX + yields + spread + fed funds). Can modify allocation, stop, target |
| **Midday Reviewer** | Profit management & risk check | Trailing stop logic. Receives morning trade stop/target/thesis. Lets winners run, cuts losers |
| **Evening Analyst** | Daily P&L review & learning | Outputs feed into next morning's PM prompt (cross-session memory) |

## Risk Management

### Hard Risk Engine (non-negotiable, per-BUY)
- Single position: max 20% (cumulative including pending same-symbol buys)
- Total exposure: max 90% (cumulative across batch, leverage-adjusted)
- Daily loss: max 3% (uses Alpaca intraday P&L)
- Sector concentration: max 40% (cumulative including pending same-sector buys)
- Stop loss required
- Inverse/leveraged ETF correction: SQQQ counted as 3x, SDS as 2x effective exposure

### Execution Safety
- Stale orders cancelled before each session
- SELLs execute before BUYs (free cash first)
- SELL uses limit price (0.5% below market for slippage protection)
- BUY limit price auto-raised to market if below (prevents unfilled orders)
- BUY attaches OTO stop-loss via Alpaca (broker-enforced)
- No hard take-profit — profit managed by midday reviewer's trailing stop logic
- Partial sell support via `allocation_pct`

### LLM Risk Manager
- Audits PM's 7-step reasoning chain for internal contradictions
- Reviews risk/reward ratios (min 1:2 preferred)
- Checks correlation and concentration risk
- Can modify trades (field aliases: `target`→`take_profit`, `stop`→`stop_loss`)
- Modifications re-validated through risk engine

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
│   │   ├── macro.py               # FRED API
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
├── tests/                         # 109 tests
├── data/
│   ├── quant_agent.db             # SQLite audit trail
│   ├── earnings/                  # Cached SEC filing analyses
│   └── news/                      # Daily reports + persistent macro narrative
└── logs/                          # Timestamped run logs
```

## Tests

```bash
pytest tests/ -v    # 109 tests
```

## Data Sources

| Source | Data | Provider |
|--------|------|----------|
| Market data | OHLCV, sector performance | yfinance |
| Macro | VIX, Treasury yields, Fed funds rate | FRED API |
| News | Real-time headlines (9 RSS feeds) | Reuters, CNBC, MarketWatch, AP, BBC, NPR, Fed |
| Earnings | 10-Q/10-K filings + strategic analysis | SEC EDGAR |
| Trading | Orders, positions, account, calendar, live quotes | Alpaca API |

## Storage

**SQLite** (`data/quant_agent.db`):
- Trades (with stop/target, reasoning)
- Position snapshots
- Agent logs (full input/output, tokens, model)
- Daily P&L records
- Evening insights (cross-session memory)

**File-based** (`data/news/`):
- `macro_narrative.json` — persistent grand backdrop, evolves daily
- `YYYY-MM-DD/full_report.json` — daily news intelligence report
- `YYYY-MM-DD/stock_alerts/` — per-symbol news alerts
- `YYYY-MM-DD/raw_headlines.json` — raw headlines for audit

**File-based** (`data/earnings/`):
- `{SYMBOL}/analysis_{10-Q}_{date}.md` — cached SEC filing analyses
