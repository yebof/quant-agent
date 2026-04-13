# quant-agent

LLM multi-agent quantitative trading system for US equities. Uses multiple AI agents to analyze markets from different angles (technical, macro, news, earnings), make portfolio decisions, and execute trades via Alpaca with hard risk circuit breakers.

## Architecture

```
Morning (06:00)          Midday (12:00)         Evening (16:30)
     |                        |                       |
 [Parallel Data Fetch]   [Sync Positions]       [Record PnL]
  Macro  News  Tech       Midday Reviewer        Evening Analyst
  Analyst Analyst Analyst     |                       |
     |                   Risk Check              Tomorrow Outlook
 Portfolio Manager       Execute SELL/REDUCE
     |
 Risk Engine (hard rules)
     |
 Risk Manager LLM (review)
     |
 Execute via Alpaca
```

### Agents

| Agent | Role | Input |
|-------|------|-------|
| **Tech Analyst** | Batch technical analysis (MA, RSI, MACD, BB, ATR) | OHLCV + indicators for all symbols |
| **News Analyst** | Market sentiment from RSS feeds | Reuters, CNBC, MarketWatch, AP, BBC, NPR, Fed |
| **Macro Analyst** | Regime assessment & equity outlook | VIX, Treasury yields, Fed funds rate (FRED) |
| **Earnings Analyst** | SEC 10-Q/10-K filing analysis | EDGAR filings, cached per filing |
| **Portfolio Manager** | Synthesize all analyses into BUY/SELL/HOLD decisions | All agent outputs + account state |
| **Risk Manager** | LLM-based trade review with veto power | Proposed trades + violations + macro context |
| **Midday Reviewer** | Intraday position health assessment | Current positions + macro |
| **Evening Analyst** | Daily P&L review & next-day outlook | Positions + trades + macro |

### Risk Management

Two layers of risk control:

**Hard circuit breakers** (non-negotiable, block BUY orders):
- Single position size: max 20% of portfolio
- Total portfolio exposure: max 90%
- Daily loss limit: max 3% (triggers emergency sell-all at midday)
- Sector concentration: max 40%
- Stop loss required on all new positions

**LLM Risk Manager** (reviews remaining trades):
- Can approve, reject, or modify trades (adjust allocation, stop loss, etc.)
- Receives hard rule violations as advisory context

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

1. Copy and edit environment variables:

```bash
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
FRED_API_KEY=...
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
EOF
```

2. Edit `config/settings.yaml` for:
   - Model selection per agent (supports both OpenAI and Anthropic)
   - Risk parameters
   - Trading universe (77 symbols by default: indices, sectors, individual stocks, inverse ETFs)
   - Schedule times

## Usage

```bash
# Source env and run
source .env

# Single morning run (analyze + trade)
python main.py --mode morning

# Single midday check (position review + risk check)
python main.py --mode midday

# Single evening report (P&L + outlook)
python main.py --mode evening

# Live scheduler (runs all three on cron: 06:00, 12:00, 16:30 weekdays)
python main.py --mode live

# Or use the wrapper script (logs to logs/ directory)
./run.sh morning
```

## Trading Universe

77 symbols across 10 sectors:

- **Index ETFs**: SPY, QQQ, IWM, DIA
- **Sector ETFs**: XLF, XLE, XLV, XLI, XLP, XLY, XLU, XLRE, XLB, SMH, DRAM
- **Inverse ETFs**: SH, SDS, PSQ, SQQQ
- **Individual stocks**: Tech, Finance, Healthcare, Energy, Consumer, Industrial, etc.

## Project Structure

```
quant-agent/
├── main.py                    # CLI entry point
├── config/
│   ├── settings.yaml          # Configuration (models, risk, universe)
│   └── prompts/               # System prompts for each agent
├── src/
│   ├── pipeline.py            # Main orchestrator (morning/midday/evening)
│   ├── scheduler.py           # APScheduler cron jobs
│   ├── config.py              # Pydantic config models
│   ├── models.py              # Data models
│   ├── agents/                # LLM agents (8 specialized agents)
│   ├── data/                  # Data providers (yfinance, FRED, RSS, EDGAR)
│   ├── execution/             # Alpaca broker wrapper
│   ├── risk/                  # Hard risk rule engine
│   └── storage/               # SQLite audit trail
├── tests/                     # 74 tests
├── scripts/                   # Utilities
└── data/                      # SQLite DB + cached earnings analyses
```

## Tests

```bash
pytest tests/ -v
```

## Data Sources

| Source | Data | Provider |
|--------|------|----------|
| Market data | OHLCV, sector performance | yfinance |
| Macro | VIX, Treasury yields, Fed funds rate | FRED API |
| News | Real-time headlines | 9 RSS feeds |
| Earnings | 10-Q/10-K filings | SEC EDGAR |
| Trading | Orders, positions, account | Alpaca API |

## Storage

SQLite database (`data/quant_agent.db`) stores:
- Trade history with reasoning
- Position snapshots
- Agent logs (input, output, tokens, model)
- Daily P&L records
