# quant-agent

LLM multi-agent quantitative trading system for US equities. 8 specialized AI agents analyze markets from different angles (technical, macro, news intelligence, SEC earnings), synthesize into trading decisions with explicit chain-of-thought reasoning, and execute via Alpaca with multi-layer risk controls.

## Architecture

Six sessions per trading day (ET, Mon-Fri), launchd-scheduled, with
their own cadence + scope:

```
08:00-09:15  earnings_preprocess  (once/day, pre-market)
             └─ Earnings Analyst runs LLM on newly-filed 10-Q/10-K — the
                ONLY session that calls the earnings LLM. Writes analysis
                to disk + confirms filing. Hot sessions below only READ
                this cache.

09:30-12:00  morning              (once/day, main trading)
             ├─ Cancel stale entry orders, keep protective exits
             ├─ Force-delever if cash<-$1 (cash-only default; safety net)
             │
             ├─── Parallel fan-out ───────────────────┐
             │  Macro     News Intel     Tech          │   Earnings
             │  (6-step)  (3-layer)      (5-step CoT)  │   (read cache)
             │                                         │
             ├─ Load L1-L8 memory + yesterday's evening insights
             ▼
          Portfolio Manager (7-step CoT; R/R-weighted sizing; drawdown-aware;
                             regime-adaptive cash floor; drift trim)
             │
             ├─ Symbol Guard (universe + analyst-coverage check)
             ├─ Hard Risk Engine (per-BUY, leverage-adjusted)
             │    cash_only / max_position / max_sector / max_total /
             │    max_daily_loss / require_stop_loss — blocking
             │    + advisory: macro_exposure_deviation / correlation_cluster
             ├─ Risk Manager LLM (6-step CoT; scale_all_buys; R/R veto)
             ▼
          Execute: SELLs first → refresh cash → BUYs
                   (OTO bracket with broker-enforced stop; no hard TP)

09:30-16:00  intra_check          (EVERY 30-MIN TICK, no LLM, ~5 sec)
             └─ Daily P&L vs -3% loss cap — emergency sell-all if breached.
                Stateless circuit breaker; not subject to once-per-day guard.

13:00-14:30  midday               (once/day, sell-only)
             ├─ Force-delever
             ├─ Auto take-profit (≥15% gain → trim 33%)
             ├─ Ex-dividend stop adjustment for held names
             ▼
          Position Reviewer (6-step CoT, session_type="midday" = patient)
             │  Default: HOLD unless named thesis trigger fires
             │  3-8% profit → consider trail to breakeven
             │  8-15% profit → consider trail to halfway
             │  > 15% profit → consider trail to 70% of move
             │
             └─ SELL / REDUCE / TRAIL_STOP only on trigger (thesis_invalid_if
                / HIGH state_change reversal / bearish earnings / cluster
                breach). Price alone is never a trigger.

15:30-15:55  close                (once/day, sell-only, 25-min window)
             └─ Same Position Reviewer, session_type="close" = act-on-trigger.
                17.5h until next intraday control — if a thesis trigger is
                firing, act NOW. But "near close" is never itself a trigger.
                Good stocks are meant to be held through the night.

20:00-22:00  evening              (once/day, post-market)
             ├─ Reconcile all submitted orders → terminal status
             ▼
          Evening Analyst (7-step CoT)
             │  previous_outlook_assessment — grade yesterday's call
             │  sell_grades / buy_grades — structured per-trade labels
             │  outlook_calibration meta-loop — own bias vs actual across
             │                                  ~10 sessions
             │  thesis_health_review — per-held 8w tech trajectory + news +
             │                         valuation + full 10-Q/10-K reasoning_chain
             │                         (via src/data/earnings_deep_dive) →
             │                         thesis_trajectory: strengthening / intact /
             │                         weakening / broken + loss_root_cause
             │  missed_opportunities — scan universe + Alpaca top_movers for
             │                         names we should have held (learn, don't chase)
             │  tomorrow_bias / conviction / key_risks — PM reads these at open
             │
             └─ Atomic write: daily_pnl + insights in one transaction.
```

See CLAUDE.md "不要违反的约定" for the locked-in invariants (cash-only default,
SELL allocation_pct semantics, ET-everywhere timezone, etc.).

## Agents

| Agent | Role | Key Feature |
|-------|------|-------------|
| **Tech Analyst** | Batch technical analysis | 5-step CoT (trend / momentum / volatility / volume / S&R). ATR-based default stop (`entry − 2*ATR`). Output rating + `conviction` (high/medium/low) + `reference_target` + `thesis_invalid_if` (soft exit condition). **Auto-computed `risk_reward`** (Python-calculated, not LLM-trusted) flows into PM sizing and RM veto logic. **Signal-age memory** (`data/tech/last_ratings.json`): prior rating surfaced to LLM as context; `signal_age_days` counted to spot stale setups — PM cuts allocation on 8+ day stale BUYs. **Valuation context** (yfinance trailing PE / forward PE / P/S) surfaced per symbol — LLM flags >40x forward PE or >15x P/S as stretched in `reasoning_chain.support_resistance`. Pre-filter thresholds normalized by ATR. Auto-chunks batch > 30 symbols. Cross-field validator: BUY stop must be below entry, SELL above. |
| **News Intelligence** | 3-layer news analysis | Layer 1: Persistent macro narrative. Layer 2: State change detection. Layer 3: Per-symbol alerts with conviction. Daily storage in `data/news/` |
| **Macro Analyst** | Regime assessment & sector guidance | 6-step CoT (vol / curve / monetary / inflation+labor+credit / cross-signal / sector). Inputs: VIX, 2Y/10Y yields, **DFF** (daily fed funds), **core & headline CPI**, **UNRATE**, **HY OAS**. Persists yesterday's regime → detects `regime_shift`. Cross-references News narrative via `alignment_with_news`. Emits bull/bear view-change triggers. |
| **Earnings Analyst** | SEC 10-Q/10-K analysis | Revenue, margins, cash flow, strategic direction, competitive positioning, strategic vs operational risks, strategy consistency across filings. `investment_implications` carries a 5-step `reasoning_chain` (fundamental_quality / growth_trajectory / strategic_risks / management_execution / valuation_context) — sentiment call is derivable from the numbers, not a vibe check. |
| **Portfolio Manager** | Central decision maker | Mandatory 7-step reasoning chain + continuity check across **8 memory layers**: L1 today's signals, L2 per-position entry context + Tech rating 7-day trajectory with `Weight:%` and `⚠️DRIFT` flag on concentrated winners, L3a rolling Portfolio Narrative (7 evenings), L3b Macro Regime Trajectory (7 days), L3c Active HIGH-conviction state_changes (14 days), **L4 Trade Calibration** — actual realized win rate + avg return on closed BUYs (45d), bucketed by size, **L5 RM Verdicts** (last 5 sessions — PM shrinks sizing when RM keeps scaling it down), **L6 Own Recent Decisions** (last 3 sessions — spot flip-flops), **L7 Projected Book Preview** — if you rubber-stamp all TA BUYs @ 5%, flags sectors nearing 35% cap. Sizing scales by TechAnalyst's `risk_reward` (R/R ≥ 3 boost, < 1.5 requires catalyst). **Regime-adaptive cash floor**: risk-off 25% / transitional 15% / risk-on 5%. **Drawdown-aware**: halves new BUYs when `in_drawdown` flagged. **Drift trim**: Weight > 12% + P&L > 10% → must trim or justify; Weight > 18% → hard trim. **Earnings-queued hard cap**: just-filed 10-Q with no analysis yet → BUY capped at 5% (enforced in pipeline). **Holding discipline** (tiered by days_held): <5d → default HOLD unless thesis_invalid_if or macro regime flipped today; 5-15d → standard; >15d profitable + trend intact → let it run. 11-row **Rule Priority** cheat sheet resolves conflicts (thesis_invalid > holding > earnings-cap > drift > cash-floor > R/R > ...). |
| **Risk Manager** | Trade review with veto power | Mandatory 6-step `reasoning_chain` (rr_audit / signal_fidelity / correlation_check / event_risk / sizing_sanity / overall) — vague approvals rejected. Enforces R/R discipline: BUYs with R/R < 1.5 must be downsized via modifications or rejected unless PM named a catalyst. Sees raw Tech ratings + R/R + full macro context. Can modify per-symbol fields OR apply portfolio-wide `scale_all_buys` (0.0-1.0). |
| **Position Reviewer** | Profit management & trailing-stop execution | Trailing-stop logic is **real**, not cosmetic — `TRAIL_STOP` action actually cancels the broker's old stop and submits a new one at the specified price via `AlpacaBroker.replace_stop_loss`. Sees VIX + HY OAS + core CPI to gauge whether to tighten stops broadly. Output is Pydantic `PositionReview` — action enum enforced (typos like `TRIAL_STOP` rejected); `TRAIL_STOP` requires `new_stop_price > 0`. |
| **Evening Analyst** | Daily P&L review & multi-layer learning | Pydantic `EveningReport` with mandatory **7-step** `EveningReasoningChain`. **Single-day outlook retrospection** — grades yesterday's `tomorrow_outlook` against today's actual. **Multi-day calibration meta-loop** — its own tomorrow_bias / tomorrow_conviction hit rate over ~10 sessions (deterministic mirror); can't self-delude. **Structured trade grading** — `sell_grades` / `buy_grades` with `correct/premature/wrong` per trade; `buy_grades` also carries `thesis_trajectory` + `loss_root_cause` so "bought expensive vs fundamentals broke" is distinguishable. **Thesis Health Review (value-investor lens)** — per held position surfaces 8-week tech rating trajectory, 8-week news events, valuation bucket (cheap / fair / stretched), AND the full 5-step fundamentals reasoning_chain from the latest 10-Q/10-K (loaded via `src/data/earnings_deep_dive.py` from `data/earnings/{SYMBOL}/analysis_*.md`, truncated to 500c for primary / 300c for risk+mgmt steps). Outputs `thesis_trajectory` ∈ {strengthening/intact/weakening/broken}. **Missed Opportunities** — universe + Alpaca top-movers scan with quality filters (liquidity, volume confirm, valuation) surfacing names we should have held (learn, don't chase). **Quarterly meta-reflector** — separate agent runs at quarter boundaries with a **7-step facts→portrait→gap→prompt-audit→proposal** CoT. The digest now also carries `agent_prompts_snapshot` (compressed persona + rules + memory + `## Learnings (system-evolved)` section for all 6 editable agents), so step 6 `existing_prompt_audit` grounds proposed edits in what's already in each target prompt rather than rediscovering rules from memory. Auto-evolves prompts under 4 guards (FIFO / Jaccard dedup / prohibited-word regex / git commit rollback). Plus 7-day portfolio narrative + 14-day HIGH state changes (same layers PM sees). Outputs feed next morning's PM. |

## Risk Management

### Hard Risk Engine (non-negotiable, per-BUY)
- Single position: max 20% (gross exposure — SQQQ 3x, SDS 2x counted at full magnitude)
- Total **net** exposure: max 90% (hedges cancel — e.g. long SPY + short SH ≈ zero net)
- Daily loss: max 3% of prior-close equity (`equity − last_equity`; includes realized fills from broker-triggered OTO stops, not just marks)
- Sector concentration: max 40% (gross, cumulative including pending same-sector buys)
- Stop loss required
- **Cash-only by default** (`risk.allow_margin: false` in settings.yaml) — BUYs cannot drive cash negative; filter pre-sums same-session SELL proceeds so legitimate rotations pass. When cash < 0, PM + midday prompts get a mandatory **DE-LEVER** directive. Flip to `true` to allow margin.
- Inverse ETFs (SH, SDS, PSQ, SQQQ) carry signed multipliers for net exposure and gross magnitude for sizing/sector caps
- **Advisory**: if projected net exposure deviates > 15pp from Macro's `target_invested_pct`, emits a non-blocking `macro_exposure_deviation` violation — RiskManager sees it and can respond with `scale_all_buys`
- **Correlation cluster** (advisory): a proposed BUY plus already-held positions correlated > 0.7 with it (120-day daily returns) must not together exceed 50% of book. Catches AI / mega-cap-growth concentration that sector caps miss when yfinance tags NVDA (Technology) and GOOGL (Communication Services) separately.
- **Correlation coverage gap** (advisory): when yfinance data is too sparse to build the matrix but the book has ≥ 2 positions, RM sees a `correlation_coverage_gap` advisory and can respond with `scale_all_buys < 1.0`. Prevents the cluster check from silently disabling when upstream data degrades.

### Execution Safety
- Stale orders cancelled before each session
- SELLs execute before BUYs (free cash first)
- SELL uses limit price (0.5% below market for slippage protection); midday emergency sells use a wider 1% buffer to ensure fill during cascades
- BUY limit price auto-raised to market if below (prevents unfilled orders)
- BUY attaches OTO stop-loss via Alpaca (broker-enforced)
- No hard take-profit — profit managed by midday reviewer's trailing stop logic
- Partial sell via `allocation_pct` (1–99 = partial, 100 = full exit; 0 is treated as a no-op)
- **Order-status gating**: every broker submission runs through `_order_accepted()` before the audit log is written — Alpaca error payloads (missing id, status rejected/expired/canceled) are refused so the trades table never records a phantom fill
- **No-price BUY skip**: if neither the broker nor in-memory OHLCV bars can sanity-check the LLM's `entry_price`, the BUY is skipped rather than submitted as a stale limit
- **Earnings-queued BUY cap**: pipeline hard-clamps any BUY on a symbol whose latest 10-Q/10-K is `queued` (fresh filing, LLM analysis still running) to ≤ 5% allocation, regardless of PM's conviction

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

**Timezone-resilient scheduling**: plists fire every 30 minutes (`StartInterval=1800`); a bash wrapper `scripts/run_if_et_window.sh` checks whether the current **US/Eastern** time is inside the target window and whether the mode already ran in the last hour. Runs the right session at the right ET moment regardless of the host's timezone (handy when traveling). Windows (Mon-Fri ET, authoritative Python table at `src/trading_calendar.py` `SESSION_WINDOWS`, locked to the bash wrapper by `test_trading_calendar.py`):
- `earnings_preprocess` 08:00-09:15 ET — pre-market LLM analysis of fresh 10-Q/10-K filings
- `morning` 09:30-12:00 ET — research + trading
- `intra_check` 12:00-13:30 ET — lightweight circuit-breaker (no LLM)
- `midday` 15:00-16:30 ET — position review + real trailing stops
- `evening` 20:00-22:00 ET — daily P&L + insights for next morning

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
│   ├── pipeline.py                # Orchestrator (morning/midday/evening/earnings_preprocess/intra_check)
│   ├── pipeline_stages.py         # MorningResearch / Decision / Risk / Execution stage classes
│   ├── pipeline_context.py        # RunContext dataclass — explicit shared state across stages
│   ├── portfolio_constructor.py   # Deterministic Target → TradeDecision translator (risk-budget sizing)
│   ├── trading_calendar.py        # ET timezone + SESSION_WINDOWS + session_date_key (single source of truth)
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
├── tests/                         # 385 tests
├── data/
│   ├── quant_agent.db             # SQLite audit trail
│   ├── earnings/                  # Cached SEC filing analyses
│   └── news/                      # Daily reports + persistent macro narrative
└── logs/                          # Timestamped run logs
```

## Tests

```bash
pytest tests/ -v    # 385 tests
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
- `{SYMBOL}/analysis_{10-Q}_{date}.md` — cached SEC filing analyses, written by `run_earnings_preprocess` (the only LLM-producing path — see **Hot / Cold Earnings Path** below). Each file carries a human-readable header + one embedded ```` ```json ```` block with the full `EarningsAnalysis` schema, including the 5-step `investment_implications.reasoning_chain`
- `manifest.json` — tracks processed filings; `failed_attempts` counter bounds preprocess retries (abandon + mark `abandoned=True` after 3 failures so a consistently-unparseable 10-Q doesn't burn tokens every run forever)

### Hot / Cold Earnings Path

LLM analysis of 10-Q/10-K filings runs **only** in the pre-market `earnings_preprocess` window (08:00-09:15 ET). Hot sessions (morning / midday / evening) are read-only consumers: `_load_earnings_analyses` surfaces the cached analysis for already-confirmed filings, and any filing that preprocess missed appears as a `queued=True` placeholder — PM then caps the BUY at 5% regardless of conviction. No background threads in hot sessions, no session-time LLM token spend on earnings.

### Evening earnings deep-dive (value-investor lens)

Morning/midday PM sees a one-line earnings sentiment snippet (~140 chars — `"bullish high — revenue +16%"`); that's enough to gate BUY sizing. **Evening** needs more: when judging `thesis_trajectory` on a held position, "bought expensive" vs "fundamentals actually broke" is the core question. So the evening path calls `src/data/earnings_deep_dive.load_earnings_deep_dive(symbol, manifest)` per held name — it parses the JSON block out of `analysis_*.md`, pulls the full 5-step reasoning_chain (fundamental_quality / growth_trajectory / valuation_context / strategic_risks / management_execution), truncates each step (500c primary, 300c risk+mgmt) to bound token cost, and injects the structured dict into `thesis_health_context[symbol]["earnings_deep_dive"]`. The evening prompt renders a `--- Earnings deep-dive (10-Q 2026-01-30, bullish/high) ---` block so the LLM can reason on actual fundamentals, not a compressed tag. Missed-opportunity symbols still use the 140-char snippet — token budget reason; deep-dive only for the ~10-15 names we actually hold.
