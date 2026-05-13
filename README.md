# quant-agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/yebof/quant-agent/actions/workflows/test.yml/badge.svg)](https://github.com/yebof/quant-agent/actions/workflows/test.yml)
[![Last Commit](https://img.shields.io/github/last-commit/yebof/quant-agent?display_timestamp=committer&label=last%20commit)](https://github.com/yebof/quant-agent/commits/main)

LLM multi-agent quantitative trading system for US equities. Eight specialized daily agents — covering technical analysis, macroeconomic regimes, real-time news intelligence, SEC 10-Q/10-K filings, portfolio management, risk review, position management, and post-market reflection — coordinate through **schema-enforced reasoning chains**: every chain-of-thought step is a Pydantic `min_length=1` mandatory field, so the LLM cannot skip steps or fake the audit trail. A separate **quarterly Meta Reflector** reviews 90 days of accumulated outcomes (themes caught vs missed, loss patterns by attributable agent, signal activity, agent hit rates) and proposes append-only edits to six of the eight agents' prompts under a 10-invariant safety system. Decisions execute via Alpaca with multi-layer risk controls — deterministic Python filters (cash-only, daily-loss circuit breaker, sector caps, correlation cluster) gate every order, and an LLM Risk Manager audits the Portfolio Manager's plan with veto power and per-symbol modifications before it reaches the broker.

> ⚠️ **Disclaimer**: This software is provided **for educational and research purposes only**. It is NOT investment advice. Trading securities involves substantial risk of loss; you can lose more than your initial deposit. Past performance — including any backtest, simulation, paper-trading result, or live result observed in this repository — does not guarantee future performance. The authors and contributors make no representation that any strategy, signal, or system implemented here will achieve any particular result, and no representation that any code path is correct, fit for purpose, or free from defects.
>
> By using this software, **you accept full responsibility for any trades it places**, including against real-money accounts. Default configuration points at Alpaca paper trading; switching to a live account is your decision, and the consequences are yours alone. The MIT license below disclaims all warranties — read it. Do not run this against capital you cannot afford to lose.

## Highlights

- **Multi-agent coordination, not just orchestration.** Each agent has a single specialty and a strict output contract. The Portfolio Manager synthesizes upstream signals through an **8-layer memory stack** (today's quantitative facts, 7-day portfolio narrative, 14-day active HIGH-conviction state changes, 45-day trade calibration bucketed by size, last-5 Risk-Manager verdicts for self-calibration, own recent decisions for flip-flop detection, and a projected book preview that flags sector concentration before any orders are placed) and emits Pydantic-validated `TargetPosition` intent. A deterministic `PortfolioConstructor` then translates intent + ATR-based stops + live broker prices into actual orders — the LLM never guesses entry prices, stop levels, or share counts.

- **Reflection that closes the loop.** Every evening grades that day's trades along two axes (`correct / premature / wrong` × `thesis_trajectory_at_sell` ∈ `strengthening / intact / weakening / broken`) and grades the previous day's outlook against the actual next-day return — **deterministically** computed by pairing predicted bias with realized P&L, not LLM self-grading. Bias hit rates over rolling 10 sessions feed forward into the next evening's prompt, so the system gets quieter when it has been wrong. Concrete loop documented in the commit history: when evening flagged `memory-pricing` and `rare-earth` as recurring missed themes, the next morning's Portfolio Manager bought MU and MP from those exact themes, and that night's grader marked both entries `correct`.

- **Quarterly self-evolution under 10-invariant safety.** A Meta Reflector runs at quarter-end with a 7-step CoT (`facts → multi-axis self-portrait → gap diagnosis → existing-prompt audit → proposal`) and proposes append-only learnings for six of the eight agents' prompt files. The Risk Manager and Position Reviewer are **schema-protected** — the `MetaReflectionAgentName` Literal in `models.py` doesn't include them, so even with `evolution.enabled=true` the reflector cannot touch the agents that encode hard discipline. Edits ship under per-agent FIFO cap (oldest learning auto-evicts), Jaccard token-similarity dedup against existing entries, prohibited-word regex (`never / always / override / ignore all` directly conflict with hard-invariant wording), atomic file writes, an audit log of accepted and rejected attempts with reasons, and an optional git auto-commit so `git revert <sha>` is your one-shot rollback for an entire quarter of evolution.

- **Production-grade broker discipline.** Full SELL protection lifecycle: cancel existing protective stops, submit limit order, validate broker acceptance via explicit status check, wait for terminal status, then reprotect on the **actual filled quantity** — not the submitted one (a partial-fill failure mode that took several iterations to fully pin down). Orphaned protection-restore intents persist to a SQLite table and drain at every subsequent session entry, so a crash mid-flight cannot leave a position naked overnight. Daily P&L and reflection insights write atomically in a single transaction, so a crash between the two never leaves the next morning's PM reading inconsistent state.

- **Risk/Reward computed in Python, not trusted to the LLM.** `TechAnalysisResult.risk_reward` is a `@computed_field` derived from entry / stop / target geometry; the Portfolio Manager scales sizing by it (R/R ≥ 3 boosts allocation, R/R < 1.5 requires an explicitly named catalyst or gets halved/skipped); the Risk Manager enforces the 1.5 minimum independently. Neither side can fudge the math against the other — and neither is the source of truth.

- **Same-day trim discipline.** A symbol that already received a sell-side action earlier today (auto-take-profit / morning emergency / midday REDUCE / force-delever) is off-limits for additional REDUCE/SELL on a second session unless the LLM cites a hard trigger in its `reason` (`thesis_invalid_if`, HIGH bearish state-change reversal, bearish earnings filing, daily-loss circuit breaker, correlation cluster breach, stop hit). Soft signals like TARGET_BREACH alone, slowing pace, geopolitical noise, or valuation stretch do **not** qualify — they were already priced into the earlier trim, and re-applying them is the mechanical loop that produced one 73 % single-day cut on a still-strengthening name before this rule landed. Enforced at both prompt + executor layers.

- **Timezone-resilient by construction.** A single `src/trading_calendar.py` module is the source of truth for ET session windows, fill timestamps, and date keys. The OS-level scheduler (systemd `quant-agent@%i.timer` on Linux, launchd plist on macOS) fires at correct US-market times regardless of the operator's host timezone — the wrapper checks ET wall clock at fire time and skips if outside the window, so the system runs correctly when the operator is in SGT, GMT, or PT. A bash test pins the wrapper's window table against the Python authoritative source.

- **Telegram session-status push (opt-in).** Every session emits a structured status message — orders, R/R-weighted sizing, degraded-data flags, daily P&L, tomorrow's bias, or the exact exception trace on failure — to a Telegram chat you control. Set `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` to enable; missing creds make the notifier a silent no-op and trading is unchanged. Per-mode noise policy hides the 14 silent `intra_check` ticks per day and pre-market `nothing_new` earnings polls while always surfacing emergency liquidations, hard-risk blocks, exceptions, and the substantive morning / midday / close / evening completions. The notifier is wired into `main.py`'s `finally` block so even a `SystemExit` from a wrapper kill still produces a push before the process exits — and HTTP failures to Telegram are swallowed so an outage on their side can never cascade into a trading failure.

- **Tested.** 874 tests pin every invariant, including regression tests for every fix in the public commit history. Per-entry isolation (one bad LLM sub-item must not drop the whole report) is now standard across all 9 agents — a discipline that surfaced after a single malformed `MissedOpportunity` entry took down a complete evening report; adding a 10th agent would inherit the same pattern.

## Architecture

Six sessions per trading day (ET, Mon-Fri), driven by a 30-min OS-level
timer (systemd on Linux / launchd on macOS) that wraps each session
with `scripts/run_if_et_window.sh` for ET-window gating and a cross-mode
session lock. Each session has its own cadence + scope:

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
             ├─ Auto take-profit (≥30% gain → trim 15%; give-back guardrail only)
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

15:30-16:00  close                (once/day, sell-only, 30-min window)
             └─ Same Position Reviewer, session_type="close" = act-on-trigger.
                17.5h until next intraday control — if a thesis trigger is
                firing, act NOW. But "near close" is never itself a trigger.
                Good stocks are meant to be held through the night.
                Window width ≥ the OS timer's 30-min tick interval so any
                phase of the tick lands inside it (PR #41 lesson — earlier
                25-min window let bad phases miss the close two days running).

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
| **Position Reviewer** | Profit management & trailing-stop execution | Trailing-stop logic is **real**, not cosmetic — `TRAIL_STOP` action actually cancels the broker's old stop and submits a new one at the specified price via `AlpacaBroker.replace_stop_loss`. Sees VIX + HY OAS + core CPI to gauge whether to tighten stops broadly. Output is Pydantic `PositionReview` — action enum enforced (typos like `TRIAL_STOP` rejected); `TRAIL_STOP` requires `new_stop_price > 0`. **Same-day trim discipline**: a symbol that already received a sell-side action earlier today (auto-take-profit / morning emergency / midday REDUCE / force-delever) is off-limits for additional REDUCE/SELL on a second session unless the LLM's `reason` cites a hard trigger (`thesis_invalid_if`, HIGH bearish state-change, bearish earnings, daily-loss / circuit breaker, correlation cluster breach, stop hit). Soft signals (TARGET_BREACH alone, slowing pace, geopolitical noise, valuation stretch) do NOT qualify — they were already priced into the earlier trim, so re-applying them is the mechanical loop that produced one 73 % single-day cut on a still-strengthening name. The Python executor enforces this independently of the prompt; TRAIL_STOP and HOLD are unaffected. |
| **Evening Analyst** | Daily P&L review & multi-layer learning | Pydantic `EveningReport` with mandatory **7-step** `EveningReasoningChain`. **Single-day outlook retrospection** — grades yesterday's `tomorrow_outlook` against today's actual. **Multi-day calibration meta-loop** — its own tomorrow_bias / tomorrow_conviction hit rate over ~10 sessions (deterministic mirror); can't self-delude. **Structured trade grading** — `sell_grades` / `buy_grades` with `correct/premature/wrong` per trade; `buy_grades` also carries `thesis_trajectory` + `loss_root_cause` so "bought expensive vs fundamentals broke" is distinguishable. **Thesis Health Review (value-investor lens)** — per held position surfaces 8-week tech rating trajectory, 8-week news events, valuation bucket (cheap / fair / stretched), AND the full 5-step fundamentals reasoning_chain from the latest 10-Q/10-K (loaded via `src/data/earnings_deep_dive.py` from `data/earnings/{SYMBOL}/analysis_*.md`, truncated to 500c for primary / 300c for risk+mgmt steps). Outputs `thesis_trajectory` ∈ {strengthening/intact/weakening/broken}. **Missed Opportunities** — universe + Alpaca top-movers scan with quality filters (liquidity, volume confirm, valuation) surfacing names we should have held (learn, don't chase). **Quarterly meta-reflector** — separate agent runs at quarter boundaries with a **7-step facts→portrait→gap→prompt-audit→proposal** CoT. The digest now also carries `agent_prompts_snapshot` (compressed persona + rules + memory + `## Learnings (system-evolved)` section for all 6 editable agents), so step 6 `existing_prompt_audit` grounds proposed edits in what's already in each target prompt rather than rediscovering rules from memory. Auto-evolves prompts under 10 invariants enforced by `src/evolution/prompt_editor.py` (allow-list, append-only, FIFO cap, per-cycle agent cap, Jaccard dedup, prohibited-word regex, length bounds, atomic file write, audit log, optional git commit + revert rollback). Plus 7-day portfolio narrative + 14-day HIGH state changes (same layers PM sees). Outputs feed next morning's PM. |

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
- **Same-day trim discipline**: once a symbol has been trimmed/sold today (auto-TP / midday / morning emergency / force-delever), the executor blocks additional REDUCE or SELL on it for the remaining sessions of the day unless the LLM cites a hard trigger keyword in its `reason` — see Position Reviewer row above for the full list. Prevents the mechanical loop where a soft flag (TARGET_BREACH, slowing pace, valuation stretch) keeps re-firing across midday + close on the same name

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
- [Anthropic](https://console.anthropic.com/) API key (default since 2026-05-11 — all 9 agents on `claude-opus-4-7`) and/or [OpenAI](https://platform.openai.com/) API key (model-name routing: any agent set to `gpt-*` / `o1-*` / `o3-*` / `o4-*` in `config/settings.yaml` uses OpenAI; everything else uses Anthropic)

### Install

```bash
git clone <repo-url> && cd quant-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure

1. Create `.env` (set `chmod 600` after — these are secrets):
```bash
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...                  # optional — only needed if any agent stays on a gpt-*/o*-* model
FRED_API_KEY=...
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...

# Optional: Telegram session-status push (see "Optional env vars" below)
# TELEGRAM_BOT_TOKEN=...
# TELEGRAM_CHAT_ID=...
EOF
chmod 600 .env
```

2. Edit `config/settings.yaml` — models per agent, risk parameters, trading universe, schedule. Default after 2026-05-11 is `claude-opus-4-7` for all 9 agents; flip individual agents to a `gpt-*` model name to route them through OpenAI instead.

### Optional env vars

- `QUANT_AGENT_MAX_RETRIES` (default `7`) — base agent LLM-call retry budget. Backoff is **exponential floor + full positive jitter**: each sleep is in `[2^attempt, 2*2^attempt)`. With N=7 the worst-case total window is ~140s (6 sleeps in `[1,2)+[2,4)+[4,8)+[8,16)+[16,32)+[32,64)` ≈ up to 126s, plus 7 fast-fail call latencies). Evolution: `3 → 5 (DNS hiccup, 2026-04-23) → 7+jitter (sustained 30s OpenAI outages, 2026-04-28+29)`. Jitter is the load-bearing piece — without it, every retry attempt fires at deterministic offsets and a 30s outage swallows all of them; with jitter, individual attempts spread over a wider window so at least one tends to land outside any short outage. Drop to 2-3 for fast tests, raise to 10 if your provider chronically misbehaves.

- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` — enable Telegram session-status push. If either is missing the notifier no-ops silently; trading runs unchanged. Set both to wire it up:

  1. Create a bot via [@BotFather](https://t.me/BotFather): `/newbot` → name → username → it hands you a token like `12345:ABC...`.
  2. Start a chat with your bot (send any message — `/start` is fine — so it can DM you back).
  3. Get your chat_id: `curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates" | jq '.result[0].message.chat.id'`. For a group, add the bot to the group then send a message; chat_id is negative.
  4. Add to `.env`:
     ```
     TELEGRAM_BOT_TOKEN=12345:ABC...
     TELEGRAM_CHAT_ID=987654321
     ```
  5. (Optional) Set `TELEGRAM_DISABLED=1` to mute without removing the creds.

  **Per-mode noise policy** (so the operator gets signal, not noise):
  - `morning` / `midday` / `close` / `evening`: always notify on completion (status + run_id + orders + degraded-data flag + elapsed).
  - `earnings_preprocess`: notify only when filings were analyzed; silent on `nothing_new` / `market_holiday` / transient SEC `fetch_error`.
  - `intra_check`: silent on the 14 OK ticks per trading day; notifies loudly when the circuit breaker fires (`emergency_sold` / `hard_risk_block`).
  - `meta`: silent on `not_quarter_end`; notifies on actual reflection runs.
  - Any session that raises an exception: always notifies, regardless of mode policy.

### Production deployment

Either OS-level scheduler can drive the 6 sessions; both wrap `scripts/run_if_et_window.sh` so the actual ET-window / last-run / cross-mode-lock logic is shared.

**Linux (systemd, recommended for headless server / Tailscale-reachable hosts)**

The repo ships with a template unit + timer at `~/.config/systemd/user/quant-agent@.service` / `quant-agent@.timer`. Enable all 6 instances and turn on user-lingering so they fire when the user isn't logged in:

```bash
systemctl --user enable --now \
  quant-agent@earnings_preprocess.timer \
  quant-agent@morning.timer \
  quant-agent@intra_check.timer \
  quant-agent@midday.timer \
  quant-agent@close.timer \
  quant-agent@evening.timer
loginctl enable-linger "$USER"
```

`TimeoutStartSec=1500` on the service is the systemd safety net above the wrapper's own `timeout --kill-after=30 1200` (so the wrapper kills Python before systemd kills the wrapper). The timer's `Persistent=true` catches up after reboots.

**macOS (launchd, legacy path — works but Sequoia has two extra hurdles)**

If you deploy via `scripts/install_plists.sh`, Sequoia adds:

1. **Full Disk Access for `/bin/bash`** — System Settings → Privacy & Security → Full Disk Access → `+` → ⌘+Shift+G → `/bin/bash` → enable. Without this, launchd-spawned bash can't read `.env` or the project files in `~/Documents/`. Symptom: `Operation not permitted` (errno 8) flooding `logs/launchd_*.log`.

2. **Plugged-in power on trading nights** — if the laptop hibernates (critical battery), launchd's `StartInterval` jobs don't fire and you'll lose the session. `sudo pmset -c sleep 0` keeps the Mac awake on AC.

If you only use the CLI modes (`python main.py --mode morning` etc.), neither of the above applies.

### Quarterly prompt auto-evolution

Set `evolution.enabled: true` in `config/settings.yaml` (default is `false`) to let the meta-reflector **write to prompt files** at end-of-quarter:

- Runs only when you invoke `python main.py --mode meta` (no OS timer schedules this — manual trigger on the last trading day of each quarter)
- Appends proposed Learnings bullets to the 6 editable agent prompts (tech / news / macro / earnings / portfolio_manager / evening_analyst)
- **`risk_manager` and `position_reviewer` are schema-protected** — the `MetaReflectionAgentName` literal in `src/models.py` doesn't include them, so even with `enabled: true` the reflector can't touch them
- 10 invariants enforce safety (full list in `src/evolution/prompt_editor.py` module docstring). User-visible ones: per-agent **FIFO cap** (10 Learnings, oldest auto-evicted), **Jaccard dedup** (0.6 threshold), **prohibited-words regex** (never / always / override / ignore all — these directly conflict with hard-invariant wording in core prompts), **per-cycle agent cap** (max 3 distinct agents edited per quarterly run), **atomic file writes** (tmp + os.replace), **audit log** at `data/evolution/edits.jsonl` with both accepted and rejected attempts + reasons, and `auto_commit: true` so each quarter's edits land as one `chore(prompts):` commit — `git revert <sha>` is your one-shot rollback

Keep `enabled: false` until you've eyeballed at least one quarterly `reflection.json` under `data/evolution/{period}/` and are comfortable with proposal quality.

## Usage

```bash
source .env

python main.py --mode morning    # Analyze + trade
python main.py --mode midday     # Position review + trailing stops
python main.py --mode evening    # PnL report + insights for tomorrow
python main.py --mode live       # APScheduler in-process (dev/legacy; production
                                 # uses systemd/launchd timers, not this)
```

**Automated scheduling**: the production path is a 30-min OS-level timer (systemd `quant-agent@.timer` on Linux, launchd plist on macOS) that calls `scripts/run_if_et_window.sh <mode>` for each session. The wrapper checks the current **US/Eastern** wall clock against the target window, applies the cross-mode session lock (one heavy LLM session at a time, except `intra_check` which is exempt), and skips if the mode already ran today. Runs the right session at the right ET moment regardless of the host's timezone — handy when traveling. Windows (Mon-Fri ET, authoritative Python table at `src/trading_calendar.py` `SESSION_WINDOWS`, locked to the bash wrapper by `test_trading_calendar.py`):
- `earnings_preprocess` 08:00-09:15 ET — pre-market LLM analysis of fresh 10-Q/10-K filings
- `morning` 09:30-12:00 ET — research + trading
- `intra_check` 09:30-16:00 ET — every 30min tick; stateless circuit-breaker (no LLM)
- `midday` 13:00-14:30 ET — position review + real trailing stops (patient disposition)
- `close` 15:30-16:00 ET — position review (act-on-trigger; window ≥ 30-min OS-timer tick so it never misses)
- `evening` 20:00-22:00 ET — daily P&L + insights for next morning

## Trading Universe

97 symbols (source of truth: `config/settings.yaml:trading.universe`):
- **Index ETFs**: SPY, QQQ, IWM, DIA
- **Sector ETFs**: XLF, XLE, XLV, XLI, XLP, XLY, XLU, XLRE, XLB, SMH, DRAM
- **Inverse ETFs**: SH, SDS, PSQ, SQQQ (leverage-corrected in risk engine)
- **Individual stocks**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, AVGO, JPM, CAT, plus ~80 single names across tech, energy / oil, infrastructure, consumer, healthcare, financials, and power-transition themes

## Project Structure

```
quant-agent/
├── main.py                        # CLI entry point
├── config/
│   ├── settings.yaml              # Models, risk params, universe, schedule
│   └── prompts/                   # System prompts for each agent
├── src/
│   ├── pipeline.py                # Orchestrator (morning/midday/close/evening/earnings_preprocess/intra_check/meta)
│   ├── pipeline_stages.py         # MorningResearch / Decision / Risk / Execution stage classes
│   ├── pipeline_context.py        # RunContext dataclass — explicit shared state across stages
│   ├── notifier.py                # Telegram session-status push (opt-in via env vars; per-mode noise policy)
│   ├── portfolio_constructor.py   # Deterministic Target → TradeDecision translator (risk-budget sizing)
│   ├── trading_calendar.py        # ET timezone + SESSION_WINDOWS + session_date_key (single source of truth)
│   ├── scheduler.py               # APScheduler — only used by --mode live (dev/legacy)
│                                  #   Production uses systemd timers (Linux) or launchd (macOS).
│   ├── config.py                  # Pydantic config with API key validation
│   ├── models.py                  # Data models (ReasoningChain, MacroNarrative, etc.)
│   ├── agents/                    # 8 daily LLM agents + 1 quarterly meta_reflector
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
├── tests/                         # 874 tests
├── data/
│   ├── quant_agent.db             # SQLite audit trail
│   ├── earnings/                  # Cached SEC filing analyses
│   └── news/                      # Daily reports + persistent macro narrative
└── logs/                          # Timestamped run logs
```

## Tests

```bash
pytest tests/ -v    # 874 tests
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

**SQLite** (`data/quant_agent.db`, WAL mode + `synchronous=NORMAL`, indexed for prune):
- Trades (with stop/target, reasoning, actual submitted fill price) — `idx_trades_timestamp` keeps the 5-year prune fast
- Position snapshots (synced each midday — rows for closed symbols are purged)
- Agent logs for all 8 daily LLM agents (full input/output, tokens, model — `idx_agent_logs_timestamp` keeps the 2-year prune fast for quarter-over-quarter learning)
- Daily P&L records
- Evening insights (cross-session memory)
- `pending_protection_restores` — orphaned protective-stop recovery queue, drained at every session entry and TTL-pruned after 30 days

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

## Author

**Yebo Feng** — [@yebof](https://github.com/yebof) · `fengyebo@gmail.com`

Designed and built end-to-end as a personal research project on multi-agent LLM coordination for swing/position trading. The architecture, prompt engineering across all 9 agents, schema-enforced reasoning chains, broker order-lifecycle state machine, and quarterly self-evolution loop are all original work. Implementation was vibe-coded throughout with Claude Code (Anthropic) and Codex (OpenAI) as pair-programmer agents — the system that runs the agents was itself built by agents. Issues and PRs welcome.

If this repository was useful for your own work — academic, professional, or personal — a star is appreciated and a citation/mention is even better.

## License

MIT — see [LICENSE](LICENSE).
