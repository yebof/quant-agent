"""Position reviewer — the sell-only agent that runs at midday (13:00 ET) and
close (15:30 ET).

Formerly MiddayReviewerAgent. Renamed because it runs twice per day and both
sessions share the same logic + memory, differing only in `session_type`
disposition (midday is patient, close is act-on-trigger because there's no
overnight control).

v3 upgrades vs the original MiddayReviewerAgent:
  - Schema: mandatory 6-step `PositionReasoningChain` (parallel depth to
    morning PM's 7-step chain). Empty strings fail Pydantic validation.
  - Memory layers: macro regime + 7-day macro trajectory + 7-day portfolio
    narrative + 14-day active HIGH state changes + 45-day trade calibration
    + PM's recent decisions + yesterday's evening insights + recent system
    performance + full earnings analyses (not just queued list).
  - Deterministic pre-compute: per-held-position `thesis_progress_pct` /
    `pace` / `distance_to_stop_pct` / `distance_to_target_pct` + three
    winner flags (parabolic / drift / target_breach). Math done in Python
    and surfaced as numbers, not asked of the LLM.
  - Session-type awareness: 'midday' and 'close' prompts differ in bias.
  - Anti-flip-flop: "Own Recent Decisions" memory shown so the agent
    doesn't reverse itself within hours without a named thesis break.
"""

import logging
from pathlib import Path

from pydantic import ValidationError

from src.agents.base import BaseAgent
from src.models import NewsIntelligenceReport, Position, PositionReview

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "position_reviewer.md"

_SESSION_LABEL = {
    "midday": "Midday (13:00 ET) — afternoon still open",
    "close": "Close (15:30 ET) — ~25 min to close, 17.5h until next intraday control",
}

_SESSION_DISPOSITION = {
    "midday": (
        "You have 2.5+ hours of trading left. Default disposition is PATIENT. "
        "Prefer TRAIL_STOP over SELL when the situation is 'drifting, not breaking'. "
        "Let afternoon resolve ambiguous signals — it usually will. "
        "Only SELL on a specific thesis trigger that won't improve by waiting."
    ),
    "close": (
        "You won't have hands on the wheel for 17.5 hours after close. That does NOT "
        "mean sell more — 'it's close time' is never a trigger. It means: if a thesis "
        "trigger is CLEARLY firing (thesis_invalid_if condition satisfied, HIGH "
        "conviction state_change reversing the thesis, momentum demonstrably dead), "
        "act NOW rather than hoping tomorrow morning catches it. If no trigger is "
        "firing, HOLD — good stocks are meant to be held over weekends and "
        "overnights alike."
    ),
}


class PositionReviewerAgent(BaseAgent):
    """Sell-only agent: reviews open positions, outputs HOLD / TRAIL_STOP /
    REDUCE / SELL. Never BUYs. Used at midday and close sessions."""

    @staticmethod
    def _trade_executed(trade: dict) -> bool:
        """Belt-and-braces guard for BUY rows surfaced to the LLM prompt.

        In practice the caller has already filtered via db.get_trades(
        executed_only=True), so by the time a row reaches here it's either
        filled, legacy NULL-status, or canceled-with-partial-fill. This
        predicate re-applies the same shape check so a future caller that
        forgets executed_only can't accidentally inject a submitted-only or
        zero-fill canceled BUY into the reviewer's trade-context map.
        """
        status = trade.get("fill_status")
        if status is None:
            return True
        if str(status).lower() == "filled":
            return True
        try:
            return float(trade.get("fill_qty") or 0) > 0
        except (TypeError, ValueError):
            return False

    @property
    def name(self) -> str:
        return "position_reviewer"

    @property
    def system_prompt(self) -> str:
        if PROMPT_PATH.exists():
            return PROMPT_PATH.read_text()
        return "You are a position reviewer. Respond with JSON."

    def build_user_message(self, **kwargs) -> str:
        positions: list[Position] = kwargs["positions"]
        macro_summary: dict = kwargs["macro_summary"]
        cash_balance: float = kwargs["cash_balance"]
        total_value: float = kwargs["total_value"]
        session_type: str = kwargs.get("session_type", "midday")
        position_facts: dict = kwargs.get("position_facts") or {}  # symbol → deterministic-metrics dict
        morning_trades: list[dict] = kwargs.get("morning_trades", [])
        news_intel: NewsIntelligenceReport | None = kwargs.get("news_intel")
        earnings_analyses: list[dict] = kwargs.get("earnings_analyses") or []
        macro_analysis: dict | None = kwargs.get("macro_analysis")
        weekly_narrative: str = kwargs.get("weekly_narrative") or ""
        macro_trajectory: str = kwargs.get("macro_trajectory") or ""
        active_state_changes: str = kwargs.get("active_state_changes") or ""
        calibration_note: str = kwargs.get("calibration_note") or ""
        own_recent_decisions: str = kwargs.get("own_recent_decisions") or ""
        yesterday_insights: dict | None = kwargs.get("yesterday_insights")
        recent_performance: dict = kwargs.get("recent_performance") or {}

        # Build morning trade context map (entry thesis, stop_loss, reference target).
        trade_context: dict[str, dict] = {}
        for t in morning_trades:
            sym = t.get("symbol", "")
            if (
                t.get("action") == "BUY"
                and sym
                and sym not in trade_context
                and self._trade_executed(t)
            ):
                trade_context[sym] = t

        # Positions block — surfaces deterministic metrics alongside the raw numbers.
        def _pnl_pct(p: Position) -> str:
            cost = p.avg_entry * p.qty
            return f"{p.unrealized_pnl / cost * 100:.1f}%" if cost else "N/A"

        positions_lines = []
        for p in positions:
            pnl_pct = _pnl_pct(p)
            header = (
                f"- **{p.symbol}** ({p.sector}): {p.qty} sh @ ${p.avg_entry:.2f} | "
                f"Now ${p.current_price:.2f} | P&L ${p.unrealized_pnl:.2f} ({pnl_pct})"
            )
            lines = [header]

            ctx = trade_context.get(p.symbol) or {}
            sl = ctx.get("stop_loss") or 0
            tp = ctx.get("take_profit") or 0
            if sl:
                lines.append(f"  Hard stop (broker): ${sl:.2f}")
            if tp:
                lines.append(
                    f"  Reference target: ${tp:.2f} (soft — you manage exit)"
                )
            entry_reasoning = (ctx.get("reasoning") or "").strip()
            if entry_reasoning:
                lines.append(f"  Entry thesis: {entry_reasoning[:220]}")

            # Deterministic per-position metrics (computed by pipeline).
            pf = position_facts.get(p.symbol) or {}
            metric_bits: list[str] = []
            if pf.get("days_held") is not None:
                metric_bits.append(f"days_held={pf['days_held']}")
            if pf.get("thesis_progress_pct") is not None:
                metric_bits.append(f"thesis_progress={pf['thesis_progress_pct']:.0f}%")
            if pf.get("pace") is not None:
                metric_bits.append(f"pace={pf['pace']:.2f}×")
            if pf.get("distance_to_stop_pct") is not None:
                metric_bits.append(f"to_stop={pf['distance_to_stop_pct']:.1f}%")
            if pf.get("distance_to_target_pct") is not None:
                metric_bits.append(f"to_target={pf['distance_to_target_pct']:.1f}%")
            if pf.get("weight_pct") is not None:
                metric_bits.append(f"weight={pf['weight_pct']:.1f}%")
            if metric_bits:
                lines.append(f"  Metrics: {' | '.join(metric_bits)}")

            flag_bits: list[str] = []
            if pf.get("parabolic_flag"):
                flag_bits.append("⚠️ PARABOLIC (+15% in <3d, momentum confirmation needed)")
            if pf.get("drift_flag"):
                flag_bits.append("⚠️ DRIFT (weight > 12% + PnL > 10%)")
            if pf.get("target_breach_flag"):
                flag_bits.append("⚠️ TARGET_BREACH (>150% of reference_target)")
            if flag_bits:
                lines.append(f"  Flags: {' '.join(flag_bits)}")

            positions_lines.append("\n".join(lines))
        positions_text = "\n".join(positions_lines) if positions_lines else "No open positions."

        # Macro block — full regime, not just raw numbers.
        vix = macro_summary.get("vix", {}) or {}
        hy = macro_summary.get("credit_spread", {}) or {}
        infl = macro_summary.get("inflation", {}) or {}
        if macro_analysis:
            regime = macro_analysis.get("regime", "N/A")
            outlook = macro_analysis.get("equity_outlook", "N/A")
            confidence = macro_analysis.get("confidence", "N/A")
            guidance = macro_analysis.get("position_guidance") or {}
            target_invested = guidance.get("target_invested_pct", "N/A")
            macro_regime_line = (
                f"Regime: **{regime}** | Outlook: **{outlook}** ({confidence}) | "
                f"Target invested: {target_invested}%"
            )
        else:
            macro_regime_line = "Regime: (no macro analysis this session)"

        # News block.
        if news_intel:
            conv_order = {"high": 0, "medium": 1, "low": 2}
            state_lines = []
            for c in news_intel.state_changes[:5]:
                state_lines.append(
                    f"- [{c.conviction.upper()}] {c.event}: {c.previous_state} → {c.new_state} "
                    f"(impact: {c.market_impact}; affects: {', '.join(c.affected_symbols[:5]) or 'broad'})"
                )
            state_text = "\n".join(state_lines) or "No significant session state changes."
            held_syms = {p.symbol for p in positions}
            stock_lines = []
            for sym, alerts in (news_intel.stock_news or {}).items():
                if sym not in held_syms:
                    continue
                for a in sorted(alerts, key=lambda x: conv_order.get(x.conviction, 9))[:2]:
                    stock_lines.append(
                        f"- {sym}: [{a.conviction.upper()}] {a.sentiment} — {a.impact_summary}"
                    )
            stock_text = "\n".join(stock_lines) or "No per-position news alerts."
            news_section = (
                f"### Session News Intelligence\n"
                f"PM Briefing: {news_intel.pm_briefing[:300]}\n\n"
                f"State changes this session:\n{state_text}\n\n"
                f"Held-position alerts:\n{stock_text}\n\n"
                f"Overall sentiment: {news_intel.market_sentiment} ({news_intel.confidence})\n"
            )
        else:
            news_section = "### Session News\n(no news report available)\n"

        # Earnings — full content, not just queued list.
        held_syms = {p.symbol for p in positions}
        earnings_lines: list[str] = []
        queued: list[str] = []
        for ea in earnings_analyses:
            sym = ea.get("symbol")
            if not sym or sym not in held_syms:
                continue
            if ea.get("queued"):
                queued.append(sym)
                continue
            analysis = ea.get("analysis") or {}
            impl = analysis.get("investment_implications") or {}
            sentiment = impl.get("sentiment", "?")
            conv = impl.get("conviction", "?")
            key_thesis = (impl.get("key_thesis") or "").strip()[:180]
            earnings_lines.append(
                f"- {sym} [{sentiment} / {conv}]: {key_thesis}"
            )
        earnings_parts: list[str] = []
        if earnings_lines:
            earnings_parts.append(
                "### Earnings Analyses (held positions)\n" + "\n".join(earnings_lines)
            )
        if queued:
            earnings_parts.append(
                f"### Just-filed (analysis queued — treat as elevated event risk): "
                f"{', '.join(queued)}"
            )
        earnings_section = "\n\n".join(earnings_parts) if earnings_parts else ""

        # Memory layers (reuse PM-style prose).
        def _opt_section(title: str, body: str) -> str:
            body = (body or "").strip()
            return f"### {title}\n{body}\n" if body else ""

        narrative_section = _opt_section("Portfolio Narrative (last 7 evenings)", weekly_narrative)
        trajectory_section = _opt_section("Macro Regime Trajectory (7 days)", macro_trajectory)
        active_changes_section = _opt_section(
            "Active HIGH-conviction State Changes (14 days)", active_state_changes
        )
        calibration_section = _opt_section("Trade Calibration (45-day realized)", calibration_note)
        decisions_section = _opt_section(
            "Your Recent Decisions (don't flip-flop without a named trigger)",
            own_recent_decisions,
        )

        if yesterday_insights:
            yi_outlook = (yesterday_insights.get("tomorrow_outlook") or "").strip()[:300]
            yi_bias = yesterday_insights.get("tomorrow_bias", "neutral")
            yi_conviction = yesterday_insights.get("tomorrow_conviction", "medium")
            yi_risk = yesterday_insights.get("risk_rating", "moderate")
            yesterday_section = (
                f"### Yesterday Evening's Outlook for Today\n"
                f"- Bias: {yi_bias} ({yi_conviction}) | Risk: {yi_risk}\n"
                f"- Outlook: {yi_outlook}\n"
            )
        else:
            yesterday_section = ""

        if recent_performance:
            r5 = recent_performance.get("rolling_5d_pct")
            r20 = recent_performance.get("rolling_20d_pct")
            dd = recent_performance.get("in_drawdown")
            dd_note = " ⚠️ IN DRAWDOWN — bias toward HOLDING quality, don't panic-sell the bottom" if dd else ""
            perf_section = (
                f"### Recent System Performance{dd_note}\n"
                f"- 5d: {r5}% | 20d: {r20}%\n"
            )
        else:
            perf_section = ""

        # Account + cash.
        cash_pct = f"{cash_balance / total_value * 100:.1f}%" if total_value else "N/A"

        # Margin mandate (carried over from v2 — sub-dollar threshold).
        allow_margin: bool = bool(kwargs.get("allow_margin", True))
        from src.risk.constants import MARGIN_DEFICIT_FLOOR_USD
        if not allow_margin and cash_balance < -MARGIN_DEFICIT_FLOOR_USD:
            deficit = -cash_balance
            margin_section = (
                f"### ⚠️ Cash-only policy — de-lever required\n"
                f"Cash is ${cash_balance:,.2f} (deficit ${deficit:,.2f}). "
                f"This account runs cash-only; prefer SELL or REDUCE on the "
                f"weakest-conviction position(s) to restore cash ≥ 0. Do NOT "
                f"TRAIL_STOP when the real problem is over-leverage.\n"
            )
        else:
            margin_section = ""

        session_label = _SESSION_LABEL.get(session_type, session_type)
        session_bias = _SESSION_DISPOSITION.get(session_type, "")

        return f"""## Position Review — {session_label}

{margin_section}
### Session Disposition
{session_bias}

### Account
- Total Value: ${total_value:,.2f}
- Cash: ${cash_balance:,.2f} ({cash_pct})

### Open Positions
{positions_text}

### Macro
{macro_regime_line}
- VIX: {vix.get('current', 'N/A')} (trend: {vix.get('trend', 'N/A')})
- HY OAS: {hy.get('current_bps', 'N/A')}bps (30d Δ: {hy.get('change_30d_bps', 'N/A')}bps)
- Core CPI YoY: {infl.get('core_cpi_yoy', 'N/A')}%

{trajectory_section}
{active_changes_section}
{news_section}
{earnings_section}

{narrative_section}
{yesterday_section}
{calibration_section}
{decisions_section}
{perf_section}

Review each position. Fill the 6-step `reasoning_chain` before emitting
any action. Remember: intraday price is noise; thesis is signal; good
stocks are meant to be held. Respond as JSON matching the PositionReview
schema."""

    def review(self, positions: list[Position], macro_summary: dict,
               cash_balance: float, total_value: float,
               session_type: str = "midday",
               position_facts: dict | None = None,
               morning_trades: list[dict] | None = None,
               news_intel: NewsIntelligenceReport | None = None,
               earnings_analyses: list[dict] | None = None,
               macro_analysis: dict | None = None,
               weekly_narrative: str = "",
               macro_trajectory: str = "",
               active_state_changes: str = "",
               calibration_note: str = "",
               own_recent_decisions: str = "",
               yesterday_insights: dict | None = None,
               recent_performance: dict | None = None,
               allow_margin: bool = True) -> tuple[PositionReview | None, "AgentResult"]:
        result = self.run(
            positions=positions,
            macro_summary=macro_summary,
            cash_balance=cash_balance,
            total_value=total_value,
            session_type=session_type,
            position_facts=position_facts or {},
            morning_trades=morning_trades or [],
            news_intel=news_intel,
            earnings_analyses=earnings_analyses or [],
            macro_analysis=macro_analysis,
            weekly_narrative=weekly_narrative,
            macro_trajectory=macro_trajectory,
            active_state_changes=active_state_changes,
            calibration_note=calibration_note,
            own_recent_decisions=own_recent_decisions,
            yesterday_insights=yesterday_insights,
            recent_performance=recent_performance or {},
            allow_margin=allow_margin,
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Position reviewer returned non-JSON response")
            return None, result
        if not isinstance(parsed, dict):
            logger.error("Position reviewer expected object, got %s", type(parsed).__name__)
            return None, result
        try:
            review = PositionReview(**parsed)
        except ValidationError as e:
            logger.error("Position review failed schema validation: %s", e)
            return None, result
        return review, result
