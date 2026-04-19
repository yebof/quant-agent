import json
import logging
import re
from pathlib import Path

from src.agents.base import BaseAgent, AgentResult
from src.models import NewsIntelligenceReport

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "news_analyst.md"

# Tokens too common to anchor an event on — they'd let any hallucinated event
# survive a keyword match. Deliberately conservative: we only want to exclude
# words that appear in virtually any headline.
_STATE_CHANGE_STOPWORDS = frozenset({
    "from", "into", "with", "that", "this", "these", "those",
    "have", "been", "will", "would", "could", "should",
    "change", "state", "event", "today", "more", "less",
    "than", "some", "many", "much", "also", "very",
    "after", "before", "during", "while", "about", "against", "between",
    "said", "says", "reports", "reported", "according",
})


class NewsAnalystAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "news_analyst"

    @property
    def system_prompt(self) -> str:
        if PROMPT_PATH.exists():
            return PROMPT_PATH.read_text()
        return "You are a news intelligence analyst. Respond with JSON."

    # Per-session mode descriptor that shapes the agent's task. Morning
    # does the full 3-layer build; midday focuses on DELTA vs morning
    # (what's new/changed); evening focuses on SUMMARY (what stuck vs
    # faded across the day). All three still emit the same schema so
    # downstream consumers don't care which mode produced the report.
    _SESSION_GUIDANCE = {
        "morning": (
            "MORNING mode — full 3-layer build. Treat today as a fresh book; "
            "produce the complete macro_narrative, state_changes, and stock_news "
            "sections. This report sets the tone for the day's trading."
        ),
        "midday": (
            "MIDDAY mode — DELTA focus. The morning report is shown below as "
            "'This morning's snapshot'. Your job is to surface what CHANGED "
            "since morning: new state changes, resolved state changes, fresh "
            "stock catalysts. Keep sections that haven't changed brief (one "
            "line saying 'unchanged from morning'). Prioritize HIGH-conviction "
            "developments touching held symbols."
        ),
        "evening": (
            "EVENING mode — SUMMARY focus. Two prior snapshots (morning, midday) "
            "may be shown below. Synthesize: which narratives STUCK (confirmed "
            "by the day's price action) vs FADED (initial interpretation didn't "
            "hold). macro_narrative should reflect where the market ACTUALLY is "
            "at end-of-day, not the morning hypothesis. state_changes should "
            "include events that closed/resolved today. This report becomes "
            "tomorrow's 'previous_narrative' — be the history you want PM to read."
        ),
    }

    def build_user_message(self, **kwargs) -> str:
        news_text: str = kwargs["news_text"]
        universe: list[str] = kwargs.get("universe", [])
        stock_mentions: dict[str, list] = kwargs.get("stock_mentions", {})
        previous_narrative: dict | None = kwargs.get("previous_narrative")
        session: str = kwargs.get("session", "morning")
        prior_session_report: dict | None = kwargs.get("prior_session_report")

        universe_text = ", ".join(universe) if universe else "N/A"

        # Session-specific guidance
        guidance = self._SESSION_GUIDANCE.get(session, self._SESSION_GUIDANCE["morning"])
        session_section = f"## Session Mode\n{guidance}\n"

        # Prior snapshot for midday/evening — lets the agent diff/summarize
        # rather than rebuild from scratch.
        if prior_session_report and session != "morning":
            prior_briefing = (prior_session_report.get("pm_briefing") or "")[:500]
            prior_sentiment = prior_session_report.get("market_sentiment", "?")
            prior_state_changes = prior_session_report.get("state_changes") or []
            sc_lines = [
                f"- [{sc.get('conviction','?').upper()}] {sc.get('event','')}: "
                f"{sc.get('previous_state','')} → {sc.get('new_state','')}"
                for sc in prior_state_changes[:5]
            ]
            sc_text = "\n".join(sc_lines) or "(none)"
            prior_section = f"""## Prior Session Snapshot (use as baseline for your delta/summary)
Sentiment at prior session: {prior_sentiment}
PM Briefing: {prior_briefing}
State changes captured earlier:
{sc_text}
"""
        else:
            prior_section = ""

        # Previous macro narrative section (evolves slowly across days)
        if previous_narrative:
            narrative_section = f"""## Previous Macro Narrative (update if needed, keep if unchanged)

```json
{json.dumps(previous_narrative, indent=2)}
```
"""
        else:
            narrative_section = "## Previous Macro Narrative\nNo previous narrative. Build one from scratch using today's news.\n"

        # Stock-specific news section
        if stock_mentions:
            stock_lines = []
            for symbol, items in sorted(stock_mentions.items()):
                for item in items[:5]:  # max 5 per symbol
                    source = getattr(item, "source", "")
                    title = getattr(item, "title", str(item))
                    summary = getattr(item, "summary", "")
                    stock_lines.append(f"  [{source}] {title}")
                    if summary:
                        stock_lines.append(f"    > {summary[:200]}")
            stock_section = f"## Stock-Specific News (mentions of universe symbols)\n\n" + "\n".join(stock_lines)
        else:
            stock_section = "## Stock-Specific News\nNo universe symbols detected in today's headlines."

        from src.trading_calendar import session_date_key
        today = session_date_key()

        return f"""## Today's Date: {today}

{session_section}
{prior_section}
{narrative_section}

## General News (last 24 hours)

{news_text}

{stock_section}

## Trading Universe
{universe_text}

Analyze all the above and produce your intelligence report as JSON."""

    @staticmethod
    def _extract_event_keywords(event: str) -> set[str]:
        """Lowercase 4+ letter tokens that aren't generic structural words.

        Used to check whether a state_change.event is actually supported by
        the input headlines. Four-letter floor keeps out a/an/is/on/etc.
        without excluding meaningful short acronyms (we accept the false-drop
        risk on a 3-letter event for the false-accept-safety).
        """
        tokens = re.findall(r"[A-Za-z]{4,}", event.lower())
        return {t for t in tokens if t not in _STATE_CHANGE_STOPWORDS}

    @classmethod
    def _filter_hallucinated_state_changes(
        cls,
        report: NewsIntelligenceReport,
        news_text: str,
        prior_session_report: dict | None = None,
    ) -> NewsIntelligenceReport:
        """Drop state_changes whose event keywords do not appear in the input
        headlines — a rough but effective guard against LLM-invented narrative
        shifts ("Iran ceasefire" when the input only had Fed news).

        A state_change is kept when either:
          - any extracted event keyword appears in the headlines text, OR
          - any ticker in `affected_symbols` appears in the headlines text, OR
          - the event has no extractable keywords AND no affected_symbols
            (can't verify either way — keep rather than silently drop), OR
          - the event was already present in `prior_session_report`
            state_changes (the reason midday/evening passes prior_session is
            to let the model carry forward / resolve a morning event even
            when fresh headlines don't repeat it verbatim).

        Matching is case-insensitive substring. Not perfect for paraphrasing,
        but dropping a correctly-interpreted-but-reworded change is far less
        costly than letting a fabricated change reach PM sizing logic.
        """
        if not report.state_changes or not news_text:
            return report

        text_lower = news_text.lower()
        # Build a supplementary token pool from the prior session's
        # state_changes so events carried forward across sessions survive.
        prior_tokens: set[str] = set()
        prior_symbols: set[str] = set()
        if prior_session_report:
            for psc in prior_session_report.get("state_changes") or []:
                event = psc.get("event") if isinstance(psc, dict) else None
                if event:
                    prior_tokens.update(cls._extract_event_keywords(event))
                syms = psc.get("affected_symbols") if isinstance(psc, dict) else None
                for s in syms or []:
                    if s:
                        prior_symbols.add(s.lower())
        kept: list = []
        dropped: list[str] = []
        for sc in report.state_changes:
            event_kws = cls._extract_event_keywords(sc.event)
            affected = [s for s in (sc.affected_symbols or []) if s]
            symbol_hits = [s for s in affected if s.lower() in text_lower]
            kw_hits = [k for k in event_kws if k in text_lower]
            prior_kw_hit = bool(event_kws & prior_tokens)
            prior_sym_hit = any(s.lower() in prior_symbols for s in affected)

            if kw_hits or symbol_hits or prior_kw_hit or prior_sym_hit:
                kept.append(sc)
            elif not event_kws and not affected:
                # Nothing verifiable either way — err on keep.
                kept.append(sc)
            else:
                dropped.append(sc.event[:80])

        if dropped:
            logger.warning(
                "news_analyst: dropped %d state_change(s) whose event "
                "keywords and affected_symbols are absent from the input "
                "headlines — likely hallucination: %s",
                len(dropped), dropped,
            )
            return report.model_copy(update={"state_changes": kept})
        return report

    def analyze(self, news_text: str, universe: list[str] | None = None,
                stock_mentions: dict | None = None,
                previous_narrative: dict | None = None,
                session: str = "morning",
                prior_session_report: dict | None = None) -> tuple[NewsIntelligenceReport | None, AgentResult]:
        result = self.run(
            news_text=news_text,
            universe=universe or [],
            stock_mentions=stock_mentions or {},
            previous_narrative=previous_narrative,
            session=session,
            prior_session_report=prior_session_report,
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("News analyst returned non-JSON response")
            return None, result
        try:
            report = NewsIntelligenceReport(**parsed)
        except Exception as e:
            logger.error("Failed to parse news intelligence report: %s", e)
            return None, result
        report = self._filter_hallucinated_state_changes(
            report, news_text, prior_session_report=prior_session_report,
        )
        return report, result
