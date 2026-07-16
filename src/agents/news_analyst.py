import json
import logging
import re
from pathlib import Path

from pydantic import ValidationError

from src.agents.base import BaseAgent, AgentResult
from src.models import NewsIntelligenceReport, StateChange, StockNewsItem

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
        # audit round 2 #24: the close session (15:30 ET) had no entry and
        # silently fell back to MORNING guidance — mislabeling the run as a
        # "fresh book" full rebuild 30 minutes before the bell.
        "close": (
            "CLOSE mode — DELTA focus, ~30 minutes to the bell. The most "
            "recent prior snapshot (midday or morning) may be shown below as "
            "the baseline. Surface what CHANGED since that snapshot: new or "
            "reversing state changes and fresh stock catalysts that could "
            "trigger an exit before the close or move held positions "
            "overnight. Keep unchanged sections to one line ('unchanged "
            "since midday'). Prioritize HIGH-conviction developments "
            "touching held symbols."
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

        # Stock-specific news section.
        # audit round 2 #12: the loop used to discard the symbol key and,
        # because tag_symbol_mentions files a multi-symbol headline under
        # EVERY matching ticker, render the same headline N times with no
        # attribution. Invert to item → [symbols]: each headline renders
        # once, prefixed with the tickers it was tagged for.
        if stock_mentions:
            grouped: dict[tuple, list[str]] = {}
            item_by_key: dict[tuple, object] = {}
            order: list[tuple] = []
            for symbol, items in sorted(stock_mentions.items()):
                for item in items[:5]:  # max 5 per symbol
                    key = (getattr(item, "source", ""), getattr(item, "title", str(item)))
                    if key not in grouped:
                        grouped[key] = []
                        item_by_key[key] = item
                        order.append(key)
                    if symbol not in grouped[key]:
                        grouped[key].append(symbol)
            stock_lines = []
            for key in order:
                source, title = key
                syms = ", ".join(grouped[key])
                stock_lines.append(f"  [{source}] ({syms}) {title}")
                summary = getattr(item_by_key[key], "summary", "")
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

        Matching: event keywords are case-insensitive substring (safe — the
        4-char floor + stopwords already exclude generic tokens); ticker
        symbols are whole-token matches (audit round 2 #25 — 1-2 letter
        tickers are substrings of almost anything). Not perfect for
        paraphrasing, but dropping a correctly-interpreted-but-reworded
        change is far less costly than letting a fabricated change reach
        PM sizing logic.
        """
        if not report.state_changes or not news_text:
            return report

        text_lower = news_text.lower()
        # audit round 2 #25: symbol hits must be whole-token matches, not raw
        # substrings — universe tickers like V / MA / GE are substrings of
        # virtually any headline blob ("nvidia" contains "v"), which let a
        # fabricated state_change tagged with a short ticker sail through
        # this filter. Mirrors the word-boundary discipline of
        # src/data/news.py:tag_symbol_mentions.
        text_tokens = set(re.findall(r"[a-z0-9.\-]+", text_lower))
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
            symbol_hits = [s for s in affected if s.lower() in text_tokens]
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
        # Per-entry isolation: a single malformed StockNewsItem (e.g. empty
        # headline) or StateChange (e.g. bad conviction enum) must not drop
        # the WHOLE news report — that report carries macro_narrative,
        # pm_briefing, and the rest of state_changes / stock_news that
        # PM relies on to brief the morning. Mirrors EveningAnalyst.
        # _drop_invalid_missed_opportunities (PR #73) and the
        # TechAnalyst.analyze_batch isolate-failures-by-symbol discipline.
        parsed = self._drop_invalid_state_changes(parsed)
        parsed = self._drop_invalid_stock_news(parsed)
        try:
            report = NewsIntelligenceReport(**parsed)
        except Exception as e:
            logger.error("Failed to parse news intelligence report: %s", e)
            return None, result
        report = self._filter_hallucinated_state_changes(
            report, news_text, prior_session_report=prior_session_report,
        )
        return report, result

    @staticmethod
    def _drop_invalid_state_changes(parsed: dict) -> dict:
        """Pre-validate each StateChange; drop malformed entries with a warning.

        StateChange has Literal validation on `conviction`; a typo in one
        item must not poison the whole report. Mutates parsed in place
        for `state_changes`; non-list shapes are normalized to [].
        """
        raw = parsed.get("state_changes")
        if raw is None:
            return parsed
        if not isinstance(raw, list):
            logger.warning(
                "News analyst: state_changes is %s, not list — replacing with []",
                type(raw).__name__,
            )
            parsed["state_changes"] = []
            return parsed
        valid: list[dict] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                logger.warning(
                    "News analyst: dropping non-dict state_changes entry at "
                    "index %d: %r", i, item,
                )
                continue
            try:
                StateChange(**item)
            except ValidationError as e:
                event = item.get("event") or f"<idx {i}>"
                logger.warning(
                    "News analyst: dropping malformed state_change %r: %s",
                    event, e,
                )
                continue
            valid.append(item)
        parsed["state_changes"] = valid
        return parsed

    @staticmethod
    def _drop_invalid_stock_news(parsed: dict) -> dict:
        """Pre-validate each StockNewsItem under each symbol bucket.

        `stock_news` is a dict[str, list[StockNewsItem]]. A single item
        with an empty headline (the most common LLM glitch) currently
        kills the whole NewsIntelligenceReport — including macro_narrative
        and pm_briefing, which PM needs even if a single per-symbol
        bullet is malformed. Drop bad items per-symbol; if a symbol's
        list ends up empty, drop the symbol entry too.
        """
        raw = parsed.get("stock_news")
        if raw is None:
            return parsed
        if not isinstance(raw, dict):
            logger.warning(
                "News analyst: stock_news is %s, not dict — replacing with {}",
                type(raw).__name__,
            )
            parsed["stock_news"] = {}
            return parsed
        cleaned: dict[str, list[dict]] = {}
        for sym, items in raw.items():
            if not isinstance(items, list):
                logger.warning(
                    "News analyst: stock_news[%s] is %s, not list — dropping",
                    sym, type(items).__name__,
                )
                continue
            valid: list[dict] = []
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    logger.warning(
                        "News analyst: dropping non-dict stock_news entry "
                        "under %s at index %d: %r", sym, i, item,
                    )
                    continue
                try:
                    StockNewsItem(**item)
                except ValidationError as e:
                    headline = (item.get("headline") or f"<idx {i}>")[:80]
                    logger.warning(
                        "News analyst: dropping malformed stock_news entry "
                        "under %s (%s): %s", sym, headline, e,
                    )
                    continue
                valid.append(item)
            if valid:
                cleaned[sym] = valid
        parsed["stock_news"] = cleaned
        return parsed
