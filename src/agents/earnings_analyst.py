"""Earnings Analyst Agent — reads SEC filings and writes structured analyses.

For new filings: reads raw text, produces analysis, saves to markdown file.
For existing filings: returns previously saved analysis.
"""

import json
import logging
import os
import re
from pathlib import Path

from pydantic import ValidationError

from src.agents.base import BaseAgent, AgentResult
from src.data.earnings import EarningsReport
from src.models import EarningsAnalysis

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "earnings_analyst.md"


class EarningsAnalystAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "earnings_analyst"

    @property
    def system_prompt(self) -> str:
        if PROMPT_PATH.exists():
            return PROMPT_PATH.read_text()
        return "You are an earnings analyst. Respond with JSON."

    def build_user_message(self, **kwargs) -> str:
        symbol: str = kwargs["symbol"]
        form_type: str = kwargs["form_type"]
        filing_date: str = kwargs["filing_date"]
        filing_text: str = kwargs["filing_text"]
        prior_analysis: str = kwargs.get("prior_analysis", "")

        prior_section = ""
        if prior_analysis:
            prior_section = f"""## Prior Analysis (for context)
{prior_analysis}

---

"""

        return f"""{prior_section}## Filing: {symbol} {form_type} (filed {filing_date})

{filing_text}

Analyze this filing and respond with JSON. Cite specific numbers from the text above."""

    def analyze_reports(self, reports: list[EarningsReport]) -> list[dict]:
        """Analyze all reports. Only schema-validated analyses are returned.

        Returns list of {symbol, analysis_dict, agent_result_or_none}.
        """
        results = []

        for report in reports:
            if report.is_new and report.text_excerpt:
                # New filing — run LLM analysis
                analysis, agent_result = self._analyze_new(report)
                if analysis:
                    # Save analysis to disk
                    self._save_analysis(report.analysis_path, report, analysis)
                    results.append({
                        "symbol": report.symbol,
                        "analysis": analysis,
                        "agent_result": agent_result,
                        "is_new": True,
                        "form_type": report.form_type,
                        "filing_date": report.filing_date,
                    })
            elif report.analysis_path and Path(report.analysis_path).exists():
                # Existing analysis — read from disk
                analysis = self._load_analysis(report)
                if analysis:
                    results.append({
                        "symbol": report.symbol,
                        "analysis": analysis,
                        "agent_result": None,
                        "is_new": False,
                        "form_type": report.form_type,
                        "filing_date": report.filing_date,
                    })

        return results

    def _analyze_new(self, report: EarningsReport) -> tuple[dict | None, AgentResult]:
        """Run LLM analysis on a new filing."""
        # Check for prior analysis to provide context
        prior = ""
        symbol_dir = Path(report.analysis_path).parent
        prior_analyses = sorted(symbol_dir.glob("analysis_*.md"), reverse=True)
        if prior_analyses:
            # Read the most recent prior analysis (skip current)
            for p in prior_analyses:
                if str(p) != report.analysis_path:
                    prior = p.read_text()[:5000]  # First 5K chars of prior analysis
                    break

        result = self.run(
            symbol=report.symbol,
            form_type=report.form_type,
            filing_date=report.filing_date,
            filing_text=report.text_excerpt,
            prior_analysis=prior,
        )
        parsed = result.parse_json()
        if parsed is None:
            logger.error("Earnings analyst returned non-JSON for %s", report.symbol)
            return None, result
        validated = self._validate_analysis(report, parsed, source="llm")
        if validated is None:
            return None, result
        return validated.model_dump(), result

    def _save_analysis(self, path: str, report: EarningsReport, analysis: dict):
        """Save analysis as markdown + JSON. Atomic write: tmp + rename so
        a SIGKILL mid-write can never leave a half-written file on disk.

        Without atomicity (the pre-2026-05-13 behavior), a kill between
        `p.write_text` and process exit produces a truncated markdown,
        which then fails JSON-block re-parse on next session →
        record_failure() ticks → 3 consecutive ticks abandon the filing
        permanently. The LLM succeeded but the disk write didn't is the
        worst kind of silent regression — wasted tokens AND lost thesis
        context. Same atomic-write discipline as news_store /
        macro_store / tech_store (those were already protected; earnings
        was the only outlier).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Write markdown with embedded JSON
        header = f"# {report.symbol} {report.form_type} Analysis ({report.filing_date})\n\n"
        header += f"Filing source: `{report.filing_path}`\n\n"
        header += f"## Investment Implications\n\n"
        impl = analysis.get("investment_implications", {})
        header += f"- Sentiment: {impl.get('sentiment', 'N/A')}\n"
        header += f"- Conviction: {impl.get('conviction', 'N/A')}\n"
        header += f"- Thesis: {impl.get('key_thesis', 'N/A')}\n\n"
        header += f"## Full Analysis\n\n```json\n{json.dumps(analysis, indent=2)}\n```\n"

        tmp = p.with_suffix(p.suffix + ".tmp")
        try:
            tmp.write_text(header)
            os.replace(tmp, p)
        except Exception:
            # Clean up tmp on failure so the next run doesn't see a
            # stale partial. Re-raise so the caller (manifest update)
            # doesn't proceed as if the analysis was saved.
            tmp.unlink(missing_ok=True)
            raise
        logger.info("Saved analysis for %s %s → %s", report.symbol, report.form_type, path)

    def _load_analysis(self, report: EarningsReport) -> dict | None:
        """Load previously saved analysis from markdown file."""
        text = Path(report.analysis_path).read_text()
        # Extract JSON from ```json ... ``` block
        match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                validated = self._validate_analysis(report, parsed, source="cache")
                return validated.model_dump() if validated else None
            except json.JSONDecodeError:
                logger.warning("Failed to parse saved analysis: %s", report.analysis_path)
        return None

    def _validate_analysis(
        self, report: EarningsReport, parsed: dict | list, source: str
    ) -> EarningsAnalysis | None:
        if not isinstance(parsed, dict):
            logger.warning("Invalid %s earnings analysis for %s: expected JSON object", source, report.symbol)
            return None

        try:
            analysis = EarningsAnalysis(**parsed)
        except ValidationError as exc:
            logger.warning("Invalid %s earnings analysis for %s: %s", source, report.symbol, exc)
            return None

        if analysis.symbol != report.symbol.upper():
            logger.warning(
                "Invalid %s earnings analysis for %s: symbol mismatch (%s)",
                source,
                report.symbol,
                analysis.symbol,
            )
            return None
        if analysis.form_type != report.form_type:
            logger.warning(
                "Invalid %s earnings analysis for %s: form mismatch (%s)",
                source,
                report.symbol,
                analysis.form_type,
            )
            return None
        if analysis.filing_date != report.filing_date:
            logger.warning(
                "Invalid %s earnings analysis for %s: filing_date mismatch (%s)",
                source,
                report.symbol,
                analysis.filing_date,
            )
            return None

        return analysis
