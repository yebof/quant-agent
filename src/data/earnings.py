"""SEC EDGAR earnings data provider.

Downloads 10-Q and 10-K filings, extracts text, and tracks what's been fetched
via a local manifest so filings are only downloaded once.
"""

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import timedelta

from src.util.time import et_now
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SEC_BASE = "https://data.sec.gov"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "quant-agent research@example.com"  # SEC requires contact info
REQUEST_DELAY = 0.12  # SEC rate limit: 10 req/s

# ETFs don't have SEC 10-Q/10-K filings — skip them at the entry point to
# avoid wasting CIK lookups + retry budget on something that will always
# fail. Keep this list in sync with `config/settings.yaml:trading.universe`
# whenever a new ETF is added there.
ETFS = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLV", "XLI", "XLP",
        "XLY", "XLU", "XLRE", "XLB", "SMH", "SOXX", "DRAM", "CHPX",
        "SH", "SDS", "PSQ", "SQQQ"}


@dataclass
class FilingInfo:
    symbol: str
    form_type: str  # "10-Q" or "10-K"
    filing_date: str
    accession_number: str
    primary_doc: str  # filename of main document


@dataclass
class EarningsReport:
    symbol: str
    form_type: str
    filing_date: str
    filing_path: str  # local path to raw HTML
    analysis_path: str | None  # local path to analysis markdown
    text_excerpt: str  # extracted text for LLM (truncated)
    is_new: bool  # True if just downloaded this run


class EarningsDataProvider:
    def __init__(self, data_dir: str = "data/earnings", lookback_days: int = 45):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.data_dir / "manifest.json"
        self._manifest_lock = threading.Lock()
        self.manifest = self._load_manifest()
        self.lookback_days = lookback_days
        self._ticker_to_cik: dict[str, str] | None = None

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Corrupt manifest, starting fresh: %s", e)
        return {}

    def save_manifest(self):
        with self._manifest_lock:
            tmp = self.manifest_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.manifest, indent=2))
            os.replace(str(tmp), str(self.manifest_path))

    def confirm_filing(self, report: "EarningsReport"):
        """Mark a filing as processed in the manifest. Call after analysis file is written."""
        with self._manifest_lock:
            manifest_key = f"{report.symbol}_{report.form_type}"
            self.manifest[manifest_key] = {
                "filing_date": report.filing_date,
                "form_type": report.form_type,
                "local_path": report.filing_path,
                "analysis_path": report.analysis_path,
                "failed_attempts": 0,
            }
        self.save_manifest()

    def record_failure(self, report: "EarningsReport", max_attempts: int = 3) -> bool:
        """Track a failed LLM analysis attempt. Abandon after `max_attempts`.

        Without bounded retries, a filing whose analysis consistently fails
        (parse error, rate limit, model overloaded) would be re-queued every
        session forever — wasting tokens indefinitely. After max_attempts we
        mark the filing abandoned so _check_symbol skips it and falls back to
        any prior analysis.

        Filing-date scoping: the manifest is keyed by symbol+form_type, but
        a single key spans multiple quarters of 10-Qs. When the entry's
        stored filing_date differs from the incoming report, this is a NEW
        filing — reset failed_attempts and abandoned flag so a one-off
        parse failure on Q1 doesn't pre-abandon Q2 on its first attempt.
        Codex r11 P2: previously the prior quarter's abandoned/attempts
        carried forward, so Q2's first transient failure landed at
        attempts=4 (abandon immediately).

        Returns True when the filing has just been abandoned (caller should
        stop queueing it).
        """
        abandoned = False
        with self._manifest_lock:
            key = f"{report.symbol}_{report.form_type}"
            entry = dict(self.manifest.get(key, {}))
            prior_filing_date = entry.get("filing_date")
            if prior_filing_date and prior_filing_date != report.filing_date:
                # Different filing_date → this is a new quarter. Reset the
                # retry budget; previous failure history doesn't apply.
                entry["failed_attempts"] = 0
                entry.pop("abandoned", None)
                entry.pop("abandoned_at", None)
                logger.info(
                    "Earnings retry budget reset for %s %s: prior filing %s "
                    "→ new filing %s",
                    report.symbol, report.form_type,
                    prior_filing_date, report.filing_date,
                )
            attempts = int(entry.get("failed_attempts", 0)) + 1
            entry["filing_date"] = report.filing_date
            entry["form_type"] = report.form_type
            entry["local_path"] = report.filing_path
            entry["failed_attempts"] = attempts
            if attempts >= max_attempts:
                entry["abandoned"] = True
                entry["abandoned_at"] = et_now().isoformat()
                abandoned = True
                logger.error(
                    "Abandoning earnings analysis for %s %s (%s) after %d attempts",
                    report.symbol, report.form_type, report.filing_date, attempts,
                )
            else:
                logger.warning(
                    "Earnings analysis for %s %s failed (attempt %d/%d); will retry next session",
                    report.symbol, report.form_type, attempts, max_attempts,
                )
            self.manifest[key] = entry
        self.save_manifest()
        return abandoned

    def _sec_get(self, url: str, max_retries: int = 3) -> bytes:
        """GET with SEC-required headers, rate limiting, and retry on
        transient SEC errors.

        SEC enforces 10 req/sec via 429 (rate-limited) and returns 503
        when the service is overloaded. Before this retry loop, both
        errors raised HTTPError uncaught — caller's broad `except
        Exception` turned them into a silent empty filing list, which
        propagated to evening's `thesis_health_review` as missing 10-Q
        context (the core input for value-investing thesis decisions).

        Retries 429 / 503 / transient URLError with exponential backoff
        (1s, 2s, 4s). 404 / 400 / other 4xx-5xx propagate immediately —
        those mean the URL itself is wrong (bad CIK, missing filing),
        not a transient rate-limit, and retrying wastes the budget.
        """
        req = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            time.sleep(REQUEST_DELAY)
            try:
                with urlopen(req, timeout=15) as resp:
                    return resp.read()
            except HTTPError as e:
                last_exc = e
                if e.code in (429, 503):
                    backoff = 1.0 * (2 ** attempt)  # 1s → 2s → 4s
                    logger.warning(
                        "SEC %d on attempt %d/%d for %s — backing off %.1fs",
                        e.code, attempt + 1, max_retries, url, backoff,
                    )
                    time.sleep(backoff)
                    continue
                # Non-transient HTTP error: don't retry, surface immediately.
                raise
            except URLError as e:
                # Network blip (DNS / connection reset / timeout). Retry
                # since these are typically transient.
                last_exc = e
                backoff = 1.0 * (2 ** attempt)
                logger.warning(
                    "SEC URLError on attempt %d/%d for %s: %s — backing off %.1fs",
                    attempt + 1, max_retries, url, e, backoff,
                )
                time.sleep(backoff)
                continue
        # All retries exhausted; surface the last exception so caller
        # (currently inside a broad except Exception) can log it.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"SEC fetch failed for {url} without exception")

    def _get_cik(self, ticker: str) -> str | None:
        """Look up CIK number for a ticker symbol."""
        if self._ticker_to_cik is None:
            try:
                data = json.loads(self._sec_get(SEC_TICKERS_URL))
                self._ticker_to_cik = {}
                for entry in data.values():
                    t = entry.get("ticker", "").upper()
                    cik = str(entry.get("cik_str", ""))
                    if t and cik:
                        self._ticker_to_cik[t] = cik
            except Exception as e:
                logger.warning("Failed to fetch SEC ticker map: %s", e)
                self._ticker_to_cik = {}
        return self._ticker_to_cik.get(ticker.upper())

    def _get_recent_filings(self, cik: str, ticker: str) -> list[FilingInfo]:
        """Get recent 10-Q/10-K filings from SEC EDGAR.

        Note on MLPs (master limited partnerships, e.g. EPD): they are SEC
        registrants and DO file 10-Q/10-K via the partnership entity —
        no special handling required. The Schedule K-1 some operators
        associate with MLPs is a tax document mailed to unit holders, not
        a substitute for the corporate filing. EPD shows up on EDGAR with
        regular quarterly 10-Qs that this method will pick up.
        """
        padded_cik = cik.zfill(10)
        url = f"{SEC_BASE}/submissions/CIK{padded_cik}.json"
        try:
            data = json.loads(self._sec_get(url))
        except Exception as e:
            logger.warning("Failed to fetch submissions for %s (CIK %s): %s", ticker, cik, e)
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        # SEC's submissions JSON returns parallel arrays; in practice they
        # always align, but an upstream truncation or partial response
        # would silently desync them. Index-based access on the previous
        # version checked only forms vs dates length and could IndexError
        # on accessions / primary_docs if those came up short. zip()
        # tolerates whichever array is shortest and exits cleanly — at
        # worst we miss a trailing filing rather than crash mid-scan.
        if not (len(forms) == len(dates) == len(accessions) == len(primary_docs)):
            logger.warning(
                "SEC submissions arrays misaligned for %s (CIK %s): "
                "forms=%d dates=%d accessions=%d primary_docs=%d — "
                "iterating over the shortest",
                ticker, cik, len(forms), len(dates),
                len(accessions), len(primary_docs),
            )

        cutoff = (et_now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        filings = []
        for form, filing_date, accession, primary_doc in zip(
            forms, dates, accessions, primary_docs,
        ):
            if form not in ("10-Q", "10-K"):
                continue
            if filing_date < cutoff:
                continue
            filings.append(FilingInfo(
                symbol=ticker,
                form_type=form,
                filing_date=filing_date,
                accession_number=accession,
                primary_doc=primary_doc or "",
            ))
        return filings

    def _download_filing(self, cik: str, filing: FilingInfo) -> str | None:
        """Download filing HTML and save to local file. Returns local path."""
        symbol_dir = self.data_dir / filing.symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        accession_clean = filing.accession_number.replace("-", "")
        url = f"{SEC_ARCHIVES}/{cik}/{accession_clean}/{filing.primary_doc}"

        local_path = symbol_dir / f"{filing.form_type}_{filing.filing_date}.html"
        if local_path.exists():
            return str(local_path)

        try:
            content = self._sec_get(url)
            local_path.write_bytes(content)
            logger.info("Downloaded %s %s (%s) → %s", filing.symbol, filing.form_type,
                        filing.filing_date, local_path)
            return str(local_path)
        except Exception as e:
            logger.warning("Failed to download %s %s: %s", filing.symbol, filing.form_type, e)
            return None

    def _extract_text(self, html_path: str, max_chars: int = 30000) -> str:
        """Extract high-signal sections from a SEC 10-Q / 10-K filing.

        A raw 10-K can be 200K+ chars; 70-80% is boilerplate the LLM doesn't
        need (properties listings, mine safety disclosures, legal notes,
        signatures, exhibit indices, XBRL footers). Dumping that to the
        earnings_analyst wastes ~30% of our total token budget and dilutes
        its attention away from what drives the investment call.

        This returns a compressed document with just:
        - Financial statements  (revenue / margins / EPS numbers)
        - MD&A                  (narrative on growth, segments, outlook)
        - Risk factors          (top risks management flagged)

        Falls back to truncated full-text when structured extraction
        can't locate any sections (non-standard filing layout).
        """
        raw = Path(html_path).read_bytes()
        soup = BeautifulSoup(raw, "html.parser")

        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Structured path
        sections = self._extract_key_sections(text)
        structured_output = ""
        if sections:
            parts: list[str] = []
            total = 0
            # Order: financials (hard numbers) → MD&A (narrative) → risks (tail)
            order = ("financial_statements", "mdna", "risk_factors")
            for label in order:
                body = sections.get(label)
                if not body:
                    continue
                # Per-section cap — MD&A on a 10-K can run 40K+ on its own.
                if len(body) > 12000:
                    body = body[:12000] + "\n[... section truncated ...]"
                header = label.replace("_", " ").upper()
                section_text = f"=== {header} ===\n{body}"
                if total + len(section_text) + 2 > max_chars:
                    remaining = max_chars - total - 30  # 30 chars for tail marker
                    if remaining > 2000:
                        parts.append(section_text[:remaining] + "\n[... truncated ...]")
                    break
                parts.append(section_text)
                total += len(section_text) + 2
            if parts:
                structured_output = "\n\n".join(parts)

        # If structured extraction produced meaningful content (≥3K chars),
        # use it. Below that the sections are either sparse ('see 10-K')
        # stubs or our patterns missed the real headers — fall back to the
        # truncated full text so the LLM still has something to work with.
        MIN_STRUCTURED_SIZE = 3000
        if structured_output and len(structured_output) >= MIN_STRUCTURED_SIZE:
            logger.info(
                "Extracted %d section(s) from filing → %d chars (down from %d)",
                len(sections), len(structured_output), len(text),
            )
            return structured_output

        # Fallback: truncated full text
        if len(text) > max_chars:
            logger.info(
                "Structured extraction too sparse (%d chars); falling back to truncated full text "
                "(%d → %d chars)",
                len(structured_output), len(text), max_chars,
            )
            text = text[:max_chars] + "\n\n[... truncated ...]"
        return text

    def _extract_key_sections(self, text: str) -> dict[str, str]:
        """Locate financial / MD&A / risk-factor section bodies via regex.

        Filings typically carry a table of contents listing 'Item 1. ...',
        'Item 2. ...' near the top — those are pointers, not the section
        bodies themselves. We prefer matches beyond the first ~15K chars
        (past the TOC) when multiple matches exist. Body extends from the
        header to the next detected section/stop marker.
        """
        # Each entry: (label, pattern, strategy)
        # - "first":    the pattern matches a distinctive heading, not a TOC
        #               line — the first occurrence is the real one. Financial
        #               statements use this because 'CONSOLIDATED STATEMENTS
        #               OF OPERATIONS' isn't something a TOC typically says.
        # - "skip_toc": the pattern matches 'Item X. Section Name', which DOES
        #               appear in a TOC — prefer the first occurrence past
        #               ~15K chars (where the TOC ends).
        patterns = [
            ("financial_statements", re.compile(
                r"(?im)(?:condensed\s+)?consolidated\s+statements?\s+of\s+(?:operations?|income)\b"
            ), "first"),
            ("mdna", re.compile(
                # [\u2019'] accepts both ASCII apostrophe and the curly
                # quote U+2019 that SEC HTML filings commonly use.
                r"(?im)^\s*(?:item\s*[27]\.?)\s*management[\u2019']?s?\s+discussion"
            ), "skip_toc"),
            ("risk_factors", re.compile(
                r"(?im)^\s*(?:item\s*1a\.?)\s*risk\s+factors"
            ), "skip_toc"),
        ]
        stop_pattern = re.compile(
            r"(?im)^\s*(?:item\s*\d+[a-z]?\.?\s|"
            r"signatures?\s*$|"
            r"exhibit\s+index|"
            r"part\s+(?:i|ii|iii|iv)\b)"
        )
        all_stops = sorted(m.start() for m in stop_pattern.finditer(text))

        found: dict[str, str] = {}
        for label, pat, strategy in patterns:
            matches = list(pat.finditer(text))
            if not matches:
                continue
            if strategy == "first":
                chosen = matches[0]
            else:  # skip_toc
                chosen = next(
                    (m for m in matches if m.start() >= 15000),
                    matches[-1],
                )
            body_start = chosen.end()
            # Next stop after (body_start + 200) — don't let the header's
            # own "Item X" mention terminate its own body.
            next_stop = None
            for stop in all_stops:
                if stop > body_start + 200:
                    next_stop = stop
                    break
            body = (
                text[body_start:next_stop].strip()
                if next_stop else text[body_start:].strip()
            )
            # Low threshold — 10-Q Risk Factors sections often read "No
            # material changes from 10-K" in ~200-400 chars, which is still
            # useful information (confirms no new risks flagged). Below 150
            # is almost certainly a false-positive match.
            if len(body) >= 150:
                found[label] = body
        return found

    def _get_analysis_path(self, symbol: str, form_type: str, filing_date: str) -> str:
        """Return path for the analysis markdown file."""
        symbol_dir = self.data_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return str(symbol_dir / f"analysis_{form_type}_{filing_date}.md")

    def check_and_fetch(self, symbols: list[str]) -> list[EarningsReport]:
        """Check for new filings for all symbols. Download new ones, return reports.

        Returns EarningsReport for each symbol that has:
        - A newly downloaded filing (is_new=True), or
        - An existing analysis from a previous run (is_new=False)
        """
        reports: list[EarningsReport] = []
        stocks = [s for s in symbols if s not in ETFS]

        for symbol in stocks:
            try:
                report = self._check_symbol(symbol)
                if report:
                    reports.append(report)
            except Exception as e:
                logger.warning("Error checking earnings for %s: %s", symbol, e)

        logger.info("Earnings check: %d reports (%d new) from %d stocks",
                     len(reports), sum(1 for r in reports if r.is_new), len(stocks))
        return reports

    def _check_symbol(self, symbol: str) -> EarningsReport | None:
        """Check a single symbol for new or existing filings."""
        cik = self._get_cik(symbol)
        if not cik:
            return None

        filings = self._get_recent_filings(cik, symbol)
        if not filings:
            # No recent filings — check for existing analysis (any form)
            return self._get_existing_analysis(symbol)

        # Take the most recent filing
        latest = filings[0]
        manifest_key = f"{symbol}_{latest.form_type}"
        entry = self.manifest.get(manifest_key, {})
        last_known = entry.get("filing_date")

        # Honor the abandoned flag: after N failed analysis attempts we stop
        # re-queueing this specific filing. Fall back to prior analysis if any.
        if entry.get("abandoned") and last_known == latest.filing_date:
            logger.info(
                "Skipping %s %s (%s) — previously abandoned after repeated LLM failures",
                symbol, latest.form_type, latest.filing_date,
            )
            return self._get_existing_analysis(symbol, form_type=latest.form_type)

        if last_known == latest.filing_date:
            # Already processed this filing — return existing analysis matching this form_type
            existing = self._get_existing_analysis(symbol, form_type=latest.form_type)
            if existing:
                return existing
            # Analysis file missing (e.g. killed mid-analysis) — re-download

        # New filing — download it
        local_path = self._download_filing(cik, latest)
        if not local_path:
            return self._get_existing_analysis(symbol, form_type=latest.form_type)

        text = self._extract_text(local_path)
        analysis_path = self._get_analysis_path(symbol, latest.form_type, latest.filing_date)

        return EarningsReport(
            symbol=symbol,
            form_type=latest.form_type,
            filing_date=latest.filing_date,
            filing_path=local_path,
            analysis_path=analysis_path,
            text_excerpt=text,
            is_new=True,
        )

    def _get_existing_analysis(
        self, symbol: str, form_type: str | None = None
    ) -> EarningsReport | None:
        """Find the latest existing analysis for a symbol.

        When form_type is given, only analyses of that form are considered; otherwise
        any form's most-recent analysis is returned. Ordering is by filing_date from
        the filename, not by lexicographic sort (so 10-K 2026-03-01 beats 10-Q 2026-02-15).
        """
        symbol_dir = self.data_dir / symbol
        if not symbol_dir.exists():
            return None

        pattern = f"analysis_{form_type}_*.md" if form_type else "analysis_*.md"

        def _filing_date(path: Path) -> str:
            # filename format: analysis_<form_type>_<YYYY-MM-DD>.md
            parts = path.stem.split("_", 2)
            return parts[2] if len(parts) > 2 else ""

        analyses = sorted(symbol_dir.glob(pattern), key=_filing_date, reverse=True)
        if not analyses:
            return None

        analysis_path = str(analyses[0])
        # Parse form type and date from filename: analysis_10-Q_2026-03-15.md
        parts = analyses[0].stem.split("_", 2)
        form_type = parts[1] if len(parts) > 1 else "unknown"
        filing_date = parts[2] if len(parts) > 2 else "unknown"

        return EarningsReport(
            symbol=symbol,
            form_type=form_type,
            filing_date=filing_date,
            filing_path="",
            analysis_path=analysis_path,
            text_excerpt="",  # No text needed — analysis already exists
            is_new=False,
        )
