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
from datetime import datetime, timedelta
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

# ETFs don't have SEC filings
ETFS = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLV", "XLI", "XLP",
        "XLY", "XLU", "XLRE", "XLB", "SMH", "DRAM", "SH", "SDS", "PSQ", "SQQQ"}


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
            }
        self.save_manifest()

    def _sec_get(self, url: str) -> bytes:
        """GET with SEC-required headers and rate limiting."""
        req = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
        time.sleep(REQUEST_DELAY)
        with urlopen(req, timeout=15) as resp:
            return resp.read()

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
        """Get recent 10-Q/10-K filings from SEC EDGAR."""
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

        cutoff = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        filings = []
        for i, form in enumerate(forms):
            if form not in ("10-Q", "10-K"):
                continue
            if i >= len(dates) or dates[i] < cutoff:
                continue
            filings.append(FilingInfo(
                symbol=ticker,
                form_type=form,
                filing_date=dates[i],
                accession_number=accessions[i],
                primary_doc=primary_docs[i] if i < len(primary_docs) else "",
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

    def _extract_text(self, html_path: str, max_chars: int = 150000) -> str:
        """Extract clean text from SEC HTML filing."""
        raw = Path(html_path).read_bytes()
        soup = BeautifulSoup(raw, "html.parser")

        # Remove script, style, and hidden elements
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()

        text = soup.get_text(separator="\n")

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)

        # Collapse multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated ...]"

        return text

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
        last_known = self.manifest.get(manifest_key, {}).get("filing_date")

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
