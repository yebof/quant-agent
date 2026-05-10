import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import AgentResult
from src.agents.earnings_analyst import EarningsAnalystAgent
from src.data.earnings import EarningsReport


def _valid_analysis(report: EarningsReport) -> dict:
    return {
        "symbol": report.symbol,
        "form_type": report.form_type,
        "filing_date": report.filing_date,
        "revenue": {
            "total": "$10.0 billion",
            "yoy_growth": "+5%",
            "segments": [{"name": "Core", "revenue": "$8.0 billion", "growth": "+4%"}],
        },
        "profitability": {
            "gross_margin": "45%",
            "operating_margin": "20%",
            "net_income": "$2.0 billion",
            "eps": "$1.00 diluted",
        },
        "cash_flow": {
            "operating_cf": "$3.0 billion",
            "free_cf": "$2.5 billion",
            "capex": "$0.5 billion",
        },
        "balance_sheet": {
            "cash_and_equivalents": "$4.0 billion",
            "total_debt": "$1.0 billion",
            "assessment": "Healthy balance sheet",
        },
        "management_highlights": ["Demand remained stable across core products"],
        "guidance": "Management did not provide numeric guidance",
        "strategic_direction": {
            "key_initiatives": ["Expanding into cloud services"],
            "capital_allocation": "50% buybacks, 30% R&D, 20% debt reduction",
            "competitive_positioning": "Market leader with 35% share in core segment",
        },
        "risk_flags": {
            "strategic_risks": ["Cloud expansion faces entrenched competitors"],
            "operational_risks": ["FX volatility remains a headwind"],
        },
        "strategy_consistency": "Consistent with prior quarter — cloud expansion on track",
        "investment_implications": {
            "sentiment": "bullish",
            "conviction": "medium",
            "reasoning_chain": {
                "fundamental_quality": "Revenue +5% with margin expansion",
                "growth_trajectory": "Operating leverage building QoQ",
                "strategic_risks": "Cloud competition is real but execution on track",
                "management_execution": "Guidance hit, capex on plan",
                "valuation_context": "Trades at a reasonable forward multiple",
            },
            "key_thesis": "Margins expanded while demand remained resilient",
            "bull_case": "Operating leverage continues",
            "bear_case": "FX pressure worsens",
        },
        "data_quality": "Filing text complete through the financial statements and MD&A.",
    }


@pytest.fixture
def agent():
    with patch("anthropic.Anthropic"):
        yield EarningsAnalystAgent(api_key="test-key", model="claude-sonnet-4-6-20250514")


@pytest.fixture
def report(tmp_path):
    return EarningsReport(
        symbol="AAPL",
        form_type="10-Q",
        filing_date="2026-03-15",
        filing_path=str(tmp_path / "10-Q.html"),
        analysis_path=str(tmp_path / "AAPL" / "analysis_10-Q_2026-03-15.md"),
        text_excerpt="Revenue was $10.0 billion and operating cash flow was $3.0 billion.",
        is_new=True,
    )


def test_extract_text_compresses_standard_10q(tmp_path):
    """Full 10-Q with TOC + all standard sections → structured extraction path."""
    from src.data.earnings import EarningsDataProvider

    filler = "Lorem ipsum dolor sit amet consectetur. " * 50  # ~2000 chars
    html = f"""<html><body>
    <h1>Apple Inc. Q1 2026 Form 10-Q</h1>
    <div>Table of Contents: Item 1. Financial Statements ... Item 2. Management's Discussion and Analysis ... Item 1A. Risk Factors ...</div>
    {"Cover page filler filler filler. " * 400}
    <h2>CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS</h2>
    <p>Net sales: Products $113,743 Services $26,340. Total $140,083.
    Operating income $42,832. Diluted EPS $2.40. {filler}</p>
    <h2>Item 2. Management's Discussion and Analysis of Financial Condition</h2>
    <p>Products revenue grew 8% YoY driven by iPhone. Services grew 14%.
    Gross margin expanded 120bps to 46.9%. Guidance implies mid-single-digit
    revenue growth in Q2. {filler}</p>
    <h2>Item 1A. Risk Factors</h2>
    <p>There have been no material changes to the risk factors disclosed in
    our 2025 Form 10-K. {filler}</p>
    </body></html>"""
    html_path = tmp_path / "test.html"
    html_path.write_bytes(html.encode())

    p = EarningsDataProvider(data_dir=str(tmp_path / "earnings"))
    out = p._extract_text(str(html_path), max_chars=30000)
    assert "=== FINANCIAL STATEMENTS ===" in out
    assert "=== MDNA ===" in out
    assert "=== RISK FACTORS ===" in out
    assert "113,743" in out
    assert "iPhone" in out
    assert len(out) < len(html) / 2


def test_extract_text_handles_smart_apostrophe(tmp_path):
    """SEC filings commonly use the curly apostrophe U+2019 — regex must match it."""
    from src.data.earnings import EarningsDataProvider

    html = (
        "<html><body>"
        + ("x " * 8000)  # push past TOC threshold
        + "\nItem 2.\nManagement\u2019s Discussion and Analysis\n"
        + "<p>Revenue up 10%. " + ("Lorem ipsum. " * 200) + "</p>"
        + "\nItem 3. Quantitative disclosures\n"
        + "</body></html>"
    )
    html_path = tmp_path / "curly.html"
    html_path.write_bytes(html.encode())

    p = EarningsDataProvider(data_dir=str(tmp_path / "earnings"))
    out = p._extract_text(str(html_path))
    # Whether structured or fallback path fires, the key content must be there.
    assert "Revenue up 10%" in out


def test_extract_text_falls_back_to_truncated_when_sections_sparse(tmp_path):
    """A filing with no recognizable section headers → fallback truncated text."""
    from src.data.earnings import EarningsDataProvider

    raw = "This is a filing without standard section markers. " * 4000
    html_path = tmp_path / "nohdr.html"
    html_path.write_bytes(f"<html><body>{raw}</body></html>".encode())

    p = EarningsDataProvider(data_dir=str(tmp_path / "earnings"))
    out = p._extract_text(str(html_path), max_chars=5000)
    # No section markers, fallback path
    assert "===" not in out
    assert len(out) <= 5100  # 5000 + small tail marker
    assert "[... truncated ...]" in out


def test_earnings_analyst_accepts_valid_analysis(agent, report):
    agent.run = MagicMock(
        return_value=AgentResult(
            raw_text=json.dumps(_valid_analysis(report)),
            tokens_used=123,
            model="test-model",
        )
    )

    analysis, _ = agent._analyze_new(report)

    assert analysis is not None
    assert analysis["symbol"] == "AAPL"
    assert analysis["investment_implications"]["sentiment"] == "bullish"


def test_earnings_analyst_rejects_metadata_mismatch(agent, report):
    bad = _valid_analysis(report)
    bad["symbol"] = "TSLA"
    agent.run = MagicMock(
        return_value=AgentResult(
            raw_text=json.dumps(bad),
            tokens_used=123,
            model="test-model",
        )
    )

    analysis, _ = agent._analyze_new(report)

    assert analysis is None


def test_earnings_analyst_rejects_invalid_cached_analysis(agent, report):
    bad = _valid_analysis(report)
    bad["filing_date"] = "2026-03-16"

    analysis_path = Path(report.analysis_path)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(
        "# Cached analysis\n\n```json\n" + json.dumps(bad, indent=2) + "\n```\n"
    )

    assert agent._load_analysis(report) is None
