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


def test_extract_text_skips_toc_for_financial_statements(tmp_path):
    """R6 audit (May 2026): logs showed 58 filings where the regex for
    'Consolidated Statements of Operations' matched the internal TOC /
    Index to Financial Statements before reaching the actual table.
    Result was ~553 chars of TOC entries vs 10,000+ chars of real data.

    Pin: when a TOC entry occurs BEFORE the real section, skip_toc
    strategy picks the later (real) one. Affected real-world filings:
    PG / SBUX / V / ABT / AMZN / CAT / COP / LLY 10-Qs."""
    from src.data.earnings import EarningsDataProvider

    # TOC entry (within first 15K chars) then later real section past 15K.
    # The early "Consolidated Statements of Operations" is a TOC pointer
    # with only ~200 chars of body before the next "Item" stop marker —
    # short enough to be dropped by the >=150 char threshold normally, but
    # NOT short enough if any "Item X" stop is found just after.
    early_toc = (
        "<html><body>"
        "<h1>Procter & Gamble 10-Q</h1>\n"
        "INDEX TO CONSOLIDATED FINANCIAL STATEMENTS\n"
        "Consolidated Statements of Operations\n"
        "(page 3)\n"
        # Filler to push the real section past 15K
        + ("Lorem cover-page boilerplate text padding the front. " * 350)
        # Real financial statement past the 15K threshold
        + "\nCONSOLIDATED STATEMENTS OF OPERATIONS\n"
        + ("Net sales $21,737 Cost of products sold $10,392 "
           "Operating income $5,148 Net earnings $4,103 Diluted EPS $1.66. " * 30)
        + "\nItem 2. Management's Discussion and Analysis\n"
        + ("Sales growth was driven by Beauty +6% and Health +8%. " * 30)
        + "\nItem 3. Quantitative disclosures\n"
        "</body></html>"
    )
    html_path = tmp_path / "pg.html"
    html_path.write_bytes(early_toc.encode())

    p = EarningsDataProvider(data_dir=str(tmp_path / "earnings"))
    out = p._extract_text(str(html_path), max_chars=30000)
    # The body of the financial_statements section must include the real
    # numerics, not just the TOC pointer.
    assert "Net sales $21,737" in out
    assert "Diluted EPS $1.66" in out
    assert "=== FINANCIAL STATEMENTS ===" in out


def test_extract_text_finds_financial_dense_region_on_fallback(tmp_path):
    """When structured extraction completely fails (filing layout that
    doesn't match any regex), fall back to the financial-data-rich
    region of the text rather than the front (which is typically
    cover page + XBRL boilerplate for iXBRL 10-Qs)."""
    from src.data.earnings import EarningsDataProvider

    # Front-loaded XBRL/cover boilerplate (no $-amount density), then a
    # rich financial table in the middle. No section headers anywhere
    # to defeat structured extraction entirely.
    boilerplate_front = (
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax "
        "contextRef=ctx_2026_q1 dimension=member dei:DocumentType "
        "ifrs-full:Assets fair-value-hierarchy Level1 Level2 Level3 "
    ) * 200
    rich_middle = (
        "Net sales $45,123 Operating income $12,456 Net income $9,876 "
        "Total assets $250,000 Cash $30,000 Diluted EPS $4.32 "
        "Free cash flow $11,234 Capital expenditures $(1,500) "
    ) * 100
    quiet_tail = "Signatures. Exhibit list. Other boilerplate. " * 200

    body = boilerplate_front + rich_middle + quiet_tail
    html_path = tmp_path / "ixbrl.html"
    html_path.write_bytes(f"<html><body>{body}</body></html>".encode())

    p = EarningsDataProvider(data_dir=str(tmp_path / "earnings"))
    out = p._extract_text(str(html_path), max_chars=10000)
    # Output must contain the rich-middle financial numbers, NOT the
    # XBRL boilerplate that dominated the front.
    assert "Net sales $45,123" in out
    assert "Diluted EPS $4.32" in out


def test_find_financial_dense_region_preserves_head_when_no_obvious_winner(tmp_path):
    """Tuning guard: if every chunk has similar low financial density
    (e.g., filing is genuinely all narrative — possibly a stub or an
    amendment with no statements), don't relocate. Returning 0 keeps
    backward compat behavior."""
    from src.data.earnings import EarningsDataProvider

    flat = "narrative text with no dollar figures or financial tables. " * 2000
    idx = EarningsDataProvider._find_financial_dense_region(flat, window=5000)
    assert idx == 0


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
