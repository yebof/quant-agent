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
        "risk_flags": ["FX volatility remains a headwind"],
        "investment_implications": {
            "sentiment": "bullish",
            "conviction": "medium",
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
