import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import pytest

from src.data.news import NewsDataProvider, NewsItem
from src.agents.news_analyst import NewsAnalystAgent


# === NewsDataProvider tests ===

def test_news_provider_format_for_prompt():
    provider = NewsDataProvider()
    items = [
        NewsItem(
            title="Fed signals pause in rate hikes",
            summary="Federal Reserve officials indicated...",
            source="Reuters Business",
            published=datetime(2026, 4, 12, 10, 0, tzinfo=timezone.utc),
            link="https://example.com/1",
        ),
        NewsItem(
            title="Tech stocks rally on earnings",
            summary="Major tech companies reported...",
            source="CNBC Top News",
            published=datetime(2026, 4, 12, 8, 0, tzinfo=timezone.utc),
            link="https://example.com/2",
        ),
    ]
    text = provider.format_for_prompt(items)
    assert "Fed signals pause" in text
    assert "Reuters Business" in text
    assert "Tech stocks rally" in text


def test_news_provider_format_empty():
    provider = NewsDataProvider()
    text = provider.format_for_prompt([])
    assert "No recent news" in text


def test_news_provider_deduplicate():
    provider = NewsDataProvider()
    items = [
        NewsItem(title="Breaking: Market rallies", summary="", source="A",
                 published=datetime(2026, 4, 12, tzinfo=timezone.utc), link=""),
        NewsItem(title="Breaking: Market rallies", summary="", source="B",
                 published=datetime(2026, 4, 12, tzinfo=timezone.utc), link=""),
        NewsItem(title="Different headline", summary="", source="C",
                 published=datetime(2026, 4, 12, tzinfo=timezone.utc), link=""),
    ]
    deduped = provider._deduplicate(items)
    assert len(deduped) == 2


def test_news_provider_format_max_items():
    provider = NewsDataProvider()
    items = [
        NewsItem(title=f"Headline {i}", summary="", source="Test",
                 published=datetime(2026, 4, 12, tzinfo=timezone.utc), link="")
        for i in range(100)
    ]
    text = provider.format_for_prompt(items, max_items=5)
    assert "Headline 0" in text
    assert "Headline 4" in text
    assert "Headline 5" not in text


# === NewsAnalystAgent tests ===

@patch("anthropic.Anthropic")
def test_news_analyst_analyze(mock_cls):
    response_json = json.dumps({
        "macro_narrative": {
            "last_updated": "2026-04-15",
            "era_themes": ["AI supercycle", "Fed easing"],
            "current_regime": "Risk-on with caution",
            "key_state_tracker": {"fed_policy": "Easing — paused at 3.6%"},
        },
        "state_changes": [
            {
                "event": "Fed signals pause in rate hikes",
                "previous_state": "Cutting rates",
                "new_state": "Pausing to assess",
                "market_impact": "Slightly bearish for rate-sensitive sectors",
                "affected_symbols": ["JPM"],
                "conviction": "high",
            }
        ],
        "stock_news": {
            "NVDA": [
                {
                    "headline": "New chip announcement",
                    "sentiment": "bullish",
                    "conviction": "medium",
                    "impact_summary": "Next-gen GPU may accelerate AI adoption",
                }
            ]
        },
        "pm_briefing": "Fed pausing. NVDA new chip bullish. Risk-on with caution.",
        "market_sentiment": "bullish",
        "confidence": "medium",
    })

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_json)]
    mock_response.usage.input_tokens = 2000
    mock_response.usage.output_tokens = 500
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = NewsAnalystAgent(api_key="test", model="claude-sonnet-4-6-20250514")
    report, agent_result = agent.analyze(
        news_text="Fed signals pause in rate hikes...",
        universe=["SPY", "NVDA", "JPM"],
    )

    assert report is not None
    assert report.market_sentiment == "bullish"
    assert report.confidence == "medium"
    assert len(report.state_changes) == 1
    assert report.state_changes[0].conviction == "high"
    assert "NVDA" in report.stock_news
    assert report.macro_narrative.current_regime == "Risk-on with caution"
    assert agent_result.tokens_used == 2500


def _make_news_intel_report(state_changes: list[dict]):
    """Helper: build a minimal NewsIntelligenceReport with custom state_changes."""
    from src.models import NewsIntelligenceReport
    return NewsIntelligenceReport.model_validate({
        "macro_narrative": {
            "last_updated": "2026-04-18",
            "era_themes": ["test"],
            "current_regime": "test regime",
            "key_state_tracker": {},
        },
        "state_changes": state_changes,
        "stock_news": {},
        "pm_briefing": "test",
        "market_sentiment": "neutral",
        "confidence": "medium",
    })


def test_filter_drops_state_change_with_no_keyword_overlap():
    """Invented event — no keyword in headlines — must be dropped."""
    report = _make_news_intel_report([
        {
            "event": "Iran ceasefire brokered",
            "previous_state": "hot war",
            "new_state": "truce",
            "market_impact": "oil down",
            "affected_symbols": ["XOM"],
            "conviction": "high",
        }
    ])
    filtered = NewsAnalystAgent._filter_hallucinated_state_changes(
        report, news_text="Fed signals pause in rate hikes; unrelated macro story."
    )
    assert filtered.state_changes == []


def test_filter_keeps_state_change_when_event_keyword_present():
    report = _make_news_intel_report([
        {
            "event": "Fed signals pause",
            "previous_state": "cutting",
            "new_state": "paused",
            "market_impact": "bearish rate-sensitive",
            "affected_symbols": [],
            "conviction": "high",
        }
    ])
    filtered = NewsAnalystAgent._filter_hallucinated_state_changes(
        report,
        news_text="The Fed today signals a pause in further rate hikes...",
    )
    assert len(filtered.state_changes) == 1
    assert "Fed signals pause" == filtered.state_changes[0].event


def test_filter_keeps_state_change_when_affected_symbol_present():
    """Event wording may be paraphrased but if an affected ticker literally
    shows up in the headlines, treat the change as grounded enough to keep."""
    report = _make_news_intel_report([
        {
            "event": "Regulatory pressure escalating",
            "previous_state": "normal",
            "new_state": "under scrutiny",
            "market_impact": "bearish",
            "affected_symbols": ["NVDA"],
            "conviction": "medium",
        }
    ])
    filtered = NewsAnalystAgent._filter_hallucinated_state_changes(
        report,
        # "Regulatory", "pressure", "escalating", "scrutiny" aren't in text,
        # but NVDA is.
        news_text="NVDA shares slipped today on industry concerns.",
    )
    assert len(filtered.state_changes) == 1


def test_filter_drops_some_keeps_others_in_mixed_batch():
    report = _make_news_intel_report([
        {
            "event": "Fed signals pause",
            "previous_state": "cutting", "new_state": "paused",
            "market_impact": "x", "affected_symbols": [], "conviction": "high",
        },
        {
            "event": "Hurricane shuts Gulf refineries",
            "previous_state": "normal", "new_state": "disrupted",
            "market_impact": "oil up", "affected_symbols": ["XOM"],
            "conviction": "high",
        },
    ])
    filtered = NewsAnalystAgent._filter_hallucinated_state_changes(
        report,
        news_text="Fed officials signal pause on policy. No weather news.",
    )
    events = [sc.event for sc in filtered.state_changes]
    assert "Fed signals pause" in events
    assert "Hurricane shuts Gulf refineries" not in events


def test_filter_preserves_state_change_carried_from_prior_session():
    """Midday/evening pass prior_session_report so the agent can carry a
    morning event forward (or mark it resolved) even when fresh headlines
    no longer repeat the phrasing verbatim. The filter must not drop the
    carry-forward as "hallucinated" just because fresh news moved on."""
    report = _make_news_intel_report([
        {
            "event": "Fed signals pause",
            "previous_state": "cutting", "new_state": "paused",
            "market_impact": "bearish rate-sensitive",
            "affected_symbols": [], "conviction": "high",
        }
    ])
    prior = {
        "state_changes": [
            {"event": "Fed signals pause in hikes",
             "previous_state": "cutting", "new_state": "paused",
             "market_impact": "bearish rate-sensitive",
             "affected_symbols": [], "conviction": "high"}
        ],
    }
    # Fresh midday headlines don't repeat the Fed language at all.
    filtered = NewsAnalystAgent._filter_hallucinated_state_changes(
        report,
        news_text="Midday headlines: commodity prices drift, no policy news.",
        prior_session_report=prior,
    )
    events = [sc.event for sc in filtered.state_changes]
    assert "Fed signals pause" in events


def test_filter_preserves_state_change_when_prior_symbol_matches():
    """Same carry-forward semantics via affected_symbols match."""
    report = _make_news_intel_report([
        {
            "event": "Regulatory probe ongoing",
            "previous_state": "opened", "new_state": "ongoing",
            "market_impact": "bearish",
            "affected_symbols": ["NVDA"], "conviction": "medium",
        }
    ])
    prior = {
        "state_changes": [
            {"event": "Regulatory probe opened",
             "previous_state": "none", "new_state": "opened",
             "market_impact": "bearish",
             "affected_symbols": ["NVDA"], "conviction": "medium"}
        ],
    }
    filtered = NewsAnalystAgent._filter_hallucinated_state_changes(
        report,
        news_text="Midday: indices drift on light volume. No chip news.",
        prior_session_report=prior,
    )
    assert len(filtered.state_changes) == 1


def test_filter_empty_news_text_keeps_all():
    """No news_text to verify against (unusual but possible) — err on keep
    rather than silently drop the LLM's whole state-change list."""
    report = _make_news_intel_report([
        {
            "event": "Something happened",
            "previous_state": "a", "new_state": "b",
            "market_impact": "x", "affected_symbols": [], "conviction": "low",
        }
    ])
    filtered = NewsAnalystAgent._filter_hallucinated_state_changes(
        report, news_text="",
    )
    assert len(filtered.state_changes) == 1


@patch("anthropic.Anthropic")
def test_news_analyst_analyze_filters_hallucinated_state_change(mock_cls):
    """Integration: analyze() now runs the hallucination filter after parsing.
    An LLM-invented state_change (event keywords not in input) must not
    survive into the returned NewsIntelligenceReport."""
    response_json = json.dumps({
        "macro_narrative": {
            "last_updated": "2026-04-18",
            "era_themes": ["AI"],
            "current_regime": "risk-on",
            "key_state_tracker": {},
        },
        "state_changes": [
            # This one is supported by the headlines.
            {
                "event": "Fed signals pause",
                "previous_state": "cutting", "new_state": "paused",
                "market_impact": "bearish rate-sensitive",
                "affected_symbols": [], "conviction": "high",
            },
            # This one is a pure hallucination — no keyword match.
            {
                "event": "Iran ceasefire brokered",
                "previous_state": "war", "new_state": "truce",
                "market_impact": "oil down",
                "affected_symbols": ["XOM"], "conviction": "high",
            },
        ],
        "stock_news": {},
        "pm_briefing": "Fed pause.",
        "market_sentiment": "neutral",
        "confidence": "medium",
    })
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_json)]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = NewsAnalystAgent(api_key="test", model="claude-sonnet-4-6")
    report, _ = agent.analyze(
        news_text="The Fed today signals a pause on further rate hikes. "
                  "No other material news.",
    )
    assert report is not None
    events = [sc.event for sc in report.state_changes]
    assert "Fed signals pause" in events
    assert "Iran ceasefire brokered" not in events


@patch("anthropic.Anthropic")
def test_news_analyst_bad_response(mock_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="I need more context...")]
    mock_response.usage.input_tokens = 1000
    mock_response.usage.output_tokens = 50
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = NewsAnalystAgent(api_key="test", model="claude-sonnet-4-6-20250514")
    analysis, agent_result = agent.analyze(news_text="Some news")

    assert analysis is None
    assert agent_result is not None
