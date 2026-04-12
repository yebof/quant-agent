import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import pytest

from src.data.news import NewsDataProvider, NewsItem
from src.agents.news_analyst import NewsAnalystAgent
from src.models import NewsAnalysisResult


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

@patch("src.agents.base.Anthropic")
def test_news_analyst_analyze(mock_cls):
    response_json = json.dumps({
        "market_sentiment": "bullish",
        "confidence": "medium",
        "key_events": [
            {
                "headline": "Fed signals pause",
                "impact": "high",
                "affected_sectors": ["Financial"],
                "affected_symbols": ["JPM"],
                "sentiment": "bullish",
                "explanation": "Lower rates support equities",
            }
        ],
        "sector_impacts": [
            {"sector": "Technology", "sentiment": "bullish", "reason": "AI spending continues"}
        ],
        "symbol_alerts": [
            {"symbol": "NVDA", "sentiment": "bullish", "reason": "New chip announcement"}
        ],
        "summary": "Market tone is bullish driven by Fed dovishness.",
    })

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_json)]
    mock_response.usage.input_tokens = 2000
    mock_response.usage.output_tokens = 500
    mock_client.messages.create.return_value = mock_response
    mock_cls.return_value = mock_client

    agent = NewsAnalystAgent(api_key="test", model="claude-sonnet-4-6-20250514")
    analysis, agent_result = agent.analyze(
        news_text="Fed signals pause in rate hikes...",
        universe=["SPY", "NVDA", "JPM"],
    )

    assert analysis is not None
    assert analysis.market_sentiment == "bullish"
    assert analysis.confidence == "medium"
    assert len(analysis.key_events) == 1
    assert analysis.key_events[0].impact == "high"
    assert len(analysis.symbol_alerts) == 1
    assert analysis.symbol_alerts[0].symbol == "NVDA"
    assert agent_result.tokens_used == 2500


@patch("src.agents.base.Anthropic")
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
