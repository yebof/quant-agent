import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen, Request
from xml.etree import ElementTree

import feedparser

logger = logging.getLogger(__name__)

RSS_FEEDS = {
    # Financial / Markets
    "Reuters Business": "https://www.reutersagency.com/feed/?best-topics=business-finance",
    "CNBC Top News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "CNBC Economy": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "MarketWatch Top": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "MarketWatch Markets": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    # Macro / Policy / Politics
    "AP Business": "https://rsshub.app/apnews/topics/business",
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "NPR Economy": "https://feeds.npr.org/1017/rss.xml",
    # Fed / Treasury
    "Fed Press Releases": "https://www.federalreserve.gov/feeds/press_all.xml",
}

USER_AGENT = "Mozilla/5.0 (quant-agent/0.1)"
FETCH_TIMEOUT = 10


@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    published: datetime | None
    link: str


class NewsDataProvider:
    def __init__(self, feeds: dict[str, str] | None = None, lookback_hours: int = 24):
        self.feeds = feeds or RSS_FEEDS
        self.lookback_hours = lookback_hours

    def fetch_news(self) -> list[NewsItem]:
        """Fetch recent news from all RSS feeds."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        all_items: list[NewsItem] = []

        for source_name, url in self.feeds.items():
            try:
                items = self._fetch_feed(source_name, url, cutoff)
                all_items.extend(items)
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", source_name, e)

        # Deduplicate by title similarity and sort by time (newest first)
        deduped = self._deduplicate(all_items)
        deduped.sort(key=lambda x: x.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

        logger.info("Fetched %d news items from %d sources (after dedup from %d)",
                     len(deduped), len(self.feeds), len(all_items))
        return deduped

    def _fetch_feed(self, source_name: str, url: str, cutoff: datetime) -> list[NewsItem]:
        """Fetch and parse a single RSS feed."""
        feed = feedparser.parse(url, agent=USER_AGENT)

        if feed.bozo and not feed.entries:
            logger.warning("Feed %s returned no entries: %s", source_name, feed.bozo_exception)
            return []

        items = []
        for entry in feed.entries:
            published = self._parse_date(entry)
            if published and published < cutoff:
                continue

            title = entry.get("title", "").strip()
            if not title:
                continue

            summary = entry.get("summary", entry.get("description", "")).strip()
            # Truncate long summaries
            if len(summary) > 300:
                summary = summary[:297] + "..."

            items.append(NewsItem(
                title=title,
                summary=summary,
                source=source_name,
                published=published,
                link=entry.get("link", ""),
            ))

        return items

    def _parse_date(self, entry) -> datetime | None:
        """Parse the published date from a feed entry."""
        parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        if parsed:
            try:
                from calendar import timegm
                ts = timegm(parsed)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, OverflowError):
                return None
        return None

    def _deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        """Remove near-duplicate headlines using word-level Jaccard similarity."""
        unique: list[NewsItem] = []
        seen_word_sets: list[set[str]] = []

        for item in items:
            words = set(item.title.lower().split())
            if not words:
                continue
            is_dup = False
            for seen in seen_word_sets:
                intersection = len(words & seen)
                union = len(words | seen)
                if union > 0 and intersection / union > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                seen_word_sets.append(words)
                unique.append(item)

        return unique

    def tag_symbol_mentions(self, items: list[NewsItem], universe: list[str]) -> dict[str, list[NewsItem]]:
        """Tag which news items mention symbols from the universe. Returns {symbol: [items]}."""
        # Build lookup: company names / common aliases
        symbol_set = {s.upper() for s in universe}
        result: dict[str, list[NewsItem]] = {}
        for item in items:
            text = f"{item.title} {item.summary}".upper()
            for sym in symbol_set:
                if sym in text:
                    result.setdefault(sym, []).append(item)
        return result

    def format_for_prompt(self, items: list[NewsItem], max_items: int = 50) -> str:
        """Format news items into a text block for the LLM prompt."""
        if not items:
            return "No recent news available."

        limited = items[:max_items]
        lines = []
        for item in limited:
            time_str = item.published.strftime("%Y-%m-%d %H:%M UTC") if item.published else "unknown"
            lines.append(f"[{item.source}] ({time_str}) {item.title}")
            if item.summary:
                lines.append(f"  > {item.summary}")

        return "\n".join(lines)
