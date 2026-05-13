import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen, Request
from xml.etree import ElementTree

import feedparser

from src.trading_calendar import et_now

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

    def fetch_news(self, lookback_hours_override: int | None = None) -> list[NewsItem]:
        """Fetch recent news from all RSS feeds.

        Default lookback is 24h, fine for Tue-Fri morning runs. On Monday
        morning the previous trading day was Friday, so a 24h window
        misses ~72h of weekend news (Fed pressers, geopolitical events,
        earnings pre-announcements all routinely land on weekends). The
        Monday-aware path: if today is Monday, automatically extend the
        lookback to cover the gap. The caller can also override via
        `lookback_hours_override` for hand-tuning / replay scenarios.
        """
        if lookback_hours_override is not None:
            effective_lookback = lookback_hours_override
        else:
            today = et_now()
            # weekday(): Monday=0 .. Sunday=6. Monday morning needs to
            # cover Fri close → Mon morning ≈ 72h. Tue after a Mon
            # holiday would also benefit but holiday awareness lives in
            # broker.is_trading_day; that's overkill here — Monday is
            # the 95% case.
            if today.weekday() == 0:  # Monday
                effective_lookback = max(self.lookback_hours, 72)
                logger.info(
                    "fetch_news: Monday detected — extending lookback "
                    "from %dh to %dh to cover weekend news",
                    self.lookback_hours, effective_lookback,
                )
            else:
                effective_lookback = self.lookback_hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=effective_lookback)
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
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=FETCH_TIMEOUT) as resp:
                raw = resp.read()
        except Exception as e:
            logger.warning("Feed %s fetch failed: %s", source_name, e)
            return []

        feed = feedparser.parse(raw)

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

    @staticmethod
    def _normalize_link(link: str) -> str:
        """Normalize an article URL for cross-source dedup.

        Strip query parameters (utm_*, ?ref=, ?source=) and fragments,
        lowercase the host, drop trailing slashes. Same article syndicated
        across Reuters / CNBC / AP carries a distinct URL per-outlet, BUT
        many outlets republish from the same wire source (AP / Reuters
        feed) with identical underlying URLs differing only in tracking
        params. Stripping those catches the exact-duplicate case before
        the noisier Jaccard pass.
        """
        if not link:
            return ""
        try:
            from urllib.parse import urlsplit, urlunsplit
            parts = urlsplit(link)
            host = (parts.netloc or "").lower()
            path = (parts.path or "").rstrip("/")
            # Drop query (tracking params) and fragment entirely. If two
            # articles legitimately differ only by a query param, we'd
            # rather lose one than have both pollute the prompt.
            return urlunsplit((parts.scheme.lower(), host, path, "", ""))
        except Exception:
            return link.strip().lower()

    def _deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        """Remove duplicates: first by normalized URL (catches exact
        cross-source republishes), then by word-level Jaccard on title
        (catches near-duplicates with different URLs).

        Word-Jaccard alone misses URL-identical duplicates because the
        Reuters / CNBC / AP boilerplate dilutes title intersection below
        the 0.7 threshold (e.g., "Stocks rise on Fed pause" vs "Markets
        rally as Fed signals pause" both link to the same AP wire URL
        but score Jaccard ~0.3). URL pre-dedup catches these cheaply.
        """
        # Pass 1: URL dedup
        url_deduped: list[NewsItem] = []
        seen_urls: set[str] = set()
        for item in items:
            key = self._normalize_link(item.link)
            if key and key in seen_urls:
                continue
            if key:
                seen_urls.add(key)
            url_deduped.append(item)

        # Pass 2: word-Jaccard on title for near-duplicates
        unique: list[NewsItem] = []
        seen_word_sets: list[set[str]] = []
        for item in url_deduped:
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
        """Tag which news items mention symbols from the universe. Uses word-boundary matching."""
        import re
        # Short symbols (1-3 chars) are prone to false positives; require word boundaries
        patterns: dict[str, re.Pattern] = {}
        for s in universe:
            sym = s.upper()
            patterns[sym] = re.compile(r'\b' + re.escape(sym) + r'\b')
        result: dict[str, list[NewsItem]] = {}
        for item in items:
            text = f"{item.title} {item.summary}".upper()
            for sym, pat in patterns.items():
                if pat.search(text):
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
