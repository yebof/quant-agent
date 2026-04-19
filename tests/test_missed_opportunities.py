"""Missed-opportunity digest — Phase 1 of the evening-agent upgrade.

Covers TradingPipeline._build_missed_opportunities_digest and its helpers.
Focused on the deterministic Python layer (market data + DB + news_store
scans); the LLM output half (MissedOpportunity classification) is covered
by model validators and the evening-prompt integration test.

Core invariants:
  - Filter by |move_pct| ≥ threshold
  - Universe ∪ Alpaca top movers
  - Source correctly tagged (universe / top_mover / both)
  - held-during-window excludes symbols we already own or traded
  - Priority sort: not-held+signal → not-held+no-signal → held
  - Graceful degradation on every data source failing
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import date

from src.pipeline import TradingPipeline, _missed_ops_quality_metrics


def _mk_ohlcv(sym_close_pairs: list[tuple[str, float]], volume: float = 0):
    """Helper: build the list[OHLCV] shape TradingPipeline expects from market.

    Volume defaults to 0 so quality-metric helpers see "no data" (None)
    instead of being fooled by MagicMock's __float__=1.0 leakage. Tests
    that care about quality metrics pass an explicit volume.
    """
    bars = []
    for i, (_sym, close) in enumerate(sym_close_pairs):
        b = MagicMock()
        b.close = close
        b.volume = volume
        b.date = date(2026, 4, 14 + i)
        bars.append(b)
    return bars


def _pipeline_with(
    *,
    universe: list[str] | None = None,
    top_movers: list[dict] | None = None,
    market_closes_by_symbol: dict[str, list[float]] | None = None,
    trades: list[dict] | None = None,
    tech_rows: list[dict] | None = None,
    news_dir_path: Path | None = None,
    earnings_manifest: dict | None = None,
    macro_state: dict | None = None,
):
    """Construct a skeleton TradingPipeline with just the dependencies
    _build_missed_opportunities_digest touches. All other TradingPipeline
    attributes are absent — any accidental access will AttributeError the
    test, which is what we want."""
    p = TradingPipeline.__new__(TradingPipeline)

    # Config — only the universe is read.
    p.config = MagicMock()
    p.config.trading.universe = universe or []

    # Broker — only get_top_movers.
    p.broker = MagicMock()
    p.broker.get_top_movers.return_value = top_movers or []

    # Market — per-symbol get_ohlcv.
    p.market = MagicMock()
    def _ohlcv(symbol: str, lookback_days: int = 10):
        closes = (market_closes_by_symbol or {}).get(symbol)
        if not closes:
            return []
        return _mk_ohlcv([(symbol, c) for c in closes])
    p.market.get_ohlcv.side_effect = _ohlcv

    # DB — only the two calls made by missed_ops helpers.
    p.db = MagicMock()
    p.db.get_trades.return_value = trades or []
    p.db.get_recent_agent_outputs.return_value = tech_rows or []

    # News store — get_missed_ops reads files under data_dir.
    p.news_store = MagicMock()
    p.news_store.data_dir = news_dir_path or Path("/tmp/does-not-exist-missed-ops-test")

    # Earnings + macro stores.
    p.earnings_provider = MagicMock()
    p.earnings_provider.manifest = earnings_manifest or {}
    p.macro_store = MagicMock()
    p.macro_store.load_last_state.return_value = macro_state

    return p


# ---------------------------------------------------------------------------
# Core digest behavior
# ---------------------------------------------------------------------------

@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_filters_below_threshold(_sec):
    """Symbols that moved less than move_threshold_pct (abs) must not appear.
    Guards against flooding evening with noise — 1% wiggle isn't a miss."""
    p = _pipeline_with(
        universe=["A", "B", "C"],
        market_closes_by_symbol={
            "A": [100, 101, 102, 103, 104, 105],   # +5% — below 8%
            "B": [100, 103, 106, 109, 112, 115],   # +15% — kept
            "C": [100, 100.5, 101, 100.7, 100.2, 100.6],  # ~flat — dropped
        },
    )
    out = p._build_missed_opportunities_digest(
        lookback_days=5, move_threshold_pct=8.0,
    )
    syms = [s.symbol for s in out]
    assert syms == ["B"]
    assert abs(out[0].move_pct - 15.0) < 0.01


@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_merges_universe_and_top_movers_with_source_tags(_sec):
    """Symbol in both sets tagged 'both'; only in one → that one. Sourcing
    is what tells the meta-reflector whether our universe covers the market."""
    p = _pipeline_with(
        universe=["A", "B"],
        top_movers=[
            {"symbol": "B", "percent_change": 18.0, "price": 50.0},   # both
            {"symbol": "Z", "percent_change": 20.0, "price": 80.0},   # top_mover only
        ],
        market_closes_by_symbol={
            "A": [100, 105, 108, 110, 113, 118],   # +18%
            "B": [50, 55, 57, 59, 60, 62],         # +24%
            "Z": [80, 84, 88, 90, 93, 96],         # +20%
        },
    )
    out = p._build_missed_opportunities_digest(
        lookback_days=5, move_threshold_pct=8.0,
    )
    by_sym = {s.symbol: s for s in out}
    assert by_sym["A"].source == "universe"
    assert by_sym["B"].source == "both"
    assert by_sym["Z"].source == "top_mover"


@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_marks_held_when_current_position(_sec):
    """Symbols in `current_position_symbols` must be flagged
    held_during_window=True, regardless of trade history."""
    p = _pipeline_with(
        universe=["HELD", "NOTHELD"],
        market_closes_by_symbol={
            "HELD":    [100, 105, 108, 110, 113, 118],
            "NOTHELD": [100, 105, 108, 110, 113, 118],
        },
    )
    out = p._build_missed_opportunities_digest(
        lookback_days=5, move_threshold_pct=8.0,
        current_position_symbols={"HELD"},
    )
    by = {s.symbol: s for s in out}
    assert by["HELD"].held_during_window is True
    assert by["NOTHELD"].held_during_window is False


@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_marks_held_when_recent_trade(_sec):
    """A symbol with an executed BUY in the last `lookback_days * 2 + 2`
    calendar days counts as held — we don't call "we missed it" if we
    actually traded it recently."""
    from src.trading_calendar import et_today
    recent_ts = et_today().isoformat() + " 10:30:00"
    p = _pipeline_with(
        universe=["R", "X"],
        market_closes_by_symbol={
            "R": [100, 105, 108, 110, 113, 118],
            "X": [100, 105, 108, 110, 113, 118],
        },
        trades=[
            {"symbol": "R", "timestamp": recent_ts, "action": "BUY",
             "fill_status": "filled"},
        ],
    )
    out = p._build_missed_opportunities_digest(
        lookback_days=5, move_threshold_pct=8.0,
    )
    by = {s.symbol: s for s in out}
    assert by["R"].held_during_window is True
    assert by["X"].held_during_window is False


@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_sort_priority_not_held_signal_first(_sec):
    """Order matters — evening LLM reads top of list first. Real 'we saw it
    didn't act' misses (not-held + signal) beat raw 'theme blindspot'
    (not-held + no-signal), which beat 'held for context' (we own it)."""
    import json as _json
    tech_json = _json.dumps({
        "analyses": [
            {"symbol": "SIGNAL_NOTHELD", "rating": "buy"},
            {"symbol": "HELD_SYM",       "rating": "buy"},
        ]
    })
    from src.trading_calendar import et_today
    today_iso = et_today().isoformat()
    p = _pipeline_with(
        universe=["SIGNAL_NOTHELD", "BLIND_NOTHELD", "HELD_SYM"],
        market_closes_by_symbol={
            "SIGNAL_NOTHELD": [100, 104, 107, 109, 112, 115],  # +15%
            "BLIND_NOTHELD":  [100, 105, 110, 115, 120, 125],  # +25% (biggest)
            "HELD_SYM":       [100, 108, 115, 118, 122, 130],  # +30% (biggest)
        },
        tech_rows=[
            {"timestamp": today_iso + " 09:35:00",
             "full_response": tech_json},
        ],
    )
    out = p._build_missed_opportunities_digest(
        lookback_days=5, move_threshold_pct=8.0,
        current_position_symbols={"HELD_SYM"},
    )
    order = [s.symbol for s in out]
    assert order[0] == "SIGNAL_NOTHELD", (
        f"not-held+signal should win regardless of move size; got {order}"
    )
    assert order[-1] == "HELD_SYM", (
        f"held should sink to bottom; got {order}"
    )


@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_top_n_caps_result(_sec):
    """Even with many candidates, digest caps at top_n — otherwise LLM
    prompt balloons."""
    syms = [f"S{i}" for i in range(30)]
    closes_by_sym = {
        s: [100] + [100 * (1 + 0.03 * (i + 1)) for i in range(5)]
        for s in syms
    }
    p = _pipeline_with(
        universe=syms,
        market_closes_by_symbol=closes_by_sym,
    )
    out = p._build_missed_opportunities_digest(
        lookback_days=5, move_threshold_pct=8.0, top_n=15,
    )
    assert len(out) == 15


@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_empty_when_no_universe_and_no_top_movers(_sec):
    """Graceful NOP when there's nothing to scan."""
    p = _pipeline_with(universe=[], top_movers=[])
    out = p._build_missed_opportunities_digest()
    assert out == []


@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_survives_top_movers_api_failure(_sec):
    """Broker screener outage must NOT crash evening. Fall back to
    universe-only and carry on."""
    p = _pipeline_with(
        universe=["A"],
        market_closes_by_symbol={
            "A": [100, 105, 108, 110, 113, 118],  # +18%
        },
    )
    p.broker.get_top_movers.side_effect = RuntimeError("screener 500")
    out = p._build_missed_opportunities_digest(move_threshold_pct=8.0)
    assert len(out) == 1
    assert out[0].symbol == "A"


# ---------------------------------------------------------------------------
# Signal-tag enrichment
# ---------------------------------------------------------------------------

@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_tech_signal_flag_handles_bare_list_shape(_sec):
    """Regression: production tech_analyst full_response is sometimes a
    BARE LIST (not ``{"analyses": [...]}``). `_missed_ops_tech_signal`
    must delegate to the same shape normalizer the quarterly digest uses,
    otherwise it silently treats every symbol as having no TA signal."""
    import json as _json
    from src.trading_calendar import et_today
    today_iso = et_today().isoformat()

    p = _pipeline_with(
        universe=["NVDA", "HOLDSYM"],
        market_closes_by_symbol={
            "NVDA":    [100, 105, 108, 110, 113, 118],  # +18%
            "HOLDSYM": [100, 105, 108, 110, 113, 118],
        },
        tech_rows=[
            # BARE LIST — matches the production shape observed 2026-04-19
            {"timestamp": today_iso + " 09:35:00",
             "full_response": _json.dumps([
                 {"symbol": "NVDA", "rating": "buy"},
                 {"symbol": "HOLDSYM", "rating": "hold"},
             ])},
        ],
    )
    out = p._build_missed_opportunities_digest(
        lookback_days=5, move_threshold_pct=8.0,
    )
    by = {s.symbol: s for s in out}
    assert by["NVDA"].had_ta_signal is True
    assert by["NVDA"].last_ta_rating == "buy"
    assert by["HOLDSYM"].had_ta_signal is False
    assert by["HOLDSYM"].last_ta_rating == "hold"


@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_populates_ta_signal_flag_from_recent_logs(_sec):
    """had_ta_signal = True when most recent tech_analyst rating in window
    is 'buy' or 'strong_buy'. Any other rating → False."""
    import json as _json
    from src.trading_calendar import et_today
    today_iso = et_today().isoformat()
    tech_json = _json.dumps({
        "analyses": [
            {"symbol": "BUY_SIG",   "rating": "buy"},
            {"symbol": "HOLD_SIG",  "rating": "hold"},
            {"symbol": "SELL_SIG",  "rating": "sell"},
        ]
    })
    p = _pipeline_with(
        universe=["BUY_SIG", "HOLD_SIG", "SELL_SIG"],
        market_closes_by_symbol={
            "BUY_SIG":  [100, 105, 108, 110, 113, 118],
            "HOLD_SIG": [100, 105, 108, 110, 113, 118],
            "SELL_SIG": [100, 105, 108, 110, 113, 118],
        },
        tech_rows=[
            {"timestamp": today_iso + " 09:35:00",
             "full_response": tech_json},
        ],
    )
    out = p._build_missed_opportunities_digest(
        lookback_days=5, move_threshold_pct=8.0,
    )
    by = {s.symbol: s for s in out}
    assert by["BUY_SIG"].had_ta_signal is True
    assert by["BUY_SIG"].last_ta_rating == "buy"
    assert by["HOLD_SIG"].had_ta_signal is False
    assert by["HOLD_SIG"].last_ta_rating == "hold"
    assert by["SELL_SIG"].had_ta_signal is False


@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_populates_news_signal_from_state_changes(_sec, tmp_path):
    """News signal harvested from state_changes.affected_symbols across all
    dated full_report.json files in the lookback window."""
    import json as _json
    from src.trading_calendar import et_today
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    day_dir = news_dir / str(et_today())
    day_dir.mkdir()
    (day_dir / "full_report.json").write_text(_json.dumps({
        "state_changes": [{
            "event": "AI capex cycle accelerating",
            "affected_symbols": ["AI_SYM"],
        }],
        "stock_news": {
            "STOCK_NEWS_SYM": [{"headline": "Q1 beat estimates"}],
        },
    }))

    p = _pipeline_with(
        universe=["AI_SYM", "STOCK_NEWS_SYM", "NO_NEWS_SYM"],
        market_closes_by_symbol={
            "AI_SYM":         [100, 105, 108, 110, 113, 118],
            "STOCK_NEWS_SYM": [100, 105, 108, 110, 113, 118],
            "NO_NEWS_SYM":    [100, 105, 108, 110, 113, 118],
        },
        news_dir_path=news_dir,
    )
    out = p._build_missed_opportunities_digest(
        lookback_days=5, move_threshold_pct=8.0,
    )
    by = {s.symbol: s for s in out}
    assert by["AI_SYM"].had_news_signal is True
    assert "AI capex" in (by["AI_SYM"].last_news_headline or "")
    assert by["STOCK_NEWS_SYM"].had_news_signal is True
    assert by["NO_NEWS_SYM"].had_news_signal is False


@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_macro_sector_tailwind_from_stored_state(_sec):
    """macro_sector_tailwind populated when macro_store has a stance for
    the symbol's sector; "unknown" otherwise (which is itself a coverage
    signal for the meta-reflector)."""
    p = _pipeline_with(
        universe=["COVERED"],
        market_closes_by_symbol={
            "COVERED": [100, 105, 108, 110, 113, 118],
        },
        macro_state={
            "sector_guidance": {"Technology": "bullish"},
        },
    )
    out = p._build_missed_opportunities_digest(
        lookback_days=5, move_threshold_pct=8.0,
    )
    assert out[0].macro_sector_tailwind == "bullish"


@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_macro_sector_tailwind_defaults_unknown(_sec):
    """When macro store has no guidance for the sector, tailwind = unknown."""
    p = _pipeline_with(
        universe=["X"],
        market_closes_by_symbol={"X": [100, 105, 108, 110, 113, 118]},
        macro_state={"sector_guidance": {"Energy": "bullish"}},
    )
    out = p._build_missed_opportunities_digest(
        lookback_days=5, move_threshold_pct=8.0,
    )
    assert out[0].macro_sector_tailwind == "unknown"


# ---------------------------------------------------------------------------
# Persistence to insights.missed_opportunities_json
# ---------------------------------------------------------------------------

def test_save_evening_snapshot_persists_missed_opportunities(tmp_path):
    """Round-trip: MissedOpportunity pydantic → json column → dict back."""
    from src.models import MissedOpportunity
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    mos = [
        MissedOpportunity(
            symbol="VST", move_pct=22.3,
            miss_category="theme_blindspot", theme_if_any="nuclear/power",
            lesson="Nuclear theme never entered news tracker; add coverage",
        ),
        MissedOpportunity(
            symbol="OKLO", move_pct=18.7,
            miss_category="trend_timing_miss", theme_if_any="nuclear/power",
            lesson="News flagged capex 9d ago, PM never sized in",
        ),
    ]
    db.save_evening_snapshot(
        date="2026-04-20", total_value=100_000, daily_pnl=200,
        daily_return_pct=0.2,
        tomorrow_outlook="x", lessons="y", suggested_actions=[],
        risk_rating="low", tomorrow_bias="neutral",
        tomorrow_conviction="medium", tomorrow_key_risks=[],
        sell_decisions_assessment="",
        missed_opportunities=mos,
    )
    row = db.get_latest_insights(before_date="2026-04-21")
    assert row is not None
    import json
    persisted = json.loads(row["missed_opportunities_json"])
    assert len(persisted) == 2
    assert persisted[0]["symbol"] == "VST"
    assert persisted[0]["miss_category"] == "theme_blindspot"
    assert persisted[1]["theme_if_any"] == "nuclear/power"


def test_save_evening_snapshot_missed_opportunities_optional(tmp_path):
    """Phase-1 backward compat: omitting `missed_opportunities` must still
    write '[]' rather than NULL so downstream readers don't have to special-
    case missing-column rows that were written post-migration."""
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()

    db.save_evening_snapshot(
        date="2026-04-20", total_value=100_000, daily_pnl=0,
        daily_return_pct=0.0,
        tomorrow_outlook="x", lessons="y", suggested_actions=[],
        risk_rating="low", tomorrow_bias="neutral",
        tomorrow_conviction="medium", tomorrow_key_risks=[],
        sell_decisions_assessment="",
    )
    row = db.get_latest_insights(before_date="2026-04-21")
    assert row is not None
    assert row["missed_opportunities_json"] == "[]"


def test_legacy_insights_row_missing_missed_opportunities_column(tmp_path):
    """Rows written before the migration on this DB should come back with
    NULL or '[]'; downstream readers must treat both as empty list without
    crashing."""
    import json as _json
    from src.storage.db import Database

    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    # Simulate a pre-migration row by inserting directly without the new col.
    db.conn.execute(
        "INSERT INTO insights (date, tomorrow_outlook, lessons, risk_rating) "
        "VALUES (?, ?, ?, ?)",
        ("2026-04-10", "legacy", "legacy lessons", "low"),
    )
    db.conn.commit()
    row = db.get_latest_insights(before_date="2026-04-11")
    assert row is not None
    # Migration's DEFAULT '[]' makes the read path return '[]' even for
    # rows that were inserted before the column existed in this session.
    val = row.get("missed_opportunities_json")
    assert val in ("[]", None)
    parsed = _json.loads(val) if val else []
    assert parsed == []


# ---------------------------------------------------------------------------
# Evening prompt rendering
# ---------------------------------------------------------------------------

def _make_evening_agent():
    from unittest.mock import patch as _patch
    from src.agents.evening_analyst import EveningAnalystAgent
    with _patch("anthropic.Anthropic"):
        return EveningAnalystAgent(api_key="k", model="claude-opus-4-6")


def _base_evening_kwargs():
    return dict(
        positions=[], macro_summary={"vix": {"current": 18}},
        total_value=100_000, daily_pnl=0, daily_return_pct=0.0,
        today_trades=[], prior_outlook=None,
        recent_sells=[], recent_buys=[],
        news_intel=None, earnings_analyses=[],
        weekly_narrative="", active_state_changes="",
        outlook_calibration={},
    )


def test_evening_prompt_renders_missed_ops_section_with_snapshots():
    """When missed_ops_snapshots has entries, the prompt must show each
    symbol with its source, move%, held flag, TA/news/earnings/macro
    context. The LLM uses these facts to classify miss_category."""
    from src.models import MissedOpportunitySnapshot
    agent = _make_evening_agent()

    snap = MissedOpportunitySnapshot(
        symbol="VST", move_pct=22.3, window_days=5,
        held_during_window=False, had_ta_signal=False,
        had_news_signal=True, had_earnings_signal=False,
        source="top_mover",
        last_ta_rating=None, last_ta_date=None,
        last_news_headline="Nuclear capex thesis accelerating",
        theme_tags=["nuclear-power"],
        recent_earnings_signal=None,
        macro_sector_tailwind="unknown",
    )
    kwargs = _base_evening_kwargs()
    kwargs["missed_ops_snapshots"] = [snap]
    msg = agent.build_user_message(**kwargs)

    assert "Missed Opportunity Review" in msg
    assert "VST" in msg
    assert "top_mover" in msg
    assert "+22.3%" in msg or "22.3%" in msg
    assert "not held" in msg
    assert "Nuclear capex" in msg
    # macro_sector_tailwind = "unknown" is itself a signal (coverage gap)
    assert "unknown" in msg


def test_evening_prompt_notes_empty_missed_ops():
    """Empty digest → explicit instruction to emit `missed_opportunities: []`.
    Prevents LLM from hallucinating rows when there's nothing to classify."""
    agent = _make_evening_agent()
    kwargs = _base_evening_kwargs()
    kwargs["missed_ops_snapshots"] = []
    msg = agent.build_user_message(**kwargs)
    assert "Missed Opportunity Review" in msg
    assert "missed_opportunities: []" in msg


def test_evening_prompt_renders_market_relative_move_on_recent_buys():
    """The SPY benchmark per-BUY lets the LLM distinguish alpha-destruction
    from systemic_drawdown without guessing."""
    agent = _make_evening_agent()
    kwargs = _base_evening_kwargs()
    kwargs["recent_buys"] = [{
        "symbol": "MU", "buy_date": "2026-04-15", "buy_price": 100,
        "current_price": 85, "pct_move_since_buy": -15.0,
        "market_relative_move_pct": -14.5,  # SPY ~ -0.5%, we fell 15% → alpha destruction
        "reasoning": "chasing memory cycle",
    }]
    msg = agent.build_user_message(**kwargs)
    assert "MU" in msg
    assert "vs SPY:" in msg
    assert "-14.50%" in msg or "-14.5" in msg


def test_evening_prompt_omits_spy_tag_when_relative_not_available():
    """market_relative_move_pct=None (SPY fetch failed or no bars) → no
    benchmark tag rendered; the row still appears with its own move."""
    agent = _make_evening_agent()
    kwargs = _base_evening_kwargs()
    kwargs["recent_buys"] = [{
        "symbol": "MU", "buy_date": "2026-04-15", "buy_price": 100,
        "current_price": 85, "pct_move_since_buy": -15.0,
        "market_relative_move_pct": None,
        "reasoning": "chasing memory cycle",
    }]
    msg = agent.build_user_message(**kwargs)
    assert "MU" in msg
    assert "vs SPY:" not in msg


# ---------------------------------------------------------------------------
# _build_recent_buys_for_grading injects market_relative_move_pct
# ---------------------------------------------------------------------------

def test_recent_buys_injects_spy_relative_move(tmp_path):
    """Regression: the helper fetches SPY bars once and tags each BUY with
    its (our_return - SPY_return) delta, so evening LLM can classify
    losing BUYs as alpha-destruction vs systemic."""
    from datetime import date as _date
    from unittest.mock import MagicMock
    from src.pipeline import TradingPipeline
    from src.storage.db import Database

    p = TradingPipeline.__new__(TradingPipeline)
    p.db = Database(str(tmp_path / "t.db"))
    p.db.initialize()
    # Insert one executed BUY from ~3 days ago with reasonable buy_date.
    from src.trading_calendar import et_today
    from datetime import timedelta
    buy_d = et_today() - timedelta(days=3)
    p.db.conn.execute(
        "INSERT INTO trades (symbol, action, qty, price, reasoning, "
        "run_id, fill_status, fill_qty, fill_price, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("MU", "BUY", 10, 100.0, "test buy", "r1", "filled", 10, 100.0,
         f"{buy_d.isoformat()} 09:35:00"),
    )
    p.db.conn.commit()

    # Market returns: MU current $85 (our -15%), SPY series flat-ish
    # ($100 at buy → $100.5 today = +0.5% SPY).
    p.broker = MagicMock()
    p.broker.get_latest_price.return_value = 85.0

    p.market = MagicMock()
    def _ohlcv(symbol, lookback_days=12):
        def _bar(d, close):
            b = MagicMock(); b.date = d; b.close = close; return b
        if symbol == "SPY":
            # oldest → newest bars; buy_date match or nearby
            return [
                _bar(et_today() - timedelta(days=10), 99.0),
                _bar(et_today() - timedelta(days=5),  99.5),
                _bar(buy_d,                            100.0),
                _bar(et_today() - timedelta(days=2),  100.3),
                _bar(et_today(),                      100.5),
            ]
        return []
    p.market.get_ohlcv.side_effect = _ohlcv

    out = p._build_recent_buys_for_grading(lookback_days=5)
    assert len(out) == 1
    row = out[0]
    assert row["symbol"] == "MU"
    assert row["pct_move_since_buy"] == -15.0
    # SPY went 100 → 100.5 = +0.5%; our relative = -15.0 - 0.5 = -15.5
    assert row["market_relative_move_pct"] is not None
    assert abs(row["market_relative_move_pct"] - (-15.5)) < 0.1


# ---------------------------------------------------------------------------
# PM L3d — _build_recent_missed_lessons
# ---------------------------------------------------------------------------

def _pipeline_with_insights_rows(rows: list[dict]):
    from src.pipeline import TradingPipeline
    p = TradingPipeline.__new__(TradingPipeline)
    p.db = MagicMock()
    p.db.get_recent_insights.return_value = rows
    return p


def test_recent_missed_lessons_surfaces_themes_seen_2plus_days():
    """A theme flagged on ≥ 2 distinct days counts as recurring and
    surfaces to PM's L3d. Single-day noise is filtered out."""
    import json
    p = _pipeline_with_insights_rows([
        {"date": "2026-04-18",
         "missed_opportunities_json": json.dumps([
             {"symbol": "VST", "move_pct": 22.3,
              "miss_category": "theme_blindspot",
              "theme_if_any": "nuclear/power",
              "lesson": "nuclear capex theme never entered news tracker"},
         ])},
        {"date": "2026-04-17",
         "missed_opportunities_json": json.dumps([
             {"symbol": "OKLO", "move_pct": 18.7,
              "miss_category": "trend_timing_miss",
              "theme_if_any": "nuclear/power",
              "lesson": "news flagged capex 9d ago, PM never sized"},
             # One-day only — should NOT surface:
             {"symbol": "TSLA", "move_pct": 11.0,
              "miss_category": "trend_timing_miss",
              "theme_if_any": "EV",
              "lesson": "single day blip"},
         ])},
    ])
    out = p._build_recent_missed_lessons(lookback_days=14)
    assert "nuclear/power" in out
    assert "VST" in out and "OKLO" in out
    # EV appeared once → filtered out
    assert "EV" not in out or "TSLA" not in out
    # Latest lesson should be rendered (newest-first → 2026-04-18 wins)
    assert "nuclear capex theme never entered" in out


def test_recent_missed_lessons_ignores_escape_hatch_categories():
    """noise_rally and risk_disciplined aren't real misses — even if they
    repeat across days they shouldn't pollute PM's memory."""
    import json
    p = _pipeline_with_insights_rows([
        {"date": "2026-04-18",
         "missed_opportunities_json": json.dumps([
             {"symbol": "X", "move_pct": 9.0,
              "miss_category": "noise_rally",
              "lesson": "no signal, legitimate skip"},
         ])},
        {"date": "2026-04-17",
         "missed_opportunities_json": json.dumps([
             {"symbol": "X", "move_pct": 9.0,
              "miss_category": "noise_rally",
              "lesson": "no signal day 2"},
         ])},
    ])
    assert p._build_recent_missed_lessons(lookback_days=14) == ""


def test_recent_missed_lessons_empty_on_no_insights():
    p = _pipeline_with_insights_rows([])
    assert p._build_recent_missed_lessons(lookback_days=14) == ""


def test_recent_missed_lessons_malformed_json_gracefully_ignored():
    p = _pipeline_with_insights_rows([
        {"date": "2026-04-18",
         "missed_opportunities_json": "{ not json"},
    ])
    # No crash, just no output.
    assert p._build_recent_missed_lessons(lookback_days=14) == ""


# ---------------------------------------------------------------------------
# PM L3f — _build_recent_loss_pits
# ---------------------------------------------------------------------------

def test_recent_loss_pits_surfaces_causes_occurring_2plus_times():
    """Two or more wrong BUYs with the same loss_root_cause → repeat
    failure pattern worth showing PM before today's sizing."""
    import json
    p = _pipeline_with_insights_rows([
        {"date": "2026-04-18",
         "buy_grades_json": json.dumps([
             {"symbol": "MU", "grade": "wrong",
              "loss_root_cause": "greed_top_chasing",
              "pct_move_since_buy": -15.0},
         ])},
        {"date": "2026-04-17",
         "buy_grades_json": json.dumps([
             {"symbol": "NVDA", "grade": "wrong",
              "loss_root_cause": "greed_top_chasing",
              "pct_move_since_buy": -12.0},
             # Only once — not a pattern
             {"symbol": "ORCL", "grade": "wrong",
              "loss_root_cause": "timing_mistake",
              "pct_move_since_buy": -6.0},
         ])},
    ])
    out = p._build_recent_loss_pits(lookback_days=14)
    assert "greed_top_chasing × 2" in out
    assert "MU" in out and "NVDA" in out
    # Non-repeating causes filtered
    assert "timing_mistake" not in out


def test_recent_loss_pits_shows_missed_warning_ref_for_macro_ignored():
    """Most self-incriminating cause — `macro_warning_ignored` — should
    surface its cited evidence so PM sees what was ignored, not just
    that something was."""
    import json
    p = _pipeline_with_insights_rows([
        {"date": "2026-04-18",
         "buy_grades_json": json.dumps([
             {"symbol": "MU", "grade": "wrong",
              "loss_root_cause": "macro_warning_ignored",
              "pct_move_since_buy": -15.0,
              "missed_warning_ref": "news 2026-04-03 HIGH: spreads +80bps"},
         ])},
        {"date": "2026-04-17",
         "buy_grades_json": json.dumps([
             {"symbol": "STX", "grade": "wrong",
              "loss_root_cause": "macro_warning_ignored",
              "pct_move_since_buy": -10.0,
              "missed_warning_ref": "macro 2026-04-02 HIGH: VIX breakout"},
         ])},
    ])
    out = p._build_recent_loss_pits(lookback_days=14)
    assert "macro_warning_ignored × 2" in out
    assert "spreads +80bps" in out


def test_recent_loss_pits_ignores_correct_and_premature_grades():
    """Only `wrong` grades count — correct / premature aren't loss pits."""
    import json
    p = _pipeline_with_insights_rows([
        {"date": "2026-04-18",
         "buy_grades_json": json.dumps([
             {"symbol": "X", "grade": "correct",
              "loss_root_cause": None, "pct_move_since_buy": 7.0},
             {"symbol": "Y", "grade": "premature",
              "loss_root_cause": None, "pct_move_since_buy": -2.0},
         ])},
    ])
    assert p._build_recent_loss_pits(lookback_days=14) == ""


# ---------------------------------------------------------------------------
# PM prompt rendering L3d + L3f
# ---------------------------------------------------------------------------

def test_pm_prompt_renders_missed_lessons_and_loss_pits_sections():
    """PM build_user_message renders the two new memory sections when the
    helpers produce content; both sections include steering language PM
    should internalize, not just the raw facts."""
    from unittest.mock import patch as _patch
    from src.agents.portfolio_manager import PortfolioManagerAgent
    with _patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="k", model="claude-opus-4-6")

    msg = agent.build_user_message(
        analyses=[], positions=[], macro_analysis=None,
        cash_balance=10_000, total_value=100_000,
        recent_missed_lessons=(
            "- nuclear/power: 3 days (symbols: VST×2, OKLO) — latest lesson: "
            "\"News flagged capex 9d ago, PM never sized\""
        ),
        recent_loss_pits=(
            "- greed_top_chasing × 3: MU (-15.0%), NVDA (-12.0%), AVGO (-9.0%)"
        ),
    )
    assert "Recurring Missed Themes" in msg
    assert "nuclear/power" in msg
    assert "coverage or timing blind-spot" in msg
    assert "Recent Loss Pits" in msg
    assert "greed_top_chasing × 3" in msg
    assert "discipline gap, not bad luck" in msg


def test_pm_prompt_shows_default_when_no_missed_or_loss_history():
    """Fresh database / no recurring patterns → default NOP text in each
    section, not an empty stub."""
    from unittest.mock import patch as _patch
    from src.agents.portfolio_manager import PortfolioManagerAgent
    with _patch("anthropic.Anthropic"):
        agent = PortfolioManagerAgent(api_key="k", model="claude-opus-4-6")

    msg = agent.build_user_message(
        analyses=[], positions=[], macro_analysis=None,
        cash_balance=10_000, total_value=100_000,
        recent_missed_lessons="",
        recent_loss_pits="",
    )
    assert "no recurring missed themes" in msg
    assert "no repeat failure modes" in msg


def test_recent_buys_relative_move_none_when_spy_fetch_fails():
    """If SPY bars fetch throws, market_relative_move_pct falls back to
    None and the function still returns the BUY row without crashing."""
    import tempfile
    from datetime import timedelta
    from unittest.mock import MagicMock
    from src.pipeline import TradingPipeline
    from src.storage.db import Database
    from src.trading_calendar import et_today

    with tempfile.TemporaryDirectory() as tmp:
        p = TradingPipeline.__new__(TradingPipeline)
        p.db = Database(f"{tmp}/t.db")
        p.db.initialize()
        buy_d = et_today() - timedelta(days=2)
        p.db.conn.execute(
            "INSERT INTO trades (symbol, action, qty, price, reasoning, "
            "run_id, fill_status, fill_qty, fill_price, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("NVDA", "BUY", 5, 200.0, "test", "r1", "filled", 5, 200.0,
             f"{buy_d.isoformat()} 09:35:00"),
        )
        p.db.conn.commit()

        p.broker = MagicMock()
        p.broker.get_latest_price.return_value = 210.0
        p.market = MagicMock()
        p.market.get_ohlcv.side_effect = RuntimeError("market down")

        out = p._build_recent_buys_for_grading(lookback_days=5)
        assert len(out) == 1
        assert out[0]["symbol"] == "NVDA"
        assert out[0]["market_relative_move_pct"] is None


# ---------------------------------------------------------------------------
# Quality metrics helper — _missed_ops_quality_metrics
# ---------------------------------------------------------------------------

def _ohlcv_with_volume(closes, volumes):
    """Build list[OHLCV-like] mocks with explicit close + volume per bar."""
    assert len(closes) == len(volumes)
    bars = []
    for i, (c, v) in enumerate(zip(closes, volumes)):
        b = MagicMock()
        b.close = c
        b.volume = v
        b.date = date(2026, 4, 1 + i)
        bars.append(b)
    return bars


def test_quality_metrics_all_none_when_bars_empty():
    """Empty bars → every metric is None. Callers must handle None everywhere
    that quality metrics flow."""
    assert _missed_ops_quality_metrics([], 5) == (None, None, None)
    assert _missed_ops_quality_metrics([MagicMock()], 5) == (None, None, None)


def test_quality_metrics_dollar_volume_averaged_over_20_bars():
    """Given 25 bars with close ~ $100 and volume ~ 1M, avg dollar volume
    should be ~ $100M, rendered in millions as 100.0."""
    bars = _ohlcv_with_volume(
        closes=[100.0] * 25,
        volumes=[1_000_000] * 25,
    )
    avg_m, vol_conf, conc = _missed_ops_quality_metrics(bars, lookback_days=5)
    assert avg_m is not None and abs(avg_m - 100.0) < 0.1
    # Today's vol == avg → ratio == 1.0
    assert vol_conf is not None and abs(vol_conf - 1.0) < 0.01


def test_quality_metrics_volume_confirmation_spike():
    """Today's dollar volume elevated vs the trailing 20d average → vol_conf
    > 1.5 (the CONFIRMED threshold the prompt uses). The ratio is computed
    over the trailing 20 bars including today, so a 3x volume spike on
    the last bar yields a ratio of ~2.7 (3M today / 1.1M avg of 19 quiet
    + 1 spike day), not 3.0 flat."""
    closes = [100.0] * 21
    # First 20 bars at volume=1M, last bar at volume=3M.
    volumes = [1_000_000] * 20 + [3_000_000]
    bars = _ohlcv_with_volume(closes, volumes)
    _, vol_conf, _ = _missed_ops_quality_metrics(bars, lookback_days=5)
    # 3M / ((19*1M + 3M)/20) = 3M / 1.1M ≈ 2.73
    assert vol_conf is not None and vol_conf >= 1.5
    assert vol_conf < 3.0


def test_quality_metrics_single_day_concentration_gap():
    """One big day (+15%) with flat days around it → 1d concentration very
    high (biggest-day move dominates the total window move)."""
    bars = _ohlcv_with_volume(
        # day 0: 100 → day 1: 100 → day 2: 115 (+15%) → day 3: 115 → day 4: 115 → day 5: 115
        closes=[100, 100, 115, 115, 115, 115],
        volumes=[1_000_000] * 6,
    )
    _, _, conc = _missed_ops_quality_metrics(bars, lookback_days=5)
    # Total window return = 15%; biggest day = 15% → concentration = 100%
    assert conc is not None and conc >= 90


def test_quality_metrics_single_day_concentration_trend():
    """Distributed move across all 5 days → concentration < 40%."""
    bars = _ohlcv_with_volume(
        closes=[100, 103, 106, 109, 112, 115],  # each day ~3%
        volumes=[1_000_000] * 6,
    )
    _, _, conc = _missed_ops_quality_metrics(bars, lookback_days=5)
    assert conc is not None and conc < 40


def test_quality_metrics_insufficient_volume_bars_returns_none():
    """Bars with no volume attribute (non-numeric) → avg_dvol_m = None.
    Single-day concentration should still compute from close prices."""
    bars = []
    for close in [100, 103, 106, 109, 112, 115]:
        b = MagicMock()
        b.close = close
        b.volume = None  # explicitly None, not MagicMock
        b.date = date(2026, 4, 1)
        bars.append(b)
    avg_m, vol_conf, conc = _missed_ops_quality_metrics(bars, lookback_days=5)
    assert avg_m is None
    assert vol_conf is None
    # Close-only metric should still work
    assert conc is not None


# ---------------------------------------------------------------------------
# Liquidity pre-filter on top-movers
# ---------------------------------------------------------------------------

@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_drops_thin_liquidity_top_movers(_sec):
    """Top-movers with 20d avg dollar volume below the threshold (default $5M)
    must be filtered OUT before reaching the LLM. Medium-long-term investor
    doesn't chase micro-cap gappers."""
    p = _pipeline_with(
        top_movers=[
            {"symbol": "THIN", "percent_change": 25.0, "price": 5.0},
            {"symbol": "LIQUID", "percent_change": 15.0, "price": 100.0},
        ],
    )
    # Override ohlcv to return bars with distinct volume profiles.
    # The digest uses the LAST `lookback_days + 1` bars for move%, so
    # put the rally at the end of the 25-bar trailing window.
    def _ohlcv(symbol, lookback_days=25):
        if symbol == "THIN":
            closes = [5.0] * 19 + [5.0, 5.2, 5.4, 5.5, 5.7, 6.25]  # +25% last 5 days
            return _ohlcv_with_volume(
                closes=closes, volumes=[100_000] * 25,  # $0.5M daily
            )
        if symbol == "LIQUID":
            closes = [100.0] * 19 + [100, 103, 106, 109, 112, 115]  # +15%
            return _ohlcv_with_volume(
                closes=closes,
                volumes=[1_000_000] * 25,  # ~$100M daily dollar vol
            )
        return []
    p.market.get_ohlcv.side_effect = _ohlcv
    out = p._build_missed_opportunities_digest(
        lookback_days=5, move_threshold_pct=8.0,
    )
    syms = [s.symbol for s in out]
    assert "LIQUID" in syms
    assert "THIN" not in syms, (
        f"thin top-mover should have been filtered; got {syms}"
    )


@patch("src.execution.broker._get_sector", return_value="Technology")
def test_digest_never_drops_universe_symbols_for_liquidity(_sec):
    """Universe symbols are curated — they bypass the liquidity filter
    even if their 20d dollar volume is below the threshold. We trust
    the human-vetted universe definition."""
    p = _pipeline_with(
        universe=["ILLIQUID_BUT_TRACKED"],
    )
    def _ohlcv(symbol, lookback_days=25):
        if symbol == "ILLIQUID_BUT_TRACKED":
            return _ohlcv_with_volume(
                closes=[5, 5.2, 5.4, 5.5, 5.7, 5.9],  # +18% over window
                volumes=[10_000] * 6,  # $50k daily — well under $5M
            )
        return []
    p.market.get_ohlcv.side_effect = _ohlcv
    out = p._build_missed_opportunities_digest(
        lookback_days=5, move_threshold_pct=8.0,
    )
    syms = [s.symbol for s in out]
    assert "ILLIQUID_BUT_TRACKED" in syms, (
        "universe symbols must bypass the liquidity filter"
    )


# ---------------------------------------------------------------------------
# Prompt rendering — new quality line
# ---------------------------------------------------------------------------

def test_evening_prompt_renders_quality_line():
    """The new quality metrics line must appear in the rendered section so
    the LLM can read volume / sustain numbers and apply the medium-long-
    term bar."""
    from src.models import MissedOpportunitySnapshot
    from unittest.mock import patch as _patch
    from src.agents.evening_analyst import EveningAnalystAgent

    with _patch("anthropic.Anthropic"):
        agent = EveningAnalystAgent(api_key="k", model="claude-opus-4-6")

    snap = MissedOpportunitySnapshot(
        symbol="VST", move_pct=22.3, window_days=5,
        held_during_window=False,
        had_ta_signal=False, had_news_signal=True, had_earnings_signal=False,
        source="top_mover",
        avg_dollar_volume_20d_m=180.0,
        volume_confirmation_ratio=2.1,
        single_day_concentration_pct=34.0,
    )
    msg = agent.build_user_message(
        positions=[], macro_summary={"vix": {"current": 18}},
        total_value=100_000, daily_pnl=0, daily_return_pct=0.0,
        missed_ops_snapshots=[snap],
    )
    assert "20d $vol 180.0M" in msg
    assert "vol_conf 2.10x" in msg and "CONFIRMED" in msg
    assert "34%" in msg and "distributed" in msg


def _pipeline_with_insights_rows_moj(rows: list[dict]):
    """Mirror of _pipeline_with_insights_rows tuned for
    `missed_opportunities_json` content."""
    from src.pipeline import TradingPipeline
    p = TradingPipeline.__new__(TradingPipeline)
    p.db = MagicMock()
    p.db.get_recent_insights.return_value = rows
    return p


# ---------------------------------------------------------------------------
# PM helper: _build_watchlist_candidates — aggregate by symbol
# ---------------------------------------------------------------------------

def test_watchlist_candidates_empty_when_no_recommendations():
    """Evening flagged everyone as 'no' (or there are no rows) → empty list.
    Script treats empty as "nothing cleared the quality bar — normal."""
    import json as _json
    p = _pipeline_with_insights_rows_moj([
        {"date": "2026-04-18",
         "missed_opportunities_json": _json.dumps([
             {"symbol": "VST", "miss_category": "noise_rally",
              "universe_addition_recommendation": "no",
              "lesson": "thin volume, no theme"},
         ])},
    ])
    assert p._build_watchlist_candidates(lookback_days=30) == []


def test_watchlist_candidates_aggregates_add_and_watch_counts():
    """Same symbol flagged 'add' on day 1 and 'watch' on day 2 → bucket
    has add_count=1, watch_count=1, total_flags=2."""
    import json as _json
    p = _pipeline_with_insights_rows_moj([
        {"date": "2026-04-18",
         "missed_opportunities_json": _json.dumps([
             {"symbol": "VST", "miss_category": "theme_blindspot",
              "theme_if_any": "nuclear/power",
              "universe_addition_recommendation": "add",
              "universe_addition_reason": "20d $vol $180M; vol_conf 2.1x",
              "lesson": "x"},
         ])},
        {"date": "2026-04-17",
         "missed_opportunities_json": _json.dumps([
             {"symbol": "VST", "miss_category": "theme_blindspot",
              "theme_if_any": "nuclear/power",
              "universe_addition_recommendation": "watch",
              "universe_addition_reason": "vol_conf 1.6x; 1d conc 45%",
              "lesson": "x"},
         ])},
    ])
    out = p._build_watchlist_candidates(lookback_days=30)
    assert len(out) == 1
    vst = out[0]
    assert vst["symbol"] == "VST"
    assert vst["add_count"] == 1
    assert vst["watch_count"] == 1
    assert vst["total_flags"] == 2
    assert vst["themes"] == ["nuclear/power"]
    # Dates sorted newest-first
    assert vst["dates"] == ["2026-04-18", "2026-04-17"]
    # Latest reason = from the newest-day entry
    assert "$180M" in vst["latest_reason"]


def test_watchlist_candidates_sorts_add_before_watch():
    """An 'add' flag outranks a 'watch' flag in the sort order —
    add clears all four quality bars, watch clears most-but-not-all."""
    import json as _json
    p = _pipeline_with_insights_rows_moj([
        {"date": "2026-04-18",
         "missed_opportunities_json": _json.dumps([
             {"symbol": "A_ONLY_WATCH", "miss_category": "theme_blindspot",
              "theme_if_any": "ai-capex",
              "universe_addition_recommendation": "watch",
              "universe_addition_reason": "vol_conf 1.5x; marginal",
              "lesson": "x"},
             {"symbol": "B_ONE_ADD", "miss_category": "theme_blindspot",
              "theme_if_any": "nuclear/power",
              "universe_addition_recommendation": "add",
              "universe_addition_reason": "$180M · 2.1x · distributed",
              "lesson": "x"},
         ])},
        {"date": "2026-04-17",
         "missed_opportunities_json": _json.dumps([
             {"symbol": "A_ONLY_WATCH", "miss_category": "theme_blindspot",
              "theme_if_any": "ai-capex",
              "universe_addition_recommendation": "watch",
              "universe_addition_reason": "vol_conf 1.5x; marginal",
              "lesson": "x"},
             {"symbol": "A_ONLY_WATCH", "miss_category": "theme_blindspot",
              "theme_if_any": "ai-capex",
              "universe_addition_recommendation": "watch",
              "universe_addition_reason": "vol_conf 1.5x; marginal",
              "lesson": "x"},
         ])},
    ])
    out = p._build_watchlist_candidates(lookback_days=30)
    # B has 1 add (beats watch); A has 3 watch
    assert [c["symbol"] for c in out] == ["B_ONE_ADD", "A_ONLY_WATCH"]


def test_watchlist_candidates_ignores_no_recommendations():
    """`no` recommendations don't contribute counts (prevent the table
    from being flooded with every noise_rally entry)."""
    import json as _json
    p = _pipeline_with_insights_rows_moj([
        {"date": "2026-04-18",
         "missed_opportunities_json": _json.dumps([
             {"symbol": "NOISE", "miss_category": "noise_rally",
              "universe_addition_recommendation": "no",
              "lesson": "thin"},
             {"symbol": "GOOD", "miss_category": "theme_blindspot",
              "theme_if_any": "nuclear/power",
              "universe_addition_recommendation": "add",
              "universe_addition_reason": "$180M · 2.1x",
              "lesson": "x"},
         ])},
    ])
    out = p._build_watchlist_candidates(lookback_days=30)
    assert [c["symbol"] for c in out] == ["GOOD"]


def test_watchlist_candidates_tolerates_malformed_json():
    """Corrupt row → skip, don't crash. Other rows still counted."""
    import json as _json
    p = _pipeline_with_insights_rows_moj([
        {"date": "2026-04-18",
         "missed_opportunities_json": "{ not valid json"},
        {"date": "2026-04-17",
         "missed_opportunities_json": _json.dumps([
             {"symbol": "VST", "miss_category": "theme_blindspot",
              "theme_if_any": "nuclear/power",
              "universe_addition_recommendation": "add",
              "universe_addition_reason": "$180M · 2.1x",
              "lesson": "x"},
         ])},
    ])
    out = p._build_watchlist_candidates(lookback_days=30)
    assert len(out) == 1
    assert out[0]["symbol"] == "VST"


# ---------------------------------------------------------------------------
# Quarterly digest: watchlist_candidates section
# ---------------------------------------------------------------------------

def test_quarterly_digest_includes_watchlist_candidates():
    """build_quarterly_digest returns a watchlist_candidates section with the
    right shape: {window_days, candidates, high_conviction, total_candidates}."""
    from datetime import date as _date
    import json as _json
    from src.evolution.quarterly_digest import build_quarterly_digest
    from unittest.mock import MagicMock as _MM

    db = _MM()
    db.get_daily_pnl.return_value = []
    db.get_recent_insights.return_value = [
        {"date": "2026-04-18",
         "missed_opportunities_json": _json.dumps([
             {"symbol": "VST", "miss_category": "theme_blindspot",
              "theme_if_any": "nuclear/power",
              "universe_addition_recommendation": "add",
              "universe_addition_reason": "$180M · 2.1x · distributed",
              "lesson": "x"},
         ])},
        {"date": "2026-04-17",
         "missed_opportunities_json": _json.dumps([
             {"symbol": "VST", "miss_category": "theme_blindspot",
              "theme_if_any": "nuclear/power",
              "universe_addition_recommendation": "add",
              "universe_addition_reason": "$180M · 2.1x · distributed",
              "lesson": "x"},
         ])},
    ]
    db.get_recent_agent_outputs.return_value = []
    db.compute_trade_calibration.return_value = {"n_closed": 0}

    digest = build_quarterly_digest(
        db, market=None, period_end=_date(2026, 3, 31), lookback_days=90,
    )
    wlc = digest["watchlist_candidates"]
    assert wlc["window_days"] == 90
    assert wlc["total_candidates"] == 1
    assert wlc["candidates"][0]["symbol"] == "VST"
    assert wlc["candidates"][0]["add_count"] == 2
    # add_count >= 2 → high conviction
    assert "VST" in wlc["high_conviction"]


def test_quarterly_digest_watchlist_high_conviction_threshold():
    """A single 'add' flag does NOT put a symbol in high_conviction —
    we need 2+ 'add's across distinct days to take it seriously."""
    from datetime import date as _date
    import json as _json
    from src.evolution.quarterly_digest import build_quarterly_digest
    from unittest.mock import MagicMock as _MM

    db = _MM()
    db.get_daily_pnl.return_value = []
    db.get_recent_insights.return_value = [
        {"date": "2026-04-18",
         "missed_opportunities_json": _json.dumps([
             {"symbol": "LONE", "miss_category": "theme_blindspot",
              "theme_if_any": "ai-capex",
              "universe_addition_recommendation": "add",
              "universe_addition_reason": "single observation",
              "lesson": "x"},
         ])},
    ]
    db.get_recent_agent_outputs.return_value = []
    db.compute_trade_calibration.return_value = {"n_closed": 0}
    digest = build_quarterly_digest(
        db, market=None, period_end=_date(2026, 3, 31), lookback_days=90,
    )
    wlc = digest["watchlist_candidates"]
    assert wlc["total_candidates"] == 1
    assert wlc["high_conviction"] == []  # single add isn't enough


def test_evening_prompt_tags_low_quality_top_movers():
    """Weak quality metrics render with explicit tags (weak / single-day
    gap) so the LLM treats them as noise_rally candidates."""
    from src.models import MissedOpportunitySnapshot
    from unittest.mock import patch as _patch
    from src.agents.evening_analyst import EveningAnalystAgent

    with _patch("anthropic.Anthropic"):
        agent = EveningAnalystAgent(api_key="k", model="claude-opus-4-6")

    snap = MissedOpportunitySnapshot(
        symbol="THIN", move_pct=22.3, window_days=5,
        held_during_window=False,
        had_ta_signal=False, had_news_signal=False, had_earnings_signal=False,
        source="top_mover",
        avg_dollar_volume_20d_m=3.5,  # thin
        volume_confirmation_ratio=0.8,  # weak
        single_day_concentration_pct=85.0,  # gap
    )
    msg = agent.build_user_message(
        positions=[], macro_summary={"vix": {"current": 18}},
        total_value=100_000, daily_pnl=0, daily_return_pct=0.0,
        missed_ops_snapshots=[snap],
    )
    assert "weak" in msg  # volume_confirmation_ratio < 1.5
    assert "single-day gap" in msg  # concentration >= 70
