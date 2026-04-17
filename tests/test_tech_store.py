"""TechStore — per-symbol rating memory with correct age semantics."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from src.data.tech_store import TechStore
from src.util.time import et_today


def _analysis(symbol, rating="buy", conviction="high",
              entry_price=500.0, stop_loss=490.0, reference_target=525.0):
    """Minimal TechAnalysisResult-shaped stub (only attributes the store reads)."""
    return SimpleNamespace(
        symbol=symbol, rating=rating, conviction=conviction,
        entry_price=entry_price, stop_loss=stop_loss, reference_target=reference_target,
    )


def test_load_returns_empty_when_file_missing(tmp_path):
    store = TechStore(data_dir=str(tmp_path / "tech"))
    assert store.load() == {}


def test_update_adds_fresh_rating_with_today_as_first_seen(tmp_path):
    store = TechStore(data_dir=str(tmp_path / "tech"))
    store.update([_analysis("NVDA", rating="buy")])
    state = store.load()
    assert state["NVDA"]["rating"] == "buy"
    assert state["NVDA"]["first_seen_date"] == str(et_today())


def test_update_keeps_first_seen_when_rating_unchanged(tmp_path):
    """A continuous rating should accumulate age, not reset."""
    store = TechStore(data_dir=str(tmp_path / "tech"))
    # Seed with an older first_seen manually
    older = (et_today() - timedelta(days=5)).isoformat()
    seed = {"NVDA": {
        "rating": "buy", "conviction": "high",
        "first_seen_date": older, "last_rating_date": older,
        "entry_price": 190, "stop_loss": 184, "reference_target": 210,
    }}
    store.save(seed)

    # Today emits same rating — first_seen must NOT move
    store.update([_analysis("NVDA", rating="buy")])
    state = store.load()
    assert state["NVDA"]["first_seen_date"] == older
    assert state["NVDA"]["last_rating_date"] == str(et_today())


def test_update_resets_first_seen_when_rating_flips(tmp_path):
    store = TechStore(data_dir=str(tmp_path / "tech"))
    older = (et_today() - timedelta(days=5)).isoformat()
    seed = {"NVDA": {
        "rating": "buy", "conviction": "high",
        "first_seen_date": older, "last_rating_date": older,
        "entry_price": 190, "stop_loss": 184, "reference_target": 210,
    }}
    store.save(seed)

    # Today flips to sell — first_seen must reset to today
    store.update([_analysis("NVDA", rating="sell",
                            entry_price=210, stop_loss=220, reference_target=190)])
    state = store.load()
    assert state["NVDA"]["first_seen_date"] == str(et_today())
    assert state["NVDA"]["rating"] == "sell"


def test_update_leaves_absent_symbols_untouched(tmp_path):
    """A symbol that's NOT in today's batch should remain in the cache."""
    store = TechStore(data_dir=str(tmp_path / "tech"))
    older = (et_today() - timedelta(days=3)).isoformat()
    store.save({"OLD_SYMBOL": {
        "rating": "buy", "conviction": "medium",
        "first_seen_date": older, "last_rating_date": older,
        "entry_price": 100, "stop_loss": 95, "reference_target": 110,
    }})

    store.update([_analysis("NEW_SYMBOL", rating="buy")])
    state = store.load()
    assert "OLD_SYMBOL" in state
    assert "NEW_SYMBOL" in state


def test_compute_ages_returns_days_since_first_seen(tmp_path):
    store = TechStore(data_dir=str(tmp_path / "tech"))
    four_days_ago = (et_today() - timedelta(days=4)).isoformat()
    store.save({"NVDA": {
        "rating": "buy", "conviction": "high",
        "first_seen_date": four_days_ago, "last_rating_date": str(et_today()),
        "entry_price": 190, "stop_loss": 184, "reference_target": 210,
    }})
    ages = store.compute_ages(["NVDA", "UNKNOWN"])
    assert ages["NVDA"] == 4
    assert "UNKNOWN" not in ages  # cached data absent → no key


def test_update_appends_to_per_symbol_history(tmp_path):
    """Each day's rating is appended to history[]. Dedup on same-day re-run."""
    store = TechStore(data_dir=str(tmp_path / "tech"))
    # First run
    a1 = _analysis("NVDA", rating="buy", conviction="medium")
    a1.risk_reward = 2.1
    store.update([a1])
    # Same-day re-run (pipeline re-executed) should replace, not duplicate
    a1b = _analysis("NVDA", rating="buy", conviction="high")
    a1b.risk_reward = 2.3
    store.update([a1b])
    hist = store.get_history("NVDA", days=7)
    assert len(hist) == 1
    assert hist[0]["conviction"] == "high"
    assert hist[0]["risk_reward"] == 2.3


def test_get_history_returns_recent_N_days(tmp_path):
    """history keeps up to 14 days; get_history(days=7) returns the last 7."""
    store = TechStore(data_dir=str(tmp_path / "tech"))
    # Seed a cache directly with 10 days of history
    seed = {"NVDA": {
        "rating": "buy",
        "conviction": "medium",
        "first_seen_date": str(et_today()),
        "last_rating_date": str(et_today()),
        "entry_price": 190, "stop_loss": 184, "reference_target": 210,
        "history": [
            {"date": (et_today() - timedelta(days=d)).isoformat(),
             "rating": "buy", "conviction": "medium", "risk_reward": 2.0}
            for d in range(9, -1, -1)
        ],
    }}
    store.save(seed)
    seven = store.get_history("NVDA", days=7)
    assert len(seven) == 7
    # Oldest first among the returned slice
    assert seven[0]["date"] <= seven[-1]["date"]


def test_compute_ages_returns_zero_for_today(tmp_path):
    store = TechStore(data_dir=str(tmp_path / "tech"))
    store.update([_analysis("NVDA", rating="buy")])  # first_seen = today
    ages = store.compute_ages(["NVDA"])
    assert ages["NVDA"] == 0
