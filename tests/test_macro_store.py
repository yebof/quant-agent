import json

from src.data.macro_store import MacroStore


def test_load_missing_returns_none(tmp_path):
    store = MacroStore(data_dir=str(tmp_path / "macro"))
    assert store.load_last_state() is None


def test_save_then_load_round_trip(tmp_path):
    store = MacroStore(data_dir=str(tmp_path / "macro"))
    analysis = {
        "regime": "risk-on",
        "confidence": "medium",
        "equity_outlook": "bullish",
        "summary": "Compressing VIX, tight credit.",
        "position_guidance": {
            "target_invested_pct": 75.0,
            "cash_recommendation_pct": 25.0,
            "reasoning": "Hold buffer for inflation tail.",
        },
        # Fields not in the snapshot subset — should be dropped.
        "reasoning_chain": {"volatility_analysis": "…"},
        "sector_guidance": [{"sector": "Technology", "stance": "overweight"}],
    }
    store.save_last_state(analysis)

    loaded = store.load_last_state()
    assert loaded["regime"] == "risk-on"
    assert loaded["equity_outlook"] == "bullish"
    assert loaded["position_guidance"]["target_invested_pct"] == 75.0
    # Ensure the large fields are NOT persisted (we want the snapshot tiny).
    assert "reasoning_chain" not in loaded
    assert "sector_guidance" not in loaded
    # Date stamp is added on save.
    assert "date" in loaded


def test_corrupt_file_returns_none(tmp_path):
    macro_dir = tmp_path / "macro"
    macro_dir.mkdir()
    (macro_dir / "last_state.json").write_text("{ not valid json")
    store = MacroStore(data_dir=str(macro_dir))
    assert store.load_last_state() is None


def test_save_non_dict_is_noop(tmp_path):
    """save_last_state should silently ignore non-dict inputs (e.g. parse errors)."""
    store = MacroStore(data_dir=str(tmp_path / "macro"))
    store.save_last_state(None)
    store.save_last_state("not a dict")
    assert store.load_last_state() is None
