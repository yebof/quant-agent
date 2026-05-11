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


# ============================================================================
# regime_shift detection — pin the cross-day comparison flow.
# Python persists yesterday's regime; macro_analyst's prompt surfaces it so
# the LLM can compare. The Python side's responsibility is:
#   - first run (no prior state) yields None from load_last_state
#   - second run sees yesterday's regime under a stable key
#   - regime change between days is observable from the persisted snapshot
# The LLM's regime_shift=True/False emission is then validated against
# whatever the LLM returns — a separate prompt-level concern.
# ============================================================================

def test_macro_store_first_day_load_returns_none_so_prompt_says_first_run(tmp_path):
    """First trading day ever (or fresh data dir) must surface None so the
    macro_analyst prompt renders the 'No prior state on file (first run)'
    section instead of falsely fabricating a yesterday."""
    store = MacroStore(data_dir=str(tmp_path / "macro"))
    assert store.load_last_state() is None


def test_macro_store_regime_change_observable_across_days(tmp_path):
    """Day 1 = risk-on; day 2 = risk-off. After saving day 2, history
    must carry both entries with distinct regimes so the LLM can see the
    transition. Day 1 save must NOT be visible after day 2 save (today
    overwrites today's slot)."""
    store = MacroStore(data_dir=str(tmp_path / "macro"))

    store.save_last_state({
        "regime": "risk-on", "confidence": "high", "equity_outlook": "bullish",
        "summary": "Tight credit + soft VIX.",
        "position_guidance": {"target_invested_pct": 80, "cash_recommendation_pct": 20, "reasoning": "x"},
    })
    day1 = store.load_last_state()
    assert day1["regime"] == "risk-on"

    # Simulate a day passing by reaching into the persisted file and
    # backdating today's history entry, then save day 2.
    history_path = store.history_path
    raw = json.loads(history_path.read_text())
    assert len(raw) == 1
    raw[0]["date"] = "2026-04-30"  # yesterday
    history_path.write_text(json.dumps(raw))

    store.save_last_state({
        "regime": "risk-off", "confidence": "high", "equity_outlook": "bearish",
        "summary": "Credit widening + VIX > 25.",
        "position_guidance": {"target_invested_pct": 30, "cash_recommendation_pct": 70, "reasoning": "x"},
    })
    day2 = store.load_last_state()
    assert day2["regime"] == "risk-off"

    # History keeps both — LLM can see the transition.
    hist = store.load_history(days=7)
    regimes_seen = [h["regime"] for h in hist]
    assert "risk-on" in regimes_seen
    assert "risk-off" in regimes_seen
    assert regimes_seen.index("risk-on") < regimes_seen.index("risk-off"), (
        "history must be ordered oldest-first so the LLM reads transitions chronologically"
    )


def test_macro_analyst_prompt_renders_first_run_banner_when_no_prior_state():
    """When MacroStore.load_last_state() returns None, build_user_message
    must render the 'No prior state on file (first run)' banner so the
    LLM doesn't hallucinate a yesterday and falsely flag regime_shift."""
    from src.agents.macro_analyst import MacroAnalystAgent

    agent = MacroAnalystAgent.__new__(MacroAnalystAgent)
    msg = agent.build_user_message(
        macro_summary={"vix": {"current": 18, "mean_5d": 17, "trend": "stable"}},
        universe=["SPY", "QQQ"],
        last_state=None,
        news_narrative=None,
    )
    assert "No prior state on file (first run)" in msg


def test_macro_analyst_prompt_surfaces_yesterday_regime_for_shift_detection():
    """When prior state is present, the prompt MUST quote yesterday's
    regime verbatim so the LLM can decide whether today represents a
    shift. Drift here (e.g., outputting the regime as a dict, mangling
    case) would silently disable shift detection."""
    from src.agents.macro_analyst import MacroAnalystAgent

    agent = MacroAnalystAgent.__new__(MacroAnalystAgent)
    msg = agent.build_user_message(
        macro_summary={"vix": {"current": 28, "mean_5d": 22, "trend": "rising"}},
        universe=["SPY"],
        last_state={
            "date": "2026-04-30",
            "regime": "risk-on",
            "confidence": "high",
            "equity_outlook": "bullish",
            "summary": "Soft VIX + tight credit.",
        },
        news_narrative=None,
    )
    assert "Yesterday's Macro State" in msg
    assert "risk-on" in msg
    assert "bullish" in msg
    assert "2026-04-30" in msg
