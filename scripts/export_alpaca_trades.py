#!/usr/bin/env python3
"""Export every Alpaca artefact reachable via the trading API.

Read-only ground truth. Bypasses local logs / SQLite (those reflect what
the agent THOUGHT it did); this hits the broker's history directly so
the output reflects what Alpaca actually accepted, filled, or rejected.

Outputs (auto-emitted as a set, alongside --output):

    data/alpaca/trades.txt              # human-readable report
    data/alpaca/trades.orders.jsonl     # every order, FULL pydantic dump
    data/alpaca/trades.orders.csv       # same orders as a CSV table
                                        # (header row = field names)
    data/alpaca/trades.daily_pnl.csv    # per-day equity + daily P&L
                                        # (broker-side portfolio_history)
    data/alpaca/trades.activities.jsonl # account activity log (FILLs +
                                        # by default DIV/JNLC/...; raw)
    data/alpaca/trades.account.json     # full account snapshot

The JSONL/JSON companions are the canonical machine-readable copies and
preserve every field the SDK exposes — nothing is dropped. The .txt is
a human view (summary + per-order table) but is NOT lossless; use the
companions for downstream analysis.

Usage (from project root):

    ./scripts/export_alpaca_trades.py
    ./scripts/export_alpaca_trades.py --output /tmp/trades.txt
    ./scripts/export_alpaca_trades.py --live              # prod endpoint
    ./scripts/export_alpaca_trades.py --since 2026-04-01
    ./scripts/export_alpaca_trades.py --skip-activities   # skip activity log
    ./scripts/export_alpaca_trades.py --no-companions     # txt only

Credentials come from .env (ALPACA_API_KEY / ALPACA_SECRET_KEY), same as
the trading pipeline. Paper vs live defaults to config/settings.yaml's
alpaca.paper unless --paper / --live overrides.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_WIDTH = 145


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def _load_env_file() -> None:
    """Best-effort .env loader so the script works without `set -a; source .env`.

    We don't pull in python-dotenv just for this — the file is line-based
    `KEY=value`, comments start with `#`. Existing env wins (so a shell
    export overrides .env, matching the trading pipeline's behavior).
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            # Tolerate the `export KEY=value` form the production .env uses
            # (the wrapper script source's it via bash). Without stripping
            # `export ` the key would land as "export ALPACA_API_KEY" and
            # os.environ.get("ALPACA_API_KEY") would still miss it.
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass  # best-effort; caller will fail at the credential check


def _read_paper_default() -> bool:
    """Fall back to config/settings.yaml's alpaca.paper when --paper/--live
    isn't given. Paper is the safer default on any load failure."""
    try:
        from src.config import load_config
        cfg = load_config(PROJECT_ROOT / "config" / "settings.yaml")
        return bool(cfg.alpaca.paper)
    except Exception:
        return True


# ---------------------------------------------------------------------------
# field extraction
# ---------------------------------------------------------------------------

def _normalize_for_json(v):
    """Recursively coerce SDK model values into JSON-safe forms WITHOUT
    losing precision or fields.

    - Enum   -> .value (string)
    - UUID   -> str
    - Decimal -> str (preserve broker-side precision; we cast to float
      only at arithmetic / formatter boundaries)
    - datetime -> left as-is so the report sort + ET formatter still
      work; the JSON encoder serializes it to UTC ISO at emit time
    - dict / list -> recurse

    This is the SINGLE place that decides "what does a field's value
    look like in the dump" — everything else just reads it.
    """
    from decimal import Decimal
    from enum import Enum
    from uuid import UUID

    if v is None:
        return None
    # IMPORTANT: check Enum BEFORE str/int. alpaca-py enums are
    # `(str, Enum)` subclasses — `isinstance(OrderStatus.FILLED, str)`
    # is True, so a leading str-shortcut would return the enum
    # unchanged. JSON serialization happens to unwrap it via
    # json.dumps, but f-string formatting in the text report calls
    # str(enum) and prints 'OrderStatus.FILLED' instead of 'filled'.
    # Putting the Enum branch first kills that whole class of bug.
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, UUID):
        return str(v)
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, datetime):
        return v
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, dict):
        return {k: _normalize_for_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_normalize_for_json(x) for x in v]
    # Unknown object (e.g., a nested pydantic model the dumper didn't
    # recurse into): try model_dump, else str().
    if hasattr(v, "model_dump"):
        try:
            return _normalize_for_json(v.model_dump())
        except Exception:
            pass
    return str(v)


def _to_full_dict(obj) -> dict:
    """Faithful per-record dump. Pydantic v2 SDK models expose every
    field via model_dump(); the normalize pass handles enums / UUIDs /
    Decimals. Falls back to a public-attribute scan for non-pydantic
    inputs (which is also why every test object built from
    SimpleNamespace continues to work)."""
    if hasattr(obj, "model_dump"):
        try:
            return _normalize_for_json(obj.model_dump())
        except Exception:
            pass
    d: dict = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if callable(val):
            continue
        d[name] = val
    return _normalize_for_json(d)


def _as_dt(v) -> datetime | None:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        s = str(v).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _et_str(dt) -> str:
    d = _as_dt(dt)
    if d is None:
        return "-"
    return d.astimezone(ET).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_qty(v) -> str:
    """Trade-quantity formatter. Alpaca supports fractional shares, so the
    column has to handle both 51.0 and 0.1234. Drops trailing zeros."""
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == 0:
        return "0"
    if f == int(f) and abs(f) < 1e12:
        return f"{int(f):,}"
    return f"{f:.4f}".rstrip("0").rstrip(".")


def _fmt_money(v) -> str:
    if v is None or v == "":
        return "-"
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _order_to_dict(o) -> dict:
    """SDK Order → faithful dict (every field the SDK exposes).

    Switched from a hand-picked subset to model_dump+normalize so no
    field is dropped — legs, hwm, asset_class, ratio_qty, position_intent,
    expires_at, source/subtag (where present), etc. all survive. The
    text report still reads a fixed subset of keys; the JSONL companion
    is now genuinely lossless.
    """
    return _to_full_dict(o)


# ---------------------------------------------------------------------------
# pagination
# ---------------------------------------------------------------------------

def fetch_all_orders(
    client,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    page_limit: int = 500,
) -> list[dict]:
    """Pull every order between `since` and `until`, dedup'd by id, sorted
    oldest first.

    Alpaca paginates by `until` (newer cap) when direction=desc. We walk
    backwards: fetch newest page → set next `until` to (oldest_submitted
    - 1µs) → repeat until empty / page shorter than limit / cursor falls
    below `since`. Dedup by id covers the boundary tie case (two orders
    sharing the same submitted_at across pages).
    """
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    seen: set[str] = set()
    out: list[dict] = []
    cursor_until = until

    while True:
        kwargs = {
            "status": QueryOrderStatus.ALL,
            "limit": page_limit,
            "nested": False,
        }
        if since is not None:
            kwargs["after"] = since
        if cursor_until is not None:
            kwargs["until"] = cursor_until
        page = client.get_orders(filter=GetOrdersRequest(**kwargs)) or []
        if not page:
            break

        oldest_in_page: datetime | None = None
        new_in_page = 0
        for o in page:
            row = _order_to_dict(o)
            oid = row["id"]
            if not oid or oid in seen:
                continue
            seen.add(oid)
            out.append(row)
            new_in_page += 1
            sub = row.get("submitted_at")
            if sub is not None and (oldest_in_page is None or sub < oldest_in_page):
                oldest_in_page = sub

        # Termination: API returned a short page (no more after this)
        # OR every id was a duplicate (we've wrapped) OR no datetimes
        # to advance the cursor.
        if len(page) < page_limit or new_in_page == 0 or oldest_in_page is None:
            break
        next_until = oldest_in_page - timedelta(microseconds=1)
        if since is not None and next_until <= since:
            break
        cursor_until = next_until

    out.sort(key=lambda r: (
        r.get("submitted_at") or datetime(1970, 1, 1, tzinfo=timezone.utc),
        r.get("id") or "",
    ))
    return out


def fetch_account_dump(client) -> dict:
    """Full account snapshot (every SDK field). Single read."""
    acct = client.get_account()
    return _to_full_dict(acct)


def fetch_portfolio_history_daily(
    client, *, since=None,
) -> dict:
    """Pull /v2/account/portfolio_history at 1D timeframe.

    Alpaca's portfolio_history endpoint is the broker-side source of
    truth for "how much did I make today" — it folds realized + open-
    position mark-to-market into a single daily equity series, which is
    what an operator actually wants when asking the question.

    `since`: datetime (or ISO 'YYYY-MM-DD' string) of the first day to
    include. Pass account.created_at to anchor the series at the
    account's opening (day 1). When None, falls back to `period='5A'`.

    Three SDK quirks worth flagging:
      1. The field is named `start` on GetPortfolioHistoryRequest (NOT
         `date_start` — pydantic silently DROPS unknown kwargs in this
         version, so a typo here looks fine and the API just returns
         its default ~30-day window with base_value = equity at the
         start of THAT window, not the account's opening).
      2. `client.get_portfolio_history` takes `history_filter=`, not
         `filter=` like `get_orders` does.
      3. `start` alone is silently capped at ~30 trading days, leaving
         a young account looking truncated. Passing `start` AND `end`
         together (end = now) lifts the cap and returns the full
         requested window with base_value = equity at the start day.
         Using `period='5A'` alone also returns long history but
         walks back ~5 years from today with pre-account-open rows
         polluting the series — not what we want here.

    Returns the model_dump'd dict (timestamp[], equity[],
    profit_loss[], profit_loss_pct[], base_value, timeframe). Empty
    dict on history-not-available; never raises.
    """
    try:
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        kwargs: dict = {"timeframe": "1D", "extended_hours": False}
        if since is not None:
            if isinstance(since, datetime):
                start_dt = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
            else:
                s = str(since)[:10]
                start_dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
            kwargs["start"] = start_dt
            # Bound the window so Alpaca returns ALL days between
            # start and now, not just its default ~30-day snap.
            kwargs["end"] = datetime.now(timezone.utc)
        else:
            kwargs["period"] = "5A"
        req = GetPortfolioHistoryRequest(**kwargs)
        history = client.get_portfolio_history(history_filter=req)
        return _to_full_dict(history)
    except Exception as exc:
        print(f"WARN: portfolio_history fetch failed: {exc}", file=sys.stderr)
        return {}


def fetch_all_activities(
    client,
    *,
    activity_types: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    page_size: int = 100,
) -> list[dict]:
    """Pull every account activity Alpaca will return.

    Alpaca's /v2/account/activities endpoint is not wrapped by alpaca-py
    0.43.x as a typed method, so we go through the SDK's raw `get(...)`.
    That keeps us using the SDK's auth + retry layer (same as orders)
    without bolting on a second HTTP client.

    Pagination: pass `page_token = <last_entry.id>` for the next page;
    stop when the page is empty or shorter than `page_size`. Returns
    activities in API order (most recent first), then sorted oldest
    first for export consistency.

    `activity_types`: None means "all types" (FILL + DIV + JNLC + ...).
    Pass an explicit list (e.g., ["FILL"]) to narrow.
    """
    params: dict = {"page_size": page_size}
    if activity_types:
        params["activity_types"] = ",".join(activity_types)
    if since is not None:
        # API accepts `after` as an RFC3339 timestamp; use UTC ISO.
        params["after"] = since.astimezone(timezone.utc).isoformat()
    if until is not None:
        params["until"] = until.astimezone(timezone.utc).isoformat()

    out: list[dict] = []
    page_token: str | None = None
    while True:
        call_params = dict(params)
        if page_token:
            call_params["page_token"] = page_token
        try:
            page = client.get("/account/activities", data=call_params) or []
        except Exception as exc:
            # Best-effort: bubble up the failure so the caller logs it
            # as a warning at the top of the report. Activities are
            # auxiliary; orders are the headline.
            raise RuntimeError(f"activities fetch failed: {exc}") from exc
        if not isinstance(page, list) or not page:
            break
        for raw in page:
            out.append(_normalize_for_json(raw))
        if len(page) < page_size:
            break
        last_id = page[-1].get("id") if isinstance(page[-1], dict) else None
        if not last_id or last_id == page_token:
            break  # guard against API loop
        page_token = last_id

    # Sort oldest first for export. transaction_time is the canonical
    # FILL timestamp; non-FILL types may use other keys, so fall back to
    # `date` then id for stable ordering.
    def _sort_key(a: dict):
        for k in ("transaction_time", "date", "id"):
            v = a.get(k)
            if v:
                return (str(v), str(a.get("id") or ""))
        return ("", str(a.get("id") or ""))

    out.sort(key=_sort_key)
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def render_report(
    orders: list[dict],
    *,
    account: dict,
    env_label: str,
    api_url: str,
    since: datetime | None,
    until: datetime | None,
    fetch_warning: str | None = None,
    companion_note: list[str] | None = None,
) -> str:
    bar = "=" * REPORT_WIDTH
    sub_bar = "-" * REPORT_WIDTH
    lines: list[str] = []

    # --- header ---
    now_utc = datetime.now(timezone.utc)
    lines.append(bar)
    lines.append("quant-agent — Alpaca trade export")
    lines.append(bar)
    lines.append(
        f"Generated:        {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC "
        f"({now_utc.astimezone(ET).strftime('%Y-%m-%d %H:%M:%S')} ET)"
    )
    lines.append(f"Account ID:       {account.get('id', '?')}")
    if account.get("account_number"):
        lines.append(f"Account number:   {account['account_number']}")
    lines.append(f"Environment:      {env_label}  ({api_url})")
    if account.get("created_at"):
        lines.append(f"Account opened:   {account['created_at']}")
    span_bits = []
    if since is not None:
        span_bits.append(f"since {since.strftime('%Y-%m-%d')}")
    if until is not None:
        span_bits.append(f"until {until.strftime('%Y-%m-%d')}")
    span_str = ", ".join(span_bits) if span_bits else "since account inception"
    lines.append(f"Orders fetched:   {len(orders):,}  ({span_str})")
    if fetch_warning:
        lines.append("")
        lines.append(f"!!! WARNING: {fetch_warning}")
    if companion_note:
        lines.append("")
        for ln in companion_note:
            lines.append(ln)
    lines.append("")

    # --- status breakdown ---
    status_counts = Counter((o.get("status") or "?") for o in orders)
    lines.append("STATUS BREAKDOWN")
    lines.append("-" * 16)
    total = max(1, sum(status_counts.values()))
    for status, n in status_counts.most_common():
        pct = 100.0 * n / total
        lines.append(f"  {status:<20} {n:>6}   ({pct:5.1f}%)")
    lines.append("")

    # --- side totals (fills only — partial fills count toward shares/notional) ---
    fills = [
        o for o in orders
        if o.get("status") == "filled" or (o.get("filled_qty") or 0)
    ]
    side_count: Counter = Counter(o.get("side") for o in fills)
    side_shares: dict[str, float] = defaultdict(float)
    side_notional: dict[str, float] = defaultdict(float)
    for o in fills:
        fq = float(o.get("filled_qty") or 0)
        px = float(o.get("filled_avg_price") or 0)
        if fq > 0 and px > 0:
            side_shares[o.get("side")] += fq
            side_notional[o.get("side")] += fq * px

    lines.append("SIDE TOTALS (filled / partially-filled orders)")
    lines.append("-" * 44)
    lines.append(
        f"  {'side':<6} {'count':>8} {'shares':>14} {'gross notional':>22}"
    )
    for side in ("buy", "sell"):
        cnt = side_count.get(side, 0)
        sh = side_shares.get(side, 0.0)
        nt = side_notional.get(side, 0.0)
        lines.append(f"  {side.upper():<6} {cnt:>8} {sh:>14,.4f} {nt:>22,.2f}")
    net_cash = side_notional.get("sell", 0.0) - side_notional.get("buy", 0.0)
    lines.append(
        f"  {'net realized cashflow (sell − buy):':<30}{net_cash:>22,.2f}"
    )
    lines.append("")

    # --- per-symbol activity ---
    sym_buys: Counter = Counter()
    sym_sells: Counter = Counter()
    sym_notional: dict[str, float] = defaultdict(float)
    for o in fills:
        sym = o.get("symbol") or "?"
        if o.get("side") == "buy":
            sym_buys[sym] += 1
        elif o.get("side") == "sell":
            sym_sells[sym] += 1
        fq = float(o.get("filled_qty") or 0)
        px = float(o.get("filled_avg_price") or 0)
        if fq and px:
            sym_notional[sym] += fq * px
    all_syms = set(sym_buys) | set(sym_sells)
    top_syms = sorted(all_syms, key=lambda s: -(sym_buys[s] + sym_sells[s]))[:20]
    if top_syms:
        lines.append("TOP 20 SYMBOLS BY FILL COUNT")
        lines.append("-" * 28)
        lines.append(
            f"  {'#':>3} {'sym':<6} {'fills':>6}  {'buy':>4} {'sell':>4}  "
            f"{'gross notional':>16}"
        )
        for i, sym in enumerate(top_syms, 1):
            b = sym_buys[sym]
            s = sym_sells[sym]
            lines.append(
                f"  {i:>3} {sym:<6} {b + s:>6}  {b:>4} {s:>4}  "
                f"{sym_notional[sym]:>16,.2f}"
            )
        lines.append("")

    # --- detail table ---
    lines.append(bar)
    lines.append("ORDER DETAIL (oldest first, all statuses)")
    lines.append(bar)
    lines.append("")
    lines.append(
        "# Times in ET. limit / stop are SUBMITTED prices (— if N/A)."
    )
    lines.append(
        "# order_id shows the leading 8 chars; full ids + client_order_id "
        "live in --jsonl output."
    )
    lines.append("")

    header = (
        f"{'submitted_at':<19}  "
        f"{'sym':<6} "
        f"{'side':<4} "
        f"{'qty':>9} "
        f"{'filled':>9} "
        f"{'avg_px':>11} "
        f"{'type':<12} "
        f"{'tif':<5} "
        f"{'status':<10} "
        f"{'class':<8} "
        f"{'limit':>11} "
        f"{'stop':>11}  "
        f"{'filled_at':<19} "
        f"{'order_id':<10}"
    )
    lines.append(header)
    lines.append(sub_bar[: len(header)])

    if not orders:
        lines.append("(no orders)")
    for o in orders:
        sub_str = _et_str(o.get("submitted_at"))
        fld_str = _et_str(o.get("filled_at"))
        oid = (o.get("id") or "")[:8] or "-"
        lines.append(
            f"{sub_str:<19}  "
            f"{(o.get('symbol') or '?'):<6} "
            f"{(o.get('side') or '?'):<4} "
            f"{_fmt_qty(o.get('qty')):>9} "
            f"{_fmt_qty(o.get('filled_qty')):>9} "
            f"{_fmt_money(o.get('filled_avg_price')):>11} "
            f"{(o.get('type') or '?'):<12} "
            f"{(o.get('time_in_force') or '?'):<5} "
            f"{(o.get('status') or '?'):<10} "
            f"{(o.get('order_class') or '-') or '-':<8} "
            f"{_fmt_money(o.get('limit_price')):>11} "
            f"{_fmt_money(o.get('stop_price')):>11}  "
            f"{fld_str:<19} "
            f"{oid:<10}"
        )

    lines.append("")
    lines.append(bar)
    lines.append(f"End of export. {len(orders):,} order(s).")
    lines.append(bar)
    return "\n".join(lines) + "\n"


def render_jsonl(rows: list[dict]) -> str:
    """Canonical machine-readable form.

    One JSON object per line, fields preserved verbatim. Datetimes are
    emitted as UTC ISO 8601. Used for orders, activities, and any other
    list-of-records the exporter produces — shared serializer keeps
    field encoding consistent across companion files.
    """
    return "\n".join(
        json.dumps(o, default=_json_default, sort_keys=True) for o in rows
    ) + ("\n" if rows else "")


def _csv_cell(v):
    """Coerce one normalized value into a CSV-safe string.

    - None             -> "" (csv writes an empty cell)
    - datetime         -> UTC ISO 8601 (matches the JSONL contract)
    - list / dict      -> JSON-encoded (legs / nested structures stay
      lossless in a single cell instead of getting flattened weirdly)
    - everything else  -> pass through; csv.DictWriter str()s it
    """
    if v is None:
        return ""
    if isinstance(v, datetime):
        return v.astimezone(timezone.utc).isoformat()
    if isinstance(v, (list, dict)):
        return json.dumps(v, default=_json_default, sort_keys=True)
    return v


def render_orders_csv(rows: list[dict]) -> str:
    """Orders as a CSV table: header row = field names, one row per
    order. Header is the alphabetical union of every key seen across
    rows, so a future SDK addition surfaces as a new column without
    requiring a code change. Empty input yields an empty string (no
    header — we don't have an authoritative field list to invent)."""
    import csv
    import io

    if not rows:
        return ""
    fields = sorted({k for r in rows for k in r.keys()})
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=fields, lineterminator="\n",
        extrasaction="ignore",  # never raise on a missing key — pad ""
    )
    writer.writeheader()
    for r in rows:
        writer.writerow({k: _csv_cell(r.get(k)) for k in fields})
    return buf.getvalue()


_DAILY_PNL_FIELDS = [
    "date_et",
    "timestamp_utc",
    "equity",
    "daily_pnl",
    "daily_return_pct",
    "cumulative_pnl_vs_base",
    "cumulative_return_pct_vs_base",
    "base_value",
]


def render_daily_pnl_csv(history: dict) -> str:
    """Per-day equity + P&L table from /v2/account/portfolio_history.

    Alpaca returns parallel arrays (timestamp / equity / profit_loss /
    profit_loss_pct) plus a scalar base_value (= equity at the START of
    the period). We zip them into rows and add a derived `daily_pnl`
    column (equity[i] − equity[i−1]) so the operator can see per-day
    deltas, not just cumulative-vs-base.

    First row's daily_pnl uses base_value as the previous reference
    point — by definition of base_value, equity[0] − base_value IS the
    first day's P&L. If base_value is missing the first row's
    daily_pnl/daily_return_pct stay blank.

    Empty history -> empty string. Column order is fixed (NOT alphabetical):
    date first, base_value last — what an operator actually reads.
    """
    import csv
    import io

    timestamps = history.get("timestamp") or []
    equity_arr = history.get("equity") or []
    pl_arr = history.get("profit_loss") or []
    pl_pct_arr = history.get("profit_loss_pct") or []
    if not timestamps:
        return ""
    try:
        base_value = float(history.get("base_value")) if history.get("base_value") not in (None, "") else None
    except (TypeError, ValueError):
        base_value = None

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=_DAILY_PNL_FIELDS, lineterminator="\n",
    )
    writer.writeheader()

    prev_equity = base_value
    for i, ts in enumerate(timestamps):
        try:
            dt_utc = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            dt_utc = None
        date_et = dt_utc.astimezone(ET).strftime("%Y-%m-%d") if dt_utc else ""

        eq = _safe_float(equity_arr, i)
        cum_pnl = _safe_float(pl_arr, i)
        cum_pct = _safe_float(pl_pct_arr, i)

        if eq is not None and prev_equity is not None:
            daily_pnl = eq - prev_equity
            daily_pct = (daily_pnl / prev_equity) if prev_equity else None
        else:
            daily_pnl = None
            daily_pct = None

        writer.writerow({
            "date_et": date_et,
            "timestamp_utc": dt_utc.isoformat() if dt_utc else "",
            "equity": _csv_cell(eq),
            "daily_pnl": _csv_cell(daily_pnl),
            "daily_return_pct": _csv_cell(daily_pct),
            "cumulative_pnl_vs_base": _csv_cell(cum_pnl),
            "cumulative_return_pct_vs_base": _csv_cell(cum_pct),
            "base_value": _csv_cell(base_value),
        })
        prev_equity = eq if eq is not None else prev_equity

    return buf.getvalue()


def _safe_float(arr, i):
    try:
        v = arr[i]
    except (IndexError, TypeError):
        return None
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Alpaca order history.")
    p.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "alpaca" / "trades.txt",
        help="Text report path (default: data/alpaca/trades.txt). "
              "Companion JSONL/JSON files derive their paths from this stem.",
    )
    p.add_argument(
        "--jsonl", type=Path, default=None,
        help="Override the orders.jsonl companion path. By default it is "
              "auto-derived from --output (e.g. data/alpaca_trades.orders.jsonl).",
    )
    p.add_argument(
        "--no-companions", action="store_true",
        help="Emit only the .txt report. By default the orders.jsonl + "
              "activities.jsonl + account.json companions are also written.",
    )
    p.add_argument(
        "--skip-activities", action="store_true",
        help="Skip the /account/activities pull. Activities can be slow "
              "on long-lived accounts; orders + account snapshot still emit.",
    )
    p.add_argument(
        "--activity-types", default=None,
        help="Comma-separated activity types to fetch (e.g. 'FILL,DIV'). "
              "Default: ALL types (FILL + DIV + JNLC + ...).",
    )
    env_group = p.add_mutually_exclusive_group()
    env_group.add_argument(
        "--paper", dest="paper", action="store_true", default=None,
        help="Force the paper endpoint.",
    )
    env_group.add_argument(
        "--live", dest="paper", action="store_false",
        help="Force the live endpoint (REAL MONEY).",
    )
    p.add_argument("--since", default=None,
                   help="ISO date (YYYY-MM-DD) inclusive lower bound.")
    p.add_argument("--until", default=None,
                   help="ISO date (YYYY-MM-DD) inclusive upper bound.")
    p.add_argument("--page-limit", type=int, default=500,
                   help="Per-page fetch size for orders (Alpaca max 500).")
    p.add_argument("--activity-page-size", type=int, default=100,
                   help="Per-page fetch size for activities (Alpaca max 100).")
    return p.parse_args(argv)


def _companion_paths(output: Path) -> dict[str, Path]:
    """Derive sibling JSONL / CSV / JSON paths from the user's --output.

    `data/alpaca/trades.txt` →
      data/alpaca/trades.orders.jsonl
      data/alpaca/trades.orders.csv
      data/alpaca/trades.activities.jsonl
      data/alpaca/trades.account.json
      data/alpaca/trades.daily_pnl.csv
    """
    stem = output.with_suffix("")  # strip .txt
    return {
        "orders": stem.with_suffix(".orders.jsonl"),
        "orders_csv": stem.with_suffix(".orders.csv"),
        "activities": stem.with_suffix(".activities.jsonl"),
        "account": stem.with_suffix(".account.json"),
        "daily_pnl_csv": stem.with_suffix(".daily_pnl.csv"),
    }


def main(argv=None) -> int:
    args = _parse_args(argv)
    _load_env_file()

    api_key = os.environ.get("ALPACA_API_KEY")
    api_sec = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not api_sec:
        print(
            "ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not found in env or .env",
            file=sys.stderr,
        )
        return 2

    paper = args.paper if args.paper is not None else _read_paper_default()
    api_url = (
        "https://paper-api.alpaca.markets/v2" if paper
        else "https://api.alpaca.markets/v2"
    )
    env_label = "PAPER" if paper else "LIVE"

    since = (
        datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        if args.since else None
    )
    until = (
        datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc)
        if args.until else None
    )

    from alpaca.trading.client import TradingClient
    client = TradingClient(api_key, api_sec, paper=paper)

    # 1) Account snapshot (full field set).
    try:
        account_full = fetch_account_dump(client)
    except Exception as exc:
        print(f"ERROR: get_account failed: {exc}", file=sys.stderr)
        return 3
    account = {
        "id": str(account_full.get("id") or "?"),
        "account_number": str(account_full.get("account_number") or ""),
        "created_at": _et_str(account_full.get("created_at")),
    }

    # 2) Orders (all statuses, paginated, full pydantic dump per row).
    fetch_warning: str | None = None
    try:
        orders = fetch_all_orders(
            client, since=since, until=until, page_limit=args.page_limit,
        )
    except Exception as exc:
        fetch_warning = f"orders fetch aborted: {exc} — report may be incomplete"
        orders = []

    # 3) Daily P&L (portfolio_history 1D since account inception).
    try:
        daily_pnl_history = fetch_portfolio_history_daily(
            client, since=account_full.get("created_at"),
        )
    except Exception as exc:
        # fetch_portfolio_history_daily already swallows; belt-and-braces.
        print(f"WARN: portfolio_history aborted: {exc}", file=sys.stderr)
        daily_pnl_history = {}

    # 4) Account activities (paginated, full passthrough). Optional.
    activities: list[dict] = []
    activities_warning: str | None = None
    if not args.skip_activities:
        types = (
            [t.strip().upper() for t in args.activity_types.split(",") if t.strip()]
            if args.activity_types else None
        )
        try:
            activities = fetch_all_activities(
                client, activity_types=types, since=since, until=until,
                page_size=args.activity_page_size,
            )
        except Exception as exc:
            activities_warning = (
                f"activities fetch aborted: {exc} — companion file may be empty"
            )

    companions = _companion_paths(args.output)
    if args.jsonl is not None:
        companions["orders"] = args.jsonl

    # Pre-compute the header note so the report tells the reader where
    # the canonical files live. --no-companions hides this line.
    companion_note: list[str] | None = None
    if not args.no_companions:
        companion_note = [
            f"Companion files (canonical, lossless):",
            f"  orders.jsonl   -> {companions['orders']}",
            f"  orders.csv     -> {companions['orders_csv']}",
            f"  daily_pnl.csv  -> {companions['daily_pnl_csv']}",
            f"  activities     -> {companions['activities']}"
            + (" (skipped via --skip-activities)" if args.skip_activities else ""),
            f"  account        -> {companions['account']}",
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined_warning = " | ".join(w for w in (fetch_warning, activities_warning) if w)
    args.output.write_text(render_report(
        orders, account=account, env_label=env_label, api_url=api_url,
        since=since, until=until,
        fetch_warning=combined_warning or None,
        companion_note=companion_note,
    ))
    print(f"Wrote {args.output}  ({len(orders):,} order(s))")

    if args.no_companions:
        return 0

    def _emit(path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
        print(f"Wrote {path}")

    _emit(companions["orders"], render_jsonl(orders))
    _emit(companions["orders_csv"], render_orders_csv(orders))
    _emit(companions["daily_pnl_csv"], render_daily_pnl_csv(daily_pnl_history))

    if args.skip_activities:
        print(f"Skipped {companions['activities']}  (--skip-activities)")
    else:
        _emit(companions["activities"], render_jsonl(activities))

    _emit(
        companions["account"],
        json.dumps(account_full, default=_json_default, sort_keys=True, indent=2)
        + "\n",
    )

    return 0


def _json_default(o):
    if isinstance(o, datetime):
        return o.astimezone(timezone.utc).isoformat()
    return str(o)


if __name__ == "__main__":
    sys.exit(main())
