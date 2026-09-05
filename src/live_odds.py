"""Fetch live/pre-match bookmaker odds for upcoming fixtures (optional API key)."""

from __future__ import annotations

import json
import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz, process

from src.utils import CACHE_DIR, ROOT, implied_probs, normalize_team

load_dotenv(ROOT / ".env", override=True)

# Free tier: https://the-odds-api.com/ (~500 req/month)
ODDS_API_KEY = (os.getenv("THE_ODDS_API_KEY") or "").strip().strip('"').strip("'")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
# Cache aggressively — free tier burns fast if every page load / match click hits the API
ODDS_CACHE_TTL_SEC = int(os.getenv("ODDS_CACHE_TTL_SEC") or "2700")  # 45 min
_CACHE_PARQUET = CACHE_DIR / "live_odds.parquet"
_CACHE_META = CACHE_DIR / "live_odds_meta.json"

_mem_cache: dict | None = None  # {"ts": float, "df": DataFrame, "note": str}


def _best_market_odds(bookmakers: list) -> tuple[float, float, float] | None:
    """Prefer Pinnacle, else first bookmaker h2h market."""
    if not bookmakers:
        return None
    ordered = sorted(
        bookmakers,
        key=lambda b: 0 if "pinnacle" in str(b.get("key", "")).lower() else 1,
    )
    for book in ordered:
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: float(o["price"]) for o in market.get("outcomes", [])}
            if len(outcomes) >= 2:
                return outcomes  # type: ignore[return-value]
    return None


def _read_disk_cache() -> tuple[pd.DataFrame, dict] | None:
    if not _CACHE_PARQUET.exists() or not _CACHE_META.exists():
        return None
    try:
        meta = json.loads(_CACHE_META.read_text(encoding="utf-8"))
        df = pd.read_parquet(_CACHE_PARQUET)
        return df, meta
    except Exception as exc:  # noqa: BLE001
        print(f"Odds cache read failed: {exc}")
        return None


def _write_disk_cache(df: pd.DataFrame, meta: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_CACHE_PARQUET, index=False)
        _CACHE_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"Odds cache write failed: {exc}")


def _cache_age_sec(meta: dict) -> float:
    return max(0.0, time.time() - float(meta.get("fetched_at", 0)))


def fetch_live_odds_table(force: bool = False) -> pd.DataFrame:
    """
    Returns DataFrame: date, home_team, away_team, odds_h, odds_d, odds_a, book_p_*.
    Empty if no API key or request fails.

    Uses in-memory + disk cache (default 45 min) so slate loads and match detail
    share one Odds API credit instead of burning one per click.
    """
    global _mem_cache
    load_dotenv(ROOT / ".env", override=True)
    api_key = (os.getenv("THE_ODDS_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key or api_key.startswith("your_"):
        return pd.DataFrame()

    ttl = int(os.getenv("ODDS_CACHE_TTL_SEC") or ODDS_CACHE_TTL_SEC)

    if not force and _mem_cache is not None:
        age = time.time() - float(_mem_cache["ts"])
        if age < ttl and isinstance(_mem_cache.get("df"), pd.DataFrame):
            return _mem_cache["df"].copy()

    if not force:
        disk = _read_disk_cache()
        if disk is not None:
            df, meta = disk
            if _cache_age_sec(meta) < ttl and not df.empty:
                _mem_cache = {"ts": float(meta["fetched_at"]), "df": df, "note": meta.get("note", "")}
                remaining = meta.get("remaining")
                age_m = int(_cache_age_sec(meta) // 60)
                print(f"Odds cache hit ({age_m}m old; remaining last-seen={remaining})")
                return df.copy()

    try:
        # Single region keeps payload smaller; still 1 credit per request
        resp = requests.get(
            ODDS_API_URL,
            params={
                "apiKey": api_key,
                "regions": "uk",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
            timeout=30,
        )
        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")
        if remaining is not None:
            print(f"Odds API usage: used={used} remaining={remaining}")
        if resp.status_code != 200:
            print(f"Odds API HTTP {resp.status_code}: {resp.text[:200]}")
            # Serve stale cache rather than empty if API errors mid-month
            disk = _read_disk_cache()
            if disk is not None and not disk[0].empty:
                print("Serving stale odds cache after API error")
                return disk[0].copy()
            return pd.DataFrame()
        payload = resp.json()
    except requests.RequestException as exc:
        print(f"Odds API failed: {exc}")
        disk = _read_disk_cache()
        if disk is not None and not disk[0].empty:
            print("Serving stale odds cache after network error")
            return disk[0].copy()
        return pd.DataFrame()

    rows = []
    for event in payload:
        home_raw = event.get("home_team", "")
        away_raw = event.get("away_team", "")
        home = normalize_team(home_raw)
        away = normalize_team(away_raw)
        commence = event.get("commence_time")
        try:
            dt = pd.to_datetime(commence, utc=True).tz_convert(None).normalize()
        except Exception:
            continue

        prices: dict[str, float] = {}
        books = event.get("bookmakers", [])
        books = sorted(books, key=lambda b: 0 if b.get("key") == "pinnacle" else 1)
        for book in books:
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for o in market.get("outcomes", []):
                    prices[o["name"]] = float(o["price"])
                break
            if prices:
                break
        if not prices:
            continue

        draw_odds = prices.get("Draw")
        home_odds = prices.get(home_raw) or prices.get(home)
        away_odds = prices.get(away_raw) or prices.get(away)
        if home_odds is None or away_odds is None:
            keys = list(prices.keys())
            for label, target in (("home", home_raw), ("away", away_raw)):
                match = process.extractOne(target, keys, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= 70 and match[0] != "Draw":
                    if label == "home":
                        home_odds = prices[match[0]]
                    else:
                        away_odds = prices[match[0]]
        if not draw_odds or not home_odds or not away_odds:
            continue
        if home_odds <= 1 or draw_odds <= 1 or away_odds <= 1:
            continue
        ph, pd_, pa = implied_probs(home_odds, draw_odds, away_odds)
        rows.append(
            {
                "date": dt,
                "home_team": home,
                "away_team": away,
                "odds_h": home_odds,
                "odds_d": draw_odds,
                "odds_a": away_odds,
                "book_p_h": ph,
                "book_p_d": pd_,
                "book_p_a": pa,
                "odds_source": "the-odds-api",
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    now = time.time()
    meta = {
        "fetched_at": now,
        "ttl_sec": ttl,
        "remaining": resp.headers.get("x-requests-remaining"),
        "used": resp.headers.get("x-requests-used"),
        "n_events": len(df),
        "note": f"Cached live odds ({len(df)} events)",
    }
    _write_disk_cache(df, meta)
    _mem_cache = {"ts": now, "df": df, "note": meta["note"]}
    return df.copy()


def attach_live_odds(matches: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Left-join live odds onto an upcoming predictions frame."""
    live = fetch_live_odds_table(force=force)
    out = matches.copy()
    if live.empty:
        out.attrs["odds_note"] = (
            "No live odds. Add a free THE_ODDS_API_KEY to .env "
            "(https://the-odds-api.com/) and restart the API."
        )
        return out

    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    live["date"] = pd.to_datetime(live["date"]).dt.normalize()
    cols = [
        "date",
        "home_team",
        "away_team",
        "odds_h",
        "odds_d",
        "odds_a",
        "book_p_h",
        "book_p_d",
        "book_p_a",
        "odds_source",
    ]
    drop = [c for c in cols if c not in ("date", "home_team", "away_team") and c in out.columns]
    out = out.drop(columns=drop, errors="ignore")
    merged = out.merge(live[cols], on=["date", "home_team", "away_team"], how="left")
    n = int(merged["odds_h"].notna().sum()) if "odds_h" in merged.columns else 0

    disk = _read_disk_cache()
    age_m = int(_cache_age_sec(disk[1]) // 60) if disk else 0
    remaining = disk[1].get("remaining") if disk else None
    note = f"Live odds attached for {n}/{len(merged)} fixtures"
    if remaining is not None:
        note += f" · API credits left: {remaining}"
    if age_m >= 0 and disk:
        note += f" · cache {age_m}m old (refresh every {ODDS_CACHE_TTL_SEC // 60}m)"
    print(note)
    merged["_odds_note"] = note
    return merged
