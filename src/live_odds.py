"""Fetch live/pre-match bookmaker odds for upcoming fixtures (optional API key)."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz, process

from src.utils import implied_probs, normalize_team

load_dotenv()

# Free tier: https://the-odds-api.com/
ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "").strip()
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"


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
            # Need home/away names from parent — handled by caller with team names
            if len(outcomes) >= 2:
                return outcomes  # type: ignore[return-value]
    return None


def fetch_live_odds_table() -> pd.DataFrame:
    """
    Returns DataFrame: date, home_team, away_team, odds_h, odds_d, odds_a, book_p_*.
    Empty if no API key or request fails.
    """
    if not ODDS_API_KEY or ODDS_API_KEY.startswith("your_"):
        return pd.DataFrame()

    try:
        resp = requests.get(
            ODDS_API_URL,
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "uk,eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"Odds API HTTP {resp.status_code}: {resp.text[:200]}")
            return pd.DataFrame()
        payload = resp.json()
    except requests.RequestException as exc:
        print(f"Odds API failed: {exc}")
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
        # Prefer Pinnacle
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

        # Map outcome names (API uses full club names) to H/D/A
        draw_odds = prices.get("Draw")
        home_odds = prices.get(home_raw) or prices.get(home)
        away_odds = prices.get(away_raw) or prices.get(away)
        if home_odds is None or away_odds is None:
            # fuzzy match keys
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
    return pd.DataFrame(rows)


def attach_live_odds(matches: pd.DataFrame) -> pd.DataFrame:
    """Left-join live odds onto an upcoming predictions frame."""
    live = fetch_live_odds_table()
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
    # Drop existing empty odds cols then merge
    drop = [c for c in cols if c not in ("date", "home_team", "away_team") and c in out.columns]
    out = out.drop(columns=drop, errors="ignore")
    merged = out.merge(live[cols], on=["date", "home_team", "away_team"], how="left")
    n = int(merged["odds_h"].notna().sum()) if "odds_h" in merged.columns else 0
    note = f"Live odds attached for {n}/{len(merged)} fixtures"
    print(note)
    # stash note for API (DataFrame.attrs can be flaky across copies)
    merged["_odds_note"] = note
    return merged
