"""Fallback finished-score ingest from football-data.co.uk (when openfootball lags)."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from src.utils import ODDS_DIR, ensure_dirs, normalize_team, outcome_from_score, season_code

_LAST_FETCH_TS = 0.0
_REFRESH_EVERY_SEC = 30 * 60
_CACHE: pd.DataFrame | None = None

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"


def _match_id(date, home: str, away: str) -> str:
    d = pd.to_datetime(date).strftime("%Y%m%d")
    return f"{d}_{home.replace(' ', '')}_{away.replace(' ', '')}"


def fetch_football_data_results(
    start_year: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Return played EPL rows: date, home_team, away_team, home_goals, away_goals,
    outcome, match_id, status=played.
    Cached ~30 min to avoid hammering football-data.
    """
    global _LAST_FETCH_TS, _CACHE
    ensure_dirs()
    now = time.time()
    if (
        not force
        and _CACHE is not None
        and (now - _LAST_FETCH_TS) < _REFRESH_EVERY_SEC
    ):
        return _CACHE.copy()

    if start_year is None:
        # Current PL season start year (Aug–May)
        today = pd.Timestamp.now()
        start_year = today.year if today.month >= 8 else today.year - 1

    code = season_code(start_year)
    dest = ODDS_DIR / f"E0_{code}_results_live.csv"
    url = BASE_URL.format(code=code)

    try:
        resp = requests.get(url, timeout=45)
        if resp.status_code != 200 or "Div" not in resp.text[:300]:
            print(f"football-data results skip: HTTP {resp.status_code}")
            if dest.exists():
                raw = pd.read_csv(dest)
            else:
                return pd.DataFrame()
        else:
            dest.write_bytes(resp.content)
            raw = pd.read_csv(dest)
    except Exception as exc:  # noqa: BLE001
        print(f"football-data results failed: {exc}")
        if dest.exists():
            raw = pd.read_csv(dest)
        else:
            return pd.DataFrame()

    if raw.empty or "FTHG" not in raw.columns:
        return pd.DataFrame()

    rows = []
    for _, r in raw.iterrows():
        try:
            hg, ag = r.get("FTHG"), r.get("FTAG")
            if pd.isna(hg) or pd.isna(ag):
                continue
            home = normalize_team(str(r.get("HomeTeam", "")))
            away = normalize_team(str(r.get("AwayTeam", "")))
            if not home or not away:
                continue
            date = pd.to_datetime(r.get("Date"), dayfirst=True, errors="coerce")
            if pd.isna(date):
                continue
            hg_i, ag_i = int(hg), int(ag)
            rows.append(
                {
                    "date": date.normalize(),
                    "home_team": home,
                    "away_team": away,
                    "home_goals": hg_i,
                    "away_goals": ag_i,
                    "outcome": outcome_from_score(hg_i, ag_i),
                    "status": "played",
                    "match_id": _match_id(date, home, away),
                    "source": "football-data",
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    _CACHE = df
    _LAST_FETCH_TS = now
    print(f"football-data results: {len(df)} played rows ({code})")
    return df.copy()


def played_match_ids(force: bool = False) -> set[str]:
    df = fetch_football_data_results(force=force)
    if df.empty or "match_id" not in df.columns:
        return set()
    return set(df["match_id"].astype(str))


def apply_results_to_features(force: bool = False) -> int:
    """
    Patch data/processed/features.parquet scheduled rows with football-data scores.
    Returns number of rows updated.
    """
    from src.utils import DATA_PROCESSED

    results = fetch_football_data_results(force=force)
    path = DATA_PROCESSED / "features.parquet"
    if results.empty or not path.exists():
        return 0

    feats = pd.read_parquet(path)
    if "match_id" not in feats.columns:
        return 0

    updated = 0
    by_id = results.set_index("match_id")
    for mid, res in by_id.iterrows():
        mask = feats["match_id"] == mid
        if not mask.any():
            # try date+teams join
            mask = (
                (pd.to_datetime(feats["date"]).dt.normalize() == res["date"])
                & (feats["home_team"] == res["home_team"])
                & (feats["away_team"] == res["away_team"])
            )
        if not mask.any():
            continue
        idx = feats.index[mask]
        for i in idx:
            if feats.at[i, "status"] == "played" and pd.notna(feats.at[i, "outcome"]):
                continue
            feats.at[i, "home_goals"] = res["home_goals"]
            feats.at[i, "away_goals"] = res["away_goals"]
            feats.at[i, "outcome"] = res["outcome"]
            feats.at[i, "status"] = "played"
            updated += 1

    if updated:
        feats.to_parquet(path, index=False)
        print(f"Patched {updated} feature rows from football-data results")
    return updated
