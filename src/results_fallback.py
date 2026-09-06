"""Fallback finished-score ingest when openfootball lags (FPL + football-data)."""

from __future__ import annotations

import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.utils import ODDS_DIR, ensure_dirs, normalize_team, outcome_from_score, season_code

_LAST_FETCH_TS = 0.0
_REFRESH_EVERY_SEC = 15 * 60
_CACHE: pd.DataFrame | None = None

_FPL_LAST_TS = 0.0
_FPL_CACHE: pd.DataFrame | None = None

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"
UK = ZoneInfo("Europe/London")
_HEADERS = {"User-Agent": "PitchPulse/2.0 (local EPL predictor)"}


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


def fetch_fpl_results(force: bool = False) -> pd.DataFrame:
    """Current-season scores from the Fantasy Premier League API (updates faster than OF)."""
    global _FPL_LAST_TS, _FPL_CACHE
    now = time.time()
    if (
        not force
        and _FPL_CACHE is not None
        and (now - _FPL_LAST_TS) < _REFRESH_EVERY_SEC
    ):
        return _FPL_CACHE.copy()

    try:
        boot = requests.get(FPL_BOOTSTRAP, timeout=30, headers=_HEADERS)
        boot.raise_for_status()
        teams = {
            int(t["id"]): normalize_team(str(t.get("name") or ""))
            for t in boot.json().get("teams", [])
        }
        fx = requests.get(FPL_FIXTURES, timeout=30, headers=_HEADERS)
        fx.raise_for_status()
        fixtures = fx.json()
    except Exception as exc:  # noqa: BLE001
        print(f"FPL results failed: {exc}")
        return pd.DataFrame() if _FPL_CACHE is None else _FPL_CACHE.copy()

    rows = []
    for f in fixtures:
        hg, ag = f.get("team_h_score"), f.get("team_a_score")
        if hg is None or ag is None:
            continue
        home = teams.get(int(f.get("team_h") or 0), "")
        away = teams.get(int(f.get("team_a") or 0), "")
        if not home or not away:
            continue
        ko = pd.Timestamp(f.get("kickoff_time"))
        if pd.isna(ko):
            continue
        if ko.tzinfo is None:
            ko = ko.tz_localize("UTC")
        local = ko.tz_convert(UK)
        date = pd.Timestamp(year=local.year, month=local.month, day=local.day)
        hg_i, ag_i = int(hg), int(ag)
        rows.append(
            {
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_goals": hg_i,
                "away_goals": ag_i,
                "outcome": outcome_from_score(hg_i, ag_i),
                "status": "played",
                "match_id": _match_id(date, home, away),
                "source": "fpl",
            }
        )

    df = pd.DataFrame(rows)
    _FPL_CACHE = df
    _FPL_LAST_TS = now
    print(f"FPL results: {len(df)} scored rows")
    return df.copy()


def fetch_live_results(force: bool = False) -> pd.DataFrame:
    """Prefer FPL (current GW), fall back to football-data.co.uk."""
    frames = []
    for fn in (fetch_fpl_results, fetch_football_data_results):
        try:
            part = fn(force=force)
            if part is not None and not part.empty:
                frames.append(part)
        except Exception as exc:  # noqa: BLE001
            print(f"{fn.__name__} skipped: {exc}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["home_team", "away_team", "date"], keep="first")


def played_match_ids(force: bool = False) -> set[str]:
    df = fetch_live_results(force=force)
    if df.empty or "match_id" not in df.columns:
        return set()
    return set(df["match_id"].astype(str))


def apply_results_to_features(force: bool = False) -> int:
    """
    Patch data/processed/features.parquet scheduled rows with live scores.
    Returns number of rows updated.
    """
    from src.utils import DATA_PROCESSED

    results = fetch_live_results(force=force)
    path = DATA_PROCESSED / "features.parquet"
    if results.empty or not path.exists():
        return 0

    feats = pd.read_parquet(path)
    if "match_id" not in feats.columns:
        return 0

    feat_dates = pd.to_datetime(feats["date"]).dt.normalize()
    updated = 0
    by_id = results.set_index("match_id")
    for mid, res in by_id.iterrows():
        mask = feats["match_id"] == mid
        if not mask.any():
            res_date = pd.Timestamp(res["date"]).normalize()
            mask = (
                (feats["home_team"] == res["home_team"])
                & (feats["away_team"] == res["away_team"])
                & ((feat_dates - res_date).abs() <= pd.Timedelta(days=1))
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
        print(f"Patched {updated} feature rows from live results")
    return updated
