"""Refresh fixture statuses from openfootball and filter truly upcoming matches."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.parser import download_openfootball, parse_season_file
from src.utils import DATA_PROCESSED, OPENFOOTBALL_DIR, ensure_dirs

_LAST_REFRESH_TS = 0.0
_REFRESH_EVERY_SEC = 30 * 60  # re-pull openfootball at most every 30 minutes


def _kickoff_ts(date_val, kickoff) -> pd.Timestamp | None:
    day = pd.to_datetime(date_val)
    if pd.isna(day):
        return None
    day = day.normalize()
    if kickoff is None or (isinstance(kickoff, float) and pd.isna(kickoff)) or str(kickoff).strip() == "":
        return day + pd.Timedelta(hours=15)  # assume mid-afternoon if unknown
    try:
        hh, mm = str(kickoff).strip().split(":")[:2]
        return day + pd.Timedelta(hours=int(hh), minutes=int(mm))
    except Exception:
        return day + pd.Timedelta(hours=15)


def is_still_upcoming(row: pd.Series, now: pd.Timestamp | None = None) -> bool:
    """True if kickoff is still in the future (with a short grace window)."""
    now = now or pd.Timestamp.now()
    # If we somehow have a final score, it's not upcoming
    status = str(row.get("status", "scheduled"))
    if status == "played" or (
        row.get("outcome") is not None and not (isinstance(row.get("outcome"), float) and pd.isna(row.get("outcome")))
        and str(row.get("outcome")) not in ("", "nan", "None")
    ):
        # outcome present means played
        if status != "scheduled" or (
            pd.notna(row.get("home_goals")) and pd.notna(row.get("away_goals"))
        ):
            return False

    ko = _kickoff_ts(row.get("date"), row.get("kickoff"))
    if ko is None:
        return True
    # Keep on slate until 2h after kickoff (in case result feed lags)
    return now < (ko + pd.Timedelta(hours=2))


def refresh_openfootball_statuses(force: bool = False) -> pd.DataFrame | None:
    """
    Optionally re-download openfootball, re-parse current season, return
    dataframe of matches that are still scheduled (no score yet).
    """
    global _LAST_REFRESH_TS
    ensure_dirs()
    now = time.time()
    if not force and (now - _LAST_REFRESH_TS) < _REFRESH_EVERY_SEC:
        return None

    try:
        download_openfootball(force=force or (now - _LAST_REFRESH_TS > _REFRESH_EVERY_SEC))
    except Exception as exc:  # noqa: BLE001
        print(f"openfootball refresh download skipped: {exc}")

    # Parse newest seasons that may contain fixtures
    frames = []
    for season_dir in sorted(OPENFOOTBALL_DIR.glob("20*-*")):
        path = season_dir / "1-premierleague.txt"
        if not path.exists():
            continue
        # Only bother with recent seasons
        if season_dir.name < "2025-26":
            continue
        try:
            df = parse_season_file(path, season=season_dir.name)
            if not df.empty:
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f"parse {season_dir.name} failed: {exc}")

    _LAST_REFRESH_TS = now
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def filter_upcoming_frame(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop matches that have kicked off / been played."""
    now = now or pd.Timestamp.now()
    if df.empty:
        return df
    mask = df.apply(lambda r: is_still_upcoming(r, now), axis=1)
    return df.loc[mask].copy()


def sync_upcoming_predictions(force_download: bool = False) -> Path | None:
    """
    Soft-refresh: mark newly scored fixtures and rewrite upcoming parquet
    without a full model retrain.
    """
    from src.predict import predict_upcoming

    fresh = refresh_openfootball_statuses(force=force_download)
    # Always re-run upcoming scoring from latest merged features if available
    try:
        predict_upcoming(merged=None, with_llm=False, as_of=None)
        return DATA_PROCESSED / "upcoming_predictions.parquet"
    except Exception as exc:  # noqa: BLE001
        print(f"upcoming rescore skipped: {exc}")
        if fresh is None:
            return None
        return None
