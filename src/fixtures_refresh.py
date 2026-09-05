"""Refresh fixture statuses from openfootball and filter truly upcoming matches."""

from __future__ import annotations

import time
from zoneinfo import ZoneInfo

import pandas as pd

from src.parser import download_openfootball, parse_season_file
from src.utils import OPENFOOTBALL_DIR, ensure_dirs

_LAST_REFRESH_TS = 0.0
_REFRESH_EVERY_SEC = 30 * 60  # re-pull openfootball at most every 30 minutes
UK = ZoneInfo("Europe/London")


def _kickoff_ts_uk(date_val, kickoff) -> pd.Timestamp | None:
    """Build a timezone-aware UK kickoff timestamp (openfootball times are local UK)."""
    day = pd.to_datetime(date_val)
    if pd.isna(day):
        return None
    day = day.normalize()
    if kickoff is None or (isinstance(kickoff, float) and pd.isna(kickoff)) or str(kickoff).strip() == "":
        hh, mm = 15, 0
    else:
        try:
            parts = str(kickoff).strip().split(":")
            hh, mm = int(parts[0]), int(parts[1])
        except Exception:
            hh, mm = 15, 0
    return pd.Timestamp(
        year=int(day.year),
        month=int(day.month),
        day=int(day.day),
        hour=hh,
        minute=mm,
        tz=UK,
    )


def is_still_upcoming(row: pd.Series, now: pd.Timestamp | None = None) -> bool:
    """True if UK kickoff is still in the future (short grace for result lag)."""
    now = now or pd.Timestamp.now(tz=UK)
    if now.tzinfo is None:
        now = now.tz_localize(UK)
    else:
        now = now.tz_convert(UK)

    status = str(row.get("status", "scheduled"))
    outcome = row.get("outcome")
    has_outcome = (
        outcome is not None
        and not (isinstance(outcome, float) and pd.isna(outcome))
        and str(outcome) not in ("", "nan", "None")
    )
    has_score = pd.notna(row.get("home_goals")) and pd.notna(row.get("away_goals"))
    if status == "played" or has_outcome or has_score:
        return False

    ko = _kickoff_ts_uk(row.get("date"), row.get("kickoff"))
    if ko is None:
        return True
    # Drop once kickoff has passed; 15 min grace only for in-progress
    return now < (ko + pd.Timedelta(minutes=15))


def refresh_openfootball_statuses(force: bool = False) -> pd.DataFrame | None:
    """Optionally re-download openfootball and return freshly parsed recent seasons."""
    global _LAST_REFRESH_TS
    ensure_dirs()
    now = time.time()
    if not force and (now - _LAST_REFRESH_TS) < _REFRESH_EVERY_SEC:
        return None

    try:
        download_openfootball(force=force or (_LAST_REFRESH_TS == 0) or (now - _LAST_REFRESH_TS > _REFRESH_EVERY_SEC))
    except Exception as exc:  # noqa: BLE001
        print(f"openfootball refresh download skipped: {exc}")

    frames = []
    for season_dir in sorted(OPENFOOTBALL_DIR.glob("20*-*")):
        path = season_dir / "1-premierleague.txt"
        if not path.exists() or season_dir.name < "2025-26":
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
    if df.empty:
        return df
    mask = df.apply(lambda r: is_still_upcoming(r, now), axis=1)
    return df.loc[mask].copy()
