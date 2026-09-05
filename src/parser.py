"""Parse openfootball england football.txt season files into a match DataFrame."""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

from src.utils import (
    OPENFOOTBALL_DIR,
    DATA_PROCESSED,
    ensure_dirs,
    normalize_team,
    outcome_from_score,
    season_label,
)

OPENFOOTBALL_ZIP = (
    "https://github.com/openfootball/england/archive/refs/heads/master.zip"
)

# Newer openfootball layout: "Home FC    v Away FC    1-0 (0-0)"
MATCH_V_RE = re.compile(
    r"^\s*(?:(\d{1,2}:\d{2})\s+)?"
    r"(.+?)\s+v\s+(.+?)\s+"
    r"(\d+)\s*-\s*(\d+)"
    r"(?:\s+\((\d+)\s*-\s*(\d+)\))?\s*$",
    re.IGNORECASE,
)

# Fixture-only (no score yet): "Home FC    v Away FC"
MATCH_V_FIXTURE_RE = re.compile(
    r"^\s*(?:(\d{1,2}:\d{2})\s+)?"
    r"(.+?)\s+v\s+(.+?)\s*$",
    re.IGNORECASE,
)

# Older layout: "Home FC 1-0 (0-0) Away FC"
MATCH_SCORE_RE = re.compile(
    r"^\s*(?:(\d{1,2}:\d{2})\s+)?"
    r"(.+?)\s+(\d+)\s*-\s*(\d+)\s+"
    r"(?:\((\d+)\s*-\s*(\d+)\)\s+)?"
    r"(.+?)\s*$"
)

DATE_RE = re.compile(
    r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(\d{1,2})(?:\s+(\d{4}))?\s*$"
)

HEADER_DATE_RE = re.compile(
    r"#\s*Date\s+\w+\s+\w+\s+\d+\s+(\d{4})\s*-\s*\w+\s+\w+\s+\d+\s+(\d{4})",
    re.IGNORECASE,
)

MATCHDAY_RE = re.compile(r"Matchday\s+(\d+)", re.IGNORECASE)

MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def download_openfootball(force: bool = False) -> Path:
    """Download openfootball/england master zip and extract premier-league txt files."""
    ensure_dirs()
    marker = OPENFOOTBALL_DIR / ".downloaded"
    if marker.exists() and not force and any(OPENFOOTBALL_DIR.rglob("*-premierleague.txt")):
        return OPENFOOTBALL_DIR

    print("Downloading openfootball/england ...")
    resp = requests.get(OPENFOOTBALL_ZIP, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            # england-master/2023-24/1-premierleague.txt
            if not name.endswith("1-premierleague.txt"):
                continue
            parts = Path(name).parts
            # Find season folder like 2023-24
            season_dirs = [p for p in parts if re.match(r"^\d{4}-\d{2}$", p)]
            if not season_dirs:
                continue
            season = season_dirs[0]
            dest = OPENFOOTBALL_DIR / season / "1-premierleague.txt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(name))

    marker.write_text("ok", encoding="utf-8")
    n = len(list(OPENFOOTBALL_DIR.rglob("*-premierleague.txt")))
    print(f"Extracted {n} Premier League season files to {OPENFOOTBALL_DIR}")
    return OPENFOOTBALL_DIR


def _resolve_year(month: int, start_year: int, end_year: int) -> int:
    # Premier League: Aug–Dec belong to start_year; Jan–Jul to end_year
    return start_year if month >= 8 else end_year


def parse_season_file(path: Path, season: str | None = None) -> pd.DataFrame:
    """Parse one football.txt premier-league file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    if season is None:
        season = path.parent.name

    start_year = int(season.split("-")[0])
    end_year = start_year + 1

    for line in lines[:20]:
        m = HEADER_DATE_RE.search(line)
        if m:
            start_year, end_year = int(m.group(1)), int(m.group(2))
            break

    rows: list[dict] = []
    current_date: pd.Timestamp | None = None
    matchday: int | None = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        md = MATCHDAY_RE.search(line)
        if md:
            matchday = int(md.group(1))
            continue

        dm = DATE_RE.match(line)
        if dm:
            month = MONTHS[dm.group(2)]
            day = int(dm.group(3))
            year = int(dm.group(4)) if dm.group(4) else _resolve_year(month, start_year, end_year)
            current_date = pd.Timestamp(year=year, month=month, day=day)
            continue

        if current_date is None:
            continue

        kickoff = home = away = hg = ag = ht_h = ht_a = None
        status = "played"
        mv = MATCH_V_RE.match(line)
        if mv:
            kickoff, home, away, hg, ag, ht_h, ht_a = mv.groups()
        else:
            ms = MATCH_SCORE_RE.match(line)
            if ms:
                kickoff, home, hg, ag, ht_h, ht_a, away = ms.groups()
                # Guard: reject lines that are clearly the v-format mis-captured
                if " v " in home.lower() or " v " in away.lower():
                    continue
            else:
                fx = MATCH_V_FIXTURE_RE.match(line)
                if not fx:
                    continue
                kickoff, home, away = fx.groups()
                status = "scheduled"
                hg = ag = ht_h = ht_a = None

        if status == "played":
            home_goals, away_goals = int(hg), int(ag)
            outcome = outcome_from_score(home_goals, away_goals)
        else:
            home_goals = away_goals = pd.NA
            outcome = pd.NA

        rows.append(
            {
                "date": current_date,
                "season": season,
                "matchday": matchday,
                "kickoff": kickoff,
                "home_team_raw": home.strip(),
                "away_team_raw": away.strip(),
                "home_team": normalize_team(home),
                "away_team": normalize_team(away),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "ht_home": int(ht_h) if ht_h is not None else pd.NA,
                "ht_away": int(ht_a) if ht_a is not None else pd.NA,
                "outcome": outcome,
                "status": status,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def parse_all_seasons(
    start_year: int = 2000,
    end_year: int = 2026,
    download: bool = True,
    include_scheduled: bool = True,
) -> pd.DataFrame:
    """Parse all available premier-league seasons in [start_year, end_year]."""
    if download:
        download_openfootball()

    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        season = season_label(year)
        path = OPENFOOTBALL_DIR / season / "1-premierleague.txt"
        if not path.exists():
            # Some repos use 2000-01 style already; also try alternate
            candidates = list(OPENFOOTBALL_DIR.glob(f"{year}-*/1-premierleague.txt"))
            if not candidates:
                continue
            path = candidates[0]
            season = path.parent.name
        df = parse_season_file(path, season=season)
        if not include_scheduled and not df.empty and "status" in df.columns:
            df = df[df["status"] != "scheduled"]
        if not df.empty:
            frames.append(df)
            n_sched = int((df["status"] == "scheduled").sum()) if "status" in df.columns else 0
            print(f"  {season}: {len(df)} matches ({n_sched} scheduled)")

    if not frames:
        raise FileNotFoundError(
            f"No openfootball season files found under {OPENFOOTBALL_DIR}. "
            "Run download_openfootball() first."
        )

    out = pd.concat(frames, ignore_index=True)
    if "status" not in out.columns:
        out["status"] = "played"
    out = out.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    out["match_id"] = (
        out["date"].dt.strftime("%Y%m%d")
        + "_"
        + out["home_team"].str.replace(" ", "", regex=False)
        + "_"
        + out["away_team"].str.replace(" ", "", regex=False)
    )
    return out


def save_matches(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (DATA_PROCESSED / "matches.parquet")
    df.to_parquet(path, index=False)
    df.to_csv(path.with_suffix(".csv"), index=False)
    print(f"Saved {len(df)} matches -> {path}")
    return path


if __name__ == "__main__":
    matches = parse_all_seasons()
    save_matches(matches)
    print(matches.head())
    print(matches["season"].value_counts().sort_index())
