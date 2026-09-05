"""Shared helpers: paths, team-name canonicalization, season utilities."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OPENFOOTBALL_DIR = DATA_RAW / "openfootball"
ODDS_DIR = DATA_RAW / "odds"
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "cache"

# Canonical short names used across the project after normalization.
TEAM_ALIASES: dict[str, str] = {
    # Arsenal
    "arsenal": "Arsenal",
    "arsenal fc": "Arsenal",
    # Aston Villa
    "aston villa": "Aston Villa",
    "aston villa fc": "Aston Villa",
    # Bournemouth
    "bournemouth": "Bournemouth",
    "afc bournemouth": "Bournemouth",
    # Brentford
    "brentford": "Brentford",
    "brentford fc": "Brentford",
    # Brighton
    "brighton": "Brighton",
    "brighton & hove albion": "Brighton",
    "brighton and hove albion": "Brighton",
    "brighton & hove albion fc": "Brighton",
    "brighton and hove albion fc": "Brighton",
    # Burnley
    "burnley": "Burnley",
    "burnley fc": "Burnley",
    # Chelsea
    "chelsea": "Chelsea",
    "chelsea fc": "Chelsea",
    # Crystal Palace
    "crystal palace": "Crystal Palace",
    "crystal palace fc": "Crystal Palace",
    # Everton
    "everton": "Everton",
    "everton fc": "Everton",
    # Fulham
    "fulham": "Fulham",
    "fulham fc": "Fulham",
    # Ipswich
    "ipswich": "Ipswich",
    "ipswich town": "Ipswich",
    "ipswich town fc": "Ipswich",
    # Leicester
    "leicester": "Leicester",
    "leicester city": "Leicester",
    "leicester city fc": "Leicester",
    # Liverpool
    "liverpool": "Liverpool",
    "liverpool fc": "Liverpool",
    # Luton
    "luton": "Luton",
    "luton town": "Luton",
    "luton town fc": "Luton",
    # Man City
    "man city": "Man City",
    "manchester city": "Man City",
    "manchester city fc": "Man City",
    # Man United
    "man united": "Man United",
    "man utd": "Man United",
    "manchester united": "Man United",
    "manchester united fc": "Man United",
    "manchester u": "Man United",
    # Newcastle
    "newcastle": "Newcastle",
    "newcastle united": "Newcastle",
    "newcastle united fc": "Newcastle",
    "newcastle utd": "Newcastle",
    # Norwich
    "norwich": "Norwich",
    "norwich city": "Norwich",
    "norwich city fc": "Norwich",
    # Nott'm Forest
    "nottingham forest": "Nott'm Forest",
    "nottingham forest fc": "Nott'm Forest",
    "nottm forest": "Nott'm Forest",
    "nott'm forest": "Nott'm Forest",
    "forest": "Nott'm Forest",
    # Sheffield United / Wednesday
    "sheffield united": "Sheffield United",
    "sheffield united fc": "Sheffield United",
    "sheffield utd": "Sheffield United",
    "sheffield wednesday": "Sheffield Weds",
    "sheffield wednesday fc": "Sheffield Weds",
    "sheffield weds": "Sheffield Weds",
    # Southampton
    "southampton": "Southampton",
    "southampton fc": "Southampton",
    # Tottenham
    "tottenham": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "tottenham hotspur fc": "Tottenham",
    "spurs": "Tottenham",
    # West Ham
    "west ham": "West Ham",
    "west ham united": "West Ham",
    "west ham united fc": "West Ham",
    # Wolves
    "wolves": "Wolves",
    "wolverhampton": "Wolves",
    "wolverhampton wanderers": "Wolves",
    "wolverhampton wanderers fc": "Wolves",
    # West Brom
    "west brom": "West Brom",
    "west bromwich albion": "West Brom",
    "west bromwich": "West Brom",
    "west bromwich albion fc": "West Brom",
    # Watford
    "watford": "Watford",
    "watford fc": "Watford",
    # Cardiff
    "cardiff": "Cardiff",
    "cardiff city": "Cardiff",
    "cardiff city fc": "Cardiff",
    # Huddersfield
    "huddersfield": "Huddersfield",
    "huddersfield town": "Huddersfield",
    "huddersfield town fc": "Huddersfield",
    # Swansea
    "swansea": "Swansea",
    "swansea city": "Swansea",
    "swansea city fc": "Swansea",
    # Stoke
    "stoke": "Stoke",
    "stoke city": "Stoke",
    "stoke city fc": "Stoke",
    # Sunderland
    "sunderland": "Sunderland",
    "sunderland afc": "Sunderland",
    # Middlesbrough
    "middlesbrough": "Middlesbrough",
    "middlesbrough fc": "Middlesbrough",
    # Hull
    "hull": "Hull",
    "hull city": "Hull",
    "hull city afc": "Hull",
    # QPR
    "qpr": "QPR",
    "queens park rangers": "QPR",
    "queens park rangers fc": "QPR",
    # Reading
    "reading": "Reading",
    "reading fc": "Reading",
    # Wigan
    "wigan": "Wigan",
    "wigan athletic": "Wigan",
    "wigan athletic fc": "Wigan",
    # Blackburn
    "blackburn": "Blackburn",
    "blackburn rovers": "Blackburn",
    "blackburn rovers fc": "Blackburn",
    # Bolton
    "bolton": "Bolton",
    "bolton wanderers": "Bolton",
    "bolton wanderers fc": "Bolton",
    # Birmingham
    "birmingham": "Birmingham",
    "birmingham city": "Birmingham",
    "birmingham city fc": "Birmingham",
    # Blackpool
    "blackpool": "Blackpool",
    "blackpool fc": "Blackpool",
    # Portsmouth
    "portsmouth": "Portsmouth",
    "portsmouth fc": "Portsmouth",
    # Charlton
    "charlton": "Charlton",
    "charlton athletic": "Charlton",
    "charlton athletic fc": "Charlton",
    # Derby
    "derby": "Derby",
    "derby county": "Derby",
    "derby county fc": "Derby",
    # Leeds
    "leeds": "Leeds",
    "leeds united": "Leeds",
    "leeds united fc": "Leeds",
    # Coventry
    "coventry": "Coventry",
    "coventry city": "Coventry",
    "coventry city fc": "Coventry",
    # Oldham / Barnsley / others historical
    "oldham": "Oldham",
    "oldham athletic": "Oldham",
    "barnsley": "Barnsley",
    "barnsley fc": "Barnsley",
    "bradford": "Bradford",
    "bradford city": "Bradford",
    "bradford city afc": "Bradford",
    "ipswich town": "Ipswich",
}


def normalize_team(name: str) -> str:
    """Map a raw team string to a canonical short name."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    key = re.sub(r"\s+", " ", str(name).strip().lower())
    key = key.replace(".", "")
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    # Strip trailing FC / AFC if still unknown
    stripped = re.sub(r"\s+(fc|afc)$", "", key).strip()
    if stripped in TEAM_ALIASES:
        return TEAM_ALIASES[stripped]
    # Title-case fallback so merges can still attempt fuzzy match
    return str(name).strip()


def season_code(start_year: int) -> str:
    """2000 -> '0001', 2023 -> '2324'."""
    end = (start_year + 1) % 100
    return f"{start_year % 100:02d}{end:02d}"


def season_label(start_year: int) -> str:
    """2000 -> '2000-01'."""
    return f"{start_year}-{ (start_year + 1) % 100:02d}"


def parse_season_label(label: str) -> int:
    """'2023-24' or '2023/24' -> 2023."""
    m = re.match(r"(\d{4})[-/](\d{2})", label.strip())
    if not m:
        raise ValueError(f"Unrecognized season label: {label}")
    return int(m.group(1))


def ensure_dirs() -> None:
    for p in (OPENFOOTBALL_DIR, ODDS_DIR, DATA_PROCESSED, MODELS_DIR, CACHE_DIR):
        p.mkdir(parents=True, exist_ok=True)


def outcome_from_score(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def implied_probs(odds_h: float, odds_d: float, odds_a: float) -> tuple[float, float, float]:
    """Convert decimal odds to overround-normalized implied probabilities."""
    inv = [1.0 / o for o in (odds_h, odds_d, odds_a)]
    total = sum(inv)
    return inv[0] / total, inv[1] / total, inv[2] / total
