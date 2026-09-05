"""Download football-data.co.uk odds CSVs and merge onto openfootball matches."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from rapidfuzz import fuzz, process

from src.utils import (
    DATA_PROCESSED,
    ODDS_DIR,
    ensure_dirs,
    implied_probs,
    normalize_team,
    season_code,
    season_label,
)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

# Prefer Pinnacle closing lines when present, else Bet365, else market average.
ODDS_PRIORITY = [
    ("PSH", "PSD", "PSA"),  # Pinnacle
    ("B365H", "B365D", "B365A"),  # Bet365
    ("AvgH", "AvgD", "AvgA"),
    ("BWH", "BWD", "BWA"),
    ("IWH", "IWD", "IWA"),
]


def odds_url(start_year: int) -> str:
    return BASE_URL.format(code=season_code(start_year))


def download_odds_season(start_year: int, force: bool = False) -> Path | None:
    ensure_dirs()
    dest = ODDS_DIR / f"E0_{season_code(start_year)}.csv"
    if dest.exists() and not force and dest.stat().st_size > 100:
        return dest

    url = odds_url(start_year)
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200 or "Div" not in resp.text[:200]:
            print(f"  skip {start_year}: HTTP {resp.status_code} / unexpected content")
            return None
        dest.write_bytes(resp.content)
        print(f"  downloaded {dest.name} ({len(resp.content)} bytes)")
        return dest
    except requests.RequestException as exc:
        print(f"  skip {start_year}: {exc}")
        return None


def download_all_odds(start_year: int = 2000, end_year: int = 2025, force: bool = False) -> list[Path]:
    paths: list[Path] = []
    print("Downloading football-data.co.uk odds ...")
    for year in range(start_year, end_year + 1):
        p = download_odds_season(year, force=force)
        if p is not None:
            paths.append(p)
    return paths


def _pick_odds_columns(df: pd.DataFrame) -> tuple[str, str, str] | None:
    for cols in ODDS_PRIORITY:
        if all(c in df.columns for c in cols):
            return cols
    return None


def load_odds_season(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    if "Date" not in df.columns or "HomeTeam" not in df.columns:
        return pd.DataFrame()

    odds_cols = _pick_odds_columns(df)
    if odds_cols is None:
        return pd.DataFrame()

    oh, od, oa = odds_cols
    out = pd.DataFrame(
        {
            "date_raw": df["Date"],
            "home_team_raw": df["HomeTeam"],
            "away_team_raw": df["AwayTeam"],
            "odds_h": pd.to_numeric(df[oh], errors="coerce"),
            "odds_d": pd.to_numeric(df[od], errors="coerce"),
            "odds_a": pd.to_numeric(df[oa], errors="coerce"),
            "odds_source": oh[: -1] if oh.endswith("H") else oh,  # rough tag
            "home_shots": pd.to_numeric(df["HS"], errors="coerce") if "HS" in df.columns else pd.NA,
            "away_shots": pd.to_numeric(df["AS"], errors="coerce") if "AS" in df.columns else pd.NA,
            "home_sot": pd.to_numeric(df["HST"], errors="coerce") if "HST" in df.columns else pd.NA,
            "away_sot": pd.to_numeric(df["AST"], errors="coerce") if "AST" in df.columns else pd.NA,
        }
    )
    # football-data dates: DD/MM/YY or DD/MM/YYYY
    # football-data uses DD/MM/YY or DD/MM/YYYY depending on season
    out["date"] = pd.to_datetime(
        out["date_raw"], dayfirst=True, errors="coerce", format="mixed"
    )
    out["home_team"] = out["home_team_raw"].map(normalize_team)
    out["away_team"] = out["away_team_raw"].map(normalize_team)
    out = out.dropna(subset=["date", "odds_h", "odds_d", "odds_a", "home_team", "away_team"])
    out = out[(out["odds_h"] > 1) & (out["odds_d"] > 1) & (out["odds_a"] > 1)]

    probs = out.apply(
        lambda r: implied_probs(r["odds_h"], r["odds_d"], r["odds_a"]),
        axis=1,
        result_type="expand",
    )
    probs.columns = ["book_p_h", "book_p_d", "book_p_a"]
    out = pd.concat([out.reset_index(drop=True), probs], axis=1)
    return out[
        [
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
            "home_shots",
            "away_shots",
            "home_sot",
            "away_sot",
            "home_team_raw",
            "away_team_raw",
        ]
    ]


def load_all_odds(start_year: int = 2000, end_year: int = 2025, download: bool = True) -> pd.DataFrame:
    if download:
        download_all_odds(start_year, end_year)

    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        path = ODDS_DIR / f"E0_{season_code(year)}.csv"
        if not path.exists():
            continue
        df = load_odds_season(path)
        if not df.empty:
            df["season"] = season_label(year)
            frames.append(df)
            print(f"  odds {season_label(year)}: {len(df)} rows")

    if not frames:
        raise FileNotFoundError(f"No odds CSVs found in {ODDS_DIR}")
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def _fuzzy_map(name: str, choices: list[str], threshold: int = 85) -> str | None:
    if not choices:
        return None
    match = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= threshold:
        return match[0]
    return None


def merge_odds(matches: pd.DataFrame, odds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Left-join odds onto matches by date + normalized team names, with fuzzy fallback."""
    if odds is None:
        odds = load_all_odds(download=False)

    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    o = odds.copy()
    o["date"] = pd.to_datetime(o["date"]).dt.normalize()

    # Exact merge first
    odds_cols = [
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
        "home_shots",
        "away_shots",
        "home_sot",
        "away_sot",
    ]
    odds_cols = [c for c in odds_cols if c in o.columns]
    merged = m.merge(
        o[odds_cols],
        on=["date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_odds"),
    )

    missing = merged["odds_h"].isna()
    if missing.any():
        # Build lookup keyed by date for fuzzy repair
        odds_by_date: dict = {
            d: g for d, g in o.groupby("date", sort=False)
        }
        repairs = 0
        repair_cols = [c for c in odds_cols if c not in ("date", "home_team", "away_team")]
        for idx in merged.index[missing]:
            row = merged.loc[idx]
            day = row["date"]
            if day not in odds_by_date:
                continue
            day_odds = odds_by_date[day]
            home_choices = day_odds["home_team"].tolist()
            fh = _fuzzy_map(row["home_team"], home_choices)
            if fh is None:
                continue
            cand = day_odds[day_odds["home_team"] == fh]
            fa = _fuzzy_map(row["away_team"], cand["away_team"].tolist())
            if fa is None:
                continue
            hit = cand[cand["away_team"] == fa].iloc[0]
            for col in repair_cols:
                merged.at[idx, col] = hit[col]
            repairs += 1
        print(f"Fuzzy-repaired {repairs} odds matches")

    n_with = merged["odds_h"].notna().sum()
    print(f"Odds coverage: {n_with}/{len(merged)} ({100 * n_with / len(merged):.1f}%)")
    if "home_shots" in merged.columns:
        n_shots = merged["home_shots"].notna().sum()
        print(f"Shots coverage: {n_shots}/{len(merged)} ({100 * n_shots / len(merged):.1f}%)")
    return merged


def save_merged(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (DATA_PROCESSED / "matches_with_odds.parquet")
    df.to_parquet(path, index=False)
    df.to_csv(path.with_suffix(".csv"), index=False)
    print(f"Saved merged matches -> {path}")
    return path


if __name__ == "__main__":
    from src.parser import parse_all_seasons, save_matches

    matches = parse_all_seasons()
    save_matches(matches)
    odds = load_all_odds()
    merged = merge_odds(matches, odds)
    save_merged(merged)
