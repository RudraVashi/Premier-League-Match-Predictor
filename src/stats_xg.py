"""Ingest Understat xG and FBref possession via soccerdata; cache under data/raw."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.utils import DATA_RAW, DATA_PROCESSED, ensure_dirs, normalize_team, season_label

XG_DIR = DATA_RAW / "understat"
POSS_DIR = DATA_RAW / "fbref"


def _season_years(start_year: int = 2014, end_year: int = 2025) -> list[str]:
    # soccerdata often wants '2023' or '2324' depending on source; Understat uses start year
    return [str(y) for y in range(start_year, end_year + 1)]


def fetch_understat_xg(
    start_year: int = 2014,
    end_year: int = 2025,
    force: bool = False,
) -> pd.DataFrame:
    """
    Pull match-level xG from Understat (ENG Premier League).
    Returns columns: date, home_team, away_team, home_xg, away_xg
    """
    ensure_dirs()
    XG_DIR.mkdir(parents=True, exist_ok=True)
    cache = XG_DIR / "match_xg.parquet"
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        print(f"Loaded cached Understat xG: {len(df)} rows")
        return df

    try:
        import soccerdata as sd
    except ImportError as exc:
        raise ImportError("Install soccerdata: pip install soccerdata") from exc

    frames: list[pd.DataFrame] = []
    seasons = _season_years(start_year, end_year)
    print(f"Fetching Understat xG for seasons {seasons[0]}-{seasons[-1]} ...")
    try:
        understat = sd.Understat(
            leagues="ENG-Premier League",
            seasons=seasons,
            data_dir=XG_DIR / "soccerdata_cache",
            no_cache=force,
        )
        schedule = understat.read_schedule()
    except Exception as exc:  # noqa: BLE001
        print(f"Understat fetch failed: {exc}")
        if cache.exists():
            return pd.read_parquet(cache)
        return pd.DataFrame(
            columns=["date", "home_team", "away_team", "home_xg", "away_xg"]
        )

    df = schedule.reset_index() if hasattr(schedule, "reset_index") else schedule.copy()
    # Column names vary by soccerdata version
    colmap_candidates = {
        "date": ["date", "game_date", "datetime"],
        "home": ["home_team", "home", "team_home"],
        "away": ["away_team", "away", "team_away"],
        "hxg": ["home_xg", "xg_home", "home_xG", "xG_home"],
        "axg": ["away_xg", "xg_away", "away_xG", "xG_away"],
    }

    def pick(cands: list[str]) -> str | None:
        lower = {c.lower(): c for c in df.columns}
        for c in cands:
            if c in df.columns:
                return c
            if c.lower() in lower:
                return lower[c.lower()]
        # fuzzy contains
        for col in df.columns:
            cl = col.lower()
            for c in cands:
                if c.lower() in cl:
                    return col
        return None

    dcol = pick(colmap_candidates["date"])
    hcol = pick(colmap_candidates["home"])
    acol = pick(colmap_candidates["away"])
    hxcol = pick(colmap_candidates["hxg"])
    axcol = pick(colmap_candidates["axg"])

    if not all([dcol, hcol, acol, hxcol, axcol]):
        print(f"Understat columns unexpected: {list(df.columns)[:30]}")
        # Try alternate: team match stats
        try:
            tms = understat.read_team_match_stats()
            tms = tms.reset_index() if hasattr(tms, "reset_index") else tms
            print(f"team_match_stats cols: {list(tms.columns)[:40]}")
        except Exception as exc2:  # noqa: BLE001
            print(f"team_match_stats also failed: {exc2}")
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_xg", "away_xg"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[dcol], errors="coerce").dt.normalize(),
            "home_team": df[hcol].map(normalize_team),
            "away_team": df[acol].map(normalize_team),
            "home_xg": pd.to_numeric(df[hxcol], errors="coerce"),
            "away_xg": pd.to_numeric(df[axcol], errors="coerce"),
        }
    ).dropna(subset=["date", "home_team", "away_team", "home_xg", "away_xg"])

    out.to_parquet(cache, index=False)
    out.to_csv(cache.with_suffix(".csv"), index=False)
    print(f"Saved Understat xG: {len(out)} rows -> {cache}")
    return out


def fetch_fbref_possession(
    start_year: int = 2017,
    end_year: int = 2025,
    force: bool = False,
) -> pd.DataFrame:
    """
    Pull match possession from FBref when available.
    Best-effort — FBref scrapes often need a browser stack; we fail soft.
    """
    ensure_dirs()
    POSS_DIR.mkdir(parents=True, exist_ok=True)
    cache = POSS_DIR / "match_possession.parquet"
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        print(f"Loaded cached FBref possession: {len(df)} rows")
        return df

    # Soft-skip by default: soccerdata FBref frequently requires a live browser
    # driver and can hang on Windows. Prefer empty + UI note over blocking the pipeline.
    if os.environ.get("EPL_FETCH_FBREF", "").lower() not in {"1", "true", "yes"}:
        print(
            "Skipping FBref possession (set EPL_FETCH_FBREF=1 to enable). "
            "xG + shots still used."
        )
        empty = pd.DataFrame(
            columns=["date", "home_team", "away_team", "home_poss", "away_poss"]
        )
        empty.to_parquet(cache, index=False)
        return empty

    try:
        import soccerdata as sd
    except ImportError:
        print("soccerdata not installed; skipping possession")
        return pd.DataFrame(
            columns=["date", "home_team", "away_team", "home_poss", "away_poss"]
        )

    season_args = [str(y) for y in range(start_year, end_year + 1)]
    print(f"Fetching FBref possession for {season_args[0]}-{season_args[-1]} ...")
    try:
        fbref = sd.FBref(
            leagues="ENG-Premier League",
            seasons=season_args,
            data_dir=POSS_DIR / "soccerdata_cache",
            no_cache=force,
        )
        try:
            stats = fbref.read_team_match_stats(stat_type="summary")
        except TypeError:
            stats = fbref.read_team_match_stats()
        except Exception:
            stats = fbref.read_schedule()
    except Exception as exc:  # noqa: BLE001
        print(f"FBref fetch failed (possession best-effort): {exc}")
        empty = pd.DataFrame(
            columns=["date", "home_team", "away_team", "home_poss", "away_poss"]
        )
        empty.to_parquet(cache, index=False)
        return empty

    df = stats.reset_index() if hasattr(stats, "reset_index") else stats.copy()
    print(f"FBref columns sample: {list(df.columns)[:40]}")

    poss_col = None
    for c in df.columns:
        if "poss" in str(c).lower():
            poss_col = c
            break

    if poss_col is None:
        print("No possession column found in FBref pull; continuing without it")
        empty = pd.DataFrame(
            columns=["date", "home_team", "away_team", "home_poss", "away_poss"]
        )
        empty.to_parquet(cache, index=False)
        return empty

    date_col = next((c for c in df.columns if str(c).lower() in ("date", "game", "datetime")), None)
    team_col = next((c for c in df.columns if str(c).lower() in ("team", "squad")), None)
    home_flag = next(
        (c for c in df.columns if "home" in str(c).lower() and df[c].dtype == bool),
        None,
    )

    if date_col and team_col and home_flag is not None:
        tmp = df[[date_col, team_col, poss_col, home_flag]].copy()
        tmp.columns = ["date", "team", "poss", "is_home"]
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.normalize()
        tmp["team"] = tmp["team"].map(normalize_team)
        tmp["poss"] = pd.to_numeric(tmp["poss"], errors="coerce")
        home = tmp[tmp["is_home"]].rename(columns={"team": "home_team", "poss": "home_poss"})
        away = tmp[~tmp["is_home"]].rename(columns={"team": "away_team", "poss": "away_poss"})
        out = home.merge(away, on="date", how="inner")
        out = out[["date", "home_team", "away_team", "home_poss", "away_poss"]].dropna()
    else:
        hposs = next((c for c in df.columns if "home" in str(c).lower() and "poss" in str(c).lower()), None)
        aposs = next((c for c in df.columns if "away" in str(c).lower() and "poss" in str(c).lower()), None)
        hteam = next((c for c in df.columns if str(c).lower() in ("home_team", "home")), None)
        ateam = next((c for c in df.columns if str(c).lower() in ("away_team", "away")), None)
        if not all([date_col, hteam, ateam, hposs, aposs]):
            empty = pd.DataFrame(
                columns=["date", "home_team", "away_team", "home_poss", "away_poss"]
            )
            empty.to_parquet(cache, index=False)
            return empty
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
                "home_team": df[hteam].map(normalize_team),
                "away_team": df[ateam].map(normalize_team),
                "home_poss": pd.to_numeric(df[hposs], errors="coerce"),
                "away_poss": pd.to_numeric(df[aposs], errors="coerce"),
            }
        ).dropna()

    out.to_parquet(cache, index=False)
    out.to_csv(cache.with_suffix(".csv"), index=False)
    print(f"Saved FBref possession: {len(out)} rows -> {cache}")
    return out


def merge_advanced_stats(
    matches: pd.DataFrame,
    xg: pd.DataFrame | None = None,
    possession: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-join xG and possession onto matches."""
    out = matches.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    if xg is None:
        xg = fetch_understat_xg()
    if possession is None:
        possession = fetch_fbref_possession()

    if not xg.empty:
        xg = xg.copy()
        xg["date"] = pd.to_datetime(xg["date"]).dt.normalize()
        before = out["home_xg"].notna().sum() if "home_xg" in out.columns else 0
        out = out.drop(columns=[c for c in ("home_xg", "away_xg") if c in out.columns], errors="ignore")
        out = out.merge(
            xg[["date", "home_team", "away_team", "home_xg", "away_xg"]],
            on=["date", "home_team", "away_team"],
            how="left",
        )
        print(
            f"xG coverage: {out['home_xg'].notna().sum()}/{len(out)} "
            f"({100 * out['home_xg'].notna().mean():.1f}%)"
        )
    else:
        out["home_xg"] = pd.NA
        out["away_xg"] = pd.NA
        print("xG coverage: 0 (Understat unavailable)")

    if not possession.empty:
        possession = possession.copy()
        possession["date"] = pd.to_datetime(possession["date"]).dt.normalize()
        out = out.drop(
            columns=[c for c in ("home_poss", "away_poss") if c in out.columns],
            errors="ignore",
        )
        out = out.merge(
            possession[["date", "home_team", "away_team", "home_poss", "away_poss"]],
            on=["date", "home_team", "away_team"],
            how="left",
        )
        print(
            f"Possession coverage: {out['home_poss'].notna().sum()}/{len(out)} "
            f"({100 * out['home_poss'].notna().mean():.1f}%)"
        )
    else:
        out["home_poss"] = pd.NA
        out["away_poss"] = pd.NA
        print("Possession coverage: 0 (FBref unavailable)")

    path = DATA_PROCESSED / "matches_with_stats.parquet"
    ensure_dirs()
    out.to_parquet(path, index=False)
    return out


if __name__ == "__main__":
    xg = fetch_understat_xg(force=False)
    poss = fetch_fbref_possession(force=False)
    print(xg.head())
    print(poss.head())
