"""Feature engineering: Elo, form, H2H, rest, plus rolling xG / shots / possession."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATA_PROCESSED, ensure_dirs

DEFAULT_ELO = 1500.0
ELO_K = 20.0
HOME_ADVANTAGE = 65.0


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _elo_update(rating: float, expected: float, score: float, k: float = ELO_K) -> float:
    return rating + k * (score - expected)


def _points(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def _form_string(hist: list[dict], n: int = 5) -> str:
    recent = hist[-n:]
    if not recent:
        return "-"
    letters = []
    for m in recent:
        if m["pts"] == 3:
            letters.append("W")
        elif m["pts"] == 1:
            letters.append("D")
        else:
            letters.append("L")
    return "-".join(letters)


def _mean_key(hist: list[dict], key: str, n: int, default: float = 0.0) -> float:
    recent = [m for m in hist[-n:] if m.get(key) is not None and not pd.isna(m.get(key))]
    if not recent:
        return default
    return float(np.mean([m[key] for m in recent]))


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Chronologically walk matches and compute pre-match features only.
    Scheduled fixtures get features but do not update Elo/history.
    """
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "status" not in df.columns:
        df["status"] = "played"
    df = df.sort_values(["date", "match_id"] if "match_id" in df.columns else ["date"]).reset_index(
        drop=True
    )

    elo: dict[str, float] = defaultdict(lambda: DEFAULT_ELO)
    history: dict[str, list[dict]] = defaultdict(list)
    last_date: dict[str, pd.Timestamp] = {}

    feature_rows: list[dict] = []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        date = row["date"]
        status = row.get("status", "played")

        home_elo = elo[home]
        away_elo = elo[away]
        home_hist = history[home]
        away_hist = history[away]

        def window_stats(hist: list[dict], n: int, home_flag: bool | None = None) -> dict:
            subset = hist
            if home_flag is True:
                subset = [m for m in hist if m["is_home"]]
            elif home_flag is False:
                subset = [m for m in hist if not m["is_home"]]
            recent = subset[-n:] if subset else []
            if not recent:
                return {"pts": 0.0, "gd": 0.0, "gf": 0.0, "ga": 0.0, "wins": 0.0, "n": 0}
            return {
                "pts": float(np.mean([m["pts"] for m in recent])),
                "gd": float(np.mean([m["gd"] for m in recent])),
                "gf": float(np.mean([m["gf"] for m in recent])),
                "ga": float(np.mean([m["ga"] for m in recent])),
                "wins": float(np.mean([1.0 if m["pts"] == 3 else 0.0 for m in recent])),
                "n": len(recent),
            }

        h5 = window_stats(home_hist, 5)
        h10 = window_stats(home_hist, 10)
        a5 = window_stats(away_hist, 5)
        a10 = window_stats(away_hist, 10)
        h_home = window_stats(home_hist, 5, home_flag=True)
        a_away = window_stats(away_hist, 5, home_flag=False)

        h2h = [m for m in home_hist if m["opponent"] == away][-5:]
        if h2h:
            h2h_pts = float(np.mean([m["pts"] for m in h2h]))
            h2h_gd = float(np.mean([m["gd"] for m in h2h]))
            h2h_n = len(h2h)
        else:
            h2h_pts, h2h_gd, h2h_n = 1.0, 0.0, 0

        rest_home = (date - last_date[home]).days if home in last_date else 7
        rest_away = (date - last_date[away]).days if away in last_date else 7
        exp_home = _elo_expected(home_elo + HOME_ADVANTAGE, away_elo)

        home_xg_for_5 = _mean_key(home_hist, "xg_for", 5)
        home_xg_against_5 = _mean_key(home_hist, "xg_against", 5)
        away_xg_for_5 = _mean_key(away_hist, "xg_for", 5)
        away_xg_against_5 = _mean_key(away_hist, "xg_against", 5)
        home_shots_5 = _mean_key(home_hist, "shots", 5)
        away_shots_5 = _mean_key(away_hist, "shots", 5)
        home_sot_5 = _mean_key(home_hist, "sot", 5)
        away_sot_5 = _mean_key(away_hist, "sot", 5)
        home_poss_5 = _mean_key(home_hist, "poss", 5, default=50.0)
        away_poss_5 = _mean_key(away_hist, "poss", 5, default=50.0)

        feat = {
            "match_id": row.get("match_id"),
            "date": date,
            "season": row.get("season"),
            "matchday": row.get("matchday"),
            "kickoff": row.get("kickoff"),
            "home_team": home,
            "away_team": away,
            "status": status,
            "outcome": row.get("outcome"),
            "home_goals": row.get("home_goals"),
            "away_goals": row.get("away_goals"),
            "home_form_str": _form_string(home_hist, 5),
            "away_form_str": _form_string(away_hist, 5),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "elo_expected_home": exp_home,
            "home_form_pts_5": h5["pts"],
            "home_form_gd_5": h5["gd"],
            "home_form_pts_10": h10["pts"],
            "home_form_gd_10": h10["gd"],
            "away_form_pts_5": a5["pts"],
            "away_form_gd_5": a5["gd"],
            "away_form_pts_10": a10["pts"],
            "away_form_gd_10": a10["gd"],
            "form_pts_diff_5": h5["pts"] - a5["pts"],
            "form_gd_diff_5": h5["gd"] - a5["gd"],
            "home_home_pts_5": h_home["pts"],
            "home_home_gd_5": h_home["gd"],
            "away_away_pts_5": a_away["pts"],
            "away_away_gd_5": a_away["gd"],
            "h2h_home_pts": h2h_pts,
            "h2h_home_gd": h2h_gd,
            "h2h_n": h2h_n,
            "rest_home": min(rest_home, 30),
            "rest_away": min(rest_away, 30),
            "rest_diff": min(rest_home, 30) - min(rest_away, 30),
            "home_matches_played": len(home_hist),
            "away_matches_played": len(away_hist),
            "home_xg_for_5": home_xg_for_5,
            "home_xg_against_5": home_xg_against_5,
            "away_xg_for_5": away_xg_for_5,
            "away_xg_against_5": away_xg_against_5,
            "xg_for_diff_5": home_xg_for_5 - away_xg_for_5,
            "xg_against_diff_5": home_xg_against_5 - away_xg_against_5,
            "home_shots_5": home_shots_5,
            "away_shots_5": away_shots_5,
            "home_sot_5": home_sot_5,
            "away_sot_5": away_sot_5,
            "shots_diff_5": home_shots_5 - away_shots_5,
            "home_poss_5": home_poss_5,
            "away_poss_5": away_poss_5,
            "poss_diff_5": home_poss_5 - away_poss_5,
        }

        for col in [
            "odds_h",
            "odds_d",
            "odds_a",
            "book_p_h",
            "book_p_d",
            "book_p_a",
            "odds_source",
            "home_xg",
            "away_xg",
            "home_shots",
            "away_shots",
            "home_sot",
            "away_sot",
            "home_poss",
            "away_poss",
        ]:
            if col in row.index:
                feat[col] = row[col]

        feature_rows.append(feat)

        # Only update state for completed matches
        if status == "scheduled" or pd.isna(row.get("home_goals")) or pd.isna(row.get("away_goals")):
            continue

        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        home_score = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        away_score = 1.0 - home_score
        exp_h = _elo_expected(home_elo + HOME_ADVANTAGE, away_elo)
        exp_a = 1.0 - exp_h
        elo[home] = _elo_update(home_elo, exp_h, home_score)
        elo[away] = _elo_update(away_elo, exp_a, away_score)

        def _num(val):
            if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
                return None
            return float(val)

        history[home].append(
            {
                "date": date,
                "opponent": away,
                "is_home": True,
                "gf": hg,
                "ga": ag,
                "gd": hg - ag,
                "pts": _points(hg, ag),
                "xg_for": _num(row.get("home_xg")),
                "xg_against": _num(row.get("away_xg")),
                "shots": _num(row.get("home_shots")),
                "sot": _num(row.get("home_sot")),
                "poss": _num(row.get("home_poss")),
            }
        )
        history[away].append(
            {
                "date": date,
                "opponent": home,
                "is_home": False,
                "gf": ag,
                "ga": hg,
                "gd": ag - hg,
                "pts": _points(ag, hg),
                "xg_for": _num(row.get("away_xg")),
                "xg_against": _num(row.get("home_xg")),
                "shots": _num(row.get("away_shots")),
                "sot": _num(row.get("away_sot")),
                "poss": _num(row.get("away_poss")),
            }
        )
        last_date[home] = date
        last_date[away] = date

    return pd.DataFrame(feature_rows)


BASE_FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_expected_home",
    "home_form_pts_5",
    "home_form_gd_5",
    "home_form_pts_10",
    "home_form_gd_10",
    "away_form_pts_5",
    "away_form_gd_5",
    "away_form_pts_10",
    "away_form_gd_10",
    "form_pts_diff_5",
    "form_gd_diff_5",
    "home_home_pts_5",
    "home_home_gd_5",
    "away_away_pts_5",
    "away_away_gd_5",
    "h2h_home_pts",
    "h2h_home_gd",
    "h2h_n",
    "rest_home",
    "rest_away",
    "rest_diff",
    "home_matches_played",
    "away_matches_played",
]

ADVANCED_FEATURE_COLUMNS = [
    "home_xg_for_5",
    "home_xg_against_5",
    "away_xg_for_5",
    "away_xg_against_5",
    "xg_for_diff_5",
    "xg_against_diff_5",
    "home_shots_5",
    "away_shots_5",
    "home_sot_5",
    "away_sot_5",
    "shots_diff_5",
    "home_poss_5",
    "away_poss_5",
    "poss_diff_5",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ADVANCED_FEATURE_COLUMNS


def save_features(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (DATA_PROCESSED / "features.parquet")
    df.to_parquet(path, index=False)
    df.to_csv(path.with_suffix(".csv"), index=False)
    print(f"Saved features ({len(df)} rows, {len(FEATURE_COLUMNS)} model cols) -> {path}")
    return path


if __name__ == "__main__":
    path = DATA_PROCESSED / "matches_with_stats.parquet"
    if not path.exists():
        path = DATA_PROCESSED / "matches_with_odds.parquet"
    matches = pd.read_parquet(path)
    feats = build_features(matches)
    save_features(feats)
