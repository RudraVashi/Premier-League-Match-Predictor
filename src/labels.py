"""Friendly football labels for Fans UI (Analyst mode keeps machine names)."""

from __future__ import annotations

FEATURE_LABELS: dict[str, str] = {
    "home_elo": "Home strength rating",
    "away_elo": "Away strength rating",
    "elo_diff": "Strength gap",
    "elo_expected_home": "Expected home edge",
    "home_form_pts_5": "Home form (pts / last 5)",
    "home_form_gd_5": "Home goal diff (last 5)",
    "home_form_pts_10": "Home form (pts / last 10)",
    "home_form_gd_10": "Home goal diff (last 10)",
    "away_form_pts_5": "Away form (pts / last 5)",
    "away_form_gd_5": "Away goal diff (last 5)",
    "away_form_pts_10": "Away form (pts / last 10)",
    "away_form_gd_10": "Away goal diff (last 10)",
    "form_pts_diff_5": "Form gap (last 5)",
    "form_gd_diff_5": "Goal-diff gap (last 5)",
    "home_home_pts_5": "Home fortress form",
    "home_home_gd_5": "Home fortress GD",
    "away_away_pts_5": "Away road form",
    "away_away_gd_5": "Away road GD",
    "h2h_home_pts": "Head-to-head (home side)",
    "h2h_home_gd": "Head-to-head GD",
    "h2h_n": "H2H sample size",
    "rest_home": "Home rest days",
    "rest_away": "Away rest days",
    "rest_diff": "Rest-day advantage",
    "home_matches_played": "Home matches logged",
    "away_matches_played": "Away matches logged",
    "home_xg_for_5": "Home xG for (last 5)",
    "home_xg_against_5": "Home xG conceded (last 5)",
    "away_xg_for_5": "Away xG for (last 5)",
    "away_xg_against_5": "Away xG conceded (last 5)",
    "xg_for_diff_5": "xG attack gap (last 5)",
    "xg_against_diff_5": "xG defence gap (last 5)",
    "home_shots_5": "Home shots (last 5)",
    "away_shots_5": "Away shots (last 5)",
    "home_sot_5": "Home shots on target (last 5)",
    "away_sot_5": "Away shots on target (last 5)",
    "shots_diff_5": "Shots gap (last 5)",
    "home_poss_5": "Home possession % (last 5)",
    "away_poss_5": "Away possession % (last 5)",
    "poss_diff_5": "Possession gap (last 5)",
}


def friendly_feature(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " "))


OUTCOME_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}
