"""Inference entry-point: upcoming fixtures + recent results scoring."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, build_features
from src.labels import friendly_feature
from src.llm_explain import explain_match
from src.odds import load_all_odds, merge_odds, save_merged
from src.parser import parse_all_seasons, save_matches
from src.shap_explain import shap_top_features_for_row
from src.stats_xg import merge_advanced_stats
from src.utils import CACHE_DIR, DATA_PROCESSED, MODELS_DIR, ensure_dirs


def refresh_data(start_year: int = 2000, end_year: int = 2026) -> pd.DataFrame:
    matches = parse_all_seasons(
        start_year=start_year, end_year=end_year, download=True, include_scheduled=True
    )
    save_matches(matches)
    odds = load_all_odds(start_year=start_year, end_year=end_year, download=True)
    merged = merge_odds(matches, odds)
    merged = merge_advanced_stats(merged)
    save_merged(merged, DATA_PROCESSED / "matches_with_stats.parquet")
    save_merged(merged, DATA_PROCESSED / "matches_with_odds.parquet")
    return merged


def load_best_model() -> tuple[str, dict]:
    metrics_path = MODELS_DIR / "metrics.json"
    name = "xgboost"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        meta = metrics.get("_meta") if isinstance(metrics.get("_meta"), dict) else {}
        if meta.get("best_model") and (MODELS_DIR / f"{meta['best_model']}.joblib").exists():
            name = meta["best_model"]
        else:
            candidates = {
                k: v["log_loss"]
                for k, v in metrics.items()
                if isinstance(v, dict)
                and "log_loss" in v
                and k in ("xgboost", "random_forest", "logistic")
            }
            if candidates:
                name = min(candidates, key=candidates.get)
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run training first.")
    return name, joblib.load(path)


def load_shap_bundle() -> tuple[str, dict]:
    """Prefer XGBoost for interactive SHAP speed."""
    xgb = MODELS_DIR / "xgboost.joblib"
    if xgb.exists():
        return "xgboost", joblib.load(xgb)
    return load_best_model()


def _score_frame(feats: pd.DataFrame, bundle: dict, model_name: str) -> pd.DataFrame:
    feature_cols = bundle.get("feature_columns", FEATURE_COLUMNS)
    out = feats.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0
    X = out[feature_cols].astype(float).fillna(0.0)
    proba = bundle["model"].predict_proba(X)
    encoder = bundle["encoder"]
    for i, lab in enumerate(encoder.classes_):
        out[f"p_{lab}"] = proba[:, i]
    out["pred"] = encoder.inverse_transform(np.argmax(proba, axis=1))
    out["model"] = model_name
    out["correct"] = out.apply(
        lambda r: (
            None
            if pd.isna(r.get("outcome")) or r.get("status") == "scheduled"
            else bool(r["pred"] == r["outcome"])
        ),
        axis=1,
    )
    return out


def predict_upcoming(
    merged: pd.DataFrame | None = None,
    with_llm: bool = False,
    as_of: str | None = None,
) -> pd.DataFrame:
    """Score scheduled fixtures (status=scheduled), defaulting to dates >= today."""
    ensure_dirs()
    if merged is None:
        for name in ("matches_with_stats.parquet", "matches_with_odds.parquet"):
            path = DATA_PROCESSED / name
            if path.exists():
                merged = pd.read_parquet(path)
                break
        if merged is None:
            raise FileNotFoundError("No merged match data. Run the pipeline first.")

    feats = build_features(merged)
    feats.to_parquet(DATA_PROCESSED / "features.parquet", index=False)

    model_name, bundle = load_best_model()
    shap_name, shap_bundle = load_shap_bundle()

    upcoming = feats[feats.get("status", "played") == "scheduled"].copy()
    if upcoming.empty:
        # Fallback: next matchdays after last played
        played = feats[feats.get("status", "played") != "scheduled"]
        cutoff = pd.Timestamp(as_of) if as_of else pd.Timestamp.now().normalize()
        upcoming = feats[feats["date"] >= cutoff].copy()
        if upcoming.empty:
            upcoming = feats.sort_values("date").tail(20).copy()

    if as_of:
        cutoff = pd.Timestamp(as_of).normalize()
        upcoming = upcoming[upcoming["date"] >= cutoff]

    upcoming = upcoming.sort_values(["date", "kickoff", "home_team"]).reset_index(drop=True)
    if upcoming.empty:
        print("No upcoming fixtures found")
        return upcoming

    scored = _score_frame(upcoming, bundle, model_name)

    explanations = []
    for _, row in scored.iterrows():
        shap_result = shap_top_features_for_row(shap_bundle, row)
        # Attach friendly labels for Fans UI
        for item in shap_result.get("top_features", []):
            item["label"] = friendly_feature(item["feature"])
        text = None
        if with_llm:
            text = explain_match(
                shap_result,
                {
                    "match_id": row["match_id"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "date": row["date"],
                },
            )
        explanations.append(
            {
                "match_id": row["match_id"],
                "pred": shap_result["predicted_class"],
                "probabilities": shap_result["probabilities"],
                "top_features": shap_result["top_features"][:8],
                "explanation": text,
                "shap_model": shap_name,
            }
        )

    out_path = DATA_PROCESSED / "upcoming_predictions.parquet"
    keep = [
        c
        for c in [
            "match_id",
            "date",
            "season",
            "matchday",
            "kickoff",
            "home_team",
            "away_team",
            "status",
            "home_form_str",
            "away_form_str",
            "pred",
            "p_H",
            "p_D",
            "p_A",
            "odds_h",
            "odds_d",
            "odds_a",
            "book_p_h",
            "book_p_d",
            "book_p_a",
            "model",
            "home_xg_for_5",
            "away_xg_for_5",
            "home_shots_5",
            "away_shots_5",
            "home_poss_5",
            "away_poss_5",
        ]
        if c in scored.columns
    ]
    scored[keep].to_parquet(out_path, index=False)
    scored[keep].to_csv(out_path.with_suffix(".csv"), index=False)

    exp_path = CACHE_DIR / "upcoming_explanations.json"
    exp_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": model_name,
                "shap_model": shap_name,
                "items": explanations,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Upcoming predictions ({len(scored)}) -> {out_path}")
    return scored[keep]


def predict_upcoming_from_history(
    merged: pd.DataFrame | None = None,
    n_recent: int = 40,
    with_llm: bool = False,
) -> pd.DataFrame:
    """Score recent played matches (for history / analyst views). Prefer current season."""
    ensure_dirs()
    if merged is None:
        path = DATA_PROCESSED / "matches_with_stats.parquet"
        if not path.exists():
            path = DATA_PROCESSED / "matches_with_odds.parquet"
        merged = pd.read_parquet(path)

    feats = build_features(merged)
    feats.to_parquet(DATA_PROCESSED / "features.parquet", index=False)
    model_name, bundle = load_best_model()

    played = feats[feats.get("status", "played") != "scheduled"].copy()
    played = played[played["outcome"].notna()] if "outcome" in played.columns else played
    if "season" in played.columns and not played.empty:
        newest = sorted(played["season"].dropna().unique())[-1]
        current = played[played["season"] == newest]
        if len(current) >= 5:
            played = current
    recent = played.sort_values("date").tail(n_recent).copy()
    scored = _score_frame(recent, bundle, model_name)

    out_path = DATA_PROCESSED / "latest_predictions.parquet"
    cols = [
        c
        for c in [
            "match_id",
            "date",
            "season",
            "matchday",
            "home_team",
            "away_team",
            "outcome",
            "pred",
            "p_H",
            "p_D",
            "p_A",
            "correct",
            "home_form_str",
            "away_form_str",
            "model",
            "odds_h",
            "odds_d",
            "odds_a",
            "book_p_h",
            "book_p_d",
            "book_p_a",
        ]
        if c in scored.columns
    ]
    scored[cols].to_parquet(out_path, index=False)
    scored[cols].to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"Latest predictions -> {out_path}")
    return scored[cols]


def main():
    parser = argparse.ArgumentParser(description="EPL outcome predictor inference")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--n-recent", type=int, default=40)
    parser.add_argument("--upcoming", action="store_true", help="Score scheduled fixtures")
    parser.add_argument("--as-of", type=str, default=None, help="YYYY-MM-DD cutoff for upcoming")
    args = parser.parse_args()

    merged = None
    if args.refresh_data:
        merged = refresh_data()
    if args.upcoming or True:
        # Always refresh upcoming artifact; also write recent history
        predict_upcoming(merged=merged, with_llm=args.with_llm, as_of=args.as_of)
    predict_upcoming_from_history(merged=merged, n_recent=args.n_recent, with_llm=False)


if __name__ == "__main__":
    main()
