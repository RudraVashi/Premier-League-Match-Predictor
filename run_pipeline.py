"""End-to-end pipeline: data + stats -> features -> train -> ablation -> upcoming."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib

from src.ablation import run_ablation
from src.backtest import run_backtest
from src.features import build_features, save_features
from src.model import train_and_evaluate
from src.odds import load_all_odds, merge_odds, save_merged
from src.parser import parse_all_seasons, save_matches
from src.predict import predict_upcoming, predict_upcoming_from_history
from src.shap_explain import compute_shap_for_frame, save_global_importance
from src.stats_xg import fetch_fbref_possession, fetch_understat_xg, merge_advanced_stats
from src.utils import MODELS_DIR, DATA_PROCESSED, ensure_dirs


def run(
    start_year: int = 2000,
    end_year: int = 2026,
    skip_download: bool = False,
    skip_stats: bool = False,
) -> None:
    ensure_dirs()
    print("\n[1/7] Parsing openfootball (incl. scheduled) + odds/shots ...")
    matches = parse_all_seasons(
        start_year=start_year,
        end_year=end_year,
        download=not skip_download,
        include_scheduled=True,
    )
    save_matches(matches)
    odds = load_all_odds(
        start_year=start_year, end_year=min(end_year, 2025), download=not skip_download
    )
    # also try current season odds
    try:
        odds2 = load_all_odds(start_year=end_year, end_year=end_year, download=not skip_download)
        odds = (
            __import__("pandas")
            .concat([odds, odds2], ignore_index=True)
            .drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
        )
    except Exception:
        pass
    merged = merge_odds(matches, odds)

    print("\n[2/7] Advanced stats (Understat xG + FBref possession) ...")
    if not skip_stats:
        try:
            fetch_understat_xg(start_year=2014, end_year=end_year, force=not skip_download)
        except Exception as exc:  # noqa: BLE001
            print(f"Understat warning: {exc}")
        try:
            fetch_fbref_possession(start_year=2017, end_year=end_year, force=False)
        except Exception as exc:  # noqa: BLE001
            print(f"FBref warning: {exc}")
    merged = merge_advanced_stats(merged)
    save_merged(merged, DATA_PROCESSED / "matches_with_stats.parquet")
    save_merged(merged, DATA_PROCESSED / "matches_with_odds.parquet")

    print("\n[3/7] Feature engineering ...")
    feats = build_features(merged)
    save_features(feats)

    print("\n[4/7] Training RF + XGBoost ...")
    result = train_and_evaluate(feats)

    print("\n[5/7] Ablation (base vs xG/shots/possession) ...")
    run_ablation(feats)

    print("\n[6/7] SHAP importance + backtest ...")
    shap_model_name = "xgboost" if (MODELS_DIR / "xgboost.joblib").exists() else result["best_name"]
    bundle = joblib.load(MODELS_DIR / f"{shap_model_name}.joblib")
    shap_out = compute_shap_for_frame(bundle, result["X_test"], max_samples=300)
    save_global_importance(shap_out["importance"])
    run_backtest(result["test_predictions"])

    print("\n[7/7] Upcoming fixture predictions ...")
    predict_upcoming(merged=merged, with_llm=False, as_of=None)
    predict_upcoming_from_history(merged=merged, n_recent=40)
    print("\nPipeline complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-stats", action="store_true", help="Skip Understat/FBref network pulls")
    args = ap.parse_args()
    run(
        start_year=args.start_year,
        end_year=args.end_year,
        skip_download=args.skip_download,
        skip_stats=args.skip_stats,
    )
