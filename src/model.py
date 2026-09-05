"""Train / evaluate Random Forest + XGBoost with calibration and baselines."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.features import FEATURE_COLUMNS
from src.utils import DATA_PROCESSED, MODELS_DIR, ensure_dirs

LABELS = ["H", "D", "A"]
TEST_SEASON_START = "2023-24"  # chronological hold-out from this season onward


def chronological_split(
    df: pd.DataFrame,
    test_season_start: str = TEST_SEASON_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True)
    # seasons are labeled YYYY-YY; lexicographic works for our format from 2000+
    train = df[df["season"] < test_season_start].copy()
    test = df[df["season"] >= test_season_start].copy()
    if train.empty or test.empty:
        # fallback: last 20% by date
        cut = int(len(df) * 0.8)
        train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    return train, test


def _encode(y: pd.Series, encoder: LabelEncoder | None = None) -> tuple[np.ndarray, LabelEncoder]:
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(LABELS)
    return encoder.transform(y), encoder


def build_models(random_state: int = 42) -> dict:
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=random_state,
        tree_method="hist",
    )
    return {"random_forest": rf, "xgboost": xgb}


def calibrate(model, X_train, y_train, method: str = "isotonic"):
    """Wrap with isotonic calibration using an internal CV on the training window."""
    # sklearn >=1.4 uses `estimator`; older used `base_estimator`
    try:
        calib = CalibratedClassifierCV(estimator=model, method=method, cv=3)
    except TypeError:
        calib = CalibratedClassifierCV(base_estimator=model, method=method, cv=3)
    calib.fit(X_train, y_train)
    return calib


def _multiclass_brier(y_true_oh: np.ndarray, proba: np.ndarray) -> float:
    return float(np.mean(np.sum((proba - y_true_oh) ** 2, axis=1)))


def evaluate_predictions(
    y_true: np.ndarray,
    proba: np.ndarray,
    encoder: LabelEncoder,
    name: str = "model",
) -> dict:
    classes = list(encoder.classes_)
    y_pred = encoder.inverse_transform(np.argmax(proba, axis=1))
    y_true_labels = encoder.inverse_transform(y_true)

    # Align proba columns to encoder.classes_
    ll = log_loss(y_true, proba, labels=list(range(len(classes))))
    y_oh = np.eye(len(classes))[y_true]
    brier = _multiclass_brier(y_oh, proba)
    acc = float(np.mean(y_pred == y_true_labels))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_labels, y_pred, labels=LABELS, zero_division=0
    )
    report = {
        "name": name,
        "log_loss": float(ll),
        "brier": float(brier),
        "accuracy": acc,
        "per_class": {
            lab: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, lab in enumerate(LABELS)
        },
    }
    print(f"\n=== {name} ===")
    print(f"log_loss={ll:.4f}  brier={brier:.4f}  accuracy={acc:.4f}")
    print(classification_report(y_true_labels, y_pred, labels=LABELS, zero_division=0))
    return report


def bookmaker_baseline(test: pd.DataFrame, encoder: LabelEncoder) -> dict | None:
    needed = ["book_p_h", "book_p_d", "book_p_a"]
    if not all(c in test.columns for c in needed):
        return None
    mask = test[needed].notna().all(axis=1)
    if mask.sum() < 50:
        print("Insufficient odds coverage for bookmaker baseline")
        return None
    sub = test.loc[mask]
    # Column order must match encoder.classes_ (H, D, A expected)
    class_to_col = {"H": "book_p_h", "D": "book_p_d", "A": "book_p_a"}
    proba = np.column_stack([sub[class_to_col[c]].to_numpy() for c in encoder.classes_])
    y_true, _ = _encode(sub["outcome"], encoder)
    return evaluate_predictions(y_true, proba, encoder, name="bookmaker")


def naive_home_baseline(test: pd.DataFrame, encoder: LabelEncoder) -> dict:
    """Always predict home win with empirical train prior if available; else 1/0/0."""
    n = len(test)
    proba = np.zeros((n, len(encoder.classes_)))
    home_idx = list(encoder.classes_).index("H")
    proba[:, home_idx] = 1.0
    y_true, _ = _encode(test["outcome"], encoder)
    return evaluate_predictions(y_true, proba, encoder, name="naive_home")


def train_and_evaluate(
    features: pd.DataFrame,
    test_season_start: str = TEST_SEASON_START,
    min_matches_played: int = 5,
) -> dict:
    ensure_dirs()
    df = features.copy()
    if "status" in df.columns:
        df = df[df["status"] != "scheduled"]
    df = df[df["outcome"].isin(LABELS)]
    # Drop early-season cold-start rows for cleaner evaluation
    df = df[
        (df["home_matches_played"] >= min_matches_played)
        & (df["away_matches_played"] >= min_matches_played)
    ].reset_index(drop=True)

    train_df, test_df = chronological_split(df, test_season_start)
    print(f"Train: {len(train_df)} matches ({train_df['season'].min()} -> {train_df['season'].max()})")
    print(f"Test:  {len(test_df)} matches ({test_df['season'].min()} -> {test_df['season'].max()})")

    feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    X_train = train_df[feature_cols].astype(float).fillna(0.0)
    X_test = test_df[feature_cols].astype(float).fillna(0.0)
    encoder = LabelEncoder().fit(LABELS)
    y_train, _ = _encode(train_df["outcome"], encoder)
    y_test, _ = _encode(test_df["outcome"], encoder)

    results = {}
    fitted = {}

    for name, model in build_models().items():
        print(f"\nTraining {name} + isotonic calibration ...")
        calibrated = calibrate(model, X_train, y_train)
        proba = calibrated.predict_proba(X_test)
        results[name] = evaluate_predictions(y_test, proba, encoder, name=name)
        fitted[name] = calibrated

        # Persist model
        joblib.dump(
            {
                "model": calibrated,
                "encoder": encoder,
                "feature_columns": feature_cols,
                "test_season_start": test_season_start,
            },
            MODELS_DIR / f"{name}.joblib",
        )

    results["bookmaker"] = bookmaker_baseline(test_df, encoder)
    results["naive_home"] = naive_home_baseline(test_df, encoder)

    # Save test predictions from best model by log loss
    model_scores = {
        k: v["log_loss"]
        for k, v in results.items()
        if v is not None and k in fitted
    }
    best_name = min(model_scores, key=model_scores.get)
    best = fitted[best_name]
    proba = best.predict_proba(X_test)
    pred_df = test_df[
        ["match_id", "date", "season", "home_team", "away_team", "outcome"]
        + [c for c in ["odds_h", "odds_d", "odds_a", "book_p_h", "book_p_d", "book_p_a"] if c in test_df.columns]
    ].copy()
    for i, lab in enumerate(encoder.classes_):
        pred_df[f"p_{lab}"] = proba[:, i]
    pred_df["pred"] = encoder.inverse_transform(np.argmax(proba, axis=1))
    pred_df["best_model"] = best_name
    pred_path = DATA_PROCESSED / "test_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    pred_df.to_csv(pred_path.with_suffix(".csv"), index=False)

    # Calibration curves for best model (home-win class)
    _plot_calibration(y_test, proba, encoder, best_name)

    metrics_path = MODELS_DIR / "metrics.json"
    # Make JSON-safe
    safe = {k: v for k, v in results.items() if v is not None}
    metrics_path.write_text(json.dumps(safe, indent=2), encoding="utf-8")
    print(f"\nBest model by log loss: {best_name}")
    print(f"Metrics -> {metrics_path}")
    print(f"Predictions -> {pred_path}")

    return {
        "results": results,
        "fitted": fitted,
        "encoder": encoder,
        "best_name": best_name,
        "test_predictions": pred_df,
        "X_test": X_test,
        "test_df": test_df,
    }


def _plot_calibration(y_true, proba, encoder, model_name: str) -> None:
    ensure_dirs()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, lab in zip(axes, LABELS):
        idx = list(encoder.classes_).index(lab)
        y_bin = (y_true == idx).astype(int)
        try:
            frac_pos, mean_pred = calibration_curve(y_bin, proba[:, idx], n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
        except ValueError:
            ax.text(0.5, 0.5, "n/a", ha="center")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title(f"Class {lab}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Calibration curves - {model_name}")
    fig.tight_layout()
    out = MODELS_DIR / "calibration_curves.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Calibration plot -> {out}")


def load_model(name: str = "xgboost") -> dict:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        # try RF
        path = MODELS_DIR / "random_forest.joblib"
    return joblib.load(path)


if __name__ == "__main__":
    feats = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    train_and_evaluate(feats)
