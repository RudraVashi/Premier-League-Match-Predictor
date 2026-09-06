"""Train / evaluate Random Forest + XGBoost with calibration and baselines."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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
ELO_FEATURE_COLUMNS = ["home_elo", "away_elo", "elo_diff", "elo_expected_home"]


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
    # Proper linear ML baseline (same feature set as trees)
    logit = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=0.5,
        class_weight="balanced",
        random_state=random_state,
    )
    return {"random_forest": rf, "xgboost": xgb, "logistic": logit}


def build_elo_baseline(random_state: int = 42) -> LogisticRegression:
    """Elo-only multinomial logistic — strength signal without form/xG stack."""
    return LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=random_state,
    )


def _make_calibrator(model, method: str):
    """Wrap a fitted estimator for calibration on a held-out chronological window."""
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(FrozenEstimator(model), method=method)
    except Exception:
        # Older sklearn: cv='prefit'
        try:
            return CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
        except TypeError:
            return CalibratedClassifierCV(base_estimator=model, method=method, cv="prefit")


def calibrate_time_aware(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    methods: tuple[str, ...] = ("isotonic", "sigmoid"),
    calib_frac: float = 0.12,
    select_frac: float = 0.08,
):
    """
    Chronological calibration:
      early fit -> mid calib (fit calibrators) -> late select (pick method).
    Avoids picking isotonic just because it overfits the calib window.
    """
    n = len(X_train)
    n_select = max(80, int(n * select_frac))
    n_calib = max(120, int(n * calib_frac))
    if n_select + n_calib + 200 >= n:
        n_select = max(60, n // 12)
        n_calib = max(100, n // 10)

    cut_fit = n - n_calib - n_select
    cut_calib = n - n_select
    X_fit = X_train.iloc[:cut_fit]
    X_cal = X_train.iloc[cut_fit:cut_calib]
    X_sel = X_train.iloc[cut_calib:]
    y_fit = y_train[:cut_fit]
    y_cal = y_train[cut_fit:cut_calib]
    y_sel = y_train[cut_calib:]

    model.fit(X_fit, y_fit)
    print(
        f"    split fit={len(X_fit)} calib={len(X_cal)} select={len(X_sel)} "
        f"(chronological)"
    )

    best = None
    best_ll = float("inf")
    best_method = methods[0]
    for method in methods:
        try:
            calib = _make_calibrator(model, method)
            calib.fit(X_cal, y_cal)
            proba = calib.predict_proba(X_sel)
            ll = float(log_loss(y_sel, proba, labels=list(range(proba.shape[1]))))
            print(f"    calib[{method}] select log_loss={ll:.4f}")
            if ll < best_ll:
                best_ll = ll
                best = calib
                best_method = method
        except Exception as exc:  # noqa: BLE001
            print(f"    calib[{method}] failed: {exc}")

    if best is None:
        try:
            best = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=3)
        except TypeError:
            best = CalibratedClassifierCV(base_estimator=model, method="sigmoid", cv=3)
        best.fit(X_train, y_train)
        best_method = "sigmoid_cv3_fallback"
        print(f"    calib fallback -> {best_method}")

    best._pitchpulse_calib_method_ = best_method  # type: ignore[attr-defined]
    return best


def calibrate(model, X_train, y_train, method: str = "isotonic"):
    """Legacy helper — prefer calibrate_time_aware for training."""
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
    class_to_col = {"H": "book_p_h", "D": "book_p_d", "A": "book_p_a"}
    proba = np.column_stack([sub[class_to_col[c]].to_numpy() for c in encoder.classes_])
    y_true, _ = _encode(sub["outcome"], encoder)
    return evaluate_predictions(y_true, proba, encoder, name="bookmaker")


def naive_home_baseline(test: pd.DataFrame, encoder: LabelEncoder) -> dict:
    """Always predict home win with certainty 1.0."""
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
    elo_cols = [c for c in ELO_FEATURE_COLUMNS if c in train_df.columns]
    X_train = train_df[feature_cols].astype(float).fillna(0.0)
    X_test = test_df[feature_cols].astype(float).fillna(0.0)
    encoder = LabelEncoder().fit(LABELS)
    y_train, _ = _encode(train_df["outcome"], encoder)
    y_test, _ = _encode(test_df["outcome"], encoder)

    results = {}
    fitted = {}
    calib_methods = {}

    for name, model in build_models().items():
        print(f"\nTraining {name} + time-aware calibration ...")
        calibrated = calibrate_time_aware(model, X_train, y_train)
        method = getattr(calibrated, "_pitchpulse_calib_method_", "unknown")
        calib_methods[name] = method
        proba = calibrated.predict_proba(X_test)
        report = evaluate_predictions(y_test, proba, encoder, name=name)
        report["calibration"] = method
        results[name] = report
        fitted[name] = calibrated

        joblib.dump(
            {
                "model": calibrated,
                "encoder": encoder,
                "feature_columns": feature_cols,
                "test_season_start": test_season_start,
                "calibration": method,
            },
            MODELS_DIR / f"{name}.joblib",
        )

    # Elo-only logistic baseline (same train/test, time-aware calib)
    print("\nTraining elo_logistic baseline + time-aware calibration ...")
    X_train_elo = train_df[elo_cols].astype(float).fillna(0.0)
    X_test_elo = test_df[elo_cols].astype(float).fillna(0.0)
    elo_model = build_elo_baseline()
    elo_cal = calibrate_time_aware(elo_model, X_train_elo, y_train)
    elo_method = getattr(elo_cal, "_pitchpulse_calib_method_", "unknown")
    elo_proba = elo_cal.predict_proba(X_test_elo)
    elo_report = evaluate_predictions(y_test, elo_proba, encoder, name="elo_logistic")
    elo_report["calibration"] = elo_method
    elo_report["features"] = elo_cols
    results["elo_logistic"] = elo_report
    fitted["elo_logistic"] = elo_cal
    calib_methods["elo_logistic"] = elo_method
    joblib.dump(
        {
            "model": elo_cal,
            "encoder": encoder,
            "feature_columns": elo_cols,
            "test_season_start": test_season_start,
            "calibration": elo_method,
        },
        MODELS_DIR / "elo_logistic.joblib",
    )

    results["bookmaker"] = bookmaker_baseline(test_df, encoder)
    results["naive_home"] = naive_home_baseline(test_df, encoder)

    # Best among production models (RF / XGB / full logistic)
    production = {
        k: v["log_loss"]
        for k, v in results.items()
        if k in ("random_forest", "xgboost", "logistic") and v
    }
    best_name = min(production, key=production.get)
    best = fitted[best_name]
    proba = best.predict_proba(X_test)
    pred_df = test_df[
        ["match_id", "date", "season", "home_team", "away_team", "outcome"]
        + [
            c
            for c in [
                "matchday",
                "odds_h",
                "odds_d",
                "odds_a",
                "book_p_h",
                "book_p_d",
                "book_p_a",
            ]
            if c in test_df.columns
        ]
    ].copy()
    for i, lab in enumerate(encoder.classes_):
        pred_df[f"p_{lab}"] = proba[:, i]
    pred_df["pred"] = encoder.inverse_transform(np.argmax(proba, axis=1))
    pred_df["correct"] = pred_df["pred"] == pred_df["outcome"]
    pred_df["best_model"] = best_name
    pred_path = DATA_PROCESSED / "test_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    pred_df.to_csv(pred_path.with_suffix(".csv"), index=False)

    _plot_calibration(y_test, proba, encoder, best_name)

    metrics_path = MODELS_DIR / "metrics.json"
    safe = {k: v for k, v in results.items() if v is not None}
    safe["_meta"] = {
        "best_model": best_name,
        "calibration_methods": calib_methods,
        "test_season_start": test_season_start,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    metrics_path.write_text(json.dumps(safe, indent=2), encoding="utf-8")
    print(f"\nBest production model by log loss: {best_name}")
    print(f"Calibration choices: {calib_methods}")
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
        "calib_methods": calib_methods,
    }


def _plot_calibration(y_true, proba, encoder, model_name: str) -> None:
    ensure_dirs()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, lab in zip(axes, LABELS):
        idx = list(encoder.classes_).index(lab)
        y_bin = (y_true == idx).astype(int)
        try:
            frac_pos, mean_pred = calibration_curve(
                y_bin, proba[:, idx], n_bins=10, strategy="quantile"
            )
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
        path = MODELS_DIR / "random_forest.joblib"
    return joblib.load(path)
