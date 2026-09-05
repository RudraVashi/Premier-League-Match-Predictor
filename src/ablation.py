"""Ablation: train with vs without advanced (xG/shots/possession) features."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.features import ADVANCED_FEATURE_COLUMNS, BASE_FEATURE_COLUMNS, FEATURE_COLUMNS
from src.model import TEST_SEASON_START, chronological_split, _multiclass_brier
from src.utils import DATA_PROCESSED, MODELS_DIR, ensure_dirs

LABELS = ["H", "D", "A"]


def _fit_eval(X_train, y_train, X_test, y_test, encoder, name: str) -> dict:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    try:
        calib = CalibratedClassifierCV(estimator=model, method="isotonic", cv=3)
    except TypeError:
        calib = CalibratedClassifierCV(base_estimator=model, method="isotonic", cv=3)
    calib.fit(X_train, y_train)
    proba = calib.predict_proba(X_test)
    ll = float(log_loss(y_test, proba, labels=list(range(len(encoder.classes_)))))
    y_oh = np.eye(len(encoder.classes_))[y_test]
    brier = _multiclass_brier(y_oh, proba)
    acc = float(np.mean(encoder.inverse_transform(np.argmax(proba, axis=1)) == encoder.inverse_transform(y_test)))
    return {"name": name, "log_loss": ll, "brier": brier, "accuracy": acc, "n_features": X_train.shape[1]}


def run_ablation(features: pd.DataFrame | None = None) -> dict:
    ensure_dirs()
    if features is None:
        features = pd.read_parquet(DATA_PROCESSED / "features.parquet")

    df = features.copy()
    df = df[df.get("status", "played") != "scheduled"] if "status" in df.columns else df
    df = df[df["outcome"].isin(LABELS)]
    df = df[
        (df["home_matches_played"] >= 5) & (df["away_matches_played"] >= 5)
    ].reset_index(drop=True)

    train_df, test_df = chronological_split(df, TEST_SEASON_START)
    encoder = LabelEncoder().fit(LABELS)
    y_train = encoder.transform(train_df["outcome"])
    y_test = encoder.transform(test_df["outcome"])

    results = {}
    for name, cols in [
        ("base_only", BASE_FEATURE_COLUMNS),
        ("full_with_xg_shots_poss", [c for c in FEATURE_COLUMNS if c in df.columns]),
    ]:
        use = [c for c in cols if c in df.columns]
        X_train = train_df[use].astype(float).fillna(0.0)
        X_test = test_df[use].astype(float).fillna(0.0)
        print(f"Ablation {name}: {len(use)} features ...")
        results[name] = _fit_eval(X_train, y_train, X_test, y_test, encoder, name)

    base_ll = results["base_only"]["log_loss"]
    full_ll = results["full_with_xg_shots_poss"]["log_loss"]
    results["delta"] = {
        "log_loss_full_minus_base": full_ll - base_ll,
        "log_loss_improvement": base_ll - full_ll,
        "advanced_features": [c for c in ADVANCED_FEATURE_COLUMNS if c in df.columns],
        "note": (
            "Negative log_loss_full_minus_base means advanced stats helped. "
            "Honest reporting matters either way."
        ),
    }
    out = MODELS_DIR / "ablation.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Ablation -> {out}")
    print(
        f"  base LL={base_ll:.4f}  full LL={full_ll:.4f}  "
        f"delta={results['delta']['log_loss_full_minus_base']:+.4f}"
    )
    return results


if __name__ == "__main__":
    run_ablation()
