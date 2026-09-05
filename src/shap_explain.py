"""SHAP explainability for tree-based match outcome models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.features import FEATURE_COLUMNS
from src.utils import CACHE_DIR, DATA_PROCESSED, MODELS_DIR, ensure_dirs


def _unwrap_base_estimator(calibrated_model):
    """Best-effort unwrap CalibratedClassifierCV -> underlying tree model."""
    if hasattr(calibrated_model, "calibrated_classifiers_"):
        cc = calibrated_model.calibrated_classifiers_[0]
        for attr in ("estimator", "base_estimator", "clf"):
            if hasattr(cc, attr):
                return getattr(cc, attr)
    if hasattr(calibrated_model, "estimator"):
        return calibrated_model.estimator
    return calibrated_model


def compute_shap_for_frame(
    model_bundle: dict,
    X: pd.DataFrame,
    max_samples: int | None = 500,
) -> dict:
    """
    Compute TreeSHAP values.

    Returns dict with:
      - values: (n, n_features, n_classes) or (n, n_features)
      - base_values
      - feature_names
      - mean_abs: mean |SHAP| per feature (importance)
    """
    ensure_dirs()
    feature_cols = model_bundle.get("feature_columns", FEATURE_COLUMNS)
    X_use = X[feature_cols].astype(float).fillna(0.0)
    if max_samples is not None and len(X_use) > max_samples:
        X_use = X_use.sample(max_samples, random_state=42)

    base = _unwrap_base_estimator(model_bundle["model"])
    explainer = shap.TreeExplainer(base)
    explanation = explainer(X_use)

    values = np.array(explanation.values)
    base_values = np.array(explanation.base_values)

    # Multi-class trees often return (n, features, classes)
    if values.ndim == 3:
        mean_abs = np.mean(np.abs(values), axis=(0, 2))
    else:
        mean_abs = np.mean(np.abs(values), axis=0)

    importance = (
        pd.Series(mean_abs, index=feature_cols)
        .sort_values(ascending=False)
        .to_dict()
    )
    return {
        "values": values,
        "base_values": base_values,
        "feature_names": feature_cols,
        "importance": importance,
        "index": X_use.index.to_list(),
        "X": X_use,
    }


def shap_top_features_for_row(
    model_bundle: dict,
    row: pd.Series | dict,
    top_k: int = 8,
    focus_class: str | None = None,
) -> list[dict]:
    """
    Return top-k feature contributions for a single match row,
    focused on the predicted (or specified) class.
    """
    feature_cols = model_bundle.get("feature_columns", FEATURE_COLUMNS)
    encoder = model_bundle["encoder"]
    X = pd.DataFrame([{c: float(row.get(c, 0) or 0) for c in feature_cols}])

    calibrated = model_bundle["model"]
    proba = calibrated.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    if focus_class is not None:
        pred_idx = list(encoder.classes_).index(focus_class)

    base = _unwrap_base_estimator(calibrated)
    explainer = shap.TreeExplainer(base)
    shap_values = explainer.shap_values(X)

    # shap_values may be list[class] of (1, F) or array (1, F, C) / (1, F)
    if isinstance(shap_values, list):
        contrib = np.array(shap_values[pred_idx]).reshape(-1)
    else:
        arr = np.array(shap_values)
        if arr.ndim == 3:
            contrib = arr[0, :, pred_idx]
        else:
            contrib = arr.reshape(-1)

    items = []
    for name, value, raw in zip(feature_cols, contrib, X.iloc[0].values):
        items.append(
            {
                "feature": name,
                "shap": float(value),
                "value": float(raw),
                "abs_shap": abs(float(value)),
            }
        )
    items.sort(key=lambda d: d["abs_shap"], reverse=True)
    top = items[:top_k]

    return {
        "predicted_class": encoder.classes_[pred_idx],
        "probabilities": {
            cls: float(proba[i]) for i, cls in enumerate(encoder.classes_)
        },
        "top_features": top,
        "all_features": items,
    }


def save_global_importance(importance: dict, path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (MODELS_DIR / "shap_importance.json")
    path.write_text(json.dumps(importance, indent=2), encoding="utf-8")
    return path


def explain_predictions_file(
    model_name: str = "xgboost",
    predictions_path: Path | None = None,
    features_path: Path | None = None,
    n: int = 50,
) -> pd.DataFrame:
    """Attach compact SHAP summaries to a sample of test predictions."""
    ensure_dirs()
    bundle = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    preds = pd.read_parquet(predictions_path or DATA_PROCESSED / "test_predictions.parquet")
    feats = pd.read_parquet(features_path or DATA_PROCESSED / "features.parquet")
    merged = preds.merge(
        feats[["match_id"] + FEATURE_COLUMNS],
        on="match_id",
        how="left",
        suffixes=("", "_feat"),
    )
    sample = merged.head(n)
    records = []
    for _, row in sample.iterrows():
        explanation = shap_top_features_for_row(bundle, row)
        records.append(
            {
                "match_id": row["match_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "pred": explanation["predicted_class"],
                "p_H": explanation["probabilities"].get("H"),
                "p_D": explanation["probabilities"].get("D"),
                "p_A": explanation["probabilities"].get("A"),
                "top_features": json.dumps(explanation["top_features"]),
            }
        )
    out = pd.DataFrame(records)
    out_path = CACHE_DIR / "shap_explanations.json"
    out.to_json(out_path, orient="records", indent=2)
    print(f"Wrote {len(out)} SHAP explanations -> {out_path}")
    return out


if __name__ == "__main__":
    explain_predictions_file()
