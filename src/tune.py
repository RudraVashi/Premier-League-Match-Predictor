"""Optional Optuna hyperparameter search for XGBoost (chronological CV)."""

from __future__ import annotations

import json

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.features import FEATURE_COLUMNS
from src.model import TEST_SEASON_START, chronological_split
from src.utils import DATA_PROCESSED, MODELS_DIR, ensure_dirs

LABELS = ["H", "D", "A"]


def objective_factory(X: pd.DataFrame, y: np.ndarray):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
        }
        tscv = TimeSeriesSplit(n_splits=3)
        losses = []
        for train_idx, val_idx in tscv.split(X):
            model = XGBClassifier(**params)
            model.fit(X.iloc[train_idx], y[train_idx])
            proba = model.predict_proba(X.iloc[val_idx])
            losses.append(log_loss(y[val_idx], proba, labels=[0, 1, 2]))
        return float(np.mean(losses))

    return objective


def run_optuna(n_trials: int = 25) -> dict:
    ensure_dirs()
    feats = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    feats = feats[
        (feats["home_matches_played"] >= 5) & (feats["away_matches_played"] >= 5)
    ]
    train_df, _ = chronological_split(feats, TEST_SEASON_START)
    X = train_df[FEATURE_COLUMNS].astype(float).fillna(0.0)
    encoder = LabelEncoder().fit(LABELS)
    y = encoder.transform(train_df["outcome"])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_factory(X, y), n_trials=n_trials, show_progress_bar=True)
    best = {"best_value": study.best_value, "best_params": study.best_params}
    out = MODELS_DIR / "optuna_best.json"
    out.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Best log loss {study.best_value:.4f} -> {out}")
    return best


if __name__ == "__main__":
    run_optuna()
