"""FastAPI backend for EPL Match Outcome Predictor v2."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labels import FEATURE_LABELS, friendly_feature, OUTCOME_LABELS
from src.llm_explain import explain_match
from src.shap_explain import shap_top_features_for_row
from src.utils import CACHE_DIR, DATA_PROCESSED, MODELS_DIR, ROOT

# Ensure API process sees project .env regardless of shell cwd
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
except Exception:
    pass

app = FastAPI(
    title="EPL Match Outcome Predictor API",
    version="2.0.0",
    description="Fans + Analyst endpoints for Premier League match predictions",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_parquet(*names: str) -> pd.DataFrame:
    for name in names:
        path = DATA_PROCESSED / name
        if path.exists():
            return pd.read_parquet(path)
    return pd.DataFrame()


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row_to_match(row: pd.Series, include_book: bool = True) -> dict:
    def fget(key, default=None):
        val = row.get(key, default)
        if pd.isna(val):
            return default
        if hasattr(val, "isoformat"):
            return val.isoformat()
        if hasattr(val, "item"):
            try:
                return val.item()
            except Exception:
                return val
        return val

    pred = fget("pred")
    outcome = fget("outcome")
    correct = fget("correct")
    if correct is None and pred is not None and outcome is not None:
        correct = pred == outcome

    payload = {
        "match_id": fget("match_id"),
        "date": str(fget("date", ""))[:10],
        "kickoff": fget("kickoff"),
        "season": fget("season"),
        "matchday": fget("matchday"),
        "home_team": fget("home_team"),
        "away_team": fget("away_team"),
        "status": fget("status", "played"),
        "home_form": fget("home_form_str"),
        "away_form": fget("away_form_str"),
        "prediction": pred,
        "prediction_label": OUTCOME_LABELS.get(pred or "", pred),
        "outcome": outcome,
        "outcome_label": OUTCOME_LABELS.get(outcome or "", outcome) if outcome else None,
        "correct": correct,
        "probabilities": {
            "home": float(fget("p_H") or 0),
            "draw": float(fget("p_D") or 0),
            "away": float(fget("p_A") or 0),
        },
        "stats_snapshot": {
            "home_xg_for_5": fget("home_xg_for_5"),
            "away_xg_for_5": fget("away_xg_for_5"),
            "home_shots_5": fget("home_shots_5"),
            "away_shots_5": fget("away_shots_5"),
            "home_poss_5": fget("home_poss_5"),
            "away_poss_5": fget("away_poss_5"),
        },
    }
    if include_book:
        ph, pd_, pa = fget("book_p_h"), fget("book_p_d"), fget("book_p_a")
        if ph is not None and pd_ is not None and pa is not None:
            payload["bookmaker"] = {
                "odds_home": fget("odds_h"),
                "odds_draw": fget("odds_d"),
                "odds_away": fget("odds_a"),
                "p_home": ph,
                "p_draw": pd_,
                "p_away": pa,
            }
        else:
            payload["bookmaker"] = None
    return payload


@app.get("/health")
def health():
    return {
        "ok": True,
        "has_upcoming": (DATA_PROCESSED / "upcoming_predictions.parquet").exists(),
        "has_metrics": (MODELS_DIR / "metrics.json").exists(),
    }


@app.get("/fixtures/upcoming")
def fixtures_upcoming(
    limit: int = Query(40, ge=1, le=100),
    days: int = Query(21, ge=1, le=120, description="Only fixtures within N days"),
    refresh: bool = Query(False, description="Force re-download openfootball fixtures"),
):
    from src.fixtures_refresh import filter_upcoming_frame, refresh_openfootball_statuses
    from src.live_odds import attach_live_odds

    # Soft refresh: pull latest openfootball scores/fixtures on a throttle
    try:
        fresh = refresh_openfootball_statuses(force=refresh)
        if fresh is not None and not fresh.empty:
            # Drop any upcoming rows that now have scores in the fresh parse
            played_ids = set(
                fresh.loc[fresh["status"] == "played", "date"].astype(str)
                + "_"
                + fresh.loc[fresh["status"] == "played", "home_team"].astype(str)
                + "_"
                + fresh.loc[fresh["status"] == "played", "away_team"].astype(str)
            )
            # match_id format uses compact team names without spaces
            played_match_ids = set()
            played = fresh[fresh["status"] == "played"].copy()
            if not played.empty:
                played["match_id"] = (
                    pd.to_datetime(played["date"]).dt.strftime("%Y%m%d")
                    + "_"
                    + played["home_team"].str.replace(" ", "", regex=False)
                    + "_"
                    + played["away_team"].str.replace(" ", "", regex=False)
                )
                played_match_ids = set(played["match_id"])
        else:
            played_match_ids = set()
    except Exception as exc:  # noqa: BLE001
        print(f"fixture refresh skipped: {exc}")
        played_match_ids = set()

    df = _read_parquet("upcoming_predictions.parquet")
    if df.empty:
        raise HTTPException(
            404,
            "No upcoming predictions. Run: python run_pipeline.py or python -m src.predict --upcoming",
        )

    if played_match_ids and "match_id" in df.columns:
        df = df[~df["match_id"].isin(played_match_ids)]

    df = df.sort_values(["date", "kickoff", "home_team"])
    # Drop kickoffs that already happened (Ipswich vs Liverpool yesterday, etc.)
    df = filter_upcoming_frame(df)

    try:
        df = attach_live_odds(df)
    except Exception as exc:  # noqa: BLE001
        print(f"live odds skipped: {exc}")

    today = pd.Timestamp.now().normalize()
    dates = pd.to_datetime(df["date"])
    near = df[dates <= today + pd.Timedelta(days=days)]
    if not near.empty:
        df = near
    df = df.head(limit)

    note = None
    if "_odds_note" in df.columns and len(df):
        note = str(df["_odds_note"].iloc[0])
    elif not ODDS_KEY_PRESENT():
        note = (
            "No live odds yet — add THE_ODDS_API_KEY to .env (free at the-odds-api.com) "
            "and restart the API."
        )
    elif len(df) and df.get("book_p_h") is not None and df["book_p_h"].notna().sum() == 0:
        note = "Odds key loaded, but no live prices matched these fixtures yet."

    return {
        "count": len(df),
        "odds_note": note,
        "matches": [_row_to_match(r) for _, r in df.iterrows()],
    }


def ODDS_KEY_PRESENT() -> bool:
    import os
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
    key = (os.getenv("THE_ODDS_API_KEY") or "").strip().strip('"').strip("'")
    return bool(key) and not key.startswith("your_")


@app.get("/fixtures/recent")
def fixtures_recent(limit: int = Query(40, ge=1, le=200)):
    df = _read_parquet("latest_predictions.parquet", "test_predictions.parquet")
    if df.empty:
        raise HTTPException(404, "No recent predictions found")
    df = df.sort_values("date", ascending=False).head(limit)
    return {"count": len(df), "matches": [_row_to_match(r) for _, r in df.iterrows()]}


@app.get("/match/{match_id}")
def match_detail(match_id: str):
    frames = []
    for name in (
        "upcoming_predictions.parquet",
        "latest_predictions.parquet",
        "test_predictions.parquet",
        "features.parquet",
    ):
        df = _read_parquet(name)
        if not df.empty and "match_id" in df.columns:
            hit = df[df["match_id"] == match_id]
            if not hit.empty:
                frames.append(hit.iloc[0])
                break
    if not frames:
        raise HTTPException(404, f"Match {match_id} not found")
    row = frames[0]

    # Enrich from features if needed
    feats = _read_parquet("features.parquet")
    if not feats.empty and "match_id" in feats.columns:
        frow = feats[feats["match_id"] == match_id]
        if not frow.empty:
            for col in frow.columns:
                if col not in row.index or pd.isna(row.get(col)):
                    row[col] = frow.iloc[0][col]

    detail = _row_to_match(row)

    # Attach live odds for scheduled fixtures when historical odds are missing
    if detail.get("bookmaker", {}).get("p_home") is None:
        try:
            from src.live_odds import fetch_live_odds_table

            live = fetch_live_odds_table()
            if not live.empty:
                live["date"] = pd.to_datetime(live["date"]).dt.normalize()
                day = pd.to_datetime(detail["date"]).normalize()
                hit = live[
                    (live["date"] == day)
                    & (live["home_team"] == detail["home_team"])
                    & (live["away_team"] == detail["away_team"])
                ]
                if not hit.empty:
                    h = hit.iloc[0]
                    detail["bookmaker"] = {
                        "odds_home": float(h["odds_h"]),
                        "odds_draw": float(h["odds_d"]),
                        "odds_away": float(h["odds_a"]),
                        "p_home": float(h["book_p_h"]),
                        "p_draw": float(h["book_p_d"]),
                        "p_away": float(h["book_p_a"]),
                    }
        except Exception as exc:  # noqa: BLE001
            print(f"live odds detail skipped: {exc}")

    # SHAP
    shap_payload = None
    xgb_path = MODELS_DIR / "xgboost.joblib"
    model_path = xgb_path if xgb_path.exists() else next(MODELS_DIR.glob("*.joblib"), None)
    if model_path and model_path.exists() and not feats.empty:
        bundle = joblib.load(model_path)
        frow = feats[feats["match_id"] == match_id]
        if not frow.empty:
            shap_result = shap_top_features_for_row(bundle, frow.iloc[0])
            top = []
            for item in shap_result.get("top_features", [])[:8]:
                top.append(
                    {
                        **item,
                        "label": friendly_feature(item["feature"]),
                    }
                )
            shap_payload = {
                "predicted_class": shap_result["predicted_class"],
                "probabilities": shap_result["probabilities"],
                "top_features": top,
                "model": model_path.stem,
            }
    detail["shap"] = shap_payload

    # Cached explanation
    for cache_name in ("upcoming_explanations.json", "latest_explanations.json", "llm_explanations.json"):
        cache = _load_json(CACHE_DIR / cache_name)
        if not cache:
            continue
        items = cache.get("items", cache) if isinstance(cache, dict) else []
        if isinstance(items, dict) and match_id in items:
            detail["explanation"] = items[match_id].get("text") if isinstance(items[match_id], dict) else items[match_id]
            break
        if isinstance(items, list):
            for it in items:
                if it.get("match_id") == match_id and it.get("explanation"):
                    detail["explanation"] = it["explanation"]
                    detail["shap"] = detail.get("shap") or {
                        "top_features": it.get("top_features"),
                        "probabilities": it.get("probabilities"),
                        "predicted_class": it.get("pred"),
                    }
                    break
    return detail


@app.post("/match/{match_id}/explain")
def explain_endpoint(match_id: str):
    detail = match_detail(match_id)
    if detail.get("explanation"):
        return {"match_id": match_id, "explanation": detail["explanation"], "cached": True}

    feats = _read_parquet("features.parquet")
    if feats.empty:
        raise HTTPException(404, "Features missing")
    frow = feats[feats["match_id"] == match_id]
    if frow.empty:
        raise HTTPException(404, "Match features not found")

    xgb_path = MODELS_DIR / "xgboost.joblib"
    model_path = xgb_path if xgb_path.exists() else next(MODELS_DIR.glob("*.joblib"), None)
    if not model_path:
        raise HTTPException(500, "No trained model")
    bundle = joblib.load(model_path)
    shap_result = shap_top_features_for_row(bundle, frow.iloc[0])
    for item in shap_result.get("top_features", []):
        item["label"] = friendly_feature(item["feature"])
    text = explain_match(
        shap_result,
        {
            "match_id": match_id,
            "home_team": frow.iloc[0]["home_team"],
            "away_team": frow.iloc[0]["away_team"],
            "date": frow.iloc[0]["date"],
        },
    )
    return {
        "match_id": match_id,
        "explanation": text,
        "cached": False,
        "top_features": shap_result.get("top_features", [])[:8],
    }


@app.get("/metrics")
def metrics():
    data = {
        "metrics": _load_json(MODELS_DIR / "metrics.json"),
        "ablation": _load_json(MODELS_DIR / "ablation.json"),
        "backtest": _load_json(MODELS_DIR / "backtest_summary.json"),
        "shap_importance": _load_json(MODELS_DIR / "shap_importance.json"),
        "feature_labels": FEATURE_LABELS,
        "artifacts": {
            "calibration_curves": (MODELS_DIR / "calibration_curves.png").exists(),
            "roi_curve": (MODELS_DIR / "roi_curve.png").exists(),
        },
    }
    if data["metrics"] is None:
        raise HTTPException(404, "metrics.json missing — run the pipeline")
    return data


@app.get("/labels")
def labels():
    return {"features": FEATURE_LABELS, "outcomes": OUTCOME_LABELS}
