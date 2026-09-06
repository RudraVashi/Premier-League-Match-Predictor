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

    # Edge vs market on the model's predicted class
    book = payload.get("bookmaker") or {}
    model_p = None
    book_p = None
    if pred == "H":
        model_p, book_p = payload["probabilities"]["home"], book.get("p_home")
    elif pred == "D":
        model_p, book_p = payload["probabilities"]["draw"], book.get("p_draw")
    elif pred == "A":
        model_p, book_p = payload["probabilities"]["away"], book.get("p_away")
    if model_p is not None and book_p is not None:
        edge = float(model_p) - float(book_p)
        payload["edge"] = {
            "pp": round(edge * 100, 1),
            "model_p": float(model_p),
            "book_p": float(book_p),
            "side": pred,
        }
    else:
        payload["edge"] = None
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
    from src.fixtures_refresh import filter_upcoming_frame
    from src.live_odds import attach_live_odds

    played_match_ids = _ingest_live_results(force=refresh)

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


def _ingest_live_results(force: bool = False) -> set[str]:
    """Pull openfootball + FPL/football-data scores into features; return played ids."""
    played_match_ids: set[str] = set()
    try:
        from src.fixtures_refresh import refresh_openfootball_statuses

        fresh = refresh_openfootball_statuses(force=force)
        if fresh is not None and not fresh.empty:
            played = fresh[fresh["status"] == "played"].copy()
            if not played.empty:
                played["match_id"] = (
                    pd.to_datetime(played["date"]).dt.strftime("%Y%m%d")
                    + "_"
                    + played["home_team"].str.replace(" ", "", regex=False)
                    + "_"
                    + played["away_team"].str.replace(" ", "", regex=False)
                )
                played_match_ids |= set(played["match_id"].astype(str))
    except Exception as exc:  # noqa: BLE001
        print(f"fixture refresh skipped: {exc}")

    try:
        from src.results_fallback import apply_results_to_features, played_match_ids as live_played

        played_match_ids |= live_played(force=force)
        apply_results_to_features(force=force)
    except Exception as exc:  # noqa: BLE001
        print(f"live results ingest skipped: {exc}")
    return played_match_ids


def _kickoff_sort_key(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.where(~s.isin(["nan", "None", "<NA>", "NaT", ""]), "15:00")


def ODDS_KEY_PRESENT() -> bool:
    import os
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
    key = (os.getenv("THE_ODDS_API_KEY") or "").strip().strip('"').strip("'")
    return bool(key) and not key.startswith("your_")


@app.get("/fixtures/recent")
def fixtures_recent(
    limit: int = Query(40, ge=1, le=200),
    season: str | None = Query(None, description="Prefer this season label, e.g. 2026-27"),
    team: str | None = Query(None, description="Filter home/away team substring, e.g. Man United"),
):
    """
    Analyst slate: newest-season played matches from features (live),
    with probs/correct joined from latest_predictions when available.
    Avoids stale May spillover from a mixed 40-row parquet artifact.
    """
    _ingest_live_results(force=False)
    feats = _read_parquet("features.parquet", "matches_with_stats.parquet", "matches_with_odds.parquet")
    scored = _read_parquet("latest_predictions.parquet", "test_predictions.parquet")
    if feats.empty and scored.empty:
        raise HTTPException(404, "No recent predictions found")

    df = feats.copy() if not feats.empty else scored.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "status" in df.columns:
        df = df[df["status"] != "scheduled"]
    if "outcome" in df.columns:
        df = df[df["outcome"].notna()]

    if season and "season" in df.columns:
        df = df[df["season"] == season]
    elif "season" in df.columns and not df.empty:
        newest = sorted(df["season"].dropna().unique())[-1]
        current = df[df["season"] == newest]
        if len(current) >= 3:
            df = current
        else:
            cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=21)
            df = df[df["date"] >= cutoff]

    if team:
        t = team.strip().lower()
        aliases = {
            "united": "man united",
            "man u": "man united",
            "mufc": "man united",
            "city": "man city",
            "spurs": "tottenham",
            "wolves": "wolves",
            "nottingham": "nott'm forest",
            "forest": "nott'm forest",
        }
        needle = aliases.get(t, t)
        mask = df["home_team"].astype(str).str.lower().str.contains(needle, regex=False) | df[
            "away_team"
        ].astype(str).str.lower().str.contains(needle, regex=False)
        df = df[mask]

    if "kickoff" in df.columns:
        df = df.assign(_ko=_kickoff_sort_key(df["kickoff"]))
        df = df.sort_values(["date", "_ko"], ascending=[False, False]).drop(columns=["_ko"])
    else:
        df = df.sort_values("date", ascending=False)
    df = df.head(limit)

    # Join model probs / correct from scored artifact when present
    if not scored.empty and "match_id" in df.columns and "match_id" in scored.columns:
        keep = [
            c
            for c in [
                "match_id",
                "pred",
                "p_H",
                "p_D",
                "p_A",
                "correct",
                "odds_h",
                "odds_d",
                "odds_a",
                "book_p_h",
                "book_p_d",
                "book_p_a",
            ]
            if c in scored.columns
        ]
        scored_slim = scored[keep].drop_duplicates("match_id")
        drop_overlap = [c for c in keep if c != "match_id" and c in df.columns]
        df = df.drop(columns=drop_overlap, errors="ignore").merge(
            scored_slim, on="match_id", how="left"
        )

    need_score = "p_H" not in df.columns or df["p_H"].isna().any()
    if need_score and not df.empty:
        try:
            from src.predict import _score_frame, load_best_model

            name, bundle = load_best_model()
            missing = df["p_H"].isna() if "p_H" in df.columns else pd.Series(True, index=df.index)
            if missing.any():
                scored_part = _score_frame(df.loc[missing].copy(), bundle, name)
                for col in ("pred", "p_H", "p_D", "p_A", "correct"):
                    if col in scored_part.columns:
                        df.loc[missing, col] = scored_part[col].values
        except Exception as exc:  # noqa: BLE001
            print(f"recent scoring skipped: {exc}")

    matches = [_row_to_match(r) for _, r in df.iterrows()]
    graded = [m for m in matches if m.get("correct") is not None]
    n_ok = sum(1 for m in graded if m.get("correct") is True)
    strip = [
        {"match_id": m["match_id"], "correct": m.get("correct"), "label": "C" if m.get("correct") else "M"}
        for m in matches[:10]
        if m.get("correct") is not None
    ]
    last10_ok = sum(1 for s in strip if s["correct"])

    return {
        "count": len(df),
        "season": season or (str(df["season"].iloc[0]) if "season" in df.columns and len(df) else None),
        "latest_matchday": int(df["matchday"].max()) if "matchday" in df.columns and df["matchday"].notna().any() else None,
        "matches": matches,
        "hit_rate": {
            "n": int(len(graded)),
            "correct": int(n_ok),
            "rate": round(float(n_ok) / len(graded), 3) if graded else None,
            "last_n": len(strip),
            "last_n_correct": last10_ok,
            "last_n_rate": round(last10_ok / len(strip), 3) if strip else None,
            "strip": strip,
        },
    }


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

    feats = _read_parquet("features.parquet")
    if not feats.empty and "match_id" in feats.columns:
        frow = feats[feats["match_id"] == match_id]
        if not frow.empty:
            for col in frow.columns:
                if col not in row.index or pd.isna(row.get(col)):
                    row[col] = frow.iloc[0][col]

    detail = _row_to_match(row)

    book = detail.get("bookmaker") or {}
    if book.get("p_home") is None:
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
                    pred = detail.get("prediction")
                    probs = detail.get("probabilities") or {}
                    bp = detail["bookmaker"]
                    mp = bp_map = None
                    if pred == "H":
                        mp, bp_map = probs.get("home"), bp.get("p_home")
                    elif pred == "D":
                        mp, bp_map = probs.get("draw"), bp.get("p_draw")
                    elif pred == "A":
                        mp, bp_map = probs.get("away"), bp.get("p_away")
                    if mp is not None and bp_map is not None:
                        edge = float(mp) - float(bp_map)
                        detail["edge"] = {
                            "pp": round(edge * 100, 1),
                            "model_p": float(mp),
                            "book_p": float(bp_map),
                            "side": pred,
                        }
        except Exception as exc:  # noqa: BLE001
            print(f"live odds detail skipped: {exc}")

    try:
        from src.poisson_scores import top_scorelines

        snap = detail.get("stats_snapshot") or {}
        detail["scorelines"] = top_scorelines(
            snap.get("home_xg_for_5"),
            snap.get("away_xg_for_5"),
            top_n=5,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scorelines skipped: {exc}")
        detail["scorelines"] = []

    shap_payload = None
    try:
        xgb_path = MODELS_DIR / "xgboost.joblib"
        model_path = xgb_path if xgb_path.exists() else next(MODELS_DIR.glob("*.joblib"), None)
        if model_path and model_path.exists() and not feats.empty:
            bundle = joblib.load(model_path)
            frow = feats[feats["match_id"] == match_id]
            if not frow.empty:
                shap_result = shap_top_features_for_row(bundle, frow.iloc[0])
                top = []
                for item in shap_result.get("top_features", [])[:8]:
                    top.append({**item, "label": friendly_feature(item["feature"])})
                shap_payload = {
                    "predicted_class": shap_result["predicted_class"],
                    "probabilities": shap_result["probabilities"],
                    "top_features": top,
                    "model": model_path.stem,
                }
    except Exception as exc:  # noqa: BLE001
        print(f"shap skipped: {exc}")
    detail["shap"] = shap_payload

    for cache_name in ("upcoming_explanations.json", "latest_explanations.json", "llm_explanations.json"):
        cache = _load_json(CACHE_DIR / cache_name)
        if not cache:
            continue
        items = cache.get("items", cache) if isinstance(cache, dict) else []
        if isinstance(items, dict) and match_id in items:
            entry = items[match_id]
            detail["explanation"] = entry.get("text") if isinstance(entry, dict) else entry
            break
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                if it.get("match_id") == match_id:
                    detail["explanation"] = it.get("text") or it.get("explanation")
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
