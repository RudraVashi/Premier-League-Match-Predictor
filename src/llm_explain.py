"""Gemini LLM layer: turn SHAP + match context into grounded natural-language explanations."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.utils import CACHE_DIR, ensure_dirs

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
CACHE_FILE = CACHE_DIR / "llm_explanations.json"


SYSTEM_PROMPT = """You are a football analytics assistant explaining Premier League match predictions.
You MUST ground every claim in the provided SHAP feature contributions and match context.
Prefer football language: xG, shots, possession, form, strength gap — not raw snake_case names.
Do not invent injuries, tactics, or news that are not in the context.
Be concise (3-5 sentences), clear, and honest about uncertainty — especially around draws.
Mention the top drivers by name in plain English.
"""


def _load_cache() -> dict:
    ensure_dirs()
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    ensure_dirs()
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


from src.labels import friendly_feature


def _friendly_feature(name: str) -> str:
    return friendly_feature(name)


def build_prompt(context: dict) -> str:
    top = context.get("top_features", [])
    lines = []
    for f in top:
        direction = "supports home" if f["shap"] > 0 else "supports away/draw shift"
        # SHAP sign depends on class; describe magnitude neutrally too
        lines.append(
            f"- {_friendly_feature(f['feature'])}: value={f['value']:.3f}, "
            f"SHAP={f['shap']:+.4f} ({direction})"
        )
    probs = context.get("probabilities", {})
    return f"""Match: {context.get('home_team')} vs {context.get('away_team')}
Date: {context.get('date')}
Model prediction: {context.get('predicted_class')}
Probabilities: Home={probs.get('H', 0):.1%}, Draw={probs.get('D', 0):.1%}, Away={probs.get('A', 0):.1%}

Top SHAP drivers for the predicted class:
{chr(10).join(lines)}

Write a short natural-language explanation of why the model leaned this way.
"""


def explain_with_gemini(
    context: dict,
    model_name: str | None = None,
    use_cache: bool = True,
) -> str:
    """
    Call Gemini to narrate a SHAP explanation.
    Caches by match_id to stay within free-tier limits.
    """
    ensure_dirs()
    match_id = str(context.get("match_id", ""))
    cache = _load_cache() if use_cache else {}
    if use_cache and match_id and match_id in cache:
        return cache[match_id]["text"]

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        # Offline fallback so the app still works without a key
        text = _fallback_explanation(context)
        if match_id:
            cache[match_id] = {"text": text, "source": "fallback"}
            _save_cache(cache)
        return text

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        model = model_name or DEFAULT_MODEL
        prompt = build_prompt(context)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.4,
            },
        )
        text = (response.text or "").strip() or _fallback_explanation(context)
        if match_id:
            cache[match_id] = {"text": text, "source": "gemini", "model": model}
            _save_cache(cache)
        return text
    except Exception as exc:  # noqa: BLE001 — surface friendly fallback
        text = _fallback_explanation(context) + f"\n\n_(LLM unavailable: {exc})_"
        if match_id:
            cache[match_id] = {"text": text, "source": "fallback_error"}
            _save_cache(cache)
        return text


def _fallback_explanation(context: dict) -> str:
    """Deterministic template when Gemini is unavailable."""
    home = context.get("home_team", "Home")
    away = context.get("away_team", "Away")
    pred = context.get("predicted_class", "?")
    probs = context.get("probabilities", {})
    top = context.get("top_features", [])[:3]
    drivers = ", ".join(_friendly_feature(t["feature"]) for t in top) if top else "form and Elo"
    pred_word = {"H": f"a {home} home win", "D": "a draw", "A": f"an {away} away win"}.get(
        pred, pred
    )
    return (
        f"The model leans toward {pred_word} "
        f"(H {probs.get('H', 0):.0%} / D {probs.get('D', 0):.0%} / A {probs.get('A', 0):.0%}). "
        f"The strongest drivers in the SHAP breakdown are {drivers}. "
        f"This explanation is template-based; set GOOGLE_API_KEY for Gemini narratives."
    )


def explain_match(shap_result: dict, meta: dict) -> str:
    """Convenience wrapper combining SHAP output + match metadata."""
    context = {
        "match_id": meta.get("match_id"),
        "home_team": meta.get("home_team"),
        "away_team": meta.get("away_team"),
        "date": str(meta.get("date", "")),
        "predicted_class": shap_result.get("predicted_class"),
        "probabilities": shap_result.get("probabilities", {}),
        "top_features": shap_result.get("top_features", []),
    }
    return explain_with_gemini(context)


if __name__ == "__main__":
    demo = {
        "match_id": "demo",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "date": "2024-04-01",
        "predicted_class": "H",
        "probabilities": {"H": 0.52, "D": 0.25, "A": 0.23},
        "top_features": [
            {"feature": "elo_diff", "value": 80.0, "shap": 0.12},
            {"feature": "home_form_pts_5", "value": 2.2, "shap": 0.08},
            {"feature": "away_form_pts_5", "value": 1.0, "shap": 0.05},
        ],
    }
    print(explain_with_gemini(demo, use_cache=False))
