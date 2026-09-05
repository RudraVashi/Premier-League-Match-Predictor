# PitchPulse — EPL Match Outcome Predictor (v2)

**Can calibrated tree models, fed with results-derived form plus real xG and shot volume, produce useful Premier League probabilities — and can a fan-facing product explain them in football language?**

Portfolio project for data science / ML engineering roles. Dual UI: **Fans** (upcoming slate, plain English) and **Analyst** (metrics, SHAP, ablation).

## Quickstart

```bash
# Python
pip install -r requirements.txt
cp .env.example .env   # optional GOOGLE_API_KEY for Gemini explanations

python run_pipeline.py
# scores models + writes upcoming_predictions (incl. 2026/27 fixtures)

# API
uvicorn api.main:app --reload --port 8000

# Web (separate terminal)
cd web
npm install
npm run dev
```

Open http://localhost:5173 — Fans mode defaults to the upcoming 2026/27 slate (e.g. Everton vs Man United, Sun 6 Sep 2026).

## Product surface

| Mode | What you get |
| --- | --- |
| Fans | Match cards, Home/Draw/Away bars, form `W-D-L`, xG/shots chips, model vs bookmaker, Gemini "why" |
| Analyst | SHAP drivers, hold-out metrics, ablation (base vs xG/shots/possession), ROI summary |

Primary UI is **React + FastAPI** (`web/` + `api/`). The old Streamlit app remains at `app/streamlit_app.py` for notebook-style exploration only.

## Data & features

- **Results + fixtures**: [openfootball/england](https://github.com/openfootball/england) (played + scheduled lines)
- **Odds + shots**: [football-data.co.uk](https://www.football-data.co.uk/englandm.php) (`HS`/`AS`/`HST`/`AST`)
- **xG**: Understat via `soccerdata` (cached under `data/raw/understat/`)
- **Possession**: FBref via `soccerdata` — optional (`EPL_FETCH_FBREF=1`); skipped by default on Windows because the scraper needs a browser driver

Leakage-safe rolling features before kickoff: Elo, form, H2H, rest days, plus rolling xG for/against, shots, and possession when present.

## Validation

- Chronological hold-out from `2023-24` onward
- Metrics: log loss, Brier, per-class PR vs bookmaker + naive home
- **Ablation** written to `models/ablation.json` — report whether xG/shots/possession moved log loss (honest either way)

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness |
| `GET /fixtures/upcoming` | Scheduled 2026/27+ predictions |
| `GET /fixtures/recent` | Recent scored picks + correctness |
| `GET /match/{id}` | Detail + SHAP with friendly labels |
| `POST /match/{id}/explain` | Gemini (or template) narrative |
| `GET /metrics` | Metrics + ablation + backtest |

## Deploy sketch

- API → Render / Railway (`uvicorn api.main:app --host 0.0.0.0 --port $PORT`)
- Web → Vercel/Netlify with `VITE_API_URL=https://your-api.example`

## Disclaimer

Research / portfolio only — not betting advice. Beating closing lines is hard; an honest miss vs the bookmaker is a valid finding.
