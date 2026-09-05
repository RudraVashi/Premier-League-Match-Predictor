"""Deprecated primary UI — use React + FastAPI instead.

  uvicorn api.main:app --reload --port 8000
  cd web && npm run dev

This Streamlit app remains as an optional analyst sandbox.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="PitchPulse (legacy Streamlit)", layout="wide")
st.title("Legacy Streamlit UI")
st.warning(
    "PitchPulse v2 ships a React Fans/Analyst dashboard. "
    "Run `uvicorn api.main:app --port 8000` and `cd web && npm run dev`."
)
st.markdown(
    """
### Why the change
Streamlit was fine for prototyping metrics, but the fan product needs:
- upcoming 2026/27 fixtures as the default view
- football language (xG, shots, form strings)
- dual Fans / Analyst modes with real layout control

See the project README for the v2 quickstart.
"""
)
