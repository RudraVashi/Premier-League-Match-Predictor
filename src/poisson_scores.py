"""Simple independent Poisson scoreline suggestions from rolling xG."""

from __future__ import annotations

import math
from typing import Any


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def top_scorelines(
    home_xg: float | None,
    away_xg: float | None,
    *,
    max_goals: int = 5,
    top_n: int = 5,
    floor: float = 0.55,
) -> list[dict[str, Any]]:
    """
    Rank most likely scorelines assuming independent Poisson goals.
    Lambdas default from L5 xG; floored so empty stats still produce a board.
    """
    lh = float(home_xg) if home_xg is not None and not (isinstance(home_xg, float) and math.isnan(home_xg)) else 1.2
    la = float(away_xg) if away_xg is not None and not (isinstance(away_xg, float) and math.isnan(away_xg)) else 1.1
    lh = max(floor, min(3.5, lh))
    la = max(floor, min(3.5, la))

    rows: list[dict[str, Any]] = []
    for hg in range(max_goals + 1):
        ph = _poisson_pmf(hg, lh)
        for ag in range(max_goals + 1):
            p = ph * _poisson_pmf(ag, la)
            rows.append(
                {
                    "home_goals": hg,
                    "away_goals": ag,
                    "score": f"{hg}-{ag}",
                    "probability": round(p, 4),
                }
            )
    rows.sort(key=lambda r: r["probability"], reverse=True)
    return rows[:top_n]
