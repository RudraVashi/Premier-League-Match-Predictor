"""Betting simulation: flat-stake and fractional Kelly backtests vs bookmaker odds."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import DATA_PROCESSED, MODELS_DIR, ensure_dirs

OUTCOME_TO_ODDS = {"H": "odds_h", "D": "odds_d", "A": "odds_a"}
OUTCOME_TO_P = {"H": "p_H", "D": "p_D", "A": "p_A"}


def _select_bet(row: pd.Series, min_edge: float = 0.02) -> tuple[str | None, float, float]:
    """
    Pick the outcome with the largest positive edge:
    edge = model_prob - implied_book_prob (from decimal odds, raw 1/odds).
    Returns (outcome, model_prob, decimal_odds) or (None, 0, 0).
    """
    best = None
    best_edge = min_edge
    best_p = 0.0
    best_odds = 0.0
    for outcome, odds_col in OUTCOME_TO_ODDS.items():
        odds = row.get(odds_col)
        p = row.get(OUTCOME_TO_P[outcome])
        if pd.isna(odds) or pd.isna(p) or odds <= 1:
            continue
        implied = 1.0 / float(odds)
        edge = float(p) - implied
        if edge > best_edge:
            best_edge = edge
            best = outcome
            best_p = float(p)
            best_odds = float(odds)
    return best, best_p, best_odds


def kelly_fraction(p: float, odds: float, fraction: float = 0.25) -> float:
    """Fractional Kelly stake as a fraction of bankroll."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    full = (b * p - q) / b
    return max(0.0, full * fraction)


def run_backtest(
    predictions: pd.DataFrame,
    stake: float = 1.0,
    starting_bankroll: float = 100.0,
    min_edge: float = 0.03,
    kelly_frac: float = 0.25,
) -> dict:
    """
    Simulate flat-stake and fractional-Kelly betting on held-out predictions.
    Only bets when model edge over raw implied odds exceeds min_edge.
    """
    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    needed = ["odds_h", "odds_d", "odds_a", "p_H", "p_D", "p_A", "outcome"]
    if not all(c in df.columns for c in needed):
        raise ValueError(f"Predictions missing required columns: {needed}")

    flat_rows = []
    kelly_rows = []
    bankroll = starting_bankroll
    flat_pnl = 0.0
    n_bets = 0
    n_wins = 0

    for _, row in df.iterrows():
        pick, p, odds = _select_bet(row, min_edge=min_edge)
        if pick is None:
            flat_rows.append(
                {"date": row["date"], "bankroll": starting_bankroll + flat_pnl, "bet": False}
            )
            kelly_rows.append({"date": row["date"], "bankroll": bankroll, "bet": False})
            continue

        won = row["outcome"] == pick
        # Flat stake
        profit = stake * (odds - 1.0) if won else -stake
        flat_pnl += profit
        n_bets += 1
        n_wins += int(won)
        flat_rows.append(
            {
                "date": row["date"],
                "match_id": row.get("match_id"),
                "pick": pick,
                "odds": odds,
                "won": won,
                "profit": profit,
                "bankroll": starting_bankroll + flat_pnl,
                "bet": True,
            }
        )

        # Kelly
        f = kelly_fraction(p, odds, fraction=kelly_frac)
        k_stake = bankroll * f
        if k_stake < 1e-6:
            kelly_rows.append({"date": row["date"], "bankroll": bankroll, "bet": False})
            continue
        k_profit = k_stake * (odds - 1.0) if won else -k_stake
        bankroll += k_profit
        kelly_rows.append(
            {
                "date": row["date"],
                "match_id": row.get("match_id"),
                "pick": pick,
                "odds": odds,
                "won": won,
                "stake": k_stake,
                "profit": k_profit,
                "bankroll": bankroll,
                "bet": True,
            }
        )

    flat = pd.DataFrame(flat_rows)
    kelly = pd.DataFrame(kelly_rows)

    flat_roi = flat_pnl / (n_bets * stake) if n_bets else 0.0
    summary = {
        "n_matches": int(len(df)),
        "n_bets": int(n_bets),
        "n_wins": int(n_wins),
        "hit_rate": float(n_wins / n_bets) if n_bets else 0.0,
        "flat_pnl": float(flat_pnl),
        "flat_roi": float(flat_roi),
        "kelly_final_bankroll": float(bankroll),
        "kelly_roi": float((bankroll - starting_bankroll) / starting_bankroll),
        "min_edge": min_edge,
        "flat_stake": stake,
        "starting_bankroll": starting_bankroll,
    }
    print("\n=== Betting backtest ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    _plot_bankroll(flat, kelly, summary)
    ensure_dirs()
    flat.to_csv(DATA_PROCESSED / "backtest_flat.csv", index=False)
    kelly.to_csv(DATA_PROCESSED / "backtest_kelly.csv", index=False)
    pd.Series(summary).to_json(MODELS_DIR / "backtest_summary.json")
    return {"summary": summary, "flat": flat, "kelly": kelly}


def _plot_bankroll(flat: pd.DataFrame, kelly: pd.DataFrame, summary: dict) -> None:
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(10, 5))
    if not flat.empty:
        ax.plot(flat["date"], flat["bankroll"], label="Flat stake (£1)", linewidth=1.8)
    if not kelly.empty:
        ax.plot(kelly["date"], kelly["bankroll"], label="¼-Kelly", linewidth=1.8)
    ax.axhline(summary["starting_bankroll"], color="gray", linestyle="--", linewidth=1)
    ax.set_title(
        f"Simulated bankroll - flat ROI {summary['flat_roi']:.1%} | "
        f"Kelly ROI {summary['kelly_roi']:.1%} ({summary['n_bets']} bets)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Bankroll")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = MODELS_DIR / "roi_curve.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"ROI plot -> {out}")


if __name__ == "__main__":
    preds = pd.read_parquet(DATA_PROCESSED / "test_predictions.parquet")
    run_backtest(preds)
