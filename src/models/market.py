"""
Market-relative evaluation: is there any alpha vs. ADP?

Raw accuracy (MAE, rank corr vs. actuals) is meaningless without the market
baseline — ADP alone rank-correlates ~0.55-0.65 with finish. The tests here
answer the only question that matters on draft day:

  Do MODEL-vs-MARKET disagreements predict outcomes?

Long/short construction (per position, per season):
  edge_i    = adp_pos_rank_i - model_pos_rank_i     (>0: model likes player
                                                     MORE than the market)
  outcome_i = adp_pos_rank_i - actual_pos_rank_i    (>0: player BEAT their
                                                     market price)
  longs  = top MARKET_LS_TOP_FRAC by edge, shorts = bottom fraction.
  ls_spread = mean(outcome | long) - mean(outcome | short)

A positive, consistent ls_spread is alpha. A model whose raw rank corr beats
ADP's but whose ls_spread is ~0 is just repricing the consensus.

Public API
----------
market_baseline(yoy_with_adp)                 — ADP's own rank IC by season/position
long_short_test(df, ...)                      — one season/position L/S evaluation
rolling_market_backtest(model_class, ...)     — walk-forward: model IC vs ADP IC
                                                + L/S spread per test season
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import MARKET_LS_TOP_FRAC, POSITIONS
from models.backtest import _predict_for_backtest


# ---------------------------------------------------------------------------
# Market baseline
# ---------------------------------------------------------------------------

def market_baseline(
    yoy_with_adp: pd.DataFrame,
    target: str = "next_fpts",
) -> pd.DataFrame:
    """
    ADP's own predictive power: Spearman(-adp, next_fpts) per (season, position).

    Input: YoY pairs with 'adp' attached (attach_adp(..., season_offset=1)).
    This is the bar the model has to clear.
    """
    rows = []
    df = yoy_with_adp.dropna(subset=["adp", target])
    for (season, pos), sub in df.groupby(["season", "position"], observed=True):
        if len(sub) < 8:
            continue
        ic, _ = spearmanr(-sub["adp"], sub[target])
        rows.append({
            "season": season, "position": pos,
            "adp_rank_ic": round(float(ic), 3), "n": len(sub),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        avg = (
            result.groupby("position", observed=True)["adp_rank_ic"]
            .mean().round(3).reset_index()
        )
        avg["season"] = "average"
        avg["n"] = result.groupby("position", observed=True)["n"].sum().values
        result = pd.concat([result, avg], ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Long/short residual-vs-market test
# ---------------------------------------------------------------------------

def long_short_test(
    df: pd.DataFrame,
    pred_col: str = "pred",
    adp_rank_col: str = "adp_pos_rank",
    actual_col: str = "next_fpts",
    top_frac: float = MARKET_LS_TOP_FRAC,
) -> dict[str, Any] | None:
    """
    One-group (single position-season) long/short evaluation.

    Requires >= 10 rows with pred, adp rank, and actual all present.
    Returns dict with ls_spread (in positional-rank units), hit rates, and
    the long/short player counts — or None if too few rows.
    """
    sub = df.dropna(subset=[pred_col, adp_rank_col, actual_col]).copy()
    if len(sub) < 10:
        return None

    sub["model_pos_rank"] = sub[pred_col].rank(ascending=False, method="min")
    sub["actual_pos_rank"] = sub[actual_col].rank(ascending=False, method="min")

    sub["edge"] = sub[adp_rank_col] - sub["model_pos_rank"]      # >0: model over market
    sub["outcome"] = sub[adp_rank_col] - sub["actual_pos_rank"]  # >0: beat market price

    k = max(2, int(round(top_frac * len(sub))))
    longs = sub.nlargest(k, "edge")
    shorts = sub.nsmallest(k, "edge")

    return {
        "ls_spread": float(longs["outcome"].mean() - shorts["outcome"].mean()),
        "long_hit_rate": float((longs["outcome"] > 0).mean()),
        "short_hit_rate": float((shorts["outcome"] < 0).mean()),
        "n": int(len(sub)),
        "k_per_side": int(k),
    }


# ---------------------------------------------------------------------------
# Rolling market backtest
# ---------------------------------------------------------------------------

def rolling_market_backtest(
    model_class: type,
    yoy_df: pd.DataFrame,
    adp_df: pd.DataFrame,
    test_seasons: list[int] | None = None,
    target: str = "next_fpts",
    top_frac: float = MARKET_LS_TOP_FRAC,
    **model_kwargs: Any,
) -> pd.DataFrame:
    """
    Walk-forward evaluation of model vs. market.

    For each test season S (feature season; ADP joined from draft year S+1):
      - train model on seasons < S
      - per position: model rank IC, ADP rank IC, IC edge, and the
        long/short spread of model-vs-ADP disagreements.

    Returns
    -------
    pd.DataFrame
        Columns: test_season, position, n, model_ic, adp_ic, ic_edge,
        ls_spread, long_hit_rate, short_hit_rate. Plus 'average' rows.
        ls_spread > 0 with consistency across seasons = actual alpha.
    """
    from data.adp import attach_adp

    if test_seasons is None:
        test_seasons = [2021, 2022, 2023]

    rows = []
    for test_season in test_seasons:
        train_df = yoy_df[yoy_df["season"] < test_season].copy()
        test_df = yoy_df[yoy_df["season"] == test_season].copy()
        if train_df.empty or test_df.empty or len(train_df["season"].unique()) < 2:
            warnings.warn(f"Skipping test season {test_season}: insufficient data.")
            continue

        model = model_class(**model_kwargs)
        try:
            model.train(train_df, target=target, fit_age=False)
        except Exception as e:
            warnings.warn(f"Training failed for {test_season}: {e}")
            continue

        # ADP from the draft year the predictions would trade in
        test_with_adp = attach_adp(test_df, adp_df, season_offset=1)

        for pos in POSITIONS:
            pos_test = (
                test_with_adp[test_with_adp["position"] == pos]
                .copy().reset_index(drop=True)
            )
            if pos_test.empty:
                continue

            pred = _predict_for_backtest(model, model_class, pos_test, pos, target)
            if pred is None:
                continue
            pos_test["pred"] = pred

            eval_df = pos_test.dropna(subset=["pred", "adp", target])
            if len(eval_df) < 10:
                continue

            model_ic, _ = spearmanr(eval_df["pred"], eval_df[target])
            adp_ic, _ = spearmanr(-eval_df["adp"], eval_df[target])
            ls = long_short_test(
                eval_df, pred_col="pred", adp_rank_col="adp_pos_rank",
                actual_col=target, top_frac=top_frac,
            )

            rows.append({
                "test_season": test_season,
                "position": pos,
                "n": int(len(eval_df)),
                "model_ic": round(float(model_ic), 3),
                "adp_ic": round(float(adp_ic), 3),
                "ic_edge": round(float(model_ic - adp_ic), 3),
                "ls_spread": round(ls["ls_spread"], 2) if ls else np.nan,
                "long_hit_rate": round(ls["long_hit_rate"], 2) if ls else np.nan,
                "short_hit_rate": round(ls["short_hit_rate"], 2) if ls else np.nan,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "test_season", "position", "n", "model_ic", "adp_ic",
            "ic_edge", "ls_spread", "long_hit_rate", "short_hit_rate",
        ])

    result = pd.DataFrame(rows)

    avg_rows = []
    for pos in result["position"].unique():
        pos_data = result[result["position"] == pos]
        avg_rows.append({
            "test_season": "average",
            "position": pos,
            "n": int(pos_data["n"].sum()),
            "model_ic": round(float(pos_data["model_ic"].mean()), 3),
            "adp_ic": round(float(pos_data["adp_ic"].mean()), 3),
            "ic_edge": round(float(pos_data["ic_edge"].mean()), 3),
            "ls_spread": round(float(pos_data["ls_spread"].mean()), 2),
            "long_hit_rate": round(float(pos_data["long_hit_rate"].mean()), 2),
            "short_hit_rate": round(float(pos_data["short_hit_rate"].mean()), 2),
        })
    return pd.concat([result, pd.DataFrame(avg_rows)], ignore_index=True)
