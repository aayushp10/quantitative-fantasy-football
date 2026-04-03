"""
Weekly scoring consistency / variance features.

A player who scores 15 fpts/game every week is more valuable (and more
predictable) than one who scores 30 one week and 0 the next at the same
seasonal average. Consistency features capture this variance:

  weekly_fpts_std  — raw weekly standard deviation
  weekly_fpts_cv   — coefficient of variation (std / mean); scale-free
  weekly_fpts_median — median weekly score; robust to boom outliers
  boom_rate        — fraction of weeks above 1.5× positional median
  bust_rate        — fraction of weeks below 0.5× positional median
  consistency_score — composite: (1 - cv) × (1 - bust_rate)

High boom_rate + low median → TD-dependent profile (volatile)
High bust_rate → injury/role risk or heavy snap rotation
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_consistency_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly scoring consistency metrics per (player_id, season).

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Cleaned weekly player stats. Must contain:
            player_id, season, week, and one of:
            fantasy_points_ppr (preferred) or fantasy_points.
        A 'position' column is required for boom/bust positional medians;
        if absent, boom_rate and bust_rate are set to NaN.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, season, weekly_fpts_std, weekly_fpts_cv,
                 weekly_fpts_median, boom_rate, bust_rate, consistency_score.
    """
    required = {"player_id", "season", "week"}
    missing = required - set(weekly_df.columns)
    if missing:
        raise ValueError(f"weekly_df missing required columns: {missing}")

    # Select fantasy points column
    if "fantasy_points_ppr" in weekly_df.columns:
        fpts_col = "fantasy_points_ppr"
    elif "fantasy_points" in weekly_df.columns:
        fpts_col = "fantasy_points"
    else:
        raise ValueError("weekly_df has neither 'fantasy_points_ppr' nor 'fantasy_points' column.")

    df = weekly_df.copy()
    df[fpts_col] = pd.to_numeric(df[fpts_col], errors="coerce").fillna(0.0)

    # Per-player-season aggregates
    agg = (
        df.groupby(["player_id", "season"], observed=True)[fpts_col]
        .agg(
            weekly_fpts_std="std",
            weekly_fpts_mean="mean",
            weekly_fpts_median="median",
        )
        .reset_index()
    )

    # Coefficient of variation (NaN when mean == 0 to avoid div/0)
    agg["weekly_fpts_cv"] = np.where(
        agg["weekly_fpts_mean"] > 0,
        agg["weekly_fpts_std"] / agg["weekly_fpts_mean"],
        np.nan,
    )

    # -----------------------------------------------------------------------
    # Boom / bust rates (require positional medians)
    # -----------------------------------------------------------------------
    has_position = "position" in weekly_df.columns

    if has_position:
        # Positional season median of per-game scores
        pos_medians = (
            df.groupby(["season", "position"], observed=True)[fpts_col]
            .median()
            .reset_index()
            .rename(columns={fpts_col: "pos_median_fpts"})
        )

        # Per-game rows with positional median attached
        per_game = df.merge(pos_medians, on=["season", "position"], how="left")
        per_game["is_boom"] = (per_game[fpts_col] > 1.5 * per_game["pos_median_fpts"]).astype(float)
        per_game["is_bust"] = (per_game[fpts_col] < 0.5 * per_game["pos_median_fpts"]).astype(float)

        boom_bust = (
            per_game.groupby(["player_id", "season"], observed=True)[["is_boom", "is_bust"]]
            .mean()
            .reset_index()
            .rename(columns={"is_boom": "boom_rate", "is_bust": "bust_rate"})
        )
        agg = agg.merge(boom_bust, on=["player_id", "season"], how="left")
    else:
        agg["boom_rate"] = np.nan
        agg["bust_rate"] = np.nan

    # Composite consistency score: higher = more consistent
    # Clamp cv to [0, 1] for the formula to be well-behaved
    cv_clamped = agg["weekly_fpts_cv"].clip(upper=1.0)
    bust_filled = agg["bust_rate"].fillna(0.5)  # neutral when position unknown
    agg["consistency_score"] = (1.0 - cv_clamped) * (1.0 - bust_filled)

    return agg[
        ["player_id", "season", "weekly_fpts_std", "weekly_fpts_cv",
         "weekly_fpts_median", "boom_rate", "bust_rate", "consistency_score"]
    ].reset_index(drop=True)
