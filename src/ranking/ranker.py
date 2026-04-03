"""
Convert projections and VORP into final rankings.

Generates:
- overall_rank: ranked by projected_fpts_season across all positions
- pos_rank: ranked within each position

Also supports re-ranking for different scoring formats by passing
different projected_fpts_season values.
"""
from __future__ import annotations

import pandas as pd

from config import POSITIONS


def generate_rankings(projections_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add overall_rank and pos_rank columns to a projections DataFrame.

    If 'vorp' is present, rankings are by VORP (draft value). Otherwise
    rankings are by projected_fpts_season.

    Parameters
    ----------
    projections_df : pd.DataFrame
        Must have 'position' and 'projected_fpts_season'. Optionally 'vorp'.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added 'overall_rank' and 'pos_rank' columns,
        sorted by overall_rank ascending.
    """
    rank_col = "vorp" if "vorp" in projections_df.columns else "projected_fpts_season"

    proj = projections_df.copy()

    # Overall rank (across all positions, by value)
    proj["overall_rank"] = (
        proj[rank_col].rank(ascending=False, method="min").astype(int)
    )

    # Positional rank
    proj["pos_rank"] = (
        proj.groupby("position", observed=True)[rank_col]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Sort by overall rank
    proj = proj.sort_values("overall_rank").reset_index(drop=True)

    return proj


def rankings_table(
    projections_df: pd.DataFrame,
    position: str | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Return a clean, display-ready rankings table.

    Parameters
    ----------
    projections_df : pd.DataFrame
        Output from generate_rankings().
    position : str, optional
        Filter to a single position. If None, returns all positions.
    top_n : int, optional
        Limit to top N players.

    Returns
    -------
    pd.DataFrame
        Selected columns, filtered and formatted.
    """
    df = projections_df.copy()

    if position is not None:
        df = df[df["position"] == position]

    display_cols = [
        col for col in [
            "overall_rank", "pos_rank", "player_name", "position", "team",
            "age", "projected_fpts_pg", "projected_games", "projected_fpts_season",
            "vorp", "tier", "trend_class",
        ]
        if col in df.columns
    ]

    df = df[display_cols]

    if top_n:
        df = df.head(top_n)

    # Format floats
    for col in ["projected_fpts_pg", "projected_fpts_season", "vorp"]:
        if col in df.columns:
            df[col] = df[col].round(1)
    if "age" in df.columns:
        df["age"] = df["age"].round(1)

    return df.reset_index(drop=True)
