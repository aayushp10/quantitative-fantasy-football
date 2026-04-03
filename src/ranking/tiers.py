"""
K-means tier assignment within positions.

Tiers identify value cliffs in the rankings — the gap between WR12 and WR13
being in different tiers signals a significant talent dropoff at that point.

Implementation: K-means on VORP (or projected_fpts_season) within each
position. Clusters are sorted by mean VORP descending and labeled
Tier 1 through Tier N (Tier 1 = best).

K-means on 1D VORP is effectively analogous to Jenks natural breaks
when the distribution has real clusters, without the extra dependency.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def assign_tiers(
    projections_df: pd.DataFrame,
    position: str,
    n_tiers: int = 8,
    value_col: str | None = None,
) -> pd.DataFrame:
    """
    Assign K-means tier labels to players within a position.

    Parameters
    ----------
    projections_df : pd.DataFrame
        Must have 'position' column and a value column (vorp or
        projected_fpts_season).
    position : str
        One of 'QB', 'RB', 'WR', 'TE', or 'ALL'.
    n_tiers : int
        Number of K-means clusters. Actual tiers may be fewer if there
        aren't enough distinct value levels.
    value_col : str, optional
        Column to cluster on. Defaults to 'vorp' if present,
        else 'projected_fpts_season'.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'tier' column added/updated for the
        specified position.
    """
    if value_col is None:
        value_col = "vorp" if "vorp" in projections_df.columns else "projected_fpts_season"

    if value_col not in projections_df.columns:
        raise ValueError(f"Column '{value_col}' not found in projections_df.")

    proj = projections_df.copy()

    if "tier" not in proj.columns:
        proj["tier"] = np.nan

    positions_to_tier = [position] if position != "ALL" else ["QB", "RB", "WR", "TE"]

    for pos in positions_to_tier:
        mask = proj["position"] == pos
        pos_df = proj.loc[mask].copy()

        if pos_df.empty:
            continue

        values = pos_df[value_col].dropna().values.reshape(-1, 1)

        if len(values) == 0:
            continue

        actual_tiers = min(n_tiers, len(values))
        if actual_tiers < 2:
            proj.loc[mask, "tier"] = 1
            continue

        km = KMeans(
            n_clusters=actual_tiers,
            n_init=10,
            random_state=42,
        )

        # Fit on non-null values
        valid_mask = pos_df[value_col].notna()
        if valid_mask.sum() < actual_tiers:
            proj.loc[mask, "tier"] = 1
            continue

        km.fit(pos_df.loc[valid_mask, value_col].values.reshape(-1, 1))
        raw_labels = km.predict(pos_df.loc[valid_mask, value_col].values.reshape(-1, 1))

        # Map cluster labels to tier numbers (Tier 1 = highest value)
        cluster_centers = km.cluster_centers_.flatten()
        # Sort clusters by center descending; cluster with highest center = Tier 1
        rank_map = {
            cluster_id: rank + 1
            for rank, cluster_id in enumerate(np.argsort(cluster_centers)[::-1])
        }
        tier_labels = np.array([rank_map[label] for label in raw_labels])

        # Assign back (only valid rows)
        valid_indices = pos_df.loc[valid_mask].index
        proj.loc[valid_indices, "tier"] = tier_labels.astype(int)

        # Players with null value col get the lowest tier
        null_indices = pos_df.loc[~valid_mask].index
        proj.loc[null_indices, "tier"] = actual_tiers

    proj["tier"] = proj["tier"].astype("Int64")  # nullable integer
    return proj


def assign_tiers_all_positions(
    projections_df: pd.DataFrame,
    n_tiers: int = 8,
    value_col: str | None = None,
) -> pd.DataFrame:
    """Assign tiers for all positions in a single call."""
    result = projections_df.copy()
    for pos in ["QB", "RB", "WR", "TE"]:
        result = assign_tiers(result, position=pos, n_tiers=n_tiers, value_col=value_col)
    return result
