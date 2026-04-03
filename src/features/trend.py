"""
Late-season momentum detection — a key alpha source.

Most projection tools use full-season averages. Comparing weeks 13-18
to full-season averages reveals role changes that serve as leading
indicators for the following season.

Classification:
  BREAKOUT  — delta > +BREAKOUT_THRESHOLD (opportunity accelerating)
  STABLE    — delta within threshold range
  DECLINING — delta < DECLINING_THRESHOLD (opportunity fading)

The functions here reuse build_opportunity_factors() and
build_efficiency_factors() from their respective modules, called with a
late-season PBP slice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BREAKOUT_THRESHOLD,
    DECLINING_THRESHOLD,
    MIN_GAMES_FOR_TREND,
    TREND_WEEK_START,
)
from features.opportunity import build_opportunity_factors
from features.efficiency import build_efficiency_factors


# ---------------------------------------------------------------------------
# Core delta computation
# ---------------------------------------------------------------------------

def _compute_delta(
    base: pd.DataFrame,
    late: pd.DataFrame,
    factor_cols: list[str],
    join_keys: list[str] = None,
) -> pd.DataFrame:
    """
    Compute per-player late-season delta relative to early-season values.

    delta = late_value - early_value

    Positive delta → player's role/efficiency was rising in the second half.
    Using early (weeks 1 to TREND_WEEK_START-1) vs. late (weeks TREND_WEEK_START+)
    gives a purer momentum signal than late vs. full (which includes the late weeks).
    """
    if join_keys is None:
        join_keys = ["player_id", "team", "season"]

    merged = base[join_keys + factor_cols].merge(
        late[join_keys + factor_cols],
        on=join_keys,
        suffixes=("_early", "_late"),
        how="inner",
    )
    for col in factor_cols:
        merged[f"{col}_delta"] = merged[f"{col}_late"] - merged[f"{col}_early"]
        merged[f"{col}_acceleration"] = merged[f"{col}_delta"]  # alias for readability

    return merged


def _late_season_games(weekly: pd.DataFrame) -> pd.DataFrame:
    """Return games_played count in the late-season window per player-season."""
    late = weekly[weekly["week"] >= TREND_WEEK_START]
    return (
        late.groupby(["player_id", "season"], observed=True)
        .agg(late_games=("week", "nunique"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Opportunity trend
# ---------------------------------------------------------------------------

def _opportunity_trend(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compare late-season opportunity factors vs. early-season (weeks 1 to TREND_WEEK_START-1).
    """
    pbp_early = pbp[pbp["week"] < TREND_WEEK_START]
    pbp_late = pbp[pbp["week"] >= TREND_WEEK_START]

    opp_early = build_opportunity_factors(pbp_early)
    opp_late = build_opportunity_factors(pbp_late)

    factor_cols = [c for c in ["target_share", "rush_share", "air_yard_share", "wopr"]
                   if c in opp_early.columns and c in opp_late.columns]

    if not factor_cols:
        return pd.DataFrame()

    delta = _compute_delta(opp_early, opp_late, factor_cols)
    return delta


# ---------------------------------------------------------------------------
# Efficiency trend
# ---------------------------------------------------------------------------

def _efficiency_trend(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compare late-season efficiency factors vs. early-season (weeks 1 to TREND_WEEK_START-1).
    """
    pbp_early = pbp[pbp["week"] < TREND_WEEK_START]
    pbp_late = pbp[pbp["week"] >= TREND_WEEK_START]

    eff_early = build_efficiency_factors(pbp_early)
    eff_late = build_efficiency_factors(pbp_late)

    factor_cols = [
        c for c in ["epa_per_target", "epa_per_carry", "epa_per_dropback", "catch_rate"]
        if c in eff_early.columns and c in eff_late.columns
    ]

    if not factor_cols:
        return pd.DataFrame()

    delta = _compute_delta(eff_early, eff_late, factor_cols)
    return delta


# ---------------------------------------------------------------------------
# Snap trend
# ---------------------------------------------------------------------------

def _snap_trend(snap_df: pd.DataFrame, season: int) -> pd.DataFrame | None:
    """
    Late-season snap share delta from snap count data.
    Returns None if snap data is unavailable.
    """
    if snap_df is None or snap_df.empty:
        return None

    snap_pct_col = next(
        (c for c in ["offense_pct", "off_pct"] if c in snap_df.columns), None
    )
    if snap_pct_col is None:
        return None

    season_snaps = snap_df[snap_df["season"] == season].copy()
    early_avg = (
        season_snaps[season_snaps["week"] < TREND_WEEK_START]
        .groupby(["player_id", "season"], observed=True)
        [snap_pct_col].mean().reset_index().rename(columns={snap_pct_col: "snap_pct_early"})
    )
    late_avg = (
        season_snaps[season_snaps["week"] >= TREND_WEEK_START]
        .groupby(["player_id", "season"], observed=True)
        [snap_pct_col].mean().reset_index().rename(columns={snap_pct_col: "snap_pct_late"})
    )
    merged = early_avg.merge(late_avg, on=["player_id", "season"])
    merged["snap_trend"] = merged["snap_pct_late"] - merged["snap_pct_early"]
    return merged[["player_id", "season", "snap_trend"]]


# ---------------------------------------------------------------------------
# Trend classification
# ---------------------------------------------------------------------------

def _classify_trend(row: pd.Series) -> str:
    """
    Assign BREAKOUT / STABLE / DECLINING based on available delta columns.

    Uses target_share_delta if present, otherwise rush_share_delta.
    """
    delta_col = None
    for candidate in ["target_share_delta", "rush_share_delta", "wopr_delta"]:
        if candidate in row.index and pd.notna(row[candidate]):
            delta_col = candidate
            break

    if delta_col is None:
        return "STABLE"

    val = row[delta_col]
    if val > BREAKOUT_THRESHOLD:
        return "BREAKOUT"
    if val < DECLINING_THRESHOLD:
        return "DECLINING"
    return "STABLE"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_trends(
    pbp: pd.DataFrame,
    weekly: pd.DataFrame,
    season: int,
    snap_df: pd.DataFrame | None = None,
    window: int = 5,
) -> pd.DataFrame:
    """
    Detect late-season momentum for all players in a given season.

    Parameters
    ----------
    pbp : pd.DataFrame
        Cleaned, play-type-filtered PBP for a single season (or multiple).
        The function filters internally to the requested season.
    weekly : pd.DataFrame
        Weekly player stats for games_played count.
    season : int
        The season year to analyze.
    snap_df : pd.DataFrame, optional
        Snap count data.
    window : int
        Not used directly; kept for API compatibility. Trend window is
        determined by TREND_WEEK_START from config.

    Returns
    -------
    pd.DataFrame
        One row per (player_id, season) with trend factor columns and
        a 'trend_class' column: BREAKOUT / STABLE / DECLINING.
    """
    pbp_season = pbp[pbp["season"] == season].copy()
    weekly_season = weekly[weekly["season"] == season].copy()

    if pbp_season.empty:
        return pd.DataFrame()

    # Minimum late-season games requirement
    late_games = _late_season_games(weekly_season)

    # Opportunity trend
    opp_trend = _opportunity_trend(pbp_season)

    # Efficiency trend
    eff_trend = _efficiency_trend(pbp_season)

    # Snap trend
    snap_trend = _snap_trend(snap_df, season) if snap_df is not None else None

    # Merge all trend signals
    result = opp_trend.copy() if not opp_trend.empty else pd.DataFrame()

    if not eff_trend.empty and not result.empty:
        # Use a thin merge on common keys
        common_keys = [c for c in ["player_id", "team", "season"] if c in eff_trend.columns]
        eff_cols = [c for c in eff_trend.columns
                    if c not in result.columns or c in common_keys]
        result = result.merge(
            eff_trend[eff_cols], on=[k for k in common_keys if k in result.columns],
            how="outer"
        )
    elif not eff_trend.empty:
        result = eff_trend.copy()

    if result.empty:
        return pd.DataFrame()

    # Attach late games count and apply minimum threshold
    if "player_id" in result.columns:
        result = result.merge(late_games, on=["player_id", "season"], how="left")
        # Players with fewer than MIN_GAMES_FOR_TREND late games get NaN trend factors
        insufficient = result["late_games"].fillna(0) < MIN_GAMES_FOR_TREND
        trend_delta_cols = [c for c in result.columns if c.endswith("_delta") or c.endswith("_acceleration")]
        result.loc[insufficient, trend_delta_cols] = np.nan

    # Snap trend
    if snap_trend is not None and "player_id" in result.columns:
        result = result.merge(snap_trend, on=["player_id", "season"], how="left")

    # Classify each player
    result["trend_class"] = result.apply(_classify_trend, axis=1)

    return result
