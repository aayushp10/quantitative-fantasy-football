"""
Fantasy point calculation for any scoring format.

The canonical scoring key names (matching config.py ScoringWeights) map to
actual DataFrame column names via an optional column_map parameter. This
decouples the scoring logic from whichever data source's naming convention
is in use.
"""
from __future__ import annotations

import pandas as pd

from config import PPR_SCORING, ScoringWeights

# Default mapping from scoring key → nfl_data_py weekly stats column name
DEFAULT_COLUMN_MAP: dict[str, str] = {
    "pass_yd": "passing_yards",
    "pass_td": "passing_tds",
    "interception": "interceptions",
    "rush_yd": "rushing_yards",
    "rush_td": "rushing_tds",
    "reception": "receptions",
    "rec_yd": "receiving_yards",
    "rec_td": "receiving_tds",
    "fumble_lost": "sack_fumbles_lost",  # nfl_data_py uses sack_fumbles_lost + rushing_fumbles_lost
    "pass_2pt": "passing_2pt_conversions",
    "rush_2pt": "rushing_2pt_conversions",
    "rec_2pt": "receiving_2pt_conversions",
}

# Some datasets split fumbles; this map handles the common nfl_data_py weekly schema
FUMBLE_COLUMNS = ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]


def calculate_fantasy_points(
    stats_df: pd.DataFrame,
    scoring_config: ScoringWeights | None = None,
    column_map: dict[str, str] | None = None,
) -> pd.Series:
    """
    Compute fantasy points for every row in stats_df.

    Parameters
    ----------
    stats_df : pd.DataFrame
        One row per player-game (or player-season). Columns must include
        the stat categories referenced by column_map.
    scoring_config : dict, optional
        Scoring weights. Defaults to PPR_SCORING from config.
    column_map : dict, optional
        Maps scoring key → DataFrame column name. Defaults to DEFAULT_COLUMN_MAP.

    Returns
    -------
    pd.Series
        Fantasy points for each row, same index as stats_df.
    """
    if scoring_config is None:
        scoring_config = PPR_SCORING
    if column_map is None:
        column_map = DEFAULT_COLUMN_MAP

    total = pd.Series(0.0, index=stats_df.index)

    for key, weight in scoring_config.items():
        if key == "fumble_lost":
            # Aggregate all fumble-lost columns that are present
            fumble_total = pd.Series(0.0, index=stats_df.index)
            for col in FUMBLE_COLUMNS:
                if col in stats_df.columns:
                    fumble_total += stats_df[col].fillna(0.0)
            total += fumble_total * weight
        else:
            col = column_map.get(key)
            if col and col in stats_df.columns:
                total += stats_df[col].fillna(0.0) * weight

    return total


def calculate_fantasy_points_from_dict(
    stats: dict,
    scoring_config: ScoringWeights | None = None,
) -> float:
    """
    Compute fantasy points for a single player's stat line provided as a dict.

    Keys must match the canonical scoring key names (pass_yd, pass_td, etc.).
    Convenience wrapper for tests and interactive use.
    """
    if scoring_config is None:
        scoring_config = PPR_SCORING

    total = 0.0
    for key, weight in scoring_config.items():
        total += stats.get(key, 0.0) * weight
    return total
