"""
Value Over Replacement Player (VORP) calculation.

VORP = projected_fpts_season - replacement_level_fpts

Replacement level = the Nth-best player at each position, where N is the
number of that position rostered across all teams.

Default (12-team, 1QB/2RB/2WR/1TE/1FLEX):
  Replacement QB  = QB #13   (12 teams × 1 starter)
  Replacement RB  = RB #25   (12 teams × 2 starters, before FLEX)
  Replacement WR  = WR #25   (12 teams × 2 starters, before FLEX)
  Replacement TE  = TE #13   (12 teams × 1 starter)
  FLEX replacement = best of RB #26 / WR #26 / TE #14

VORP is what determines actual draft value, not raw projected points.
A TE who scores 200 pts may have more VORP than a QB who scores 300 pts
if the positional scarcity at TE makes every point at the position rare.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ROSTER_SPOTS


# ---------------------------------------------------------------------------
# Replacement rank computation
# ---------------------------------------------------------------------------

def get_replacement_levels(
    projections_df: pd.DataFrame,
    league_size: int = 12,
    roster_config: dict | None = None,
) -> dict[str, float]:
    """
    Determine the replacement-level fantasy points for each position.

    Parameters
    ----------
    projections_df : pd.DataFrame
        Must have 'position' and 'projected_fpts_season' columns.
    league_size : int
        Number of teams in the league. Used to look up default roster config.
    roster_config : dict, optional
        Custom roster configuration dict with keys 'QB', 'RB', 'WR', 'TE',
        'FLEX', 'SUPERFLEX' and optionally 'league_size'. If None, looks
        up from ROSTER_SPOTS in config using the closest match to league_size.

    Returns
    -------
    dict
        {position: replacement_level_fpts}
    """
    if roster_config is None:
        # Find the closest default config
        key_map = {
            10: "10team",
            12: "12team",
            14: "14team",
        }
        config_key = key_map.get(league_size, "12team")
        roster_config = ROSTER_SPOTS[config_key]

    n_teams = roster_config.get("league_size", league_size)
    is_superflex = roster_config.get("SUPERFLEX", 0) > 0
    flex_slots = roster_config.get("FLEX", 1)

    # Starter counts per position
    starters = {
        "QB": roster_config.get("QB", 1) * n_teams,
        "RB": roster_config.get("RB", 2) * n_teams,
        "WR": roster_config.get("WR", 2) * n_teams,
        "TE": roster_config.get("TE", 1) * n_teams,
    }

    # SUPERFLEX adds another QB slot
    if is_superflex:
        starters["QB"] += n_teams

    replacement_levels = {}

    for pos, n_starters in starters.items():
        pos_proj = (
            projections_df[projections_df["position"] == pos]
            ["projected_fpts_season"]
            .dropna()
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )
        if len(pos_proj) > n_starters:
            replacement_levels[pos] = float(pos_proj.iloc[n_starters])
        elif len(pos_proj) > 0:
            replacement_levels[pos] = float(pos_proj.iloc[-1])
        else:
            replacement_levels[pos] = 0.0

    # FLEX replacement: best of RB/(n_starters_rb)+1, WR/(n_starters_wr)+1, TE/(n_starters_te)+1
    flex_candidates = []
    for pos in ["RB", "WR", "TE"]:
        n = starters[pos]
        pos_proj = (
            projections_df[projections_df["position"] == pos]
            ["projected_fpts_season"]
            .dropna()
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )
        # First player past the starter threshold is the first FLEX candidate
        if len(pos_proj) > n:
            flex_candidates.append(float(pos_proj.iloc[n]))

    if flex_candidates:
        flex_replacement = max(flex_candidates)
        # Downward adjust replacement for positions whose extras are below FLEX level
        for pos in ["RB", "WR", "TE"]:
            if replacement_levels[pos] < flex_replacement:
                replacement_levels[pos] = flex_replacement

    return replacement_levels


# ---------------------------------------------------------------------------
# VOR calculation
# ---------------------------------------------------------------------------

def calculate_vor(
    projections_df: pd.DataFrame,
    league_size: int = 12,
    roster_config: dict | None = None,
) -> pd.DataFrame:
    """
    Add VORP column to a projections DataFrame.

    Parameters
    ----------
    projections_df : pd.DataFrame
        Must have 'position' and 'projected_fpts_season'.
    league_size : int
        Number of teams. Used to look up default roster config.
    roster_config : dict, optional
        Custom roster config (see get_replacement_levels).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
        'replacement_level', 'vorp'.
    """
    required = {"position", "projected_fpts_season"}
    missing = required - set(projections_df.columns)
    if missing:
        raise ValueError(f"projections_df missing columns: {missing}")

    replacement_levels = get_replacement_levels(projections_df, league_size, roster_config)

    proj = projections_df.copy()
    proj["replacement_level"] = proj["position"].map(replacement_levels)
    proj["vorp"] = proj["projected_fpts_season"] - proj["replacement_level"]

    return proj


# ---------------------------------------------------------------------------
# Multi-format VOR
# ---------------------------------------------------------------------------

def calculate_vor_all_formats(
    projections_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Compute VORP for standard league configurations.

    Returns
    -------
    dict
        {config_name: projections_with_vorp_df}
    """
    configs = {
        "10team": (10, ROSTER_SPOTS["10team"]),
        "12team": (12, ROSTER_SPOTS["12team"]),
        "14team": (14, ROSTER_SPOTS["14team"]),
        "12team_superflex": (12, ROSTER_SPOTS["12team_superflex"]),
    }

    results = {}
    for name, (size, config) in configs.items():
        results[name] = calculate_vor(projections_df, league_size=size, roster_config=config)

    return results
