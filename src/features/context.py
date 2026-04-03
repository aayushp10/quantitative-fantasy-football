"""
Situational / environmental context factors.

These frame the opportunity and efficiency factors:
- How much of a player's production came in garbage time (inflated)?
- What is the team's pace, pass rate, and overall offensive quality?
- What was the strength of schedule they faced?

Garbage time:  wp < 0.10 OR wp > 0.90 AND game_half == 'Half2'
Neutral script: 0.25 <= wp <= 0.75 AND qtr in [1, 2, 3]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    return np.where(denom > 0, num / denom, np.nan)


# ---------------------------------------------------------------------------
# Garbage time and neutral-script masking
# ---------------------------------------------------------------------------

def _garbage_time_mask(pbp: pd.DataFrame) -> pd.Series:
    """Boolean mask: True for plays that are in garbage time."""
    if "wp" not in pbp.columns or "game_half" not in pbp.columns:
        return pd.Series(False, index=pbp.index)
    return (
        ((pbp["wp"] < 0.10) | (pbp["wp"] > 0.90)) &
        (pbp["game_half"] == "Half2")
    )


def _neutral_script_mask(pbp: pd.DataFrame) -> pd.Series:
    """Boolean mask: True for game-script-neutral plays (Q1-Q3, wp 25-75%)."""
    if "wp" not in pbp.columns or "qtr" not in pbp.columns:
        return pd.Series(True, index=pbp.index)  # can't filter, keep all
    return (
        (pbp["wp"] >= 0.25) & (pbp["wp"] <= 0.75) &
        (pbp["qtr"].isin([1, 2, 3]))
    )


# ---------------------------------------------------------------------------
# Per-player context factors
# ---------------------------------------------------------------------------

def _player_game_script(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Garbage time share and neutral-script EPA per player.
    """
    # Fantasy-relevant players: identified receiver, rusher, or passer
    # We compute this at the play level then aggregate
    gt_mask = _garbage_time_mask(pbp)
    ns_mask = _neutral_script_mask(pbp)

    rows = []

    # Receivers
    recv = pbp[(pbp["pass"] == 1) & pbp["receiver_player_id"].notna()].copy()
    recv["player_id"] = recv["receiver_player_id"]
    recv["team"] = recv["posteam"]
    recv["in_garbage"] = gt_mask.reindex(recv.index, fill_value=False)
    recv["in_neutral"] = ns_mask.reindex(recv.index, fill_value=True)
    rows.append(recv[["player_id", "team", "season", "epa", "in_garbage", "in_neutral"]])

    # Rushers
    rush = pbp[(pbp["rush"] == 1) & pbp["rusher_player_id"].notna()].copy()
    rush["player_id"] = rush["rusher_player_id"]
    rush["team"] = rush["posteam"]
    rush["in_garbage"] = gt_mask.reindex(rush.index, fill_value=False)
    rush["in_neutral"] = ns_mask.reindex(rush.index, fill_value=True)
    rows.append(rush[["player_id", "team", "season", "epa", "in_garbage", "in_neutral"]])

    all_plays = pd.concat(rows, ignore_index=True)

    def _agg(g):
        n = len(g)
        n_garbage = g["in_garbage"].sum()
        neutral_epa = g.loc[g["in_neutral"], "epa"].mean() if g["in_neutral"].any() else np.nan
        return pd.Series({
            "garbage_time_share": n_garbage / n if n > 0 else np.nan,
            "neutral_script_epa": neutral_epa,
            "total_plays": n,
        })

    result = (
        all_plays.groupby(["player_id", "team", "season"], observed=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    return result


# ---------------------------------------------------------------------------
# Team-level context factors
# ---------------------------------------------------------------------------

def _team_context(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Team pace, pass rate, and overall offensive EPA — joined to players later.
    """
    team_season = (
        pbp.groupby(["posteam", "season"], observed=True)
        .agg(
            team_plays=("play_id", "count"),
            team_pass_plays=("pass", "sum"),
            team_offensive_epa=("epa", "mean"),
            team_games=("week", "nunique"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    team_season["team_pace"] = _safe_divide(team_season["team_plays"], team_season["team_games"])
    team_season["team_pass_rate"] = _safe_divide(
        team_season["team_pass_plays"], team_season["team_plays"]
    )
    return team_season[["team", "season", "team_pace", "team_pass_rate", "team_offensive_epa"]]


def _strength_of_schedule(pbp: pd.DataFrame, schedules: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute average opponent defensive EPA faced per player's team.

    Uses defensive EPA allowed per team from PBP. If schedules DataFrame
    is provided, computes forward-looking SOS for projection; otherwise
    uses historical SOS from actual games played.
    """
    # Defensive EPA: from the opponent's perspective, defteam stops offteam
    # We measure defensive quality as mean EPA allowed per game
    if "defteam" not in pbp.columns:
        return pd.DataFrame(columns=["team", "season", "sos_opp_def_epa"])

    def_quality = (
        pbp.groupby(["defteam", "season"], observed=True)
        .agg(def_epa_allowed=("epa", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    # Opponent faced by posteam in each game
    game_matchups = (
        pbp[["game_id", "posteam", "defteam", "season"]]
        .drop_duplicates()
    )
    sos = (
        game_matchups.merge(def_quality, left_on=["defteam", "season"],
                            right_on=["team", "season"])
        .groupby(["posteam", "season"], observed=True)
        .agg(sos_opp_def_epa=("def_epa_allowed", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    return sos


# ---------------------------------------------------------------------------
# Games played and availability
# ---------------------------------------------------------------------------

def _availability(weekly: pd.DataFrame) -> pd.DataFrame:
    """Compute games played per player-season from weekly data."""
    return (
        weekly.groupby(["player_id", "season"], observed=True)
        .agg(games_played=("week", "nunique"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_context_factors(
    pbp: pd.DataFrame,
    weekly: pd.DataFrame,
    snap_df: pd.DataFrame | None = None,
    schedules: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Assemble all situational/environmental context factors.

    Parameters
    ----------
    pbp : pd.DataFrame
        Cleaned PBP (play_type filtered to pass/run).
    weekly : pd.DataFrame
        Weekly player stats (for games_played).
    snap_df : pd.DataFrame, optional
        Snap count data (for snap_percentage).
    schedules : pd.DataFrame, optional
        Schedule data (for forward-looking SOS).

    Returns
    -------
    pd.DataFrame
        One row per (player_id, team, season) with context factor columns.
    """
    game_script = _player_game_script(pbp)
    team_ctx = _team_context(pbp)
    sos = _strength_of_schedule(pbp, schedules)
    avail = _availability(weekly)

    # Join team context to players
    result = game_script.merge(team_ctx, on=["team", "season"], how="left")
    result = result.merge(sos, on=["team", "season"], how="left")

    # Games played from weekly data — weekly uses player_id directly
    if "player_id" in weekly.columns:
        result = result.merge(avail, on=["player_id", "season"], how="left")
    else:
        result["games_played"] = np.nan

    # Snap percentage
    # loader.load_snap_counts() bridges pfr_player_id → player_id (gsis_id) via
    # roster data, so snap_df should have a player_id column when available.
    if snap_df is not None and not snap_df.empty:
        snap_pct_col = next(
            (c for c in ["offense_pct", "off_pct"] if c in snap_df.columns), None
        )
        if snap_pct_col and "player_id" in snap_df.columns and "season" in snap_df.columns:
            snap_season = (
                snap_df.groupby(["player_id", "season"], observed=True)
                [snap_pct_col].mean()
                .reset_index()
                .rename(columns={snap_pct_col: "snap_percentage"})
            )
            result = result.merge(snap_season, on=["player_id", "season"], how="left")
        else:
            result["snap_percentage"] = np.nan
    else:
        result["snap_percentage"] = np.nan

    return result
