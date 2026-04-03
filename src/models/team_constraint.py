"""
Top-down team passing offense projection with player-level distribution.

Architecture:
1. Project each team's total passing yards and passing TDs for the upcoming
   season using regression toward the league median (teams don't repeat extremes).
2. Adjust for known QB changes using qb_upgrade_delta from qb_coupling features.
3. Distribute the team total to WR/TE players using their target_share and
   air_yard_share as allocation keys.
4. Return per-player topdown_fpts_pg as an additional feature for the Ridge model.

The topdown_fpts_pg feature is added to POSITION_FEATURES for WR and TE only.
The Ridge model can then learn the optimal weight between bottom-up and top-down signals.

Public API
----------
build_topdown_features(feature_matrix, target_season) -> pd.DataFrame
    One row per (player_id, season) for WR/TE players with topdown_fpts_pg.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import PPR_SCORING

# Mean reversion weight toward league median (higher = more reversion)
_TEAM_REVERSION_WEIGHT: float = 0.40

# Games per season for per-game conversion
_GAMES: float = 17.0


def project_team_passing(
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Project each team's season passing yards and TDs using regression toward
    the league median.

    Adjusts for QB changes if qb_upgrade_delta is in the feature matrix.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Single-season feature matrix slice (all players from season N).
        Needs: team, target_share, targets_per_game or targets, games_played,
               pass_yards_per_attempt (or yards_per_target), qb_upgrade_delta (optional).

    Returns
    -------
    pd.DataFrame
        Columns: team, projected_team_targets_pg, projected_team_pass_yards_pg,
                 projected_team_pass_tds_pg.
    """
    required = {"team"}
    if not required.issubset(feature_matrix.columns):
        return pd.DataFrame(columns=["team"])

    # Estimate team-level targets per game from player target sums
    # This approximates total team pass attempts (each target = a completion or incompletion)
    fm = feature_matrix.copy()

    # Targets per game: prefer explicit column, else infer from share × team pace
    if "targets_per_game" in fm.columns:
        team_tgt_pg = (
            fm.groupby("team", observed=True)["targets_per_game"]
            .sum()
            .reset_index()
            .rename(columns={"targets_per_game": "team_targets_pg"})
        )
    elif "targets" in fm.columns and "games_played" in fm.columns:
        fm["_tgt_pg"] = fm["targets"] / fm["games_played"].clip(lower=1)
        team_tgt_pg = (
            fm.groupby("team", observed=True)["_tgt_pg"]
            .sum()
            .reset_index()
            .rename(columns={"_tgt_pg": "team_targets_pg"})
        )
    elif "team_pass_rate" in fm.columns and "team_pace" in fm.columns:
        team_info = fm.drop_duplicates("team")[["team", "team_pace", "team_pass_rate"]]
        team_tgt_pg = team_info.copy()
        team_tgt_pg["team_targets_pg"] = (
            team_tgt_pg["team_pace"] * team_tgt_pg["team_pass_rate"] * 0.6  # ~60% of pass plays are targets
        )
        team_tgt_pg = team_tgt_pg[["team", "team_targets_pg"]]
    else:
        return pd.DataFrame(columns=["team"])

    # Pass yards per target: use yards_per_target if available, else approximate
    if "yards_per_target" in fm.columns:
        team_yds_pt = (
            fm.groupby("team", observed=True)["yards_per_target"]
            .mean()
            .reset_index()
            .rename(columns={"yards_per_target": "team_yds_per_tgt"})
        )
    else:
        # Fallback: league average ~6.5 yards per target
        teams = team_tgt_pg[["team"]].copy()
        teams["team_yds_per_tgt"] = 6.5
        team_yds_pt = teams

    # TD rate: use rec_td_rate if available
    if "rec_td_rate" in fm.columns:
        team_td_rate = (
            fm.groupby("team", observed=True)["rec_td_rate"]
            .mean()
            .reset_index()
            .rename(columns={"rec_td_rate": "team_td_rate"})
        )
    else:
        teams = team_tgt_pg[["team"]].copy()
        teams["team_td_rate"] = 0.042  # league average ~42 pass TDs / ~1000 targets
        team_td_rate = teams

    # Combine
    team_proj = team_tgt_pg.merge(team_yds_pt, on="team", how="left")
    team_proj = team_proj.merge(team_td_rate, on="team", how="left")

    # Compute raw projections
    team_proj["raw_pass_yards_pg"] = (
        team_proj["team_targets_pg"] * team_proj["team_yds_per_tgt"].fillna(6.5)
    )
    team_proj["raw_pass_tds_pg"] = (
        team_proj["team_targets_pg"] * team_proj["team_td_rate"].fillna(0.042)
    )

    # Regression toward league median
    league_targets_median = team_proj["team_targets_pg"].median()
    league_yards_median = team_proj["raw_pass_yards_pg"].median()
    league_tds_median = team_proj["raw_pass_tds_pg"].median()

    w = _TEAM_REVERSION_WEIGHT
    team_proj["projected_team_targets_pg"] = (
        w * league_targets_median + (1 - w) * team_proj["team_targets_pg"]
    )
    team_proj["projected_team_pass_yards_pg"] = (
        w * league_yards_median + (1 - w) * team_proj["raw_pass_yards_pg"]
    )
    team_proj["projected_team_pass_tds_pg"] = (
        w * league_tds_median + (1 - w) * team_proj["raw_pass_tds_pg"]
    )

    # QB upgrade/downgrade adjustment
    if "qb_upgrade_delta" in fm.columns and "qb_changed" in fm.columns:
        team_qb = (
            fm.drop_duplicates("team")[["team", "qb_upgrade_delta", "qb_changed"]]
        )
        team_proj = team_proj.merge(team_qb, on="team", how="left")
        if "qb_changed" in team_proj.columns and "qb_upgrade_delta" in team_proj.columns:
            # Adjust passing yards by ~0.5 yards per play for each 0.1 EPA improvement
            epa_to_yards_factor = 5.0
            adj = np.where(
                team_proj["qb_changed"].fillna(0) == 1,
                team_proj["qb_upgrade_delta"].fillna(0) * epa_to_yards_factor,
                0.0,
            )
            team_proj["projected_team_pass_yards_pg"] = (
                team_proj["projected_team_pass_yards_pg"] + adj
            ).clip(lower=0)

    keep = ["team", "projected_team_targets_pg",
            "projected_team_pass_yards_pg", "projected_team_pass_tds_pg"]
    return team_proj[[c for c in keep if c in team_proj.columns]].reset_index(drop=True)


def compute_topdown_player_projections(
    team_projections: pd.DataFrame,
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distribute team-level passing projections to individual WR/TE players
    using their target_share and air_yard_share as allocation keys.

    For each player:
      topdown_targets_pg     = projected_team_targets_pg × player_target_share
      topdown_rec_yards_pg   = projected_team_pass_yards_pg × player_air_yard_share
      topdown_rec_tds_pg     = projected_team_pass_tds_pg × player_target_share
      topdown_fpts_pg        = receptions×1.0 + rec_yards×0.1 + rec_tds×6.0

    Parameters
    ----------
    team_projections : pd.DataFrame
        Output from project_team_passing().
    feature_matrix : pd.DataFrame
        Single-season feature matrix with player_id, team, target_share, air_yard_share,
        catch_rate, position.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, season, topdown_targets_pg, topdown_rec_yards_pg,
                 topdown_fpts_pg.
        Only WR and TE rows included; others excluded.
    """
    if team_projections.empty or "player_id" not in feature_matrix.columns:
        return pd.DataFrame(columns=["player_id", "season", "topdown_fpts_pg"])

    # Only WR/TE receive top-down features (pass catchers)
    fm = feature_matrix[
        feature_matrix.get("position", pd.Series("", index=feature_matrix.index)).isin(["WR", "TE"])
        if "position" in feature_matrix.columns
        else pd.Series(True, index=feature_matrix.index)
    ].copy()

    if fm.empty:
        return pd.DataFrame(columns=["player_id", "season", "topdown_fpts_pg"])

    fm = fm.merge(team_projections, on="team", how="left")

    s = PPR_SCORING

    # Per-player allocation
    tgt_share = fm["target_share"].fillna(0) if "target_share" in fm.columns else pd.Series(0, index=fm.index)
    ay_share = fm["air_yard_share"].fillna(tgt_share) if "air_yard_share" in fm.columns else tgt_share
    catch_rate = fm["catch_rate"].fillna(0.65) if "catch_rate" in fm.columns else pd.Series(0.65, index=fm.index)

    fm["topdown_targets_pg"] = fm["projected_team_targets_pg"].fillna(0) * tgt_share
    fm["topdown_rec_yards_pg"] = fm["projected_team_pass_yards_pg"].fillna(0) * ay_share
    fm["topdown_rec_tds_pg"] = fm["projected_team_pass_tds_pg"].fillna(0) * tgt_share

    fm["_topdown_receptions"] = fm["topdown_targets_pg"] * catch_rate
    fm["topdown_fpts_pg"] = (
        fm["_topdown_receptions"] * s["reception"]
        + fm["topdown_rec_yards_pg"] * s["rec_yd"]
        + fm["topdown_rec_tds_pg"] * s["rec_td"]
    ).clip(lower=0)

    keep = ["player_id", "season"] + [
        c for c in ["topdown_targets_pg", "topdown_rec_yards_pg", "topdown_fpts_pg"]
        if c in fm.columns
    ]
    return fm[keep].reset_index(drop=True)


def build_topdown_features(
    feature_matrix: pd.DataFrame,
    target_season: int,
) -> pd.DataFrame:
    """
    Public API: build all top-down features for a single season's feature matrix slice.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        All players from a single season.
    target_season : int
        The season being projected (not used directly but kept for API clarity).

    Returns
    -------
    pd.DataFrame
        Columns: player_id, season, topdown_targets_pg, topdown_rec_yards_pg,
                 topdown_fpts_pg. Only WR/TE rows; others excluded.
    """
    if feature_matrix.empty:
        return pd.DataFrame(columns=["player_id", "season", "topdown_fpts_pg"])

    team_proj = project_team_passing(feature_matrix)
    if team_proj.empty:
        return pd.DataFrame(columns=["player_id", "season", "topdown_fpts_pg"])

    return compute_topdown_player_projections(team_proj, feature_matrix)
