"""
Vacated target and carry share computation.

When a player leaves a team (free agency, trade, retirement, cut), their
usage share becomes available for remaining and incoming players. A WR2
entering a team whose WR1 departed gets a meaningful opportunity upgrade
that the model should recognize.

Vacated share = sum of (target_share × games_played/17) for players who
were on Team T in season N but are NOT on Team T in season N+1.

Weighted by availability because a player who played 8 games vacates half
as much as one who played a full 17 games.

Public API
----------
compute_vacated_shares(feature_matrix, rosters_df) -> pd.DataFrame
    One row per (team, season N) with vacated target/carry share totals.

assign_vacated_shares_to_players(vacated_df, rosters_df, feature_matrix) -> pd.DataFrame
    One row per (player_id, season N) with the vacated shares on their
    next season's team (destination for changers, current for stayers).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_GAMES_PER_SEASON: float = 17.0


def _detect_departures(
    feature_matrix: pd.DataFrame,
    rosters_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify players who were on a team in season N but NOT on that team in N+1.

    Returns one row per departure with columns:
        player_id, team, season (= N), target_share, rush_share, games_played.
    """
    required_fm = {"player_id", "team", "season"}
    if not required_fm.issubset(feature_matrix.columns):
        return pd.DataFrame(columns=["player_id", "team", "season"])

    # Determine next-season team from rosters
    roster_base = (
        rosters_df[["player_id", "team", "season"]]
        .drop_duplicates(subset=["player_id", "season"])
        .sort_values(["player_id", "season"])
        .copy()
    )
    roster_base["next_team"] = roster_base.groupby("player_id", observed=True)["team"].shift(-1)
    roster_base["next_season"] = roster_base.groupby("player_id", observed=True)["season"].shift(-1)

    # A player departed team T if in season N their next season is N+1 on a DIFFERENT team
    # or if they have no next season (retired/cut) — treated as departed from their last team
    departed = roster_base[
        (roster_base["next_season"].isna() | (roster_base["next_season"] == roster_base["season"] + 1))
        & (roster_base["next_team"].isna() | (roster_base["next_team"] != roster_base["team"]))
    ].copy()

    # Pull target_share, rush_share, games_played from feature_matrix
    share_cols = ["player_id", "season", "team"]
    for col in ["target_share", "rush_share", "games_played"]:
        if col in feature_matrix.columns:
            share_cols.append(col)

    departed = departed.merge(
        feature_matrix[share_cols],
        on=["player_id", "team", "season"],
        how="inner",  # only players with feature data
    )

    return departed


def compute_vacated_shares(
    feature_matrix: pd.DataFrame,
    rosters_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute vacated target and carry shares per (team, season N).

    A player's share is "vacated" from team T if they were on team T in
    season N but not on team T in season N+1 (departed via free agency,
    trade, retirement, or cut). Weighted by games played / 17 to account
    for partial-season availability.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Must have: player_id, team, season, and optionally target_share,
        rush_share, games_played.
    rosters_df : pd.DataFrame
        Must have: player_id, team, season.

    Returns
    -------
    pd.DataFrame
        Columns: team, season (N),
                 vacated_target_share, vacated_carry_share,
                 top_departed_target_share, n_departed_skill.
    """
    departed = _detect_departures(feature_matrix, rosters_df)
    if departed.empty:
        return pd.DataFrame(columns=["team", "season", "vacated_target_share",
                                     "vacated_carry_share", "top_departed_target_share",
                                     "n_departed_skill"])

    # Availability weight: games_played / 17
    games = departed["games_played"].fillna(_GAMES_PER_SEASON) if "games_played" in departed.columns \
        else pd.Series(_GAMES_PER_SEASON, index=departed.index)
    avail_weight = (games / _GAMES_PER_SEASON).clip(upper=1.0)

    if "target_share" in departed.columns:
        departed["_eff_target_share"] = departed["target_share"].fillna(0) * avail_weight
    else:
        departed["_eff_target_share"] = 0.0

    if "rush_share" in departed.columns:
        departed["_eff_carry_share"] = departed["rush_share"].fillna(0) * avail_weight
    else:
        departed["_eff_carry_share"] = 0.0

    agg = (
        departed.groupby(["team", "season"], observed=True)
        .agg(
            vacated_target_share=("_eff_target_share", "sum"),
            vacated_carry_share=("_eff_carry_share", "sum"),
            top_departed_target_share=("_eff_target_share", "max"),
            n_departed_skill=("player_id", "count"),
        )
        .reset_index()
    )

    return agg


def assign_vacated_shares_to_players(
    vacated_df: pd.DataFrame,
    rosters_df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each player in season N, look up the vacated shares on their
    season N+1 team — their destination team for movers, current team for stayers.

    This tells the model: "this player is entering a team that lost X% of
    its target share from departing players."

    Parameters
    ----------
    vacated_df : pd.DataFrame
        Output from compute_vacated_shares(): team, season, vacated_target_share, etc.
    rosters_df : pd.DataFrame
        Roster data: player_id, team, season.
    feature_matrix : pd.DataFrame
        Current feature matrix: player_id, team, season (to get current team).

    Returns
    -------
    pd.DataFrame
        Columns: player_id, season (N), team_vacated_target_share,
                 team_vacated_carry_share, top_departed_target_share.
    """
    if vacated_df.empty:
        return pd.DataFrame(columns=["player_id", "season", "team_vacated_target_share",
                                     "team_vacated_carry_share", "top_departed_target_share"])

    # Base: all player-seasons in the feature matrix
    base = (
        feature_matrix[["player_id", "team", "season"]]
        .drop_duplicates(subset=["player_id", "season"])
        .copy()
    )

    # Determine next-season team from rosters
    roster_base = (
        rosters_df[["player_id", "team", "season"]]
        .drop_duplicates(subset=["player_id", "season"])
        .sort_values(["player_id", "season"])
        .copy()
    )
    roster_base["next_team"] = roster_base.groupby("player_id", observed=True)["team"].shift(-1)
    roster_base["next_season"] = roster_base.groupby("player_id", observed=True)["season"].shift(-1)

    # Merge next_team onto base
    base = base.merge(
        roster_base[["player_id", "season", "next_team", "next_season"]],
        on=["player_id", "season"],
        how="left",
    )

    # Use next_team if player is on a consecutive season; else current team
    base["lookup_team"] = np.where(
        base["next_season"] == base["season"] + 1,
        base["next_team"].fillna(base["team"]),
        base["team"],
    )

    # Look up vacated shares for the lookup_team in season N (the departures happened between N and N+1)
    vacated_renamed = vacated_df.rename(columns={
        "team": "lookup_team",
        "vacated_target_share": "team_vacated_target_share",
        "vacated_carry_share": "team_vacated_carry_share",
    })

    result = base.merge(
        vacated_renamed[["lookup_team", "season", "team_vacated_target_share",
                          "team_vacated_carry_share", "top_departed_target_share"]],
        on=["lookup_team", "season"],
        how="left",
    )

    keep_cols = ["player_id", "season"] + [
        c for c in ["team_vacated_target_share", "team_vacated_carry_share", "top_departed_target_share"]
        if c in result.columns
    ]
    return result[keep_cols].reset_index(drop=True)
