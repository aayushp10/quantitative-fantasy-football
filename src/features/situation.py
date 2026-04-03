"""
Situation change detection and encoding.

Detects team changes between consecutive seasons. For players who changed
teams, the model should see the DESTINATION team's context factors (pace,
pass rate, offensive EPA) rather than the old team's — because we're
predicting next-season output on the new team.

Public API
----------
build_situation_features(rosters_df, team_context_df) -> pd.DataFrame
    One row per (player_id, season) for ALL players with:
      - team_changed (0/1 binary flag)
      - new_team_pace, new_team_pass_rate, new_team_offensive_epa (NaN if unchanged)
      - context_delta_pace, context_delta_pass_rate, context_delta_epa (NaN if unchanged)
"""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Step 1: Detect team changes
# ---------------------------------------------------------------------------

def detect_team_changes(rosters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify players whose team changed between season N and season N+1.

    Parameters
    ----------
    rosters_df : pd.DataFrame
        Cleaned roster data with columns: player_id, team, season.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, season (= N, the earlier season), old_team, new_team.
        Only includes rows where a change occurred AND consecutive seasons exist.
    """
    required = {"player_id", "team", "season"}
    missing = required - set(rosters_df.columns)
    if missing:
        return pd.DataFrame(columns=["player_id", "season", "old_team", "new_team"])

    df = (
        rosters_df[["player_id", "team", "season"]]
        .drop_duplicates(subset=["player_id", "season"])
        .sort_values(["player_id", "season"])
        .copy()
    )

    # Shift to get next season's team for each player
    df["next_team"] = df.groupby("player_id", observed=True)["team"].shift(-1)
    df["next_season"] = df.groupby("player_id", observed=True)["season"].shift(-1)

    # Keep only consecutive-season pairs where the team changed
    changes = df[
        (df["next_season"] == df["season"] + 1)
        & (df["next_team"].notna())
        & (df["next_team"] != df["team"])
    ].copy()

    changes = changes.rename(columns={"team": "old_team", "next_team": "new_team"})
    return changes[["player_id", "season", "old_team", "new_team"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Get new team context
# ---------------------------------------------------------------------------

def build_new_team_context(
    team_context_df: pd.DataFrame,
    team_changes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each team-change row, look up the DESTINATION team's context factors.

    Parameters
    ----------
    team_context_df : pd.DataFrame
        Team-level context: team, season, team_pace, team_pass_rate, team_offensive_epa.
    team_changes_df : pd.DataFrame
        Output from detect_team_changes(): player_id, season, old_team, new_team.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, season, new_team_pace, new_team_pass_rate, new_team_offensive_epa,
                 old_team_pace, old_team_pass_rate, old_team_offensive_epa.
    """
    ctx_cols = ["team", "season", "team_pace", "team_pass_rate", "team_offensive_epa"]
    available_ctx = [c for c in ctx_cols if c in team_context_df.columns]
    if len(available_ctx) < 2:
        # Return empty if context data is missing
        return pd.DataFrame(columns=["player_id", "season"])

    ctx = team_context_df[available_ctx].copy()

    # New team context: join on new_team + season
    new_ctx = team_changes_df.merge(
        ctx.rename(columns={
            "team": "new_team",
            "team_pace": "new_team_pace",
            "team_pass_rate": "new_team_pass_rate",
            "team_offensive_epa": "new_team_offensive_epa",
        }),
        on=["new_team", "season"],
        how="left",
    )

    # Old team context: join on old_team + season (for delta computation)
    old_ctx_cols = {
        "team": "old_team",
        "team_pace": "old_team_pace",
        "team_pass_rate": "old_team_pass_rate",
        "team_offensive_epa": "old_team_offensive_epa",
    }
    old_rename = {k: v for k, v in old_ctx_cols.items() if k in available_ctx}
    new_ctx = new_ctx.merge(
        ctx.rename(columns=old_rename),
        on=["old_team", "season"],
        how="left",
    )

    keep_cols = ["player_id", "season"]
    for col in ["new_team_pace", "new_team_pass_rate", "new_team_offensive_epa",
                "old_team_pace", "old_team_pass_rate", "old_team_offensive_epa"]:
        if col in new_ctx.columns:
            keep_cols.append(col)

    return new_ctx[keep_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_situation_features(
    rosters_df: pd.DataFrame,
    team_context_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build situation-change features for the feature matrix.

    For team-change players, the context delta columns capture the magnitude
    of the environment upgrade/downgrade. The assembler then overwrites the
    old team's context columns with the new team's values so the Ridge model
    sees the destination environment.

    Parameters
    ----------
    rosters_df : pd.DataFrame
        Cleaned roster data: player_id, team, season.
    team_context_df : pd.DataFrame
        Team-level context: team, season, team_pace, team_pass_rate, team_offensive_epa.

    Returns
    -------
    pd.DataFrame
        One row per (player_id, season) for ALL unique player-seasons in rosters_df.
        Columns:
            player_id, season,
            team_changed (int: 0 or 1),
            new_team_pace, new_team_pass_rate, new_team_offensive_epa,
            context_delta_pace, context_delta_pass_rate, context_delta_epa.
        All new_team_* and context_delta_* columns are NaN for unchanged players.
    """
    # All unique player-seasons as the base
    all_player_seasons = (
        rosters_df[["player_id", "season"]]
        .drop_duplicates()
        .copy()
    )

    # Detect team changes
    changes = detect_team_changes(rosters_df)

    if changes.empty:
        all_player_seasons["team_changed"] = 0
        for col in ["new_team_pace", "new_team_pass_rate", "new_team_offensive_epa",
                    "context_delta_pace", "context_delta_pass_rate", "context_delta_epa"]:
            all_player_seasons[col] = float("nan")
        return all_player_seasons

    # Get new team + old team context for changers
    ctx_for_changers = build_new_team_context(team_context_df, changes)

    # Compute deltas where both sides are available
    if "new_team_pace" in ctx_for_changers.columns and "old_team_pace" in ctx_for_changers.columns:
        ctx_for_changers["context_delta_pace"] = (
            ctx_for_changers["new_team_pace"] - ctx_for_changers["old_team_pace"]
        )
    if "new_team_pass_rate" in ctx_for_changers.columns and "old_team_pass_rate" in ctx_for_changers.columns:
        ctx_for_changers["context_delta_pass_rate"] = (
            ctx_for_changers["new_team_pass_rate"] - ctx_for_changers["old_team_pass_rate"]
        )
    if "new_team_offensive_epa" in ctx_for_changers.columns and "old_team_offensive_epa" in ctx_for_changers.columns:
        ctx_for_changers["context_delta_epa"] = (
            ctx_for_changers["new_team_offensive_epa"] - ctx_for_changers["old_team_offensive_epa"]
        )

    ctx_for_changers["team_changed"] = 1

    # Drop old team columns before merging into full roster
    drop_old = [c for c in ctx_for_changers.columns if c.startswith("old_team_")]
    ctx_for_changers = ctx_for_changers.drop(columns=drop_old, errors="ignore")

    # Merge changers onto all player-seasons (left join → NaN for non-changers)
    result = all_player_seasons.merge(ctx_for_changers, on=["player_id", "season"], how="left")
    result["team_changed"] = result["team_changed"].fillna(0).astype(int)

    # Ensure all expected columns exist
    for col in ["new_team_pace", "new_team_pass_rate", "new_team_offensive_epa",
                "context_delta_pace", "context_delta_pass_rate", "context_delta_epa"]:
        if col not in result.columns:
            result[col] = float("nan")

    return result.reset_index(drop=True)
