"""
Volume / opportunity factor construction from play-by-play data.

These are the "beta" factors — a player's share of their team's production.
All factors are computed at the (player_id, season) level.

Key implementation notes:
- Share metrics: compute player totals, then team totals, then join back.
  Never divide before joining — this avoids unit-test gotchas.
- Filter receiver_player_id.notna() before counting targets (null on throwaways).
- Red zone definitions are baked into helper functions for reuse.
- The input PBP must already be filtered to play_type in ['pass', 'run'].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_JOIN_KEYS = ["player_id", "team", "season"]


def _safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Vectorized division returning NaN where denominator is 0."""
    return np.where(denom > 0, num / denom, np.nan)


def _pass_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    return pbp[pbp["pass"] == 1]


def _rush_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    return pbp[pbp["rush"] == 1]


def _targeted_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Pass plays with an identifiable receiver (excludes throwaways/spikes)."""
    return _pass_plays(pbp)[pbp["receiver_player_id"].notna()]


def _rz_pass(pbp: pd.DataFrame, yardline: int = 20) -> pd.DataFrame:
    return _targeted_plays(pbp)[pbp["yardline_100"] <= yardline]


def _rz_rush(pbp: pd.DataFrame, yardline: int = 20) -> pd.DataFrame:
    return _rush_plays(pbp)[pbp["yardline_100"] <= yardline]


# ---------------------------------------------------------------------------
# Receiving opportunity factors
# ---------------------------------------------------------------------------

def _receiving_opportunity(pbp: pd.DataFrame) -> pd.DataFrame:
    tgt = _targeted_plays(pbp)

    # Player-level targets and air yards
    player = (
        tgt.assign(player_id=tgt["receiver_player_id"], team=tgt["posteam"])
        .groupby(["player_id", "team", "season"], observed=True)
        .agg(
            targets=("play_id", "count"),
            player_air_yards=("air_yards", "sum"),
            deep_targets=("air_yards", lambda x: (x >= 20).sum()),
            rz_targets=("yardline_100", lambda x: (x <= 20).sum()),
            ez_targets=("yardline_100", lambda x: (x <= 10).sum()),
            games=("week", "nunique"),
        )
        .reset_index()
    )

    # Team-level targets and air yards
    team = (
        tgt.assign(team=tgt["posteam"])
        .groupby(["team", "season"], observed=True)
        .agg(
            team_targets=("play_id", "count"),
            team_air_yards=("air_yards", "sum"),
        )
        .reset_index()
    )

    df = player.merge(team, on=["team", "season"])

    df["target_share"] = _safe_divide(df["targets"], df["team_targets"])
    df["air_yard_share"] = _safe_divide(df["player_air_yards"], df["team_air_yards"])
    df["wopr"] = 1.5 * df["target_share"].fillna(0) + 0.7 * df["air_yard_share"].fillna(0)
    df["deep_target_share"] = _safe_divide(df["deep_targets"], df["targets"])
    df["targets_per_game"] = _safe_divide(df["targets"], df["games"])

    # Red zone / end zone target share (relative to team red zone targets)
    rz_team = (
        _rz_pass(pbp)
        .assign(team=lambda x: x["posteam"])
        .groupby(["team", "season"], observed=True)
        .agg(team_rz_targets=("play_id", "count"))
        .reset_index()
    )
    df = df.merge(rz_team, on=["team", "season"], how="left")
    df["rz_target_share"] = _safe_divide(df["rz_targets"], df["team_rz_targets"])

    ez_team = (
        _rz_pass(pbp, yardline=10)
        .assign(team=lambda x: x["posteam"])
        .groupby(["team", "season"], observed=True)
        .agg(team_ez_targets=("play_id", "count"))
        .reset_index()
    )
    df = df.merge(ez_team, on=["team", "season"], how="left")
    df["end_zone_target_share"] = _safe_divide(df["ez_targets"], df["team_ez_targets"])

    keep = [
        "player_id", "team", "season",
        "targets", "targets_per_game",
        "target_share", "air_yard_share", "wopr",
        "deep_target_share", "rz_target_share", "end_zone_target_share",
    ]
    return df[keep]


# ---------------------------------------------------------------------------
# Rushing opportunity factors
# ---------------------------------------------------------------------------

def _rushing_opportunity(pbp: pd.DataFrame) -> pd.DataFrame:
    rush = _rush_plays(pbp)

    # Exclude QB scrambles for designed rush share calculation
    designed = rush[rush.get("qb_scramble", pd.Series(0, index=rush.index)) != 1] if "qb_scramble" in rush.columns else rush

    player = (
        rush.assign(player_id=rush["rusher_player_id"], team=rush["posteam"])
        .dropna(subset=["player_id"])
        .groupby(["player_id", "team", "season"], observed=True)
        .agg(
            carries=("play_id", "count"),
            rz_carries=("yardline_100", lambda x: (x <= 20).sum()),
            goal_line_carries=("yardline_100", lambda x: (x <= 5).sum()),
            games=("week", "nunique"),
        )
        .reset_index()
    )

    # Designed rush share (no scrambles)
    designed_player = (
        designed.assign(player_id=designed["rusher_player_id"], team=designed["posteam"])
        .dropna(subset=["player_id"])
        .groupby(["player_id", "team", "season"], observed=True)
        .agg(designed_carries=("play_id", "count"))
        .reset_index()
    )
    player = player.merge(designed_player, on=["player_id", "team", "season"], how="left")

    # Two-minute carry share (end of halves)
    two_min_mask = (
        (rush["qtr"] == 2) & (rush.get("game_seconds_remaining", pd.Series(np.nan, index=rush.index)) <= 120) |
        (rush["qtr"] == 4) & (rush.get("game_seconds_remaining", pd.Series(np.nan, index=rush.index)) <= 120)
    ) if "game_seconds_remaining" in rush.columns else pd.Series(False, index=rush.index)

    two_min_player = (
        rush[two_min_mask]
        .assign(player_id=rush["rusher_player_id"], team=rush["posteam"])
        .dropna(subset=["player_id"])
        .groupby(["player_id", "team", "season"], observed=True)
        .agg(two_min_carries=("play_id", "count"))
        .reset_index()
    )
    player = player.merge(two_min_player, on=["player_id", "team", "season"], how="left")
    player["two_min_carries"] = player["two_min_carries"].fillna(0)

    # Team totals
    team = (
        rush.assign(team=rush["posteam"])
        .groupby(["team", "season"], observed=True)
        .agg(
            team_carries=("play_id", "count"),
            team_rz_carries=("yardline_100", lambda x: (x <= 20).sum()),
        )
        .reset_index()
    )

    designed_team = (
        designed.assign(team=designed["posteam"])
        .groupby(["team", "season"], observed=True)
        .agg(team_designed_carries=("play_id", "count"))
        .reset_index()
    )

    two_min_team = (
        rush[two_min_mask].assign(team=rush["posteam"])
        .groupby(["team", "season"], observed=True)
        .agg(team_two_min_carries=("play_id", "count"))
        .reset_index()
    ) if two_min_mask.any() else pd.DataFrame(columns=["team", "season", "team_two_min_carries"])

    df = player.merge(team, on=["team", "season"])
    df = df.merge(designed_team, on=["team", "season"], how="left")
    df = df.merge(two_min_team, on=["team", "season"], how="left")

    df["rush_share"] = _safe_divide(df["carries"], df["team_carries"])
    df["rz_rush_share"] = _safe_divide(df["rz_carries"], df["team_rz_carries"])
    df["goal_line_carry_share"] = _safe_divide(
        df["goal_line_carries"],
        df["carries"].where(df["carries"] > 0)
    )
    df["designed_rush_share"] = _safe_divide(df["designed_carries"], df["team_designed_carries"])
    df["carries_per_game"] = _safe_divide(df["carries"], df["games"])
    df["two_minute_carry_share"] = _safe_divide(
        df["two_min_carries"], df.get("team_two_min_carries", pd.Series(np.nan))
    )

    keep = [
        "player_id", "team", "season",
        "carries", "carries_per_game",
        "rush_share", "rz_rush_share", "goal_line_carry_share",
        "designed_rush_share", "two_minute_carry_share",
    ]
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------------
# QB opportunity factors
# ---------------------------------------------------------------------------

def _qb_opportunity(pbp: pd.DataFrame) -> pd.DataFrame:
    pass_plays = _pass_plays(pbp)

    # Dropbacks: pass attempts + sacks + scrambles
    dropback_mask = (
        (pass_plays.get("pass_attempt", pd.Series(0, index=pass_plays.index)) == 1) |
        (pass_plays.get("sack", pd.Series(0, index=pass_plays.index)) == 1) |
        (pass_plays.get("qb_scramble", pd.Series(0, index=pass_plays.index)) == 1)
    )
    dropbacks = pass_plays[dropback_mask]

    player = (
        dropbacks.assign(player_id=dropbacks["passer_player_id"], team=dropbacks["posteam"])
        .dropna(subset=["player_id"])
        .groupby(["player_id", "team", "season"], observed=True)
        .agg(
            dropbacks=("play_id", "count"),
            games=("week", "nunique"),
        )
        .reset_index()
    )

    # Team pass rate
    all_plays = pbp.assign(team=pbp["posteam"])
    team_plays = (
        all_plays.groupby(["team", "season"], observed=True)
        .agg(team_plays=("play_id", "count"))
        .reset_index()
    )
    team_pass = (
        pass_plays.assign(team=pass_plays["posteam"])
        .groupby(["team", "season"], observed=True)
        .agg(team_dropbacks=("play_id", "count"))
        .reset_index()
    )
    team_summary = team_plays.merge(team_pass, on=["team", "season"])
    team_summary["team_pass_rate"] = _safe_divide(
        team_summary["team_dropbacks"], team_summary["team_plays"]
    )

    # Play action rate
    if "play_action" in pbp.columns:
        pa_player = (
            pass_plays[pass_plays["play_action"] == 1]
            .assign(player_id=lambda x: x["passer_player_id"], team=lambda x: x["posteam"])
            .dropna(subset=["player_id"])
            .groupby(["player_id", "team", "season"], observed=True)
            .agg(pa_dropbacks=("play_id", "count"))
            .reset_index()
        )
        player = player.merge(pa_player, on=["player_id", "team", "season"], how="left")
        player["pa_dropbacks"] = player["pa_dropbacks"].fillna(0)
        player["play_action_rate"] = _safe_divide(player["pa_dropbacks"], player["dropbacks"])
    else:
        player["play_action_rate"] = np.nan

    # QB rush attempts (designed runs, from rush plays)
    rush = _rush_plays(pbp)
    qb_rush = (
        rush.assign(player_id=rush["passer_player_id"], team=rush["posteam"])
        .dropna(subset=["player_id"])
        .groupby(["player_id", "team", "season"], observed=True)
        .agg(qb_rushes=("play_id", "count"))
        .reset_index()
    )
    player = player.merge(qb_rush, on=["player_id", "team", "season"], how="left")
    player["qb_rushes"] = player["qb_rushes"].fillna(0)

    df = player.merge(team_summary[["team", "season", "team_pass_rate", "team_dropbacks"]], on=["team", "season"])

    df["dropbacks_per_game"] = _safe_divide(df["dropbacks"], df["games"])
    df["rush_attempt_share"] = _safe_divide(df["qb_rushes"], df["dropbacks"] + df["qb_rushes"])

    keep = [
        "player_id", "team", "season",
        "dropbacks", "dropbacks_per_game",
        "team_pass_rate", "play_action_rate", "rush_attempt_share",
    ]
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_opportunity_factors(
    pbp: pd.DataFrame,
    position: str = "ALL",
) -> pd.DataFrame:
    """
    Compute all opportunity/volume factors from play-by-play data.

    Parameters
    ----------
    pbp : pd.DataFrame
        Cleaned PBP DataFrame (play_type already filtered to pass/run).
        Must include columns: play_type, pass, rush, receiver_player_id,
        rusher_player_id, passer_player_id, posteam, season, week,
        yardline_100, air_yards.
    position : str
        'QB', 'RB', 'WR', 'TE', or 'ALL'. When 'ALL', all factor sets
        are computed and outer-merged.

    Returns
    -------
    pd.DataFrame
        One row per (player_id, team, season) with opportunity factor columns.
    """
    receiving = _receiving_opportunity(pbp)
    rushing = _rushing_opportunity(pbp)
    qb = _qb_opportunity(pbp)

    if position == "QB":
        return qb
    if position == "RB":
        return rushing.merge(
            receiving[["player_id", "team", "season",
                        "targets", "targets_per_game",
                        "target_share", "rz_target_share"]],
            on=["player_id", "team", "season"], how="left"
        )
    if position in ("WR", "TE"):
        return receiving

    # ALL: outer merge all three
    result = receiving.merge(
        rushing, on=["player_id", "team", "season"], how="outer"
    ).merge(
        qb, on=["player_id", "team", "season"], how="outer"
    )
    return result
