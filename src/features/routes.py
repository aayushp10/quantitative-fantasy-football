"""
Routes run and route participation from nflverse participation data.

Targets per route run (TPRR) is materially STICKIER year-over-year than raw
target share: it separates "on the field and being targeted" from "on the
field blocking" and is a leading indicator of role expansion. Route counts
come from play-level participation data (offense players on the field during
pass plays).

Coverage caveat: participation releases cover ~2016-2023 (the NFL pulled the
feed for 2024). For uncovered seasons the assembler falls back to a snap-based
proxy: route_participation ≈ offense snap %, routes ≈ snap% × team dropbacks.
route_participation here is computed against COVERED pass plays only, so
partial-season coverage does not bias the rate downward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_route_features(
    pbp: pd.DataFrame,
    participation: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Compute per (player_id, team, season): routes, route_participation.

    Parameters
    ----------
    pbp : pd.DataFrame
        Cleaned PBP (pass/run) with game_id, play_id, posteam, season.
    participation : pd.DataFrame | None
        nflverse participation frame with nflverse_game_id (or game_id),
        play_id, offense_players (";"-separated GSIS ids). None → empty result.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, team, season, routes, route_participation.
        Empty when participation is unavailable.
    """
    empty = pd.DataFrame(columns=["player_id", "team", "season", "routes", "route_participation"])
    if participation is None or len(participation) == 0:
        return empty

    part = participation.copy()
    if "game_id" not in part.columns and "nflverse_game_id" in part.columns:
        part = part.rename(columns={"nflverse_game_id": "game_id"})
    needed = {"game_id", "play_id", "offense_players"}
    if not needed.issubset(part.columns):
        return empty

    pass_plays = pbp[pbp["pass"] == 1][["game_id", "play_id", "posteam", "season"]]

    merged = pass_plays.merge(
        part[["game_id", "play_id", "offense_players"]],
        on=["game_id", "play_id"],
        how="inner",
    )
    merged = merged[
        merged["offense_players"].notna() & (merged["offense_players"] != "")
    ]
    if merged.empty:
        return empty

    # Denominator: pass plays WITH participation coverage, per team-season
    team_covered = (
        merged.groupby(["posteam", "season"], observed=True)
        .agg(team_covered_pass_plays=("play_id", "count"))
        .reset_index()
    )

    exploded = (
        merged.assign(player_id=merged["offense_players"].str.split(";"))
        .explode("player_id")
    )
    exploded = exploded[exploded["player_id"].notna() & (exploded["player_id"] != "")]

    routes = (
        exploded.groupby(["player_id", "posteam", "season"], observed=True)
        .agg(routes=("play_id", "count"))
        .reset_index()
    )

    routes = routes.merge(team_covered, on=["posteam", "season"], how="left")
    routes["route_participation"] = np.where(
        routes["team_covered_pass_plays"] > 0,
        routes["routes"] / routes["team_covered_pass_plays"],
        np.nan,
    )

    routes = routes.rename(columns={"posteam": "team"})
    return routes[["player_id", "team", "season", "routes", "route_participation"]]
