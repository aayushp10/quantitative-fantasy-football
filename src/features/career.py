"""
Multi-year (career-to-date) features.

The YoY pairs previously gave the model exactly one season of memory:
features from season N predict N+1, with nothing about the player's
career before N beyond age and years_in_league. These features add the
missing memory, computed strictly from seasons ≤ N (expanding windows,
so there is no lookahead into the target season):

    seasons_played_todate   number of qualifying seasons through N
    career_fpts_pg          games-weighted career points per game
    peak_fpts_pg            best season-level points per game to date
    peak_target_share       best target share to date
    peak_rush_share         best rush share to date
    durability_todate       career mean of games_played / season_length
    fpts_pg_prev            previous season's points per game
    target_share_prev       previous season's target share
    rush_share_prev         previous season's rush share
    fpts_pg_yoy_change      season N minus season N−1 points per game

"prev" columns require strict season continuity (a row for N−1); a
missed season leaves them NaN — the pipelines median-impute, which is
exactly the "average player" prior we want for a gap year.

add_career_features() is idempotent and cheap (pure pandas on the
assembled matrix), so the assembler applies it on every load — no
feature-cache invalidation needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import season_length

CAREER_FEATURES: list[str] = [
    "seasons_played_todate",
    "career_fpts_pg",
    "peak_fpts_pg",
    "peak_target_share",
    "peak_rush_share",
    "durability_todate",
    "fpts_pg_prev",
    "target_share_prev",
    "rush_share_prev",
    "fpts_pg_yoy_change",
]


def add_career_features(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return the matrix with career-to-date columns added (recomputed if
    already present). Requires player_id, season, fpts_per_game."""
    required = {"player_id", "season", "fpts_per_game"}
    if not required.issubset(feature_matrix.columns):
        return feature_matrix

    fm = feature_matrix.drop(
        columns=[c for c in CAREER_FEATURES if c in feature_matrix.columns]
    )
    # Duplicate (player, season) rows exist for mid-season team changes;
    # compute on a deduped view and merge back so every stint row gets the
    # same career values.
    usage = (
        fm.get("targets", pd.Series(0, index=fm.index)).fillna(0)
        + fm.get("carries", pd.Series(0, index=fm.index)).fillna(0)
        + fm.get("dropbacks", pd.Series(0, index=fm.index)).fillna(0)
    )
    base = (
        fm.assign(_usage=usage.values)
        .sort_values("_usage", ascending=False)
        .drop_duplicates(subset=["player_id", "season"], keep="first")
        .sort_values(["player_id", "season"])
        .reset_index(drop=True)
    )

    g = base.groupby("player_id", observed=True)

    out = pd.DataFrame({
        "player_id": base["player_id"],
        "season": base["season"],
    })
    out["seasons_played_todate"] = g.cumcount() + 1

    games = base.get("games_played", pd.Series(np.nan, index=base.index)).fillna(0)
    fpts_total = base["fpts_per_game"].fillna(0) * games
    out["career_fpts_pg"] = (
        fpts_total.groupby(base["player_id"], observed=True).cumsum()
        / games.groupby(base["player_id"], observed=True).cumsum().replace(0, np.nan)
    )
    out["peak_fpts_pg"] = g["fpts_per_game"].cummax()

    for col, name in [("target_share", "peak_target_share"),
                      ("rush_share", "peak_rush_share")]:
        if col in base.columns:
            out[name] = g[col].cummax()
        else:
            out[name] = np.nan

    season_len = base["season"].map(season_length)
    games_frac = (games / season_len).clip(0, 1)
    out["durability_todate"] = (
        games_frac.groupby(base["player_id"], observed=True).expanding().mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )

    # Strict previous-season lags: only valid when a row exists for N−1
    prev_season = g["season"].shift(1)
    contiguous = prev_season == base["season"] - 1
    for col, name in [("fpts_per_game", "fpts_pg_prev"),
                      ("target_share", "target_share_prev"),
                      ("rush_share", "rush_share_prev")]:
        if col in base.columns:
            lag = g[col].shift(1)
            out[name] = lag.where(contiguous)
        else:
            out[name] = np.nan
    out["fpts_pg_yoy_change"] = base["fpts_per_game"] - out["fpts_pg_prev"]

    return fm.merge(out, on=["player_id", "season"], how="left")
