"""
Survivor-complete training pairs: (season-N predictors) -> (season-N+1
outcome), where thresholds apply ONLY to season N.

This replaces the shift-within-filtered-matrix construction of
features.assembler.build_yoy_pairs for outcome-facing work: there, a
player who qualified in season N but fell below usage thresholds (or out
of the league) in N+1 vanished from training entirely — conditioning the
target on next-season survival and censoring exactly the outcomes
(injury busts, role collapses) a market model most needs to learn.

Here the outcome join is a left merge against the unfiltered outcomes
table. Predictors with no outcome row get a TRUE ZERO season: the player
did not log a regular-season week. Because predictors and outcomes share
GSIS ids, a missing id means "did not play," not a broken join — the
one exception is the final feature season (no outcome data exists yet),
which is dropped rather than zero-filled.
"""
from __future__ import annotations

import pandas as pd


def build_training_pairs(
    predictors: pd.DataFrame,
    outcomes: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Left-join next-season outcomes onto predictor rows.

    Adds: next_season_points, next_games_active, next_active_ppg,
    next_scheduled_ppg, outcome_status ("played"/"low_usage"/"no_games").
    Rows whose target season has no outcome data at all (the running
    season) are dropped, not zero-filled.
    """
    if not {"player_id", "season"}.issubset(predictors.columns):
        raise ValueError("predictors must have player_id and season columns")

    known_seasons = set(outcomes["season"].unique())

    pairs = predictors.copy()
    pairs["target_season"] = pairs["season"] + horizon
    pairs = pairs[pairs["target_season"].isin(known_seasons)]

    right = outcomes.rename(columns={
        "season": "target_season",
        "season_points": "next_season_points",
        "games_active": "next_games_active",
        "active_ppg": "next_active_ppg",
        "scheduled_ppg": "next_scheduled_ppg",
    })[["player_id", "target_season", "next_season_points", "next_games_active",
        "next_active_ppg", "next_scheduled_ppg", "outcome_status"]]

    pairs = pairs.merge(right, on=["player_id", "target_season"], how="left")

    missing = pairs["next_season_points"].isna()
    pairs.loc[missing, "next_season_points"] = 0.0
    pairs.loc[missing, "next_games_active"] = 0
    pairs.loc[missing, "next_scheduled_ppg"] = 0.0
    pairs.loc[missing, "outcome_status"] = "no_games"
    # next_active_ppg stays NaN for zero-game seasons: 0/0 is undefined,
    # and imputing 0 would conflate "didn't play" with "played terribly"
    return pairs.reset_index(drop=True)
