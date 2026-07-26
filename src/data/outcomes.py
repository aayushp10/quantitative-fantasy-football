"""
Survivor-complete player outcomes (PR 1 of the model roadmap).

One row per (player_id, season) for EVERY player who logged a regular-season
week — no usage thresholds, no qualification filters. This is the outcome
side of training: predictors may be threshold-filtered (a model needs
signal to predict from), but outcomes must never be, or the training
target becomes "points, given that you stayed fantasy-relevant" and the
bust tail — the market's largest error class — is censored out.

Players absent from a season entirely (injured all year, cut, retired)
have no row here; the training-pairs layer records them as a true
zero-point outcome. That is correct for this id-space: predictors and
weekly data share GSIS ids, so a missing id means "did not play," not an
unmatched join.
"""
from __future__ import annotations

import pandas as pd

# 17 scheduled games from 2021 on, 16 before
def team_games(season: int) -> int:
    return 17 if season >= 2021 else 16

LOW_USAGE_GAMES = 4  # <= this many active games -> "low_usage" status


def build_outcomes(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly logs into unfiltered season outcomes.

    Returns one row per (player_id, season):
        season_points   total PPR points, regular season
        games_active    weeks with a stat line
        active_ppg      season_points / games_active
        scheduled_ppg   season_points / scheduled team games — the
                        availability-inclusive rate a drafter actually gets
        outcome_status  "played" or "low_usage" (<= LOW_USAGE_GAMES games)
    """
    w = weekly
    if "season_type" in w.columns:
        w = w[w["season_type"] == "REG"]
    fp_col = "fantasy_points_ppr" if "fantasy_points_ppr" in w.columns else "fantasy_points"
    w = w.dropna(subset=[fp_col])

    grp = w.groupby(["player_id", "season"], observed=True)
    out = grp.agg(
        season_points=(fp_col, "sum"),
        games_active=("week", "nunique"),
        player_name=("player_name", "first"),
        position=("position", "first"),
    ).reset_index()

    out["active_ppg"] = out["season_points"] / out["games_active"]
    out["scheduled_ppg"] = out["season_points"] / out["season"].map(team_games)
    out["outcome_status"] = (out["games_active"] <= LOW_USAGE_GAMES).map(
        {True: "low_usage", False: "played"}
    )
    return out
