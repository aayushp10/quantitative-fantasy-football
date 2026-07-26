"""Survivor-complete outcomes and training pairs (synthetic data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.outcomes import build_outcomes, team_games  # noqa: E402
from data.training_pairs import build_training_pairs  # noqa: E402


def _weekly(rows):
    return pd.DataFrame(rows, columns=[
        "player_id", "player_name", "position", "season", "week",
        "season_type", "fantasy_points_ppr"])


def test_build_outcomes_aggregates_and_statuses():
    w = _weekly(
        [("a", "A", "WR", 2022, wk, "REG", 10.0) for wk in range(1, 15)]
        + [("b", "B", "RB", 2022, wk, "REG", 5.0) for wk in range(1, 4)]
        + [("a", "A", "WR", 2022, 19, "POST", 30.0)]  # playoffs excluded
    )
    out = build_outcomes(w).set_index("player_id")
    assert out.at["a", "season_points"] == 140.0          # POST week dropped
    assert out.at["a", "games_active"] == 14
    assert out.at["a", "outcome_status"] == "played"
    assert out.at["a", "scheduled_ppg"] == 140.0 / 17
    assert out.at["b", "outcome_status"] == "low_usage"   # 3 games
    assert out.at["b", "active_ppg"] == 5.0


def test_team_games_era():
    assert team_games(2020) == 16
    assert team_games(2021) == 17


def test_training_pairs_keep_vanished_players_as_true_zeros():
    predictors = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "season": [2022, 2022, 2022],
        "feature_x": [1.0, 2.0, 3.0],
    })
    # 'b' has no 2023 season at all; outcomes table knows 2023 exists
    w = _weekly(
        [("a", "A", "WR", 2023, wk, "REG", 12.0) for wk in range(1, 18)]
        + [("c", "C", "TE", 2023, 1, "REG", 2.0)]
    )
    outcomes = build_outcomes(w)
    pairs = build_training_pairs(predictors, outcomes).set_index("player_id")

    assert len(pairs) == 3                                 # nobody dropped
    assert pairs.at["b", "next_season_points"] == 0.0      # true zero
    assert pairs.at["b", "outcome_status"] == "no_games"
    assert np.isnan(pairs.at["b", "next_active_ppg"])      # 0/0 stays NaN
    assert pairs.at["a", "outcome_status"] == "played"
    assert pairs.at["c", "outcome_status"] == "low_usage"


def test_training_pairs_drop_unknown_target_season():
    """The running season (no outcome data yet) is dropped, not zero-filled."""
    predictors = pd.DataFrame({
        "player_id": ["a", "a"], "season": [2022, 2023], "feature_x": [1.0, 1.5]})
    w = _weekly([("a", "A", "WR", 2023, 1, "REG", 9.0)])   # only 2023 outcomes exist
    pairs = build_training_pairs(predictors, build_outcomes(w))
    assert list(pairs["season"]) == [2022]                 # 2023->2024 unknown, dropped


def test_training_pairs_no_gap_bridging():
    """A missed season is a zero outcome for that season — the following
    season's real stats must not be pulled backward."""
    predictors = pd.DataFrame({
        "player_id": ["a"], "season": [2021], "feature_x": [1.0]})
    w = _weekly(
        [("a", "A", "WR", 2023, wk, "REG", 20.0) for wk in range(1, 18)]   # missed 2022
        + [("z", "Z", "RB", 2022, 1, "REG", 1.0)]          # 2022 outcomes exist
    )
    pairs = build_training_pairs(predictors, build_outcomes(w))
    assert pairs.at[0, "outcome_status"] == "no_games"     # 2022 = zero, not 2023's 340
    assert pairs.at[0, "next_season_points"] == 0.0
