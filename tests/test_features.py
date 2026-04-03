"""
Tests for feature engineering modules.

Uses synthetic PBP DataFrames with manually computable expected values.
This validates the aggregation logic before running on real data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.opportunity import build_opportunity_factors
from features.efficiency import build_efficiency_factors


# ---------------------------------------------------------------------------
# Fixtures: synthetic PBP DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_pass_pbp():
    """
    10 pass plays in one game/season:
    - 4 targeted to player_A (3 complete, 1 incomplete)
    - 6 targeted to player_B (5 complete, 1 incomplete)
    No rush plays.
    Team = KC, Season = 2024, Week = 1.

    Expected:
    - player_A target_share = 4/10 = 0.40
    - player_B target_share = 6/10 = 0.60
    """
    plays = []
    # player_A: 4 targets, 3 completions, air_yards = [5, 10, 15, 25]
    for i, (air, comp, yards) in enumerate([(5, 1, 8), (10, 1, 15), (15, 1, 22), (25, 0, 0)]):
        plays.append({
            "play_id": i + 1,
            "game_id": "2024_01_KC_HOU",
            "play_type": "pass",
            "pass": 1,
            "rush": 0,
            "receiver_player_id": "A",
            "rusher_player_id": None,
            "passer_player_id": "QB1",
            "posteam": "KC",
            "season": 2024,
            "week": 1,
            "complete_pass": comp,
            "incomplete_pass": 1 - comp,
            "pass_touchdown": 0,
            "rush_touchdown": 0,
            "air_yards": air,
            "yards_gained": yards,
            "yards_after_catch": 3 if comp else 0,
            "epa": 0.3 if comp else -0.2,
            "yac_epa": 0.1 if comp else 0,
            "wp": 0.5,
            "game_half": "Half1",
            "qtr": 2,
            "yardline_100": 30,
            "first_down_pass": 1 if comp else 0,
        })

    # player_B: 6 targets, 5 completions, air_yards = [3, 7, 8, 12, 6, 9]
    for j, (air, comp, yards) in enumerate([(3, 1, 5), (7, 1, 12), (8, 1, 18), (12, 1, 20), (6, 1, 9), (9, 0, 0)]):
        plays.append({
            "play_id": 10 + j + 1,
            "game_id": "2024_01_KC_HOU",
            "play_type": "pass",
            "pass": 1,
            "rush": 0,
            "receiver_player_id": "B",
            "rusher_player_id": None,
            "passer_player_id": "QB1",
            "posteam": "KC",
            "season": 2024,
            "week": 1,
            "complete_pass": comp,
            "incomplete_pass": 1 - comp,
            "pass_touchdown": 0,
            "rush_touchdown": 0,
            "air_yards": air,
            "yards_gained": yards,
            "yards_after_catch": 2 if comp else 0,
            "epa": 0.25 if comp else -0.15,
            "yac_epa": 0.05 if comp else 0,
            "wp": 0.5,
            "game_half": "Half1",
            "qtr": 2,
            "yardline_100": 35,
            "first_down_pass": 1 if comp else 0,
        })

    return pd.DataFrame(plays)


@pytest.fixture
def rz_pbp():
    """
    Mix of red zone and non-red zone pass plays to test RZ target share.

    RZ = yardline_100 <= 20
    8 total targets: 4 to player_A, 4 to player_B
    3 of player_A's 4 targets are in RZ
    1 of player_B's 4 targets are in RZ
    Team RZ targets = 4
    player_A RZ share = 3/4 = 0.75
    player_B RZ share = 1/4 = 0.25
    """
    plays = []
    # player_A: yardlines [15, 18, 12, 40] → 3 in RZ
    for i, (yl, comp) in enumerate([(15, 1), (18, 1), (12, 1), (40, 0)]):
        plays.append({
            "play_id": i + 1, "game_id": "g1", "play_type": "pass",
            "pass": 1, "rush": 0,
            "receiver_player_id": "A", "rusher_player_id": None, "passer_player_id": "QB1",
            "posteam": "KC", "season": 2024, "week": 1,
            "complete_pass": comp, "pass_touchdown": 0,
            "air_yards": 5, "yards_gained": 8 if comp else 0,
            "epa": 0.1, "wp": 0.5, "game_half": "Half1", "qtr": 2,
            "yardline_100": yl,
        })
    # player_B: yardlines [10, 35, 45, 50] → 1 in RZ
    for j, (yl, comp) in enumerate([(10, 1), (35, 1), (45, 1), (50, 0)]):
        plays.append({
            "play_id": 10 + j + 1, "game_id": "g1", "play_type": "pass",
            "pass": 1, "rush": 0,
            "receiver_player_id": "B", "rusher_player_id": None, "passer_player_id": "QB1",
            "posteam": "KC", "season": 2024, "week": 1,
            "complete_pass": comp, "pass_touchdown": 0,
            "air_yards": 5, "yards_gained": 8 if comp else 0,
            "epa": 0.1, "wp": 0.5, "game_half": "Half1", "qtr": 2,
            "yardline_100": yl,
        })
    return pd.DataFrame(plays)


@pytest.fixture
def garbage_time_pbp():
    """
    Mix of neutral-script and garbage-time plays for the same player.

    Garbage time: wp < 0.10 AND game_half == 'Half2'
    3 garbage plays, 5 neutral plays for player_A.
    """
    plays = []
    # Neutral script plays (wp=0.5, Half1)
    for i in range(5):
        plays.append({
            "play_id": i + 1, "game_id": "g1", "play_type": "pass",
            "pass": 1, "rush": 0,
            "receiver_player_id": "A", "rusher_player_id": None, "passer_player_id": "QB1",
            "posteam": "KC", "season": 2024, "week": 1,
            "complete_pass": 1, "pass_touchdown": 0,
            "air_yards": 5, "yards_gained": 10,
            "epa": 0.5, "yac_epa": 0.1,
            "wp": 0.5, "game_half": "Half1", "qtr": 2, "yardline_100": 35,
        })
    # Garbage time plays (wp=0.05, Half2)
    for j in range(3):
        plays.append({
            "play_id": 10 + j + 1, "game_id": "g1", "play_type": "pass",
            "pass": 1, "rush": 0,
            "receiver_player_id": "A", "rusher_player_id": None, "passer_player_id": "QB1",
            "posteam": "KC", "season": 2024, "week": 1,
            "complete_pass": 1, "pass_touchdown": 0,
            "air_yards": 5, "yards_gained": 15,
            "epa": 0.8, "yac_epa": 0.2,
            "wp": 0.05, "game_half": "Half2", "qtr": 4, "yardline_100": 25,
        })
    return pd.DataFrame(plays)


@pytest.fixture
def rush_pbp():
    """
    Rush plays for two RBs.
    RB_A: 8 carries (2 in RZ), RB_B: 4 carries (1 in RZ).
    Team total: 12 carries, 3 in RZ.
    RB_A rush_share = 8/12 ≈ 0.667
    RB_B rush_share = 4/12 ≈ 0.333
    RB_A rz_rush_share = 2/3 ≈ 0.667
    RB_B rz_rush_share = 1/3 ≈ 0.333
    """
    plays = []
    # RB_A: 8 carries
    for i, (yl, yards) in enumerate([(30, 5), (25, 8), (15, 3), (40, -1), (18, 4), (35, 12), (20, 6), (10, 2)]):
        plays.append({
            "play_id": i + 1, "game_id": "g1", "play_type": "run",
            "pass": 0, "rush": 1,
            "receiver_player_id": None, "rusher_player_id": "RB_A", "passer_player_id": None,
            "posteam": "KC", "season": 2024, "week": 1,
            "complete_pass": 0, "pass_touchdown": 0, "rush_touchdown": 0,
            "air_yards": 0, "yards_gained": yards, "rushing_yards": yards,
            "epa": 0.1 if yards > 3 else -0.2,
            "wp": 0.5, "game_half": "Half1", "qtr": 2, "yardline_100": yl,
            "qb_scramble": 0,
        })
    # RB_B: 4 carries
    for j, (yl, yards) in enumerate([(45, 6), (10, 3), (50, -2), (22, 8)]):
        plays.append({
            "play_id": 20 + j + 1, "game_id": "g1", "play_type": "run",
            "pass": 0, "rush": 1,
            "receiver_player_id": None, "rusher_player_id": "RB_B", "passer_player_id": None,
            "posteam": "KC", "season": 2024, "week": 1,
            "complete_pass": 0, "pass_touchdown": 0, "rush_touchdown": 0,
            "air_yards": 0, "yards_gained": yards, "rushing_yards": yards,
            "epa": 0.1 if yards > 3 else -0.2,
            "wp": 0.5, "game_half": "Half1", "qtr": 2, "yardline_100": yl,
            "qb_scramble": 0,
        })
    return pd.DataFrame(plays)


# ---------------------------------------------------------------------------
# Opportunity factor tests
# ---------------------------------------------------------------------------

class TestTargetShare:
    def test_basic_target_share(self, simple_pass_pbp):
        result = build_opportunity_factors(simple_pass_pbp)
        a = result[result["player_id"] == "A"]
        b = result[result["player_id"] == "B"]
        assert len(a) == 1
        assert abs(a["target_share"].iloc[0] - 0.40) < 0.01, \
            f"Expected 0.40, got {a['target_share'].iloc[0]}"
        assert abs(b["target_share"].iloc[0] - 0.60) < 0.01

    def test_target_share_sums_to_one(self, simple_pass_pbp):
        result = build_opportunity_factors(simple_pass_pbp)
        # For KC 2024, all target share should sum to 1.0
        season_df = result[(result["team"] == "KC") & (result["season"] == 2024)]
        total_share = season_df["target_share"].sum()
        assert abs(total_share - 1.0) < 0.01

    def test_no_duplicate_player_season_rows(self, simple_pass_pbp):
        result = build_opportunity_factors(simple_pass_pbp)
        dup = result.duplicated(subset=["player_id", "season"])
        assert not dup.any(), f"Found duplicate (player_id, season) rows: {result[dup]}"

    def test_wopr_formula(self, simple_pass_pbp):
        result = build_opportunity_factors(simple_pass_pbp)
        a = result[result["player_id"] == "A"].iloc[0]
        expected_wopr = 1.5 * a["target_share"] + 0.7 * a["air_yard_share"]
        assert abs(a["wopr"] - expected_wopr) < 0.001


class TestRedZoneTargetShare:
    def test_rz_target_share_player_a(self, rz_pbp):
        result = build_opportunity_factors(rz_pbp)
        a = result[result["player_id"] == "A"]
        assert len(a) == 1
        # 3 RZ targets out of 4 total team RZ targets
        assert abs(a["rz_target_share"].iloc[0] - 0.75) < 0.01, \
            f"Expected 0.75, got {a['rz_target_share'].iloc[0]}"

    def test_rz_target_share_player_b(self, rz_pbp):
        result = build_opportunity_factors(rz_pbp)
        b = result[result["player_id"] == "B"]
        assert abs(b["rz_target_share"].iloc[0] - 0.25) < 0.01


class TestRushShare:
    def test_rush_share(self, rush_pbp):
        result = build_opportunity_factors(rush_pbp)
        a = result[result["player_id"] == "RB_A"]
        b = result[result["player_id"] == "RB_B"]
        assert abs(a["rush_share"].iloc[0] - 8/12) < 0.01
        assert abs(b["rush_share"].iloc[0] - 4/12) < 0.01

    def test_rush_share_sums_to_one(self, rush_pbp):
        result = build_opportunity_factors(rush_pbp)
        season_df = result[(result["team"] == "KC") & (result["season"] == 2024)]
        total_rush_share = season_df["rush_share"].dropna().sum()
        assert abs(total_rush_share - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Efficiency factor tests
# ---------------------------------------------------------------------------

class TestReceiverEfficiency:
    def test_catch_rate_player_a(self, simple_pass_pbp):
        result = build_efficiency_factors(simple_pass_pbp)
        a = result[result["player_id"] == "A"]
        assert len(a) == 1
        # 3 completions out of 4 targets = 0.75
        assert abs(a["catch_rate"].iloc[0] - 0.75) < 0.01

    def test_catch_rate_player_b(self, simple_pass_pbp):
        result = build_efficiency_factors(simple_pass_pbp)
        b = result[result["player_id"] == "B"]
        # 5 completions out of 6 targets ≈ 0.833
        assert abs(b["catch_rate"].iloc[0] - 5/6) < 0.01

    def test_epa_per_target_sign(self, simple_pass_pbp):
        result = build_efficiency_factors(simple_pass_pbp)
        for pid in ["A", "B"]:
            row = result[result["player_id"] == pid].iloc[0]
            # With mostly completions and positive EPA per completion, should be positive
            assert row["epa_per_target"] > 0

    def test_avg_depth_of_target(self, simple_pass_pbp):
        result = build_efficiency_factors(simple_pass_pbp)
        a = result[result["player_id"] == "A"].iloc[0]
        expected_adot = np.mean([5, 10, 15, 25])  # 13.75
        assert abs(a["avg_depth_of_target"] - expected_adot) < 0.01


class TestRBEfficiency:
    def test_rush_success_rate(self, rush_pbp):
        result = build_efficiency_factors(rush_pbp)
        a = result[result["player_id"] == "RB_A"]
        assert len(a) == 1
        # Success rate based on positive EPA
        assert 0 <= a["rush_success_rate"].iloc[0] <= 1.0

    def test_stuff_rate(self, rush_pbp):
        result = build_efficiency_factors(rush_pbp)
        a = result[result["player_id"] == "RB_A"]
        # RB_A has one carry with yards=-1 out of 8 carries
        assert abs(a["stuff_rate"].iloc[0] - 1/8) < 0.01

    def test_ypc(self, rush_pbp):
        result = build_efficiency_factors(rush_pbp)
        a = result[result["player_id"] == "RB_A"].iloc[0]
        expected_ypc = np.mean([5, 8, 3, -1, 4, 12, 6, 2])
        assert abs(a["ypc"] - expected_ypc) < 0.01


# ---------------------------------------------------------------------------
# Context factor tests (garbage time)
# ---------------------------------------------------------------------------

class TestContextFactors:
    def test_garbage_time_detection(self, garbage_time_pbp):
        """Plays with wp < 0.10 in Half2 should be classified as garbage time."""
        from features.context import build_context_factors

        weekly_stub = pd.DataFrame({
            "player_id": ["A"],
            "season": [2024],
            "week": [1],
            "fantasy_points_ppr": [25.0],
        })
        result = build_context_factors(garbage_time_pbp, weekly_stub)
        a = result[result["player_id"] == "A"]
        assert len(a) == 1
        # 3 garbage plays out of 8 total = 0.375
        assert abs(a["garbage_time_share"].iloc[0] - 3/8) < 0.01

    def test_neutral_script_epa(self, garbage_time_pbp):
        """Neutral script EPA should only use non-garbage-time plays."""
        from features.context import build_context_factors

        weekly_stub = pd.DataFrame({
            "player_id": ["A"],
            "season": [2024],
            "week": [1],
            "fantasy_points_ppr": [25.0],
        })
        result = build_context_factors(garbage_time_pbp, weekly_stub)
        a = result[result["player_id"] == "A"].iloc[0]
        # Neutral script plays have epa=0.5, garbage has epa=0.8
        # So neutral_script_epa should be ~0.5 (only Half1, wp=0.5, qtr=2 plays)
        assert abs(a["neutral_script_epa"] - 0.5) < 0.01


# ---------------------------------------------------------------------------
# Assembler integration test
# ---------------------------------------------------------------------------

class TestAssembler:
    def test_no_duplicate_player_season(self, simple_pass_pbp):
        """After full factor assembly, no player-season duplicates."""
        opp = build_opportunity_factors(simple_pass_pbp)
        eff = build_efficiency_factors(simple_pass_pbp)
        # Merge manually (assembler tested indirectly via unit tests above)
        merged = opp.merge(eff, on=["player_id", "team", "season"], how="outer")
        assert not merged.duplicated(subset=["player_id", "season"]).any()
