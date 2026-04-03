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
from features.situation import detect_team_changes, build_situation_features
from features.pedigree import build_pedigree_features
from features.consistency import build_consistency_features


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


# ---------------------------------------------------------------------------
# Situation change detection tests
# ---------------------------------------------------------------------------

class TestDetectTeamChanges:
    @pytest.fixture
    def rosters_with_change(self):
        return pd.DataFrame({
            "player_id": ["p1", "p1", "p2", "p2"],
            "team":      ["NYG", "PHI", "KC",  "KC"],
            "season":    [2022,   2023,  2022,  2023],
        })

    def test_team_change_detected(self, rosters_with_change):
        changes = detect_team_changes(rosters_with_change)
        assert len(changes) == 1
        assert changes.iloc[0]["player_id"] == "p1"
        assert changes.iloc[0]["old_team"] == "NYG"
        assert changes.iloc[0]["new_team"] == "PHI"

    def test_same_team_not_detected(self, rosters_with_change):
        changes = detect_team_changes(rosters_with_change)
        p2_changes = changes[changes["player_id"] == "p2"]
        assert len(p2_changes) == 0

    def test_output_columns(self, rosters_with_change):
        changes = detect_team_changes(rosters_with_change)
        required = {"player_id", "season", "old_team", "new_team"}
        assert required.issubset(set(changes.columns))

    def test_season_is_earlier_season(self, rosters_with_change):
        changes = detect_team_changes(rosters_with_change)
        assert changes.iloc[0]["season"] == 2022  # season N, not N+1

    def test_empty_returns_empty(self):
        empty = detect_team_changes(pd.DataFrame(columns=["player_id", "team", "season"]))
        assert len(empty) == 0

    def test_nonconsecutive_seasons_skipped(self):
        # Player skipped 2022 (2021 → 2023 gap) — should NOT be flagged as a change
        rosters = pd.DataFrame({
            "player_id": ["p1", "p1"],
            "team":      ["NE",  "NYJ"],
            "season":    [2021,   2023],
        })
        changes = detect_team_changes(rosters)
        assert len(changes) == 0


# ---------------------------------------------------------------------------
# Pedigree features tests
# ---------------------------------------------------------------------------

class TestBuildPedigreeFeatures:
    @pytest.fixture
    def roster_df(self):
        return pd.DataFrame({
            "player_id":    ["r1_pick",  "r2_pick",  "udfa",    "r6_pick"],
            "season":       [2023,        2023,        2023,       2023],
            "draft_number": [5.0,         40.0,        float("nan"), 200.0],
            "years_exp":    [0,            1,           3,          5],
        })

    def test_round1_pick_gets_bucket_5(self, roster_df):
        result = build_pedigree_features(roster_df)
        r1 = result[result["player_id"] == "r1_pick"]
        assert r1["draft_round_bucket"].iloc[0] == 5

    def test_round2_pick_gets_bucket_4(self, roster_df):
        result = build_pedigree_features(roster_df)
        r2 = result[result["player_id"] == "r2_pick"]
        assert r2["draft_round_bucket"].iloc[0] == 4

    def test_udfa_gets_bucket_0(self, roster_df):
        result = build_pedigree_features(roster_df)
        udfa = result[result["player_id"] == "udfa"]
        assert udfa["draft_round_bucket"].iloc[0] == 0

    def test_udfa_draft_capital_score_zero(self, roster_df):
        result = build_pedigree_features(roster_df)
        udfa = result[result["player_id"] == "udfa"]
        assert udfa["draft_capital_score"].iloc[0] == 0.0

    def test_r1_draft_capital_score_high(self, roster_df):
        result = build_pedigree_features(roster_df)
        r1 = result[result["player_id"] == "r1_pick"]
        assert r1["draft_capital_score"].iloc[0] > 0.9

    def test_is_rookie_flag(self, roster_df):
        result = build_pedigree_features(roster_df)
        r1 = result[result["player_id"] == "r1_pick"]
        assert r1["is_rookie"].iloc[0] == 1

    def test_sophomore_flag(self, roster_df):
        result = build_pedigree_features(roster_df)
        r2 = result[result["player_id"] == "r2_pick"]
        assert r2["sophomore_flag"].iloc[0] == 1

    def test_veteran_not_rookie_or_sophomore(self, roster_df):
        result = build_pedigree_features(roster_df)
        udfa = result[result["player_id"] == "udfa"]
        assert udfa["is_rookie"].iloc[0] == 0
        assert udfa["sophomore_flag"].iloc[0] == 0

    def test_output_columns_present(self, roster_df):
        result = build_pedigree_features(roster_df)
        expected = {"player_id", "season", "draft_round_bucket", "draft_capital_score",
                    "years_in_league", "is_rookie", "sophomore_flag"}
        assert expected.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Consistency features tests
# ---------------------------------------------------------------------------

class TestBuildConsistencyFeatures:
    @pytest.fixture
    def weekly_consistent(self):
        """Player who scores close to 15 every week (consistent)."""
        return pd.DataFrame({
            "player_id": ["p_c"] * 16,
            "season":    [2023] * 16,
            "week":      list(range(1, 17)),
            "position":  ["WR"] * 16,
            "fantasy_points_ppr": [14.5, 15.5, 15.0, 14.8, 15.2, 14.9, 15.1, 15.3,
                                   14.7, 15.0, 15.4, 14.6, 15.1, 14.8, 15.0, 15.1],
        })

    @pytest.fixture
    def weekly_volatile(self):
        """Player with boom-or-bust profile (high variance)."""
        return pd.DataFrame({
            "player_id": ["p_v"] * 16,
            "season":    [2023] * 16,
            "week":      list(range(1, 17)),
            "position":  ["WR"] * 16,
            "fantasy_points_ppr": [0.0, 35.0, 0.0, 40.0, 0.0, 30.0, 0.0, 25.0,
                                   0.0, 35.0, 0.0, 38.0, 0.0, 32.0, 0.0, 27.0],
        })

    def test_consistent_player_low_cv(self, weekly_consistent):
        result = build_consistency_features(weekly_consistent)
        cv = result["weekly_fpts_cv"].iloc[0]
        assert cv < 0.05  # consistent player has very low CV

    def test_volatile_player_high_cv(self, weekly_volatile):
        result = build_consistency_features(weekly_volatile)
        cv = result["weekly_fpts_cv"].iloc[0]
        assert cv > 0.5  # volatile player has high CV

    def test_consistent_higher_consistency_score(self, weekly_consistent, weekly_volatile):
        r_c = build_consistency_features(weekly_consistent)
        r_v = build_consistency_features(weekly_volatile)
        assert r_c["consistency_score"].iloc[0] > r_v["consistency_score"].iloc[0]

    def test_boom_bust_rates_between_0_and_1(self, weekly_volatile):
        result = build_consistency_features(weekly_volatile)
        assert 0.0 <= result["boom_rate"].iloc[0] <= 1.0
        assert 0.0 <= result["bust_rate"].iloc[0] <= 1.0

    def test_consistency_score_between_0_and_1(self, weekly_consistent):
        result = build_consistency_features(weekly_consistent)
        score = result["consistency_score"].iloc[0]
        assert 0.0 <= score <= 1.0

    def test_output_columns_present(self, weekly_consistent):
        result = build_consistency_features(weekly_consistent)
        expected = {"player_id", "season", "weekly_fpts_std", "weekly_fpts_cv",
                    "weekly_fpts_median", "boom_rate", "bust_rate", "consistency_score"}
        assert expected.issubset(set(result.columns))

    def test_no_position_column_still_works(self):
        """If position column is absent, boom/bust rates are NaN but other cols computed."""
        weekly_no_pos = pd.DataFrame({
            "player_id": ["p1"] * 10,
            "season":    [2023] * 10,
            "week":      list(range(1, 11)),
            "fantasy_points_ppr": [10.0] * 10,
        })
        result = build_consistency_features(weekly_no_pos)
        assert len(result) == 1
        assert result["weekly_fpts_cv"].iloc[0] == 0.0  # zero variance
        assert pd.isna(result["boom_rate"].iloc[0])
        assert pd.isna(result["bust_rate"].iloc[0])


# ---------------------------------------------------------------------------
# Vacated share tests
# ---------------------------------------------------------------------------

class TestComputeVacatedShares:
    @pytest.fixture
    def feature_matrix_with_shares(self):
        """Feature matrix with target_share and rush_share for two seasons."""
        return pd.DataFrame({
            "player_id":    ["wr1",  "wr2",  "wr1",  "wr3"],
            "team":         ["KC",   "KC",   "KC",   "NE"],
            "season":       [2022,   2022,   2023,   2022],
            "target_share": [0.30,   0.25,   0.28,   0.20],
            "rush_share":   [0.0,    0.0,    0.0,    0.0],
            "games_played": [17,     17,     17,     17],
        })

    @pytest.fixture
    def rosters_with_departure(self):
        """wr2 was on KC in 2022 but left — not on KC in 2023."""
        return pd.DataFrame({
            "player_id": ["wr1",  "wr2",  "wr1",  "wr3",  "wr2"],
            "team":      ["KC",   "KC",   "KC",   "NE",   "BUF"],
            "season":    [2022,   2022,   2023,   2022,   2023],
        })

    def test_departed_player_share_counted(self, feature_matrix_with_shares, rosters_with_departure):
        from features.vacated import compute_vacated_shares

        result = compute_vacated_shares(feature_matrix_with_shares, rosters_with_departure)
        kc_2022 = result[(result["team"] == "KC") & (result["season"] == 2022)]
        assert len(kc_2022) == 1
        # wr2 left KC with target_share=0.25 and full 17 games → vacated = 0.25 * 1.0 = 0.25
        assert abs(kc_2022["vacated_target_share"].iloc[0] - 0.25) < 0.01

    def test_stayer_not_counted(self, feature_matrix_with_shares, rosters_with_departure):
        from features.vacated import compute_vacated_shares

        result = compute_vacated_shares(feature_matrix_with_shares, rosters_with_departure)
        # wr1 stayed on KC from 2022 → 2023, should not be in vacated shares
        kc_2022 = result[(result["team"] == "KC") & (result["season"] == 2022)]
        # Only wr2 departed, so top_departed = wr2's share (0.25), not wr1's (0.30)
        assert abs(kc_2022["top_departed_target_share"].iloc[0] - 0.25) < 0.01

    def test_availability_weighting(self):
        """A player who played only 8 games vacates half their share."""
        from features.vacated import compute_vacated_shares

        fm = pd.DataFrame({
            "player_id":    ["wr_part", "wr_part"],
            "team":         ["MIA",     "BUF"],
            "season":       [2022,      2023],
            "target_share": [0.30,      0.20],
            "rush_share":   [0.0,       0.0],
            "games_played": [8,         17],
        })
        rosters = pd.DataFrame({
            "player_id": ["wr_part", "wr_part"],
            "team":      ["MIA",     "BUF"],
            "season":    [2022,      2023],
        })
        result = compute_vacated_shares(fm, rosters)
        mia = result[(result["team"] == "MIA") & (result["season"] == 2022)]
        assert len(mia) == 1
        # 8/17 ≈ 0.471 availability weight → vacated = 0.30 * (8/17) ≈ 0.141
        expected = 0.30 * (8 / 17)
        assert abs(mia["vacated_target_share"].iloc[0] - expected) < 0.01


# ---------------------------------------------------------------------------
# QB coupling feature tests
# ---------------------------------------------------------------------------

class TestBuildQbQualityByTeam:
    @pytest.fixture
    def qb_pbp(self):
        """
        Two QBs on KC in 2022: QB_A has 8 dropbacks, QB_B has 4.
        QB_A is the primary QB.
        """
        plays = []
        # QB_A: 8 dropbacks (pass attempts)
        for i in range(8):
            plays.append({
                "passer_player_id": "QB_A",
                "posteam": "KC",
                "season": 2022,
                "week": 1 + i,
                "pass_attempt": 1,
                "sack": 0,
                "qb_scramble": 0,
                "epa": 0.3,
                "cpoe": 5.0,
                "air_yards": 10.0,
                "pass_touchdown": 0,
                "complete_pass": 1,
                "incomplete_pass": 0,
            })
        # QB_B: 4 dropbacks
        for j in range(4):
            plays.append({
                "passer_player_id": "QB_B",
                "posteam": "KC",
                "season": 2022,
                "week": 1 + j,
                "pass_attempt": 1,
                "sack": 0,
                "qb_scramble": 0,
                "epa": 0.1,
                "cpoe": 2.0,
                "air_yards": 8.0,
                "pass_touchdown": 0,
                "complete_pass": 1,
                "incomplete_pass": 0,
            })
        return pd.DataFrame(plays)

    @pytest.fixture
    def qb_change_pbp(self, qb_pbp):
        """Add a 2023 season where KC's primary QB changed to QB_C."""
        extra = []
        for i in range(10):
            extra.append({
                "passer_player_id": "QB_C",
                "posteam": "KC",
                "season": 2023,
                "week": 1 + i,
                "pass_attempt": 1,
                "sack": 0,
                "qb_scramble": 0,
                "epa": 0.4,
                "cpoe": 6.0,
                "air_yards": 11.0,
                "pass_touchdown": 0,
                "complete_pass": 1,
                "incomplete_pass": 0,
            })
        return pd.concat([qb_pbp, pd.DataFrame(extra)], ignore_index=True)

    def test_primary_qb_selected_by_dropbacks(self, qb_pbp):
        from features.qb_coupling import build_qb_quality_by_team

        result = build_qb_quality_by_team(qb_pbp)
        kc_2022 = result[(result["team"] == "KC") & (result["season"] == 2022)]
        assert len(kc_2022) == 1
        assert kc_2022["primary_qb_id"].iloc[0] == "QB_A"

    def test_qb_change_detected(self, qb_change_pbp):
        from features.qb_coupling import build_qb_coupling_features

        result = build_qb_coupling_features(qb_change_pbp)
        kc_2022 = result[(result["team"] == "KC") & (result["season"] == 2022)]
        assert len(kc_2022) == 1
        assert kc_2022["qb_changed"].iloc[0] == 1

    def test_output_columns(self, qb_pbp):
        from features.qb_coupling import build_qb_coupling_features

        result = build_qb_coupling_features(qb_pbp)
        required = {"team", "season", "qb_epa_per_dropback"}
        assert required.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Regressed efficiency tests
# ---------------------------------------------------------------------------

class TestRegressedEfficiency:
    @pytest.fixture
    def minimal_yoy(self):
        """Minimal YoY DataFrame with WR efficiency columns for training."""
        np.random.seed(42)
        n = 60
        return pd.DataFrame({
            "player_id": [f"p{i}" for i in range(n)],
            "season": [2020 + i // 20 for i in range(n)],
            "position": ["WR"] * n,
            "yards_per_target": np.random.uniform(5.0, 12.0, n),
            "rec_td_rate": np.random.uniform(0.02, 0.12, n),
            "next_fpts": np.random.uniform(50, 250, n),
            "next_yards_per_target": np.random.uniform(5.0, 12.0, n),
            "next_rec_td_rate": np.random.uniform(0.02, 0.12, n),
            "games_played": [16] * n,
        })

    def test_output_between_player_and_mean(self, minimal_yoy):
        from models.two_stage import TwoStageProjectionModel

        model = TwoStageProjectionModel(age_adjust=False)
        model.train(minimal_yoy, fit_age=False, use_ridge_efficiency=False)

        test_row = pd.DataFrame({
            "player_id": ["test_player"],
            "yards_per_target": [10.0],  # above average
            "rec_td_rate": [0.10],
            "games_played": [16],
        })

        preds = model._regressed_efficiency("WR", test_row)
        pos_mean = model._positional_means["WR"]["yards_per_target"]

        # Prediction must be between player value and mean (blended)
        lo = min(10.0, pos_mean)
        hi = max(10.0, pos_mean)
        assert lo <= preds["yards_per_target"][0] <= hi + 0.01

    def test_missing_values_fall_back_to_mean(self, minimal_yoy):
        from models.two_stage import TwoStageProjectionModel

        model = TwoStageProjectionModel(age_adjust=False)
        model.train(minimal_yoy, fit_age=False, use_ridge_efficiency=False)

        test_row = pd.DataFrame({
            "player_id": ["test_player"],
            "yards_per_target": [float("nan")],  # missing
            "rec_td_rate": [float("nan")],
            "games_played": [16],
        })

        preds = model._regressed_efficiency("WR", test_row)
        pos_mean_ypt = model._positional_means["WR"]["yards_per_target"]
        # NaN player rate → fill with pos_mean → regressed = pos_mean (fully at mean)
        assert abs(preds["yards_per_target"][0] - pos_mean_ypt) < 0.01

    def test_td_rate_heavily_regressed(self, minimal_yoy):
        from models.two_stage import TwoStageProjectionModel
        from config import EFFICIENCY_REGRESSION_WEIGHTS

        model = TwoStageProjectionModel(age_adjust=False)
        model.train(minimal_yoy, fit_age=False, use_ridge_efficiency=False)

        pos_mean_td = model._positional_means["WR"]["rec_td_rate"]
        player_td = 0.12  # high TD rate player

        test_row = pd.DataFrame({
            "player_id": ["test_player"],
            "yards_per_target": [8.0],
            "rec_td_rate": [player_td],
            "games_played": [16],
        })

        preds = model._regressed_efficiency("WR", test_row)
        reg_w = EFFICIENCY_REGRESSION_WEIGHTS["rec_td_rate"]  # should be 0.55
        expected = reg_w * pos_mean_td + (1 - reg_w) * player_td

        assert abs(preds["rec_td_rate"][0] - expected) < 0.001
        # With 0.55 regression weight, prediction should be closer to mean than to player value
        dist_to_mean = abs(preds["rec_td_rate"][0] - pos_mean_td)
        dist_to_player = abs(preds["rec_td_rate"][0] - player_td)
        assert dist_to_mean < dist_to_player
