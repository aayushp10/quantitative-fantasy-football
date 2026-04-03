"""
Tests for src/utils/scoring.py

Ground-truth verification against known player stat lines.
All expected values computed from the PPR scoring formula manually.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import HALF_PPR_SCORING, PPR_SCORING, STANDARD_SCORING
from utils.scoring import calculate_fantasy_points, calculate_fantasy_points_from_dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_stats_row(**kwargs) -> pd.DataFrame:
    """Build a single-row DataFrame with stat values keyed by canonical names."""
    canonical_to_col = {
        "pass_yd": "passing_yards",
        "pass_td": "passing_tds",
        "interception": "interceptions",
        "rush_yd": "rushing_yards",
        "rush_td": "rushing_tds",
        "reception": "receptions",
        "rec_yd": "receiving_yards",
        "rec_td": "receiving_tds",
        "fumble_lost": "sack_fumbles_lost",
        "pass_2pt": "passing_2pt_conversions",
        "rush_2pt": "rushing_2pt_conversions",
        "rec_2pt": "receiving_2pt_conversions",
    }
    row = {v: 0.0 for v in canonical_to_col.values()}
    for key, val in kwargs.items():
        col = canonical_to_col.get(key, key)
        row[col] = float(val)
    return pd.DataFrame([row])


def manual_ppr(stats: dict) -> float:
    total = 0.0
    for key, weight in PPR_SCORING.items():
        total += stats.get(key, 0.0) * weight
    return total


# ---------------------------------------------------------------------------
# Ground-truth player stat lines (approximate 2022 season totals)
# ---------------------------------------------------------------------------

MAHOMES_2022 = {
    "pass_yd": 5250, "pass_td": 41, "interception": 12,
    "rush_yd": 358, "rush_td": 4, "fumble_lost": 2,
}

JEFFERSON_2022 = {
    "reception": 128, "rec_yd": 1809, "rec_td": 8,
}

EKELER_2022 = {
    "rush_yd": 915, "rush_td": 13,
    "reception": 107, "rec_yd": 722, "rec_td": 5,
}

KELCE_2022 = {
    "reception": 110, "rec_yd": 1338, "rec_td": 12,
}


# ---------------------------------------------------------------------------
# Basic correctness tests
# ---------------------------------------------------------------------------

class TestPprGroundTruth:
    def test_mahomes_2022(self):
        expected = manual_ppr(MAHOMES_2022)
        result = calculate_fantasy_points_from_dict(MAHOMES_2022, PPR_SCORING)
        assert abs(result - expected) < 0.01

    def test_jefferson_2022(self):
        expected = manual_ppr(JEFFERSON_2022)
        result = calculate_fantasy_points_from_dict(JEFFERSON_2022, PPR_SCORING)
        assert abs(result - expected) < 0.01

    def test_ekeler_2022(self):
        expected = manual_ppr(EKELER_2022)
        result = calculate_fantasy_points_from_dict(EKELER_2022, PPR_SCORING)
        assert abs(result - expected) < 0.01

    def test_kelce_2022(self):
        expected = manual_ppr(KELCE_2022)
        result = calculate_fantasy_points_from_dict(KELCE_2022, PPR_SCORING)
        assert abs(result - expected) < 0.01


class TestScoringFormats:
    def test_ppr_higher_than_half_ppr_for_pass_catcher(self):
        stats = {"reception": 100, "rec_yd": 1000, "rec_td": 8}
        ppr = calculate_fantasy_points_from_dict(stats, PPR_SCORING)
        half = calculate_fantasy_points_from_dict(stats, HALF_PPR_SCORING)
        std = calculate_fantasy_points_from_dict(stats, STANDARD_SCORING)
        assert ppr > half > std

    def test_all_formats_equal_for_qb_no_receptions(self):
        stats = {"pass_yd": 300, "pass_td": 3, "interception": 1}
        ppr = calculate_fantasy_points_from_dict(stats, PPR_SCORING)
        half = calculate_fantasy_points_from_dict(stats, HALF_PPR_SCORING)
        std = calculate_fantasy_points_from_dict(stats, STANDARD_SCORING)
        assert abs(ppr - half) < 0.001
        assert abs(ppr - std) < 0.001

    def test_two_point_conversions(self):
        stats = {"pass_2pt": 1, "rush_2pt": 1, "rec_2pt": 1}
        result = calculate_fantasy_points_from_dict(stats, PPR_SCORING)
        assert abs(result - 6.0) < 0.001  # 3 × 2.0 pts

    def test_negative_stats(self):
        stats = {"interception": 2, "fumble_lost": 1}
        result = calculate_fantasy_points_from_dict(stats, PPR_SCORING)
        assert abs(result - (-6.0)) < 0.001  # 2×(-2) + 1×(-2)

    def test_zero_stats(self):
        result = calculate_fantasy_points_from_dict({}, PPR_SCORING)
        assert result == 0.0


class TestDataFrameInterface:
    def test_vectorized_basic(self):
        df = make_stats_row(pass_yd=300, pass_td=2, interception=1)
        result = calculate_fantasy_points(df)
        expected = 300 * 0.04 + 2 * 4.0 + 1 * (-2.0)
        assert abs(result.iloc[0] - expected) < 0.001

    def test_multiple_rows(self):
        rows = [
            make_stats_row(pass_yd=300, pass_td=2),
            make_stats_row(reception=10, rec_yd=100, rec_td=1),
        ]
        df = pd.concat(rows, ignore_index=True)
        result = calculate_fantasy_points(df, PPR_SCORING)
        assert len(result) == 2
        assert result.iloc[0] > 0
        assert result.iloc[1] > 0

    def test_nan_columns_treated_as_zero(self):
        df = make_stats_row(pass_yd=300)
        df["passing_tds"] = float("nan")
        result = calculate_fantasy_points(df)
        expected = 300 * 0.04
        assert abs(result.iloc[0] - expected) < 0.001

    def test_missing_columns_ignored(self):
        df = pd.DataFrame([{"passing_yards": 300}])
        result = calculate_fantasy_points(df)
        assert abs(result.iloc[0] - 300 * 0.04) < 0.001

    def test_performance_10k_rows(self):
        rng = np.random.default_rng(42)
        n = 10_000
        df = pd.DataFrame({
            "passing_yards": rng.uniform(0, 400, n),
            "passing_tds": rng.integers(0, 5, n).astype(float),
            "interceptions": rng.integers(0, 4, n).astype(float),
            "rushing_yards": rng.uniform(0, 150, n),
            "rushing_tds": rng.integers(0, 3, n).astype(float),
            "receptions": rng.integers(0, 15, n).astype(float),
            "receiving_yards": rng.uniform(0, 200, n),
            "receiving_tds": rng.integers(0, 3, n).astype(float),
            "sack_fumbles_lost": rng.integers(0, 2, n).astype(float),
        })
        start = time.perf_counter()
        result = calculate_fantasy_points(df)
        elapsed = time.perf_counter() - start
        assert len(result) == n
        assert elapsed < 1.0, f"Scoring 10k rows took {elapsed:.2f}s (should be < 1s)"

    def test_fumbles_aggregated_from_multiple_columns(self):
        df = pd.DataFrame([{
            "passing_yards": 0,
            "passing_tds": 0,
            "interceptions": 0,
            "rushing_yards": 0,
            "rushing_tds": 0,
            "receptions": 0,
            "receiving_yards": 0,
            "receiving_tds": 0,
            "sack_fumbles_lost": 1,
            "rushing_fumbles_lost": 1,
            "receiving_fumbles_lost": 1,
        }])
        result = calculate_fantasy_points(df)
        assert abs(result.iloc[0] - (-6.0)) < 0.001  # 3 fumbles × -2.0
