"""
Tests for the v5 model upgrades: career features, market ensemble, GBM.
Synthetic data only — no network, no cache dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from features.career import CAREER_FEATURES, add_career_features  # noqa: E402
from models.market_ensemble import MarketEnsembleModel  # noqa: E402


# ---------------------------------------------------------------------------
# Career features
# ---------------------------------------------------------------------------

def _career_fm() -> pd.DataFrame:
    # Player A: 2019–2022 with a gap at 2021. Player B: one season.
    return pd.DataFrame([
        {"player_id": "A", "season": 2019, "fpts_per_game": 10.0, "games_played": 16,
         "target_share": 0.10, "rush_share": 0.0, "targets": 50, "carries": 0, "dropbacks": 0},
        {"player_id": "A", "season": 2020, "fpts_per_game": 20.0, "games_played": 8,
         "target_share": 0.30, "rush_share": 0.0, "targets": 80, "carries": 0, "dropbacks": 0},
        {"player_id": "A", "season": 2022, "fpts_per_game": 15.0, "games_played": 17,
         "target_share": 0.20, "rush_share": 0.0, "targets": 90, "carries": 0, "dropbacks": 0},
        {"player_id": "B", "season": 2022, "fpts_per_game": 5.0, "games_played": 10,
         "target_share": 0.05, "rush_share": 0.0, "targets": 20, "carries": 0, "dropbacks": 0},
    ])


def test_career_features_expanding_and_lags():
    out = add_career_features(_career_fm()).set_index(["player_id", "season"])

    a19 = out.loc[("A", 2019)]
    a20 = out.loc[("A", 2020)]
    a22 = out.loc[("A", 2022)]

    assert a19["seasons_played_todate"] == 1
    assert a20["seasons_played_todate"] == 2
    assert a22["seasons_played_todate"] == 3

    # Games-weighted career mean at 2020: (10*16 + 20*8) / 24 = 13.33
    assert a20["career_fpts_pg"] == pytest.approx(13.333, abs=0.01)
    assert a20["peak_fpts_pg"] == 20.0
    assert a22["peak_target_share"] == pytest.approx(0.30)

    # Contiguous lag exists 2019→2020…
    assert a20["fpts_pg_prev"] == 10.0
    assert a20["fpts_pg_yoy_change"] == pytest.approx(10.0)
    # …but the 2021 gap voids the lag at 2022
    assert np.isnan(a22["fpts_pg_prev"])
    assert np.isnan(a22["fpts_pg_yoy_change"])

    # First season has no lag; one-season player well-defined
    assert np.isnan(a19["fpts_pg_prev"])
    b22 = out.loc[("B", 2022)]
    assert b22["seasons_played_todate"] == 1
    assert b22["career_fpts_pg"] == pytest.approx(5.0)


def test_career_features_no_lookahead():
    """Values at season N must not change when later seasons are added."""
    fm = _career_fm()
    early = add_career_features(fm[fm["season"] <= 2020])
    full = add_career_features(fm)
    for col in CAREER_FEATURES:
        e = early[(early.player_id == "A") & (early.season == 2020)][col].iloc[0]
        f = full[(full.player_id == "A") & (full.season == 2020)][col].iloc[0]
        assert (np.isnan(e) and np.isnan(f)) or e == pytest.approx(f), col


def test_career_features_idempotent_and_dedup():
    fm = _career_fm()
    dup = pd.concat([fm, fm.iloc[[2]]], ignore_index=True)  # duplicate A-2022 stint
    once = add_career_features(dup)
    twice = add_career_features(once)
    assert len(once) == len(dup)  # stint rows preserved, values merged back
    a22 = once[(once.player_id == "A") & (once.season == 2022)]
    assert a22["seasons_played_todate"].nunique() == 1  # both stints agree
    pd.testing.assert_frame_equal(
        once.sort_index(axis=1), twice.sort_index(axis=1), check_dtype=False
    )


# ---------------------------------------------------------------------------
# Market ensemble
# ---------------------------------------------------------------------------

class _StubBase:
    """Deterministic base model: predicts the 'base_pred' column."""

    def train(self, df, target="next_fpts", fit_age=True):
        return self

    def predict_position(self, pos, pos_df):
        return pos_df["base_pred"].values.astype(float)


def _ensemble_pairs(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    adp = np.sort(rng.uniform(1, 180, n))
    next_fpts = 22 - 0.09 * adp + rng.normal(0, 2, n)  # decreasing in ADP
    return pd.DataFrame({
        "player_name": [f"p{i}" for i in range(n)],
        "position": "WR",
        "season": np.repeat(np.arange(2015, 2023), n // 8),
        "adp": adp,
        "next_fpts": next_fpts,
        "base_pred": rng.uniform(0, 25, n),
    })


def test_market_prior_is_monotone_decreasing():
    pairs = _ensemble_pairs()
    m = MarketEnsembleModel(market_weight=1.0, base_factory=_StubBase)
    m.train(pairs, fit_age=False)
    grid = np.array([1.0, 20.0, 60.0, 120.0, 175.0])
    implied = m.market_implied("WR", grid)
    assert np.all(np.diff(implied) <= 1e-9)
    assert implied[0] > implied[-1]


def test_ensemble_blend_and_fallback():
    pairs = _ensemble_pairs()
    m = MarketEnsembleModel(market_weight=0.6, base_factory=_StubBase)
    m.train(pairs, fit_age=False)

    test = pairs.head(10).copy()
    prior = m.market_implied("WR", test["adp"].values)
    blended = m.predict_position("WR", test)
    expected = 0.6 * prior + 0.4 * test["base_pred"].values
    np.testing.assert_allclose(blended, expected, rtol=1e-9)

    # Missing ADP → pure base model
    no_adp = test.copy()
    no_adp["adp"] = np.nan
    np.testing.assert_allclose(
        m.predict_position("WR", no_adp), no_adp["base_pred"].values
    )


def test_ensemble_per_position_weights():
    m = MarketEnsembleModel(market_weight={"QB": 0.8, "WR": 0.5}, base_factory=_StubBase)
    assert m.weight_for("QB") == 0.8
    assert m.weight_for("WR") == 0.5
    assert 0 <= m.weight_for("RB") <= 1  # falls back to a sane default


def test_ensemble_ordering_of_disagreements_preserved():
    """Blending shrinks model-vs-market disagreements but never flips their sign."""
    pairs = _ensemble_pairs()
    m = MarketEnsembleModel(market_weight=0.7, base_factory=_StubBase)
    m.train(pairs, fit_age=False)
    test = pairs.head(50).copy()
    prior = m.market_implied("WR", test["adp"].values)
    blended = m.predict_position("WR", test)
    base_dev = test["base_pred"].values - prior
    blend_dev = blended - prior
    assert np.all(np.sign(base_dev) == np.sign(blend_dev))
    assert np.all(np.abs(blend_dev) <= np.abs(base_dev) + 1e-9)


# ---------------------------------------------------------------------------
# GBM
# ---------------------------------------------------------------------------

def _gbm_pairs(seed=0) -> pd.DataFrame:
    lgbm = pytest.importorskip("lightgbm")  # noqa: F841
    rng = np.random.default_rng(seed)
    rows = []
    for season in range(2014, 2024):
        for i in range(60):
            ts = rng.uniform(0, 0.35)
            epa = rng.normal(0, 0.5)
            rows.append({
                "player_id": f"wr{i}", "position": "WR", "season": season,
                "target_share": ts, "epa_per_target": epa,
                "games_played": rng.integers(8, 18),
                "next_fpts": 40 * ts + 2 * epa + rng.normal(0, 1.5),
            })
    return pd.DataFrame(rows)


def test_gbm_trains_and_predicts():
    from models.gbm import GBMProjectionModel

    pairs = _gbm_pairs()
    m = GBMProjectionModel(age_adjust=False, standardize=False)
    m.train(pairs, fit_age=False)
    assert "WR" in m._models
    test = pairs[pairs["season"] == 2023]
    pred = m.predict_position("WR", test.reset_index(drop=True))
    assert pred is not None and np.isfinite(pred).all()
    # Signal check: predictions should rank-correlate with the true driver
    from scipy.stats import spearmanr

    ic, _ = spearmanr(pred, test["next_fpts"].values)
    assert ic > 0.5

    imp = m.feature_importance("WR")
    assert imp.iloc[0]["feature"] == "target_share"  # dominant simulated driver
