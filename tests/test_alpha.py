"""
Tests for the Stage B market-residual ("alpha") model.
Synthetic data only — no network, no cache, no real pipeline.
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

from models.alpha import (  # noqa: E402
    ALPHA_FEATURES,
    AlphaModel,
    attach_market,
    estimate_lambda,
    market_expected_oos,
)


# ---------------------------------------------------------------------------
# Synthetic frames
# ---------------------------------------------------------------------------

def _synthetic_pairs(seasons=range(2001, 2006), n_per_pos=40, seed=0) -> pd.DataFrame:
    """ADP-joined pairs where next_fpts declines in ADP plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for pos in ["QB", "RB", "WR", "TE"]:
            adp = np.sort(rng.uniform(1, 180, n_per_pos))
            fpts = 22 - 8 * np.log(adp) / np.log(180) + rng.normal(0, 2, n_per_pos)
            for i in range(n_per_pos):
                rows.append({
                    "player_id": f"{pos}{i}",
                    "player_name": f"{pos} player{i}",
                    "position": pos,
                    "season": s,
                    "adp": adp[i],
                    "next_fpts": fpts[i],
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# market_expected_oos: strictly out-of-sample protocol
# ---------------------------------------------------------------------------

def test_market_oos_excludes_early_seasons():
    pairs = _synthetic_pairs(seasons=range(2001, 2006))
    out = market_expected_oos(pairs, min_train_seasons=2)
    # 2001/2002 can never be scored OOS with min_train_seasons=2
    assert sorted(out["season"].unique()) == [2003, 2004, 2005]
    assert out["market_expected"].notna().all()


def test_market_oos_no_future_leakage():
    """Perturbing the TEST season's targets must not move its predictions."""
    pairs = _synthetic_pairs()
    base = market_expected_oos(pairs, min_train_seasons=2)

    poisoned = pairs.copy()
    mask = poisoned["season"] == 2005
    poisoned.loc[mask, "next_fpts"] = poisoned.loc[mask, "next_fpts"] + 100.0
    out = market_expected_oos(poisoned, min_train_seasons=2)

    a = base[base["season"] == 2005].sort_values(["position", "player_id"])
    b = out[out["season"] == 2005].sort_values(["position", "player_id"])
    np.testing.assert_allclose(
        a["market_expected"].values, b["market_expected"].values, atol=1e-10
    )


def test_market_oos_uses_only_past_regime():
    """A regime break in the test season must NOT be reflected in its map."""
    pairs = _synthetic_pairs()
    # Test season 2005: flat 5.0 for everyone. The OOS expectation must
    # still look like the historical downward-sloping curve.
    pairs.loc[pairs["season"] == 2005, "next_fpts"] = 5.0
    out = market_expected_oos(pairs, min_train_seasons=2)
    s5 = out[(out["season"] == 2005) & (out["position"] == "WR")].sort_values("adp")
    # Early picks priced well above the flat 5.0 the season realized
    assert s5["market_expected"].iloc[0] > 10
    # Monotone decreasing in ADP
    assert (np.diff(s5["market_expected"].values) <= 1e-9).all()


def test_market_oos_is_recency_weighted():
    pairs = _synthetic_pairs(seasons=range(2001, 2005))
    # Recent training season (2004) shifted up by +6: with recency
    # weighting the 2005 test prediction should sit closer to the
    # recent level than the old one.
    shifted = pairs.copy()
    shifted.loc[shifted["season"] == 2004, "next_fpts"] += 6.0
    test = _synthetic_pairs(seasons=[2005])
    both = pd.concat([shifted, test], ignore_index=True)
    hi = market_expected_oos(both, min_train_seasons=2, recency=0.5)
    lo = market_expected_oos(both, min_train_seasons=2, recency=1.0)
    m_hi = hi[hi["season"] == 2005]["market_expected"].mean()
    m_lo = lo[lo["season"] == 2005]["market_expected"].mean()
    assert m_hi > m_lo  # stronger recency decay -> closer to shifted recent season


# ---------------------------------------------------------------------------
# attach_market
# ---------------------------------------------------------------------------

def test_attach_market_offset_and_stdev():
    pairs = pd.DataFrame([
        {"player_id": "A", "player_name": "Odell Beckham Jr.", "position": "WR",
         "season": 2020, "next_fpts": 15.0},
        {"player_id": "B", "player_name": "Nobody Man", "position": "RB",
         "season": 2020, "next_fpts": 5.0},
    ])
    adp = pd.DataFrame([
        {"season": 2021, "player_name": "Odell Beckham", "position": "WR",
         "adp": 24.5, "adp_pos_rank": 8, "stdev": 3.3},
        {"season": 2020, "player_name": "Odell Beckham", "position": "WR",
         "adp": 99.0, "adp_pos_rank": 40, "stdev": 9.9},
    ])
    out = attach_market(pairs, adp, season_offset=1)
    row = out[out["player_id"] == "A"].iloc[0]
    assert row["adp"] == 24.5          # joined at season+1, suffix-normalized name
    assert row["adp_stdev"] == 3.3
    assert bool(row["adp_matched"])
    assert not bool(out[out["player_id"] == "B"]["adp_matched"].iloc[0])


# ---------------------------------------------------------------------------
# Shrinkage slope recovery
# ---------------------------------------------------------------------------

def test_lambda_recovers_true_slope():
    rng = np.random.default_rng(42)
    pred = rng.normal(0, 2.0, 4000)
    realized = 0.4 * pred + rng.normal(0, 1.0, 4000)
    est = estimate_lambda(pred, realized)
    assert abs(est["slope"] - 0.4) < 0.05
    assert est["lam"] == pytest.approx(est["slope"])
    assert 0 < est["se"] < 0.05
    assert est["n"] == 4000


def test_lambda_clipping_and_degenerate():
    rng = np.random.default_rng(1)
    pred = rng.normal(0, 1, 500)
    # slope > 1 clips to 1; anti-signal clips to 0
    assert estimate_lambda(pred, 1.7 * pred)["lam"] == 1.0
    assert estimate_lambda(pred, -0.5 * pred + rng.normal(0, 0.1, 500))["lam"] == 0.0
    # too few rows -> NaN
    assert np.isnan(estimate_lambda(pred[:5], pred[:5])["lam"])
    # NaNs are dropped, not propagated
    p = pred.copy()
    p[:50] = np.nan
    assert np.isfinite(estimate_lambda(p, 0.5 * pred)["lam"])


# ---------------------------------------------------------------------------
# AlphaModel
# ---------------------------------------------------------------------------

def _alpha_frame(n=400, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos = rng.choice(["QB", "RB", "WR", "TE"], n)
    df = pd.DataFrame({
        "position": pos,
        "fund_gap": rng.normal(0, 2, n),
        "age": rng.uniform(21, 34, n),
        "log_adp": rng.uniform(0, 5.2, n),
        "adp_stdev": rng.uniform(0.5, 12, n),
        "team_vacated_target_share": rng.uniform(0, 0.4, n),
        "target_share_delta": rng.normal(0, 0.05, n),
        "fpts_pg_yoy_change": rng.normal(0, 3, n),
        "years_in_league": rng.integers(0, 12, n).astype(float),
    })
    df["residual"] = 0.6 * df["fund_gap"] + rng.normal(0, 1.5, n)
    return df


def test_alpha_model_shape_and_columns():
    df = _alpha_frame()
    model = AlphaModel(ridge_alpha=1.0).fit(df)
    pred = model.predict(df)
    assert pred.shape == (len(df),)
    assert np.isfinite(pred).all()
    # Design = declared features + position dummies (QB reference)
    assert model.feature_names_ == ALPHA_FEATURES + ["pos_RB", "pos_WR", "pos_TE"]
    coefs = model.coefficients()
    assert list(coefs.index) == model.feature_names_
    # The planted signal must dominate
    assert coefs["fund_gap"] > 0
    assert abs(coefs["fund_gap"]) == coefs.abs().max()


def test_alpha_model_handles_nans_and_small_samples():
    df = _alpha_frame()
    df.loc[df.index[:80], "adp_stdev"] = np.nan
    df.loc[df.index[10:60], "target_share_delta"] = np.nan
    model = AlphaModel().fit(df)
    holdout = _alpha_frame(n=50, seed=9)
    holdout.loc[holdout.index[:10], "age"] = np.nan
    pred = model.predict(holdout)
    assert np.isfinite(pred).all()

    with pytest.raises(ValueError):
        AlphaModel().fit(_alpha_frame(n=10))

    with pytest.raises(RuntimeError):
        AlphaModel().predict(df)
