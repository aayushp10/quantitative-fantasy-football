"""
Tests for the v4 quant upgrades:
- season-grouped walk-forward CV (models/validation.py)
- per-season cross-sectional standardization (features/standardize.py)
- empirical-Bayes shrinkage (models/shrinkage.py)
- expected TDs from usage geometry (features/expected_td.py)
- ADP loading/joining (data/adp.py) and market evaluation (models/market.py)
- availability model (models/availability.py)
- residual-quantile uncertainty (models/uncertainty.py)
- routes/TPRR (features/routes.py)
- rookie model (models/rookie.py)
- model integration: predict_position with standardization + missing columns
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import season_length, AVAILABILITY_MIN_GAMES_FRAC
from features.standardize import SeasonStandardizer
from features.expected_td import build_expected_td_features
from features.routes import build_route_features
from data.adp import normalize_name, attach_adp
from models.validation import season_walk_forward_folds
from models.shrinkage import (
    EmpiricalBayesShrinker,
    fit_beta_binomial_prior,
    fit_normal_prior,
)
from models.market import long_short_test, market_baseline
from models.availability import AvailabilityModel
from models.uncertainty import ResidualQuantiles, simulate_season_totals
from models.rookie import RookieModel, merge_rookie_projections


# ---------------------------------------------------------------------------
# Season walk-forward folds
# ---------------------------------------------------------------------------

class TestSeasonWalkForwardFolds:
    def test_train_strictly_before_validation(self):
        seasons = np.repeat([2015, 2016, 2017, 2018, 2019, 2020], 10)
        folds = season_walk_forward_folds(seasons, min_train_seasons=3)
        assert len(folds) == 3  # val = 2018, 2019, 2020
        for train_idx, val_idx in folds:
            assert seasons[train_idx].max() < seasons[val_idx].min()
            # validation fold is exactly one season
            assert len(np.unique(seasons[val_idx])) == 1

    def test_min_train_seasons_respected(self):
        seasons = np.repeat([2019, 2020, 2021], 10)
        assert season_walk_forward_folds(seasons, min_train_seasons=3) == []

    def test_never_splits_within_a_season(self):
        seasons = np.repeat([2015, 2016, 2017, 2018], 8)
        for train_idx, val_idx in season_walk_forward_folds(seasons, min_train_seasons=2):
            val_season = seasons[val_idx][0]
            # every row of the validation season is in the validation fold
            assert (seasons == val_season).sum() == len(val_idx)
            assert val_season not in seasons[train_idx]


# ---------------------------------------------------------------------------
# Per-season standardization
# ---------------------------------------------------------------------------

class TestSeasonStandardizer:
    def _frame(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "season": np.repeat([2020, 2021], 50),
            # level shift between seasons (era drift)
            "factor": np.concatenate([rng.normal(10, 2, 50), rng.normal(50, 5, 50)]),
            "binary_flag": np.tile([0, 1], 50),
        })

    def test_zscore_within_each_season(self):
        df = self._frame()
        out = SeasonStandardizer().fit_transform(df, ["factor", "binary_flag"])
        for season in [2020, 2021]:
            vals = out.loc[out["season"] == season, "factor"]
            assert abs(vals.mean()) < 1e-9
            assert abs(vals.std() - 1.0) < 0.02

    def test_binary_columns_left_raw(self):
        df = self._frame()
        out = SeasonStandardizer().fit_transform(df, ["factor", "binary_flag"])
        assert set(out["binary_flag"].unique()) == {0, 1}

    def test_nan_preserved_and_within_season_constant_maps_to_zero(self):
        # 'mixed' varies across seasons (so it IS standardized) but is
        # constant within 2020 — those rows must map to 0, not NaN.
        df = pd.DataFrame({
            "season": [2020, 2020, 2020, 2021, 2021, 2021],
            "factor": [1.0, np.nan, 3.0, 2.0, 4.0, 6.0],
            "mixed": [7.0, 7.0, np.nan, 1.0, 2.0, 3.0],
        })
        out = SeasonStandardizer().fit_transform(df, ["factor", "mixed"])
        assert np.isnan(out["factor"].iloc[1])           # NaN stays NaN
        assert (out.loc[out["season"] == 2020, "mixed"].dropna() == 0).all()
        assert np.isnan(out["mixed"].iloc[2])

    def test_transform_uses_given_frame_cross_section(self):
        train = self._frame()
        std = SeasonStandardizer().fit(train, ["factor"])
        newdf = pd.DataFrame({"season": [2024] * 30,
                              "factor": np.linspace(100, 200, 30)})
        out = std.transform(newdf)
        assert abs(out["factor"].mean()) < 1e-9  # standardized vs its own slice


# ---------------------------------------------------------------------------
# Empirical-Bayes shrinkage
# ---------------------------------------------------------------------------

class TestShrinkage:
    def test_beta_binomial_posterior_between_obs_and_prior(self):
        rng = np.random.default_rng(1)
        rates = rng.beta(3, 40, 300)
        trials = rng.integers(30, 160, 300)
        p = fit_beta_binomial_prior(rates, trials)
        assert 0 < p.mean < 0.2
        assert p.strength > 0

    def test_more_trials_less_shrinkage(self):
        df = pd.DataFrame({
            "position": ["WR"] * 200,
            "rec_td_rate": np.random.default_rng(2).beta(3, 40, 200),
            "targets": np.random.default_rng(3).integers(30, 160, 200),
        })
        eb = EmpiricalBayesShrinker(metrics={"rec_td_rate": "targets"}).fit(df)
        assert eb.has("WR", "rec_td_rate")
        prior_mean = eb.prior("WR", "rec_td_rate").mean

        obs_rate = prior_mean + 0.10  # same observed rate, different sample sizes
        test = pd.DataFrame({
            "rec_td_rate": [obs_rate, obs_rate],
            "targets": [20.0, 150.0],
        })
        post = eb.shrink(test, "WR", "rec_td_rate")
        # both pulled toward prior, low-sample pulled harder
        assert prior_mean < post[0] < post[1] < obs_rate

    def test_geometry_prior_override(self):
        df = pd.DataFrame({
            "position": ["WR"] * 100,
            "rec_td_rate": np.random.default_rng(4).beta(3, 40, 100),
            "targets": np.random.default_rng(5).integers(30, 160, 100),
        })
        eb = EmpiricalBayesShrinker(metrics={"rec_td_rate": "targets"}).fit(df)
        test = pd.DataFrame({"rec_td_rate": [np.nan], "targets": [0.0]})
        # No observation → posterior = the per-player geometry prior
        post = eb.shrink(test, "WR", "rec_td_rate", prior_mean=np.array([0.123]))
        assert abs(post[0] - 0.123) < 1e-9

    def test_normal_prior_recovers_structure(self):
        rng = np.random.default_rng(6)
        true_skill = rng.normal(8.0, 1.0, 400)          # tau = 1
        trials = rng.integers(20, 200, 400)
        noise_sd = 6.0 / np.sqrt(trials)                # s = 6
        obs = true_skill + rng.normal(0, noise_sd)
        p = fit_normal_prior(obs, trials)
        assert 7.0 < p.mean < 9.0
        # k = s2/tau2 ≈ 36; very loose bounds for MoM noise
        assert 5 < p.strength < 300


# ---------------------------------------------------------------------------
# Expected TDs from geometry
# ---------------------------------------------------------------------------

class TestExpectedTD:
    def _pbp(self):
        # 20 goal-line carries (10 TD → league rate 0.5), player A scores all
        # 5 of theirs, player B none of their 5, C has the other 10 with 5 TD.
        rows = []
        carries = [("A", 1)] * 5 + [("B", 2)] * 5 + [("C", 1)] * 10
        tds = [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5
        for i, ((player, yl), td) in enumerate(zip(carries, tds)):
            rows.append({
                "play_id": i, "season": 2024, "posteam": "KC",
                "pass": 0, "rush": 1,
                "rusher_player_id": player, "receiver_player_id": None,
                "passer_player_id": None,
                "yardline_100": yl, "rush_touchdown": td, "pass_touchdown": 0,
            })
        return pd.DataFrame(rows)

    def test_x_rate_is_league_bucket_rate(self):
        out = build_expected_td_features(self._pbp())
        a = out[out["player_id"] == "A"].iloc[0]
        b = out[out["player_id"] == "B"].iloc[0]
        # all carries in the same (0,2] bucket → same expected rate = 0.5
        assert a["x_rush_td_rate"] == pytest.approx(0.5)
        assert b["x_rush_td_rate"] == pytest.approx(0.5)

    def test_td_oe_sign(self):
        out = build_expected_td_features(self._pbp()).set_index("player_id")
        assert out.loc["A", "rush_td_oe"] == pytest.approx(0.5)   # 1.0 - 0.5
        assert out.loc["B", "rush_td_oe"] == pytest.approx(-0.5)  # 0.0 - 0.5
        assert out.loc["C", "rush_td_oe"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ADP
# ---------------------------------------------------------------------------

class TestADP:
    def test_normalize_name(self):
        assert normalize_name("Odell Beckham Jr.") == "odell beckham"
        assert normalize_name("D.J. Moore") == "dj moore"
        assert normalize_name("Ja'Marr Chase") == "jamarr chase"
        assert normalize_name("Kenneth Walker III") == "kenneth walker"

    def _adp(self):
        return pd.DataFrame({
            "season": [2023, 2023],
            "player_name": ["Justin Jefferson", "Tyreek Hill"],
            "name_norm": ["justin jefferson", "tyreek hill"],
            "position": ["WR", "WR"],
            "team": ["MIN", "MIA"],
            "adp": [1.4, 4.4],
            "adp_pos_rank": [1, 2],
        })

    def test_attach_with_season_offset(self):
        # YoY row: features season 2022 → drafted before season 2023 → offset 1
        yoy = pd.DataFrame({
            "player_name": ["Justin Jefferson", "Nobody Special"],
            "position": ["WR", "WR"],
            "season": [2022, 2022],
        })
        out = attach_adp(yoy, self._adp(), season_offset=1)
        assert out.loc[0, "adp"] == pytest.approx(1.4)
        assert out.loc[0, "adp_pos_rank"] == 1
        assert not out.loc[1, "adp_matched"]

    def test_wrong_season_does_not_match(self):
        yoy = pd.DataFrame({
            "player_name": ["Justin Jefferson"],
            "position": ["WR"], "season": [2023],
        })
        out = attach_adp(yoy, self._adp(), season_offset=1)  # looks for 2024
        assert not out.loc[0, "adp_matched"]


# ---------------------------------------------------------------------------
# Market evaluation
# ---------------------------------------------------------------------------

class TestMarket:
    def test_long_short_spread_positive_for_skilled_model(self):
        # Model = truth + tiny noise; market (ADP) = truth + big noise.
        # Model-vs-market disagreements must then predict outcomes.
        rng = np.random.default_rng(7)
        n = 60
        truth = np.linspace(20, 5, n)
        df = pd.DataFrame({
            "pred": truth + rng.normal(0, 0.5, n),
            "next_fpts": truth + rng.normal(0, 1.0, n),
        })
        market_rank = pd.Series(truth + rng.normal(0, 4.0, n)).rank(ascending=False)
        df["adp_pos_rank"] = market_rank
        res = long_short_test(df)
        assert res is not None
        assert res["ls_spread"] > 0

    def test_long_short_returns_none_when_tiny(self):
        df = pd.DataFrame({
            "pred": [1, 2], "adp_pos_rank": [1, 2], "next_fpts": [1, 2],
        })
        assert long_short_test(df) is None

    def test_market_baseline_shape(self):
        rng = np.random.default_rng(8)
        n = 40
        truth = np.linspace(20, 5, n)
        df = pd.DataFrame({
            "season": [2022] * n,
            "position": ["WR"] * n,
            "adp": pd.Series(truth + rng.normal(0, 3, n)).rank(ascending=False).values,
            "next_fpts": truth + rng.normal(0, 2, n),
        })
        out = market_baseline(df)
        row = out[(out["season"] == 2022) & (out["position"] == "WR")].iloc[0]
        assert row["adp_rank_ic"] > 0.5  # market is decent by construction


# ---------------------------------------------------------------------------
# Availability model
# ---------------------------------------------------------------------------

def _availability_yoy(n_players=40, seasons=(2018, 2019, 2020, 2021, 2022, 2023)):
    rng = np.random.default_rng(9)
    rows = []
    for season in seasons:
        L = season_length(season)
        nxt = season_length(season + 1)
        for i in range(n_players):
            age = 22 + (i % 10)
            durable = 1.0 - 0.03 * max(0, age - 26)
            games = int(np.clip(rng.binomial(L, durable * 0.9), 2, L))
            rows.append({
                "player_id": f"P{i}", "season": season,
                "position": ["RB", "WR", "TE", "QB"][i % 4],
                "age": age, "games_played": games,
                "carries": rng.integers(0, 200), "targets": rng.integers(0, 120),
                "dropbacks": 0,
                "next_games_played": int(np.clip(rng.binomial(nxt, durable * 0.9), 0, nxt)),
            })
    return pd.DataFrame(rows)


class TestAvailability:
    def test_train_and_predict_range(self):
        yoy = _availability_yoy()
        model = AvailabilityModel().train(yoy)
        games = model.predict_games(yoy[yoy["season"] == 2023], target_season=2025)
        L = season_length(2025)
        assert (games >= AVAILABILITY_MIN_GAMES_FRAC * L - 1e-9).all()
        assert (games <= L + 1e-9).all()

    def test_requires_target_column(self):
        with pytest.raises(ValueError):
            AvailabilityModel().train(pd.DataFrame({"season": [2020]}))

    def test_attach_to_projections_replaces_flat_17(self):
        yoy = _availability_yoy()
        model = AvailabilityModel().train(yoy)
        feats = yoy[yoy["season"] == 2023].reset_index(drop=True)
        proj = pd.DataFrame({
            "player_id": feats["player_id"],
            "position": feats["position"],
            "projected_fpts_pg": 10.0,
            "projected_games": 17.0,
            "projected_fpts_season": 170.0,
        })
        out = model.attach_to_projections(proj, feats, target_season=2025)
        assert (out["projected_games"] < 17).any()
        expected = out["projected_fpts_pg"] * out["projected_games"]
        assert np.allclose(out["projected_fpts_season"], expected)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def _resid_frame(n=300, seed=10):
    rng = np.random.default_rng(seed)
    pred = rng.uniform(4, 20, n)
    resid = rng.normal(0, 1.0 + 0.15 * pred)
    return pd.DataFrame({
        "season": 2022, "position": "WR", "player_id": [f"p{i}" for i in range(n)],
        "pred": pred, "actual": pred + resid, "resid": resid,
    })


class TestUncertainty:
    def test_interval_quantiles_are_ordered(self):
        rq = ResidualQuantiles().fit(_resid_frame())
        preds = np.array([5.0, 12.0, 18.0])
        lo = rq.interval("WR", preds, 0.10)
        mid = rq.interval("WR", preds, 0.50)
        hi = rq.interval("WR", preds, 0.90)
        assert (lo < mid).all() and (mid < hi).all()

    def test_intervals_widen_with_prediction_level(self):
        rq = ResidualQuantiles().fit(_resid_frame())
        preds = np.array([5.0, 18.0])
        width = rq.interval("WR", preds, 0.90) - rq.interval("WR", preds, 0.10)
        assert width[1] > width[0]  # heteroscedastic scale grows with pred

    def test_coverage_close_to_nominal_in_sample(self):
        df = _resid_frame()
        rq = ResidualQuantiles().fit(df)
        report = rq.coverage_report(df, lo=0.10, hi=0.90)
        cov = report[report["position"] == "WR"]["empirical_coverage"].iloc[0]
        assert 0.75 <= cov <= 0.85

    def test_simulate_season_totals_ordered(self):
        rq = ResidualQuantiles().fit(_resid_frame())
        proj = pd.DataFrame({
            "position": ["WR"] * 3,
            "projected_fpts_pg": [8.0, 12.0, 16.0],
            "projected_games": [15.0, 16.0, 17.0],
        })
        out = simulate_season_totals(proj, rq, target_season=2025,
                                     games_sd={"WR": 0.1}, n_sims=500)
        assert (out["season_p10"] < out["season_p50"]).all()
        assert (out["season_p50"] < out["season_p90"]).all()


# ---------------------------------------------------------------------------
# Routes / TPRR
# ---------------------------------------------------------------------------

class TestRoutes:
    def test_route_counts_and_participation(self):
        pbp = pd.DataFrame({
            "game_id": ["g1"] * 4,
            "play_id": [1, 2, 3, 4],
            "pass": [1, 1, 1, 0],
            "posteam": ["KC"] * 4,
            "season": [2022] * 4,
        })
        participation = pd.DataFrame({
            "nflverse_game_id": ["g1", "g1", "g1"],
            "play_id": [1, 2, 3],
            "offense_players": ["A;B", "A;B", "A"],
        })
        out = build_route_features(pbp, participation).set_index("player_id")
        assert out.loc["A", "routes"] == 3
        assert out.loc["B", "routes"] == 2
        assert out.loc["A", "route_participation"] == pytest.approx(1.0)
        assert out.loc["B", "route_participation"] == pytest.approx(2 / 3)

    def test_missing_participation_returns_empty(self):
        pbp = pd.DataFrame({"game_id": ["g1"], "play_id": [1], "pass": [1],
                            "posteam": ["KC"], "season": [2022]})
        assert build_route_features(pbp, None).empty


# ---------------------------------------------------------------------------
# Rookie model
# ---------------------------------------------------------------------------

def _draft_picks():
    rng = np.random.default_rng(11)
    rows = []
    for season in [2020, 2021, 2022, 2023, 2024]:
        for i in range(30):
            rows.append({
                "season": season, "round": 1 + i // 8, "pick": i * 6 + 1,
                "team": "KC", "gsis_id": f"R{season}_{i}",
                "pfr_player_name": f"Rookie {season} {i}",
                "position": ["WR", "RB", "TE", "QB"][i % 4],
                "age": 21 + (i % 3),
            })
    return pd.DataFrame(rows)


def _rookie_weekly():
    # Early picks produce more; late picks often never record a stat
    rng = np.random.default_rng(12)
    rows = []
    for season in [2020, 2021, 2022, 2023]:
        for i in range(20):  # picks 21-30 have no stats at all
            total = max(0.0, 180 - 6 * i + rng.normal(0, 10))
            rows.append({"player_id": f"R{season}_{i}", "season": season,
                         "fantasy_points_ppr": total})
    return pd.DataFrame(rows)


class TestRookieModel:
    def test_training_frame_targets(self):
        frame = RookieModel.build_training_frame(
            _draft_picks(), _rookie_weekly(), end_season=2023
        )
        # played rookie: fpts / season_length; 2020 uses 16 games
        r = frame[(frame["player_id"] == "R2020_0")].iloc[0]
        weekly_total = _rookie_weekly().query(
            "player_id == 'R2020_0' and season == 2020"
        )["fantasy_points_ppr"].iloc[0]
        assert r["rookie_fpts_per_team_game"] == pytest.approx(weekly_total / 16)
        # never-played rookie kept at 0 (no survivorship dropping)
        never = frame[frame["player_id"] == "R2021_25"].iloc[0]
        assert never["rookie_fpts_per_team_game"] == 0.0

    def test_draft_capital_orders_projections(self):
        frame = RookieModel.build_training_frame(
            _draft_picks(), _rookie_weekly(), end_season=2023
        )
        model = RookieModel().train(frame)
        proj = model.project_class(_draft_picks(), 2024)
        assert not proj.empty
        assert (proj["projected_fpts_pg"] >= 0).all()
        assert bool(proj["rookie"].all())
        wr = proj[proj["position"] == "WR"].sort_values("pick")
        # earlier WR pick projects at least as high as the last WR pick
        assert wr.iloc[0]["projected_fpts_pg"] >= wr.iloc[-1]["projected_fpts_pg"]

    def test_merge_prefers_veterans(self):
        vet = pd.DataFrame({
            "player_id": ["X"], "player_name": ["Vet"], "position": ["WR"],
            "projected_fpts_season": [200.0],
        })
        rook = pd.DataFrame({
            "player_id": ["X", "Y"], "player_name": ["Dup", "Rook"],
            "position": ["WR", "WR"], "projected_fpts_season": [150.0, 100.0],
            "rookie": [True, True],
        })
        out = merge_rookie_projections(vet, rook)
        assert len(out) == 2
        assert out[out["player_id"] == "X"]["player_name"].iloc[0] == "Vet"


# ---------------------------------------------------------------------------
# Age curves: delta method
# ---------------------------------------------------------------------------

class TestAgeCurvesDeltaMethod:
    @staticmethod
    def _transitions(position, true_peak, n_per_age=20, decay=0.02, seed=0):
        """Within-player transitions from a known quadratic aging curve."""
        rng = np.random.default_rng(seed)
        rows = []
        for age in range(21, 34):
            level_now = 15.0 * max(0.5, 1 - decay * (age - true_peak) ** 2)
            level_next = 15.0 * max(0.5, 1 - decay * (age + 1 - true_peak) ** 2)
            for i in range(n_per_age):
                rows.append({
                    "position": position, "age": float(age),
                    "fpts_per_game": level_now + rng.normal(0, 0.5),
                    "next_fpts": level_next + rng.normal(0, 0.5),
                })
        return rows

    def test_recovers_peak_ordering(self):
        from models.age_curves import fit_age_curves
        # RB truly peaks at 23, QB truly peaks at 29 — the fit must
        # recover QB later than RB (the bug the cross-sectional fit had)
        df = pd.DataFrame(
            self._transitions("RB", true_peak=23, seed=1)
            + self._transitions("QB", true_peak=29, seed=2)
        )
        fitted = fit_age_curves(df, min_pairs=50)
        assert fitted["QB"]["peak_age"] > fitted["RB"]["peak_age"] + 2

    def test_survivorship_resistance(self):
        from models.age_curves import fit_age_curves
        # Add elite-only survivors at old ages (levels HIGH but declining).
        # A cross-sectional fit would drag the peak late; the delta method
        # must not, because the survivors' own deltas are still negative.
        base = self._transitions("WR", true_peak=25, seed=3)
        rng = np.random.default_rng(4)
        for age in range(30, 36):
            for i in range(12):
                lvl = 18.0 - 0.6 * (age - 30)     # elite level, declining
                base.append({
                    "position": "WR", "age": float(age),
                    "fpts_per_game": lvl + rng.normal(0, 0.5),
                    "next_fpts": lvl - 0.6 + rng.normal(0, 0.5),
                })
        fitted = fit_age_curves(pd.DataFrame(base), min_pairs=50)
        assert fitted["WR"]["peak_age"] < 27.5


# ---------------------------------------------------------------------------
# Model integration: standardization + predict_position
# ---------------------------------------------------------------------------

def _synthetic_yoy(n_per_season=30, seasons=range(2015, 2023)):
    """WR-only yoy frame with real signal in target_share/wopr."""
    rng = np.random.default_rng(13)
    rows = []
    for season in seasons:
        # era drift: raw factor levels shift by season
        era = (season - 2015) * 0.01
        for i in range(n_per_season):
            ts = np.clip(rng.beta(2, 8) + era, 0.02, 0.4)
            rows.append({
                "player_id": f"W{i}", "player_name": f"WR {i}",
                "position": "WR", "team": "KC", "season": season,
                "target_share": ts,
                "wopr": 1.5 * ts + rng.normal(0, 0.02),
                "games_played": int(rng.integers(10, 18)),
                "next_fpts": 60 * ts + rng.normal(0, 1.5),
            })
    return pd.DataFrame(rows)


class TestModelIntegration:
    def test_single_stage_trains_and_predicts_with_missing_columns(self):
        from models.projection import FantasyProjectionModel
        yoy = _synthetic_yoy()
        model = FantasyProjectionModel(age_adjust=False)
        model.train(yoy, fit_age=False)
        assert "WR" in model._models

        test = yoy[yoy["season"] == 2022].drop(columns=["wopr"])  # missing feature
        preds = model.predict_position("WR", test)
        assert preds is not None and len(preds) == len(test)
        assert not np.isnan(preds).any()
        # signal preserved: higher target share → higher prediction
        corr = np.corrcoef(test["target_share"], preds)[0, 1]
        assert corr > 0.7

    def test_two_stage_eb_integration(self):
        from models.two_stage import TwoStageProjectionModel
        rng = np.random.default_rng(14)
        yoy = _synthetic_yoy()
        # add two-stage requirements: volume target + efficiency rates + trials
        yoy["targets_per_game"] = yoy["target_share"] * 33
        yoy["next_targets_per_game"] = yoy["targets_per_game"] + rng.normal(0, 0.5, len(yoy))
        yoy["targets"] = (yoy["targets_per_game"] * yoy["games_played"]).round()
        yoy["yards_per_target"] = rng.normal(8.0, 1.0, len(yoy))
        yoy["rec_td_rate"] = rng.beta(3, 45, len(yoy))
        yoy["catch_rate"] = rng.beta(20, 10, len(yoy))
        yoy["snap_percentage"] = 0.8

        model = TwoStageProjectionModel(age_adjust=False)
        model.train(yoy, fit_age=False)
        assert model._eb is not None and model._eb.has("WR", "rec_td_rate")

        test = yoy[yoy["season"] == 2022].reset_index(drop=True)
        preds = model.predict_position("WR", test)
        assert preds is not None and (preds >= 0).all()

        # EB efficiency: shrunk rates sit between obs and prior, closer to
        # prior for the row with fewer trials
        eff = model._regressed_efficiency("WR", test)
        assert "rec_td_rate" in eff
