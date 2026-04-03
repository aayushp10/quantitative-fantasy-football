"""
Hybrid projection model: blends single-stage and two-stage predictions.

The single-stage Ridge captures holistic patterns well (overall R²=0.471).
The fixed two-stage model captures structural volume × efficiency decomposition
with stable regressed efficiency and proper TD mean reversion.

Blending the two should outperform either alone — each model has different
failure modes that partially cancel out in the blend.

The blend weight can be optimized via optimize_blend_weight() using the
rolling backtest framework.

Public API
----------
Same as FantasyProjectionModel: train(), project(), backtest().
Additional: optimize_blend_weight(yoy_df, test_seasons) -> float
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

from config import DEFAULT_BLEND_WEIGHT, POSITIONS
from models.projection import FantasyProjectionModel
from models.two_stage import TwoStageProjectionModel, ALL_RATE_TARGET_COLS


class HybridProjectionModel:
    """
    Blended projection: blend_weight × single_stage + (1-blend_weight) × two_stage.

    Players present in only one model's output get 100% of that model.
    """

    def __init__(self, blend_weight: float = DEFAULT_BLEND_WEIGHT, age_adjust: bool = True):
        """
        Parameters
        ----------
        blend_weight : float
            Weight allocated to the single-stage model (0 to 1).
            1 - blend_weight goes to the two-stage model.
            Default 0.55 gives a slight preference to the single-stage model.
        age_adjust : bool
            Whether to apply age curve adjustments in both sub-models.
        """
        self.blend_weight = blend_weight
        self.age_adjust = age_adjust
        self._single = FantasyProjectionModel(age_adjust=age_adjust)
        self._two_stage = TwoStageProjectionModel(age_adjust=age_adjust)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        yoy_df: pd.DataFrame,
        target: str = "next_fpts",
        fit_age: bool = True,
    ) -> "HybridProjectionModel":
        """
        Train both internal models on the same data.

        The two-stage model uses use_ridge_efficiency=False (regressed historical rates).
        """
        print("Training single-stage model...")
        self._single.train(yoy_df, target=target, fit_age=fit_age)

        print("Training two-stage model (regressed efficiency)...")
        self._two_stage.train(yoy_df, target=target, fit_age=fit_age, use_ridge_efficiency=False)

        return self

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(
        self,
        features_df: pd.DataFrame,
        season: int,
        projected_games: int = 17,
    ) -> pd.DataFrame:
        """
        Generate blended projections.

        Both models project independently; their fpts_per_game predictions are
        blended. Players missing from one model get 100% of the other.
        """
        single_proj = pd.DataFrame()
        two_proj = pd.DataFrame()

        try:
            single_proj = self._single.project(features_df, season, projected_games)
        except Exception as e:
            warnings.warn(f"Single-stage projection failed: {e}")

        try:
            two_proj = self._two_stage.project(features_df, season, projected_games)
        except Exception as e:
            warnings.warn(f"Two-stage projection failed: {e}")

        if single_proj.empty and two_proj.empty:
            return pd.DataFrame()
        if single_proj.empty:
            return two_proj
        if two_proj.empty:
            return single_proj

        # Merge on player_id to blend
        merged = single_proj.merge(
            two_proj[["player_id", "projected_fpts_pg"]].rename(
                columns={"projected_fpts_pg": "two_stage_fpts_pg"}
            ),
            on="player_id",
            how="outer",
        )

        # Blend: use both where available, fall back to whichever is present
        has_single = merged["projected_fpts_pg"].notna()
        has_two = merged["two_stage_fpts_pg"].notna()

        merged["blended_fpts_pg"] = np.where(
            has_single & has_two,
            self.blend_weight * merged["projected_fpts_pg"] + (1 - self.blend_weight) * merged["two_stage_fpts_pg"],
            np.where(has_single, merged["projected_fpts_pg"], merged["two_stage_fpts_pg"]),
        )

        merged["projected_fpts_pg"] = merged["blended_fpts_pg"]
        merged["projected_fpts_season"] = merged["projected_fpts_pg"] * projected_games
        merged = merged.drop(columns=["two_stage_fpts_pg", "blended_fpts_pg"], errors="ignore")

        return merged.sort_values("projected_fpts_season", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def backtest(
        self,
        yoy_df: pd.DataFrame,
        test_season: int,
        target: str = "next_fpts",
    ) -> dict[str, Any]:
        """
        Backtest all three outputs: single-stage, two-stage, and hybrid blend.

        Returns
        -------
        dict
            Keys: 'single_stage', 'two_stage', 'hybrid'.
            Each maps to {position: {mae, r2, rank_corr, n}} plus 'overall'.
        """
        train_df = yoy_df[yoy_df["season"] < test_season].copy()
        test_df = yoy_df[yoy_df["season"] == test_season].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(f"Not enough data to backtest season {test_season}.")

        # Train fresh models for backtest (no age adjustment for fairness)
        bt_single = FantasyProjectionModel(age_adjust=False)
        bt_single.train(train_df, target=target, fit_age=False)

        bt_two = TwoStageProjectionModel(age_adjust=False)
        bt_two.train(train_df, target=target, fit_age=False, use_ridge_efficiency=False)

        results = {
            "single_stage": bt_single.backtest(yoy_df, test_season, target),
            "two_stage": bt_two.backtest(yoy_df, test_season, target),
        }

        # Compute hybrid blend manually
        from config import POSITION_FEATURES
        hybrid_result: dict[str, Any] = {}
        all_pred, all_actual = [], []

        for pos in POSITIONS:
            pos_test = test_df[test_df["position"] == pos].copy().reset_index(drop=True) \
                if "position" in test_df.columns else test_df.copy().reset_index(drop=True)
            if pos_test.empty:
                continue

            y_actual = pos_test[target].values if target in pos_test.columns else None
            if y_actual is None:
                continue

            # Single-stage predictions
            features = [f for f in POSITION_FEATURES.get(pos, []) if f in pos_test.columns]
            single_pred = None
            if features and pos in bt_single._models:
                try:
                    single_pred = bt_single._models[pos].predict(pos_test[features].values)
                except Exception:
                    pass

            # Two-stage predictions
            two_pred = None
            try:
                n = len(pos_test)
                vol_p = bt_two._predict_volume(pos, pos_test)
                eff_p = bt_two._regressed_efficiency(pos, pos_test)
                if vol_p or eff_p:
                    cr = bt_two._get_catch_rate(pos, pos_test)
                    two_pred = bt_two._combine_to_fpts(pos, vol_p, eff_p, cr, n)
            except Exception:
                pass

            if single_pred is None and two_pred is None:
                continue

            # Blend
            if single_pred is None:
                y_pred = two_pred
            elif two_pred is None:
                y_pred = single_pred
            else:
                y_pred = self.blend_weight * single_pred + (1 - self.blend_weight) * two_pred

            valid = ~np.isnan(y_actual) & ~np.isnan(y_pred)
            if valid.sum() < 3:
                continue

            mae = mean_absolute_error(y_actual[valid], y_pred[valid])
            r2 = r2_score(y_actual[valid], y_pred[valid])
            rank_corr, _ = spearmanr(y_actual[valid], y_pred[valid])

            hybrid_result[pos] = {
                "mae": round(float(mae), 3),
                "r2": round(float(r2), 3),
                "rank_corr": round(float(rank_corr), 3),
                "n": int(valid.sum()),
            }
            all_pred.extend(y_pred[valid].tolist())
            all_actual.extend(y_actual[valid].tolist())

        if all_pred:
            overall_mae = mean_absolute_error(all_actual, all_pred)
            overall_r2 = r2_score(all_actual, all_pred)
            overall_rc, _ = spearmanr(all_actual, all_pred)
            hybrid_result["overall"] = {
                "mae": round(float(overall_mae), 3),
                "r2": round(float(overall_r2), 3),
                "rank_corr": round(float(overall_rc), 3),
                "n": len(all_pred),
            }

        results["hybrid"] = hybrid_result
        return results

    # ------------------------------------------------------------------
    # Blend weight optimization
    # ------------------------------------------------------------------

    def optimize_blend_weight(
        self,
        yoy_df: pd.DataFrame,
        test_seasons: list[int] | None = None,
        target: str = "next_fpts",
        grid: list[float] | None = None,
    ) -> float:
        """
        Grid search over blend weights to minimize average MAE across test seasons.

        Trains both sub-models once per test season, then evaluates all blend
        weights without re-training (just re-blending predictions).

        Parameters
        ----------
        yoy_df : pd.DataFrame
            Full YoY pairs.
        test_seasons : list[int]
            Seasons to hold out. Default [2021, 2022, 2023].
        grid : list[float]
            Blend weight candidates. Default [0.0, 0.1, ..., 1.0].

        Returns
        -------
        float
            The optimal blend weight (also stored in self.blend_weight).
        """
        if test_seasons is None:
            test_seasons = [2021, 2022, 2023]
        if grid is None:
            grid = [round(w * 0.1, 1) for w in range(11)]  # 0.0, 0.1, ..., 1.0

        from config import POSITION_FEATURES

        # Pre-compute single-stage and two-stage predictions for each test season
        # to avoid re-training for each blend weight
        season_preds: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        # {test_season: {pos: (single_pred, two_pred, y_actual)}}

        for test_season in test_seasons:
            train_df = yoy_df[yoy_df["season"] < test_season]
            test_df = yoy_df[yoy_df["season"] == test_season]
            if train_df.empty or test_df.empty or len(train_df["season"].unique()) < 2:
                continue

            bt_single = FantasyProjectionModel(age_adjust=False)
            bt_two = TwoStageProjectionModel(age_adjust=False)
            try:
                bt_single.train(train_df, target=target, fit_age=False)
                bt_two.train(train_df, target=target, fit_age=False, use_ridge_efficiency=False)
            except Exception as e:
                warnings.warn(f"Training failed for test_season={test_season}: {e}")
                continue

            pos_preds = {}
            for pos in POSITIONS:
                pos_test = test_df[test_df["position"] == pos].copy().reset_index(drop=True) \
                    if "position" in test_df.columns else test_df.copy().reset_index(drop=True)
                if pos_test.empty or target not in pos_test.columns:
                    continue

                y_actual = pos_test[target].values

                features = [f for f in POSITION_FEATURES.get(pos, []) if f in pos_test.columns]
                single_pred = None
                if features and pos in bt_single._models:
                    try:
                        single_pred = bt_single._models[pos].predict(pos_test[features].values)
                    except Exception:
                        pass

                two_pred = None
                try:
                    n = len(pos_test)
                    vp = bt_two._predict_volume(pos, pos_test)
                    ep = bt_two._regressed_efficiency(pos, pos_test)
                    if vp or ep:
                        cr = bt_two._get_catch_rate(pos, pos_test)
                        two_pred = bt_two._combine_to_fpts(pos, vp, ep, cr, n)
                except Exception:
                    pass

                if single_pred is not None or two_pred is not None:
                    pos_preds[pos] = (single_pred, two_pred, y_actual)

            if pos_preds:
                season_preds[test_season] = pos_preds

        if not season_preds:
            warnings.warn("No valid test seasons for blend optimization.")
            return self.blend_weight

        # Evaluate each blend weight
        best_w = self.blend_weight
        best_mae = float("inf")

        for w in grid:
            maes = []
            for ts, pos_preds in season_preds.items():
                for pos, (single_p, two_p, y_actual) in pos_preds.items():
                    if single_p is None and two_p is None:
                        continue
                    if single_p is None:
                        y_pred = two_p
                    elif two_p is None:
                        y_pred = single_p
                    else:
                        y_pred = w * single_p + (1 - w) * two_p

                    valid = ~np.isnan(y_actual) & ~np.isnan(y_pred)
                    if valid.sum() >= 3:
                        maes.append(mean_absolute_error(y_actual[valid], y_pred[valid]))

            if maes:
                avg_mae = float(np.mean(maes))
                if avg_mae < best_mae:
                    best_mae = avg_mae
                    best_w = w

        print(f"Optimal blend weight: {best_w:.1f} (avg MAE={best_mae:.3f})")
        self.blend_weight = best_w
        return best_w
