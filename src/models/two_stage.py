"""
Two-stage volume × efficiency projection model.

Stage 1 (Volume): Predict per-game opportunity volume.
  - Targets per game (WR/TE/RB)
  - Carries per game (RB)
  - Dropbacks per game (QB)

Stage 2 (Efficiency): Predict per-play output rates.
  - yards_per_target, rec_td_rate (WR/TE/RB receiving)
  - ypc, rush_td_rate (RB rushing)
  - pass_yards_per_attempt, pass_td_rate (QB)

Combination: volume × efficiency → fpts_per_game

Key design choices:
- Volume models use LOW alpha (persistent signals: target_share IC≈0.30)
- Efficiency models use HIGH alpha (noisy signals: EPA per target IC≈0.10)
- TD rates regressed 55% toward positional mean to correct for luck
- Catch rate regressed 30% toward positional mean

Public API is identical to FantasyProjectionModel:
  model.train(yoy_df)
  model.project(features_df, season=2025)
  model.backtest(yoy_df, test_season=2023)
  model.feature_importance(position)
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from config import (
    CATCH_RATE_REGRESSION_WEIGHT,
    CV_GAP,
    CV_N_SPLITS,
    EFFICIENCY_FEATURES,
    EFFICIENCY_REGRESSION_WEIGHTS,
    EFFICIENCY_RIDGE_ALPHA_GRID,
    POSITIONS,
    PPR_SCORING,
    PROJECTION_CAPS,
    RECENCY_DECAY,
    TD_REGRESSION_WEIGHT,
    VOLUME_FEATURES,
    VOLUME_RIDGE_ALPHA_GRID,
)
from features.assembler import build_yoy_pairs
from models.age_curves import apply_age_adjustments, fit_age_curves


# ---------------------------------------------------------------------------
# Volume and efficiency target column names per position
# ---------------------------------------------------------------------------

VOLUME_TARGETS: dict[str, list[str]] = {
    "QB": ["dropbacks_per_game"],
    "RB": ["carries_per_game", "targets_per_game"],
    "WR": ["targets_per_game"],
    "TE": ["targets_per_game"],
}

EFFICIENCY_TARGETS: dict[str, list[str]] = {
    "QB": ["pass_yards_per_attempt", "pass_td_rate"],
    "RB": ["ypc", "yards_per_target", "rec_td_rate", "rush_td_rate"],
    "WR": ["yards_per_target", "rec_td_rate"],
    "TE": ["yards_per_target", "rec_td_rate"],
}

# All target columns that need to be shifted in build_yoy_pairs
ALL_RATE_TARGET_COLS: list[str] = [
    "targets_per_game", "carries_per_game", "dropbacks_per_game",
    "yards_per_target", "ypc", "rec_td_rate", "rush_td_rate",
    "pass_yards_per_attempt", "pass_td_rate",
    "catch_rate",
]


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _build_pipeline(alpha_grid: list[float], n_splits: int) -> GridSearchCV | Pipeline:
    """Build a GridSearchCV pipeline with the given alpha grid."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])
    if n_splits < 2:
        return pipe
    cv = TimeSeriesSplit(n_splits=n_splits, gap=CV_GAP)
    return GridSearchCV(
        pipe,
        {"ridge__alpha": alpha_grid},
        cv=cv,
        scoring="neg_mean_absolute_error",
        refit=True,
    )


def _compute_sample_weights(seasons: pd.Series, max_season: int) -> np.ndarray:
    return np.power(RECENCY_DECAY, (max_season - seasons.values)).astype(float)


def _fit_model(
    pos_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    alpha_grid: list[float],
    max_season: int,
    label: str = "",
) -> tuple[Any | None, float | None]:
    """
    Fit a single Ridge model for a position-target pair.
    Returns (fitted_model_or_pipeline, best_alpha) or (None, None) on failure.
    """
    avail_features = [f for f in feature_cols if f in pos_df.columns]
    avail_target = target_col in pos_df.columns

    if not avail_features or not avail_target:
        return None, None

    df = pos_df[avail_features + [target_col, "season"]].dropna(subset=[target_col])
    if len(df) < 5:
        return None, None

    df = df.sort_values("season")
    X = df[avail_features].values
    y = df[target_col].values
    weights = _compute_sample_weights(df["season"], max_season)

    n_splits = min(CV_N_SPLITS, len(df) // 2)
    model = _build_pipeline(alpha_grid, n_splits)

    try:
        if isinstance(model, GridSearchCV):
            try:
                model.fit(X, y, ridge__sample_weight=weights)
            except TypeError:
                model.fit(X, y)
            best_alpha = model.best_params_["ridge__alpha"]
            fitted = model.best_estimator_
        else:
            model.fit(X, y, ridge__sample_weight=weights)
            fitted = model
            best_alpha = alpha_grid[len(alpha_grid) // 2]  # middle of grid
    except Exception as e:
        warnings.warn(f"Model fitting failed for {label}: {e}")
        return None, None

    return fitted, best_alpha


# ---------------------------------------------------------------------------
# TwoStageProjectionModel
# ---------------------------------------------------------------------------

class TwoStageProjectionModel:
    """
    Volume × efficiency decomposition projection model.

    Decomposes fantasy point prediction into:
      1. Volume stage: how many targets/carries/dropbacks?
      2. Efficiency stage: how much output per opportunity?

    Applies TD rate and catch rate mean reversion to reduce noise.

    Public API is identical to FantasyProjectionModel.
    """

    def __init__(self, age_adjust: bool = True):
        self.age_adjust = age_adjust
        # {pos: {target_col: fitted_pipeline}}
        self._volume_models: dict[str, dict[str, Any]] = {}
        self._efficiency_models: dict[str, dict[str, Any]] = {}
        # Mean TD rates and catch rates computed from training data
        self._mean_td_rates: dict[str, dict[str, float]] = {}
        self._mean_catch_rates: dict[str, float] = {}
        # Positional medians for all efficiency targets (for regressed efficiency)
        self._positional_means: dict[str, dict[str, float]] = {}
        self._fitted_age_params: dict | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        yoy_df: pd.DataFrame,
        target: str = "next_fpts",
        fit_age: bool = True,
        use_ridge_efficiency: bool = False,
    ) -> "TwoStageProjectionModel":
        """
        Fit volume and efficiency models for all positions.

        Parameters
        ----------
        yoy_df : pd.DataFrame
            YoY pairs. Must contain all VOLUME_FEATURES, EFFICIENCY_FEATURES,
            and 'next_{target_col}' columns (created by build_yoy_pairs with
            extra_target_cols=ALL_RATE_TARGET_COLS).
        use_ridge_efficiency : bool
            If True, fit Ridge models for the efficiency stage (noisy, not recommended).
            If False (default), use regressed historical rates — much more stable.
        """
        if "position" not in yoy_df.columns:
            raise ValueError("yoy_df must have a 'position' column.")

        if fit_age and self.age_adjust:
            print("Fitting age curves...")
            try:
                self._fitted_age_params = fit_age_curves(
                    yoy_df.rename(columns={target: "fpts_per_game"})
                )
            except Exception as e:
                warnings.warn(f"Age curve fitting failed: {e}. Using hardcoded priors.")

        max_season = int(yoy_df["season"].max())

        for pos in POSITIONS:
            pos_df = yoy_df[yoy_df["position"] == pos].copy()
            if pos_df.empty:
                continue

            print(f"  Training {pos} (n={len(pos_df)})...")

            # ----- Volume stage -----
            self._volume_models[pos] = {}
            for vol_target in VOLUME_TARGETS.get(pos, []):
                shifted_col = f"next_{vol_target}"
                model, alpha = _fit_model(
                    pos_df,
                    feature_cols=VOLUME_FEATURES.get(pos, []),
                    target_col=shifted_col,
                    alpha_grid=VOLUME_RIDGE_ALPHA_GRID,
                    max_season=max_season,
                    label=f"{pos}/volume/{vol_target}",
                )
                if model is not None:
                    self._volume_models[pos][vol_target] = model
                    print(f"    vol/{vol_target}: alpha={alpha}")

            # ----- Positional medians for all efficiency metrics -----
            self._positional_means[pos] = {}
            for eff_col in EFFICIENCY_TARGETS.get(pos, []):
                if eff_col in pos_df.columns:
                    med = float(pos_df[eff_col].dropna().median())
                    if not np.isnan(med):
                        self._positional_means[pos][eff_col] = med

            # ----- Efficiency stage (Ridge, optional) -----
            self._efficiency_models[pos] = {}
            if use_ridge_efficiency:
                for eff_target in EFFICIENCY_TARGETS.get(pos, []):
                    shifted_col = f"next_{eff_target}"
                    model, alpha = _fit_model(
                        pos_df,
                        feature_cols=EFFICIENCY_FEATURES.get(pos, []),
                        target_col=shifted_col,
                        alpha_grid=EFFICIENCY_RIDGE_ALPHA_GRID,
                        max_season=max_season,
                        label=f"{pos}/efficiency/{eff_target}",
                    )
                    if model is not None:
                        self._efficiency_models[pos][eff_target] = model
                        print(f"    eff/{eff_target}: alpha={alpha}")
            else:
                print(f"    eff: using regressed historical rates (use_ridge_efficiency=False)")

            # ----- TD rate means (for legacy _predict_efficiency_ridge) -----
            self._mean_td_rates[pos] = {}
            for td_col in ["rec_td_rate", "rush_td_rate", "pass_td_rate"]:
                if td_col in pos_df.columns:
                    mean_val = float(pos_df[td_col].dropna().mean())
                    if not np.isnan(mean_val):
                        self._mean_td_rates[pos][td_col] = mean_val

            # ----- Catch rate mean -----
            if "catch_rate" in pos_df.columns:
                mean_cr = float(pos_df["catch_rate"].dropna().mean())
                if not np.isnan(mean_cr):
                    self._mean_catch_rates[pos] = mean_cr

        return self

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _predict_volume(self, pos: str, features_df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return {vol_target: array_of_predictions} for all volume targets."""
        preds = {}
        for vol_target, model in self._volume_models.get(pos, {}).items():
            feat_cols = [f for f in VOLUME_FEATURES.get(pos, []) if f in features_df.columns]
            if not feat_cols:
                continue
            X = features_df[feat_cols].values
            raw = model.predict(X)
            preds[vol_target] = np.maximum(0.0, raw)
        return preds

    def _regressed_efficiency(self, pos: str, features_df: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Return per-player efficiency estimates via positional median regression.

        For each efficiency metric:
          regressed = reg_weight × positional_median + (1 - reg_weight) × player_historical_rate

        Regression weights are from EFFICIENCY_REGRESSION_WEIGHTS in config.
        Higher weights pull more strongly toward the mean (used for noisier metrics like TD rates).
        Missing player values fall back to the positional median.
        """
        preds = {}
        n = len(features_df)
        for eff_col in EFFICIENCY_TARGETS.get(pos, []):
            reg_w = EFFICIENCY_REGRESSION_WEIGHTS.get(eff_col, 0.40)
            pos_mean = self._positional_means.get(pos, {}).get(eff_col, np.nan)
            if np.isnan(pos_mean):
                continue
            if eff_col in features_df.columns:
                player_rate = features_df[eff_col].fillna(pos_mean).values.astype(float)
            else:
                player_rate = np.full(n, pos_mean)
            preds[eff_col] = np.maximum(0.0, reg_w * pos_mean + (1 - reg_w) * player_rate)
        return preds

    def _predict_efficiency_ridge(self, pos: str, features_df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Ridge-based efficiency predictions (kept for comparison; unreliable due to small samples)."""
        preds = {}
        for eff_target, model in self._efficiency_models.get(pos, {}).items():
            feat_cols = [f for f in EFFICIENCY_FEATURES.get(pos, []) if f in features_df.columns]
            if not feat_cols:
                continue
            X = features_df[feat_cols].values
            raw = np.maximum(0.0, model.predict(X))

            # TD rate mean reversion
            if eff_target in self._mean_td_rates.get(pos, {}):
                pos_mean = self._mean_td_rates[pos][eff_target]
                raw = TD_REGRESSION_WEIGHT * pos_mean + (1 - TD_REGRESSION_WEIGHT) * raw

            preds[eff_target] = raw
        return preds

    def _get_catch_rate(self, pos: str, features_df: pd.DataFrame) -> np.ndarray:
        """Get catch rate with mean reversion, falling back to positional mean."""
        pos_mean = self._mean_catch_rates.get(pos, 0.65)
        if "catch_rate" in features_df.columns:
            cr = features_df["catch_rate"].fillna(pos_mean).values.astype(float)
            cr = np.clip(cr, 0.0, 1.0)
            # Regress toward positional mean
            cr = CATCH_RATE_REGRESSION_WEIGHT * pos_mean + (1 - CATCH_RATE_REGRESSION_WEIGHT) * cr
        else:
            cr = np.full(len(features_df), pos_mean)
        return cr

    def _combine_to_fpts(
        self,
        pos: str,
        vol_preds: dict[str, np.ndarray],
        eff_preds: dict[str, np.ndarray],
        catch_rate: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Combine volume and efficiency predictions into fpts_per_game."""
        s = PPR_SCORING
        fpts = np.zeros(n)

        if pos == "QB":
            dpg = vol_preds.get("dropbacks_per_game", np.zeros(n))
            py_per_db = eff_preds.get("pass_yards_per_attempt", np.zeros(n))
            ptd_rate = eff_preds.get("pass_td_rate", np.zeros(n))
            fpts += dpg * py_per_db * s["pass_yd"]
            fpts += dpg * ptd_rate * s["pass_td"]

        elif pos == "RB":
            cpg = vol_preds.get("carries_per_game", np.zeros(n))
            tpg = vol_preds.get("targets_per_game", np.zeros(n))
            ypc = eff_preds.get("ypc", np.zeros(n))
            rush_td_rate = eff_preds.get("rush_td_rate", np.zeros(n))
            yards_per_tgt = eff_preds.get("yards_per_target", np.zeros(n))
            rec_td_rate = eff_preds.get("rec_td_rate", np.zeros(n))
            receptions = tpg * catch_rate

            # Rushing
            fpts += cpg * ypc * s["rush_yd"]
            fpts += cpg * rush_td_rate * s["rush_td"]
            # Receiving
            fpts += receptions * s["reception"]
            fpts += tpg * yards_per_tgt * s["rec_yd"]
            fpts += tpg * rec_td_rate * s["rec_td"]

        else:  # WR / TE
            tpg = vol_preds.get("targets_per_game", np.zeros(n))
            yards_per_tgt = eff_preds.get("yards_per_target", np.zeros(n))
            rec_td_rate = eff_preds.get("rec_td_rate", np.zeros(n))
            receptions = tpg * catch_rate

            fpts += receptions * s["reception"]
            fpts += tpg * yards_per_tgt * s["rec_yd"]
            fpts += tpg * rec_td_rate * s["rec_td"]

        return np.maximum(0.0, fpts)

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
        Generate season projections for all players in features_df.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix for the input season.
        season : int
            Target projection season.
        projected_games : int
            Games to multiply fpts_per_game by for season total.
        """
        if not self._volume_models and not self._efficiency_models:
            raise RuntimeError("Model not trained. Call train() first.")

        result_rows = []

        for pos in POSITIONS:
            pos_df = features_df[features_df["position"] == pos].copy().reset_index(drop=True)
            if pos_df.empty:
                continue
            n = len(pos_df)

            vol_preds = self._predict_volume(pos, pos_df)
            eff_preds = self._regressed_efficiency(pos, pos_df)

            if not vol_preds and not eff_preds:
                continue

            catch_rate = self._get_catch_rate(pos, pos_df)
            fpts_pg = self._combine_to_fpts(pos, vol_preds, eff_preds, catch_rate, n)

            # Apply per-game cap
            cap_pg = PROJECTION_CAPS.get(pos, 400.0) / projected_games
            fpts_pg = np.clip(fpts_pg, 0.0, cap_pg)

            resid_std = fpts_pg.std() * 0.4
            ci_low = np.maximum(0.0, fpts_pg - 1.28 * resid_std)
            ci_high = fpts_pg + 1.28 * resid_std

            for i in range(n):
                row = {
                    "player_id":   pos_df["player_id"].iloc[i]   if "player_id"   in pos_df.columns else np.nan,
                    "player_name": pos_df["player_name"].iloc[i]  if "player_name" in pos_df.columns else np.nan,
                    "position":    pos,
                    "team":        pos_df["team"].iloc[i]         if "team"        in pos_df.columns else np.nan,
                    "age":         pos_df["age"].iloc[i]          if "age"         in pos_df.columns else np.nan,
                    "projected_fpts_pg":        float(fpts_pg[i]),
                    "projected_games":          projected_games,
                    "confidence_interval_low":  float(ci_low[i]),
                    "confidence_interval_high": float(ci_high[i]),
                }
                result_rows.append(row)

        if not result_rows:
            return pd.DataFrame()

        proj = pd.DataFrame(result_rows)
        proj["projected_season"] = season

        if self.age_adjust and "age" in proj.columns:
            proj_for_age = proj.copy()
            proj_for_age["age"] = proj_for_age["age"] + 1
            proj_adjusted = apply_age_adjustments(proj_for_age, self._fitted_age_params)
            proj["projected_fpts_pg"] = proj_adjusted["projected_fpts_pg"]
            proj["age_multiplier"] = proj_adjusted["age_multiplier"]

        proj["projected_fpts_season"] = proj["projected_fpts_pg"] * proj["projected_games"]

        for pos in POSITIONS:
            mask = proj["position"] == pos
            cap = PROJECTION_CAPS.get(pos, 400.0)
            proj.loc[mask, "projected_fpts_season"] = proj.loc[mask, "projected_fpts_season"].clip(0, cap)
            proj.loc[mask, "projected_fpts_pg"] = proj.loc[mask, "projected_fpts_season"] / projected_games

        return proj.sort_values("projected_fpts_season", ascending=False).reset_index(drop=True)

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
        Hold out test_season, train on prior seasons, evaluate.

        Prints both final fpts accuracy AND intermediate volume/efficiency R²
        per position for diagnostics.

        Returns
        -------
        dict
            {position: {mae, r2, rank_corr, n, volume_r2, efficiency_r2}}
            plus 'overall' key.
        """
        train_df = yoy_df[yoy_df["season"] < test_season].copy()
        test_df = yoy_df[yoy_df["season"] == test_season].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(f"Not enough data to backtest season {test_season}.")

        bt_model = TwoStageProjectionModel(age_adjust=False)
        bt_model.train(train_df, target=target, fit_age=False)

        results = {}
        all_pred, all_actual = [], []

        for pos in POSITIONS:
            pos_test = test_df[test_df["position"] == pos].copy().reset_index(drop=True)
            if pos_test.empty:
                continue

            n = len(pos_test)
            vol_preds = bt_model._predict_volume(pos, pos_test)
            eff_preds = bt_model._regressed_efficiency(pos, pos_test)

            if not vol_preds and not eff_preds:
                continue

            catch_rate = bt_model._get_catch_rate(pos, pos_test)
            fpts_pg = bt_model._combine_to_fpts(pos, vol_preds, eff_preds, catch_rate, n)

            y_actual = pos_test[target].values
            valid = ~np.isnan(y_actual) & ~np.isnan(fpts_pg)

            if valid.sum() < 3:
                continue

            mae = mean_absolute_error(y_actual[valid], fpts_pg[valid])
            r2 = r2_score(y_actual[valid], fpts_pg[valid])
            rank_corr, _ = spearmanr(y_actual[valid], fpts_pg[valid])

            pos_result: dict[str, Any] = {
                "mae": round(mae, 3),
                "r2": round(r2, 3),
                "rank_corr": round(rank_corr, 3),
                "n": int(valid.sum()),
            }

            # Volume stage diagnostics
            for vol_target in VOLUME_TARGETS.get(pos, []):
                shifted = f"next_{vol_target}"
                if shifted in pos_test.columns and vol_target in vol_preds:
                    y_v = pos_test[shifted].values
                    y_p = vol_preds[vol_target]
                    v_valid = ~np.isnan(y_v) & ~np.isnan(y_p)
                    if v_valid.sum() >= 3:
                        pos_result[f"volume_r2_{vol_target}"] = round(r2_score(y_v[v_valid], y_p[v_valid]), 3)

            # Efficiency stage diagnostics
            for eff_target in EFFICIENCY_TARGETS.get(pos, []):
                shifted = f"next_{eff_target}"
                if shifted in pos_test.columns and eff_target in eff_preds:
                    y_e = pos_test[shifted].values
                    y_p = eff_preds[eff_target]
                    e_valid = ~np.isnan(y_e) & ~np.isnan(y_p)
                    if e_valid.sum() >= 3:
                        pos_result[f"efficiency_r2_{eff_target}"] = round(r2_score(y_e[e_valid], y_p[e_valid]), 3)

            results[pos] = pos_result
            all_pred.extend(fpts_pg[valid].tolist())
            all_actual.extend(y_actual[valid].tolist())

        if all_pred:
            overall_mae = mean_absolute_error(all_actual, all_pred)
            overall_r2 = r2_score(all_actual, all_pred)
            overall_rc, _ = spearmanr(all_actual, all_pred)
            results["overall"] = {
                "mae": round(overall_mae, 3),
                "r2": round(overall_r2, 3),
                "rank_corr": round(overall_rc, 3),
                "n": len(all_pred),
            }

        return results

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def feature_importance(self, position: str) -> pd.DataFrame:
        """
        Return Ridge coefficients for volume and efficiency models.

        Returns a DataFrame with model_stage, feature, coefficient columns,
        sorted by abs_coefficient descending.
        """
        rows = []

        for stage, models_dict, features_dict in [
            ("volume", self._volume_models, VOLUME_FEATURES),
            ("efficiency", self._efficiency_models, EFFICIENCY_FEATURES),
        ]:
            for target_col, model in models_dict.get(position, {}).items():
                pipe = model.best_estimator_ if hasattr(model, "best_estimator_") else model
                ridge: Ridge = pipe.named_steps["ridge"]
                feat_names = [f for f in features_dict.get(position, []) if f in (
                    # approximate: use the feature list from config
                    features_dict.get(position, [])
                )]
                coefs = ridge.coef_
                for fname, coef in zip(feat_names[:len(coefs)], coefs[:len(feat_names)]):
                    rows.append({
                        "model_stage": f"{stage}/{target_col}",
                        "feature": fname,
                        "coefficient": coef,
                        "abs_coefficient": abs(coef),
                    })

        if not rows:
            return pd.DataFrame(columns=["model_stage", "feature", "coefficient", "abs_coefficient"])

        return (
            pd.DataFrame(rows)
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )
