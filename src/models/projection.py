"""
Core fantasy football projection model.

Architecture: two-stage factor model
  1. Season N factor exposures → Ridge regression → Season N+1 fpts_per_game
  2. Post-hoc multiplicative age curve adjustment

One Ridge pipeline per position (QB, RB, WR, TE). Positions have different
feature sets, regularization needs, and sample sizes.

Key implementation decisions:
- Pipeline: SimpleImputer → StandardScaler → Ridge (order matters)
  StandardScaler is mandatory — factors have vastly different scales.
  SimpleImputer fills NaN feature values with per-position medians.
- TimeSeriesSplit(n_splits=4, gap=1) prevents data leakage.
- Exponential sample weights: 0.7^(max_season - season) prioritizes recent seasons.
- Ridge alpha grid-searched over [0.1, 1.0, 10.0, 100.0, 1000.0].
- Age curve applied as multiplicative post-adjustment, not as a feature.
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
    CV_GAP,
    CV_N_SPLITS,
    POSITION_FEATURES,
    POSITIONS,
    PROJECTION_CAPS,
    RECENCY_DECAY,
    RIDGE_ALPHA_GRID,
)
from models.age_curves import apply_age_adjustments, fit_age_curves


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])


def _compute_sample_weights(seasons: pd.Series, max_season: int) -> np.ndarray:
    years_ago = max_season - seasons.values
    return np.power(RECENCY_DECAY, years_ago).astype(float)


# ---------------------------------------------------------------------------
# FantasyProjectionModel
# ---------------------------------------------------------------------------

class FantasyProjectionModel:
    """
    Position-specific Ridge regression projection models.

    Usage
    -----
    >>> model = FantasyProjectionModel()
    >>> model.train(yoy_pairs)
    >>> projections = model.project(feature_matrix_2024, season=2025)
    >>> backtest_stats = model.backtest(yoy_pairs, test_season=2024)
    """

    def __init__(self, age_adjust: bool = True):
        self.age_adjust = age_adjust
        self._models: dict[str, Any] = {}    # position → fitted GridSearchCV
        self._best_alphas: dict[str, float] = {}
        self._fitted_age_params: dict | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        yoy_df: pd.DataFrame,
        target: str = "next_fpts",
        fit_age: bool = True,
    ) -> "FantasyProjectionModel":
        """
        Fit position-specific Ridge models using TimeSeriesSplit CV.

        Parameters
        ----------
        yoy_df : pd.DataFrame
            YoY pairs from assembler.build_yoy_pairs(). Must have
            'position', 'season', and all POSITION_FEATURES columns.
        target : str
            Column name for the prediction target (fpts_per_game N+1).
        fit_age : bool
            Whether to fit empirical age curves from this data.
        """
        if "position" not in yoy_df.columns:
            raise ValueError("yoy_df must have a 'position' column.")
        if target not in yoy_df.columns:
            raise ValueError(f"Target column '{target}' not found in yoy_df.")

        if fit_age and self.age_adjust:
            print("Fitting age curves...")
            try:
                self._fitted_age_params = fit_age_curves(
                    yoy_df.rename(columns={target: "fpts_per_game"})
                )
            except Exception as e:
                warnings.warn(f"Age curve fitting failed: {e}. Using hardcoded priors.")
                self._fitted_age_params = None

        cv = TimeSeriesSplit(n_splits=CV_N_SPLITS, gap=CV_GAP)
        max_season = int(yoy_df["season"].max())

        for pos in POSITIONS:
            pos_df = yoy_df[yoy_df["position"] == pos].copy()
            features = [f for f in POSITION_FEATURES[pos] if f in pos_df.columns]

            if len(features) == 0:
                warnings.warn(f"No features available for {pos}. Skipping.")
                continue

            pos_df = pos_df.sort_values("season")
            X = pos_df[features].values
            y = pos_df[target].values
            weights = _compute_sample_weights(pos_df["season"], max_season)

            n_splits = min(CV_N_SPLITS, len(pos_df) // 2)
            if n_splits < 2:
                warnings.warn(f"{pos}: too few samples ({len(pos_df)}) for CV. Fitting without CV.")
                pipe = _build_pipeline()
                pipe.fit(X, y, ridge__sample_weight=weights)
                self._models[pos] = pipe
                self._best_alphas[pos] = 1.0
                print(f"  {pos}: no CV (n={len(pos_df)}), features={len(features)}")
                continue

            cv_local = TimeSeriesSplit(n_splits=n_splits, gap=CV_GAP)
            param_grid = {"ridge__alpha": RIDGE_ALPHA_GRID}
            gs = GridSearchCV(
                _build_pipeline(),
                param_grid,
                cv=cv_local,
                scoring="neg_mean_absolute_error",
                refit=True,
            )

            # Pass sample weights through GridSearchCV via fit_params
            try:
                gs.fit(X, y, ridge__sample_weight=weights)
            except TypeError:
                # Older sklearn versions don't support fit_params in GridSearchCV
                gs.fit(X, y)

            self._models[pos] = gs.best_estimator_
            best_alpha = gs.best_params_["ridge__alpha"]
            self._best_alphas[pos] = best_alpha

            # Cross-validation score
            cv_mae = -gs.best_score_
            print(f"  {pos}: n={len(pos_df)}, features={len(features)}, "
                  f"alpha={best_alpha}, CV MAE={cv_mae:.2f}")

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
        Generate season projections for all players in features_df.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix for the input season (season N → project N+1).
            Use the most recent available season (e.g., 2024 features → 2025 projections).
        season : int
            The target projection season (e.g., 2025).
        projected_games : int
            Number of games to multiply fpts_per_game by for season total.

        Returns
        -------
        pd.DataFrame
            Columns: player_id, player_name, position, team,
                     projected_fpts_pg, projected_games, projected_fpts_season,
                     confidence_interval_low, confidence_interval_high,
                     age_multiplier (if age_adjust=True).
        """
        if not self._models:
            raise RuntimeError("Model not trained. Call train() first.")

        result_rows = []

        for pos in POSITIONS:
            if pos not in self._models:
                continue

            pos_df = features_df[features_df["position"] == pos].copy()
            if pos_df.empty:
                continue

            features = [f for f in POSITION_FEATURES[pos] if f in pos_df.columns]
            if not features:
                continue

            X = pos_df[features].values
            predictions = self._models[pos].predict(X)

            # Apply projection caps
            cap = PROJECTION_CAPS.get(pos, 400.0) / projected_games  # per-game cap
            predictions = np.clip(predictions, 0, cap)

            # Rough 80% confidence interval from residual std
            # (simple heuristic — not a proper prediction interval)
            resid_std = predictions.std() * 0.4
            ci_low = np.maximum(0, predictions - 1.28 * resid_std)
            ci_high = predictions + 1.28 * resid_std

            pos_df = pos_df.reset_index(drop=True)
            for i in range(len(pos_df)):
                row = {
                    "player_id":   pos_df["player_id"].iloc[i]   if "player_id"   in pos_df.columns else np.nan,
                    "player_name": pos_df["player_name"].iloc[i]  if "player_name" in pos_df.columns else np.nan,
                    "position":    pos,
                    "team":        pos_df["team"].iloc[i]         if "team"        in pos_df.columns else np.nan,
                    "age":         pos_df["age"].iloc[i]          if "age"         in pos_df.columns else np.nan,
                    "projected_fpts_pg":        float(predictions[i]),
                    "projected_games":          projected_games,
                    "confidence_interval_low":  float(ci_low[i]),
                    "confidence_interval_high": float(ci_high[i]),
                }
                result_rows.append(row)

        if not result_rows:
            return pd.DataFrame()

        proj = pd.DataFrame(result_rows)
        proj["projected_season"] = season

        # Apply age curve adjustment
        if self.age_adjust and "age" in proj.columns:
            proj_for_age = proj.copy()
            proj_for_age["age"] = proj_for_age["age"] + 1  # age in the projection season
            proj_adjusted = apply_age_adjustments(proj_for_age, self._fitted_age_params)
            proj["projected_fpts_pg"] = proj_adjusted["projected_fpts_pg"]
            proj["age_multiplier"] = proj_adjusted["age_multiplier"]

        proj["projected_fpts_season"] = proj["projected_fpts_pg"] * proj["projected_games"]

        # Apply season-level caps too
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

        Parameters
        ----------
        yoy_df : pd.DataFrame
            Full YoY pairs dataset.
        test_season : int
            Season to hold out as the test set.
        target : str
            Target column name.

        Returns
        -------
        dict
            {position: {mae, r2, rank_corr, n}} plus 'overall' key.
        """
        train_df = yoy_df[yoy_df["season"] < test_season].copy()
        test_df = yoy_df[yoy_df["season"] == test_season].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(f"Not enough data to backtest season {test_season}.")

        # Temporarily fit a new model on training data only
        bt_model = FantasyProjectionModel(age_adjust=False)
        bt_model.train(train_df, target=target, fit_age=False)

        results = {}
        all_pred, all_actual = [], []

        for pos in POSITIONS:
            pos_test = test_df[test_df["position"] == pos].copy()
            if pos_test.empty or pos not in bt_model._models:
                continue

            features = [f for f in POSITION_FEATURES[pos] if f in pos_test.columns]
            if not features:
                continue

            X = pos_test[features].values
            y_actual = pos_test[target].values
            y_pred = bt_model._models[pos].predict(X)

            valid = ~np.isnan(y_actual) & ~np.isnan(y_pred)
            if valid.sum() < 3:
                continue

            mae = mean_absolute_error(y_actual[valid], y_pred[valid])
            r2 = r2_score(y_actual[valid], y_pred[valid])
            rank_corr, _ = spearmanr(y_actual[valid], y_pred[valid])

            results[pos] = {
                "mae": round(mae, 3),
                "r2": round(r2, 3),
                "rank_corr": round(rank_corr, 3),
                "n": int(valid.sum()),
            }
            all_pred.extend(y_pred[valid])
            all_actual.extend(y_actual[valid])

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
        Return Ridge coefficients (post-scaling) as feature importance.

        Larger absolute coefficient = stronger influence on the prediction
        after features are standardized.

        Parameters
        ----------
        position : str
            One of 'QB', 'RB', 'WR', 'TE'.

        Returns
        -------
        pd.DataFrame
            Columns: feature, coefficient, abs_coefficient.
            Sorted by abs_coefficient descending.
        """
        if position not in self._models:
            raise ValueError(f"No model fitted for {position}. Train the model first.")

        model = self._models[position]

        # Handle both raw Pipeline and GridSearchCV best_estimator_
        if hasattr(model, "best_estimator_"):
            pipe = model.best_estimator_
        else:
            pipe = model

        ridge: Ridge = pipe.named_steps["ridge"]
        features = [f for f in POSITION_FEATURES[position]
                    if f in (self._models.get("_feature_names", {}).get(position, POSITION_FEATURES[position]))]

        # Use POSITION_FEATURES as ground truth for column names
        feature_names = [f for f in POSITION_FEATURES[position]]
        coefs = ridge.coef_

        n = min(len(feature_names), len(coefs))
        df = pd.DataFrame({
            "feature": feature_names[:n],
            "coefficient": coefs[:n],
        })
        df["abs_coefficient"] = df["coefficient"].abs()
        return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
