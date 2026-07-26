"""
Gradient-boosted benchmark model (LightGBM).

Everything else in the pipeline is linear (Ridge). This model answers,
inside the exact same walk-forward harness, whether there is nonlinear
signal the linear stack leaves on the table (age × usage interactions,
draft-capital thresholds, saturation in shares).

Design mirrors FantasyProjectionModel deliberately:
  - same per-position training, same POSITION_FEATURES sets,
  - same per-season cross-sectional standardization (era comparability
    matters for trees too: an unstandardized target-share split learned
    in 2014 means something different in 2024),
  - same recency sample weights,
  - number of trees chosen by early stopping on the most recent training
    season (a proper temporal holdout), then refit on all data.

It subclasses FantasyProjectionModel so predict_position / project /
backtest are inherited unchanged — only train() and feature_importance()
are boosted-specific. LightGBM consumes NaN natively, so no imputer.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from config import (
    POSITION_FEATURES,
    POSITIONS,
    STANDARDIZE_BY_SEASON,
)
from features.standardize import SeasonStandardizer
from models.age_curves import fit_age_curves
from models.projection import FantasyProjectionModel, _compute_sample_weights

try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    _HAS_LGBM = False

GBM_PARAMS: dict[str, Any] = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 25,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 60
MIN_TREES = 50


class GBMProjectionModel(FantasyProjectionModel):
    """Per-position LightGBM with the FantasyProjectionModel interface."""

    def __init__(self, age_adjust: bool = True, standardize: bool = STANDARDIZE_BY_SEASON,
                 params: dict[str, Any] | None = None):
        if not _HAS_LGBM:
            raise ImportError("lightgbm is required for GBMProjectionModel "
                              "(pip install lightgbm)")
        super().__init__(age_adjust=age_adjust, standardize=standardize)
        self.params = {**GBM_PARAMS, **(params or {})}
        self.best_trees_: dict[str, int] = {}

    # ------------------------------------------------------------------

    def train(
        self,
        yoy_df: pd.DataFrame,
        target: str = "next_fpts",
        fit_age: bool = True,
    ) -> "GBMProjectionModel":
        if "position" not in yoy_df.columns:
            raise ValueError("yoy_df must have a 'position' column.")
        if target not in yoy_df.columns:
            raise ValueError(f"Target column '{target}' not found in yoy_df.")

        if fit_age and self.age_adjust:
            try:
                self._fitted_age_params = fit_age_curves(yoy_df)
            except Exception as e:
                warnings.warn(f"Age curve fitting failed: {e}. Using hardcoded priors.")
                self._fitted_age_params = None

        max_season = int(yoy_df["season"].max())

        for pos in POSITIONS:
            pos_df = yoy_df[yoy_df["position"] == pos].copy()
            if len(pos_df) < 50:
                warnings.warn(f"GBM {pos}: only {len(pos_df)} rows. Skipping.")
                continue

            features = [f for f in POSITION_FEATURES[pos] if f in pos_df.columns]
            if not features:
                continue

            pos_df = pos_df.sort_values("season").reset_index(drop=True)
            if self.standardize:
                std = SeasonStandardizer().fit(pos_df, features)
                self._standardizers[pos] = std
                pos_df_t = std.transform(pos_df)
            else:
                pos_df_t = pos_df

            X = pos_df_t[features].values
            y = pos_df_t[target].values
            seasons = pos_df["season"].values
            weights = _compute_sample_weights(pos_df["season"], max_season)
            self._feature_names[pos] = features

            # Temporal holdout for tree count: the most recent training season
            val_mask = seasons == max_season
            n_trees = self.params["n_estimators"]
            if val_mask.sum() >= 15 and (~val_mask).sum() >= 100:
                probe = LGBMRegressor(**self.params)
                probe.fit(
                    X[~val_mask], y[~val_mask],
                    sample_weight=weights[~val_mask],
                    eval_set=[(X[val_mask], y[val_mask])],
                    eval_metric="l1",
                    callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                               log_evaluation(0)],
                )
                n_trees = max(MIN_TREES, probe.best_iteration_ or n_trees)

            model = LGBMRegressor(**{**self.params, "n_estimators": n_trees})
            model.fit(X, y, sample_weight=weights)
            self._models[pos] = model
            self.best_trees_[pos] = n_trees
            self._best_alphas[pos] = float(n_trees)  # slot reused for reporting
            print(f"  GBM {pos}: n={len(pos_df)}, features={len(features)}, trees={n_trees}")

        return self

    # ------------------------------------------------------------------

    def feature_importance(self, position: str) -> pd.DataFrame:
        """Gain-based importances for one position."""
        model = self._models.get(position)
        if model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        imp = model.booster_.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({
                "feature": self._feature_names[position],
                "importance": imp,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
