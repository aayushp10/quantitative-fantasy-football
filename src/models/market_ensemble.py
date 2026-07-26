"""
Market-as-prior ensemble: blend the model with the price.

The market backtest shows ADP alone out-ranks the model (mean IC edge
≈ −0.10) while model-vs-market DISAGREEMENTS still predict outcomes
(L/S spread ≈ +13 ranks, positive every held-out season). The rational
combination is therefore not "model instead of market" but "market as
the base rate, model as the correction":

    final_pg = w · market_implied_pg + (1 − w) · base_model_pg

where market_implied_pg is a per-position isotonic regression of
next-season points per game on ADP (monotone decreasing, recency-
weighted), fitted on training pairs only. Players without an ADP fall
back to the pure base model. The blend preserves the sign and ordering
of every model-vs-market disagreement — it shrinks them, it doesn't
erase them — so the L/S alpha survives while the base ordering inherits
the market's superior IC.

Public API mirrors the other models: train / predict_position / project.
Compatible with models.backtest.rolling_backtest and
models.market.rolling_market_backtest (pass adp_df via model kwargs).
"""
from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from config import POSITIONS, RECENCY_DECAY

DEFAULT_MARKET_WEIGHT = 0.65

# Per-position defaults from the 2020–2023 walk-forward sweep (w ∈ {0.5,
# 0.65, 0.8}): every weight beat the pure model at every position, so
# sensitivity is low; these pick the best point per position (QB's weak
# standalone model wants more market, TE's break-even model wants less).
DEFAULT_MARKET_WEIGHTS: dict[str, float] = {
    "QB": 0.80, "RB": 0.65, "WR": 0.50, "TE": 0.50,
}


class MarketEnsembleModel:
    """
    w · isotonic(ADP) + (1−w) · base model, per position.

    Parameters
    ----------
    adp_history : pd.DataFrame | None
        Historical ADP from data.adp.load_adp(). Needed at train time
        (joined to pairs at season+1) and for frames that arrive without
        an 'adp' column. If None, frames must already carry 'adp'.
    market_weight : float | dict[str, float]
        Market share of the blend (global float or per-position dict).
    base_factory : callable | None
        Zero-arg factory for the base model. Default: HybridProjectionModel
        with the given age_adjust.
    """

    def __init__(
        self,
        adp_history: pd.DataFrame | None = None,
        market_weight: float | dict[str, float] | None = None,
        base_factory: Callable[[], Any] | None = None,
        age_adjust: bool = True,
    ):
        if base_factory is None:
            from models.hybrid import HybridProjectionModel

            base_factory = lambda: HybridProjectionModel(age_adjust=age_adjust)  # noqa: E731
        self.adp_history = adp_history
        self.market_weight = (
            dict(DEFAULT_MARKET_WEIGHTS) if market_weight is None else market_weight
        )
        self.age_adjust = age_adjust
        self._base = base_factory()
        self._priors: dict[str, IsotonicRegression] = {}
        self.n_prior_: dict[str, int] = {}

    # ------------------------------------------------------------------

    def weight_for(self, position: str) -> float:
        if isinstance(self.market_weight, dict):
            return float(self.market_weight.get(position, DEFAULT_MARKET_WEIGHT))
        return float(self.market_weight)

    def _attach_adp(self, df: pd.DataFrame, season_offset: int,
                    season_col: str = "season") -> pd.DataFrame:
        """Join ADP by (name, position, season+offset) unless already present."""
        if "adp" in df.columns:
            return df
        if self.adp_history is None or self.adp_history.empty:
            out = df.copy()
            out["adp"] = np.nan
            return out
        from data.adp import attach_adp

        return attach_adp(df, self.adp_history, season_offset=season_offset,
                          season_col=season_col)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        yoy_df: pd.DataFrame,
        target: str = "next_fpts",
        fit_age: bool = True,
    ) -> "MarketEnsembleModel":
        """Train the base model on the full pairs, then fit the market prior
        per position on the subset with an ADP (drafted before season N+1)."""
        self._base.train(yoy_df, target=target, fit_age=fit_age)

        with_adp = self._attach_adp(yoy_df, season_offset=1)
        max_season = int(yoy_df["season"].max())

        for pos in POSITIONS:
            sub = with_adp[
                (with_adp["position"] == pos)
                & with_adp["adp"].notna()
                & with_adp[target].notna()
            ]
            if len(sub) < 30:
                warnings.warn(f"MarketEnsemble {pos}: only {len(sub)} ADP rows; "
                              "market prior unavailable (pure base model).")
                continue
            weights = np.power(RECENCY_DECAY, max_season - sub["season"].values)
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            iso.fit(sub["adp"].values, sub[target].values, sample_weight=weights)
            self._priors[pos] = iso
            self.n_prior_[pos] = len(sub)
            print(f"  Market prior {pos}: n={len(sub)}, w_market={self.weight_for(pos):.2f}")
        return self

    # ------------------------------------------------------------------
    # Prediction (the ONE path used by backtests and project)
    # ------------------------------------------------------------------

    def market_implied(self, position: str, adp: np.ndarray) -> np.ndarray:
        """Market-implied points per game for an ADP vector (NaN → NaN)."""
        iso = self._priors.get(position)
        out = np.full(len(adp), np.nan)
        if iso is None:
            return out
        mask = ~pd.isna(adp)
        if mask.any():
            out[mask] = iso.predict(np.asarray(adp, dtype=float)[mask])
        return out

    def predict_position(self, position: str, pos_df: pd.DataFrame) -> np.ndarray | None:
        base_pred = self._base.predict_position(position, pos_df)
        if base_pred is None:
            return None

        df = self._attach_adp(pos_df.reset_index(drop=True), season_offset=1)
        prior = self.market_implied(position, df["adp"].values)

        w = self.weight_for(position)
        blended = np.where(
            np.isnan(prior),
            base_pred,
            w * prior + (1.0 - w) * base_pred,
        )
        return blended

    def set_uncertainty(self, residual_quantiles) -> "MarketEnsembleModel":
        self._base.set_uncertainty(residual_quantiles)
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
        """Base-model projection with the per-game prediction re-blended
        toward the market price for the projection season (draft year =
        `season`, offset 0)."""
        proj = self._base.project(features_df, season, projected_games)
        if proj.empty or self.adp_history is None:
            return proj

        out = proj.copy()
        out["_season"] = season
        out = self._attach_adp(out, season_offset=0, season_col="_season")
        for pos in POSITIONS:
            mask = (out["position"] == pos).values
            if not mask.any():
                continue
            prior = self.market_implied(pos, out.loc[mask, "adp"].values)
            base = out.loc[mask, "projected_fpts_pg"].values
            w = self.weight_for(pos)
            out.loc[mask, "projected_fpts_pg"] = np.where(
                np.isnan(prior), base, w * prior + (1 - w) * base
            )
        out["projected_fpts_season"] = out["projected_fpts_pg"] * out["projected_games"]
        drop = [c for c in ["_season", "adp_pos_rank", "adp_matched"] if c in out.columns]
        return (
            out.drop(columns=drop)
            .sort_values("projected_fpts_season", ascending=False)
            .reset_index(drop=True)
        )
