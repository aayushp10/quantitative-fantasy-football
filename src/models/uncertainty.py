"""
Calibrated prediction intervals from walk-forward residuals.

The old "confidence interval" was predictions.std() * 0.4 — the cross-sectional
spread of the predictions themselves, not predictive uncertainty, and its
coverage was never validated. Here:

1. walk_forward_residuals() re-trains the model on seasons < S and predicts
   season S, for every eligible S — producing genuinely OUT-OF-SAMPLE
   (pred, actual) pairs per position.
2. ResidualQuantiles fits a heteroscedastic scale |resid| ~ a + b*pred per
   position (uncertainty grows with projection level), normalizes residuals
   by that scale, and stores their empirical quantiles.
3. interval(pos, preds, q) reconstitutes calibrated quantile bounds; the
   conformal-style guarantee is empirical: coverage_report() checks it.
4. simulate_season_totals() composes per-game uncertainty with games-played
   uncertainty (models/availability.py) via Monte Carlo to give season-total
   P10/P50/P90 — the distribution drafting actually needs.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from config import POSITIONS, season_length

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


# ---------------------------------------------------------------------------
# Walk-forward out-of-sample residual collection
# ---------------------------------------------------------------------------

def walk_forward_residuals(
    model_factory,
    yoy_df: pd.DataFrame,
    target: str = "next_fpts",
    min_train_seasons: int = 4,
) -> pd.DataFrame:
    """
    Collect out-of-sample residuals via walk-forward re-training.

    Parameters
    ----------
    model_factory : callable
        Zero-arg callable returning a FRESH untrained model exposing
        .train(df, target=..., fit_age=...) and .predict_position(pos, df)
        (all three model classes qualify), e.g.
        lambda: HybridProjectionModel(age_adjust=False).
    yoy_df : pd.DataFrame
        Full YoY pairs.
    min_train_seasons : int
        Seasons required before the first validation season.

    Returns
    -------
    pd.DataFrame
        Columns: season, position, player_id, pred, actual, resid.
    """
    seasons = sorted(yoy_df["season"].unique())
    rows = []

    for val_season in seasons:
        train_df = yoy_df[yoy_df["season"] < val_season]
        val_df = yoy_df[yoy_df["season"] == val_season]
        if len(train_df["season"].unique()) < min_train_seasons or val_df.empty:
            continue

        model = model_factory()
        try:
            model.train(train_df, target=target, fit_age=False)
        except Exception as e:
            warnings.warn(f"walk_forward_residuals: training failed for {val_season}: {e}")
            continue

        for pos in POSITIONS:
            pos_val = val_df[val_df["position"] == pos].copy().reset_index(drop=True)
            if pos_val.empty or target not in pos_val.columns:
                continue
            try:
                pred = model.predict_position(pos, pos_val)
            except Exception:
                pred = None
            if pred is None:
                continue

            actual = pos_val[target].values
            valid = ~np.isnan(actual) & ~np.isnan(pred)
            for i in np.flatnonzero(valid):
                rows.append({
                    "season": val_season,
                    "position": pos,
                    "player_id": pos_val["player_id"].iloc[i] if "player_id" in pos_val.columns else None,
                    "pred": float(pred[i]),
                    "actual": float(actual[i]),
                    "resid": float(actual[i] - pred[i]),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Residual quantile model
# ---------------------------------------------------------------------------

class ResidualQuantiles:
    """
    Per-position empirical residual quantiles with a heteroscedastic scale.

    scale(pred) = a + b * pred        (b >= 0; fitted by least squares on |resid|)
    u_i = resid_i / scale(pred_i)     (normalized residuals)
    interval(pos, pred, q) = pred + quantile_q(u) * scale(pred)
    """

    def __init__(self, quantiles: tuple[float, ...] = DEFAULT_QUANTILES):
        self.quantiles = tuple(quantiles)
        self._scale: dict[str, tuple[float, float]] = {}       # pos -> (a, b)
        self._u_quantiles: dict[str, dict[float, float]] = {}  # pos -> {q: value}
        self._u_samples: dict[str, np.ndarray] = {}            # pos -> normalized resids
        self.n_fit_: dict[str, int] = {}

    # ------------------------------------------------------------------

    def fit(self, resid_df: pd.DataFrame) -> "ResidualQuantiles":
        """Fit from walk_forward_residuals() output."""
        for pos in POSITIONS:
            sub = resid_df[resid_df["position"] == pos]
            if len(sub) < 30:
                continue

            pred = sub["pred"].values.astype(float)
            resid = sub["resid"].values.astype(float)

            # Heteroscedastic scale: |resid| ~ a + b*pred, b clipped >= 0
            try:
                b, a = np.polyfit(pred, np.abs(resid), 1)
            except Exception:
                b, a = 0.0, float(np.std(resid))
            b = max(0.0, float(b))
            a = max(1e-3, float(a))
            self._scale[pos] = (a, b)

            u = resid / (a + b * pred)
            self._u_samples[pos] = u
            self._u_quantiles[pos] = {
                q: float(np.quantile(u, q)) for q in self.quantiles
            }
            self.n_fit_[pos] = len(sub)
        return self

    def fitted_positions(self) -> list[str]:
        return list(self._u_quantiles.keys())

    # ------------------------------------------------------------------

    def interval(self, position: str, preds: np.ndarray, q: float) -> np.ndarray:
        """Quantile-q bound of the predictive distribution around preds."""
        preds = np.asarray(preds, dtype=float)
        if position not in self._u_quantiles:
            # Unfitted position: fall back to a symmetric normal-ish band from
            # the pooled scale of fitted positions (never silently zero-width)
            pooled = np.concatenate(list(self._u_samples.values())) if self._u_samples else None
            if pooled is None:
                return preds
            scales = [a + b * preds for (a, b) in self._scale.values()]
            scale = np.mean(scales, axis=0)
            return preds + float(np.quantile(pooled, q)) * scale

        if q not in self._u_quantiles[position]:
            raise ValueError(
                f"Quantile {q} not fitted. Available: {sorted(self._u_quantiles[position])}"
            )
        a, b = self._scale[position]
        return preds + self._u_quantiles[position][q] * (a + b * preds)

    # ------------------------------------------------------------------

    def coverage_report(
        self,
        resid_df: pd.DataFrame,
        lo: float = 0.10,
        hi: float = 0.90,
    ) -> pd.DataFrame:
        """
        Empirical coverage of the [lo, hi] interval on a residual frame —
        run on held-out residuals to VALIDATE the claimed coverage.
        """
        rows = []
        for pos in self.fitted_positions():
            sub = resid_df[resid_df["position"] == pos]
            if sub.empty:
                continue
            pred = sub["pred"].values.astype(float)
            actual = sub["actual"].values.astype(float)
            low = self.interval(pos, pred, lo)
            high = self.interval(pos, pred, hi)
            inside = (actual >= low) & (actual <= high)
            rows.append({
                "position": pos,
                "nominal_coverage": hi - lo,
                "empirical_coverage": float(np.mean(inside)),
                "n": len(sub),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Season-total Monte Carlo (per-game dist × games-played dist)
# ---------------------------------------------------------------------------

def simulate_season_totals(
    projections: pd.DataFrame,
    residual_quantiles: ResidualQuantiles,
    target_season: int,
    games_sd: dict[str, float] | None = None,
    n_sims: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Add season-total quantile columns to a projections frame.

    Composes per-game uncertainty (bootstrap of normalized walk-forward
    residuals) with games-played uncertainty (truncated normal around
    projected_games; sd per position from AvailabilityModel.residual_std_,
    in games-frac units).

    Adds: season_p10, season_p25, season_p50, season_p75, season_p90.
    """
    rng = np.random.default_rng(seed)
    L = season_length(target_season)
    out = projections.copy()
    for col in ["season_p10", "season_p25", "season_p50", "season_p75", "season_p90"]:
        out[col] = np.nan

    for pos in POSITIONS:
        mask = out["position"] == pos
        if not mask.any():
            continue

        preds = out.loc[mask, "projected_fpts_pg"].values.astype(float)
        games = (
            out.loc[mask, "projected_games"].values.astype(float)
            if "projected_games" in out.columns
            else np.full(mask.sum(), float(L))
        )

        u = residual_quantiles._u_samples.get(pos)
        if u is None or len(u) == 0:
            continue
        a, b = residual_quantiles._scale.get(pos, (1.0, 0.0))

        g_sd_frac = (games_sd or {}).get(pos, (games_sd or {}).get("overall", 0.15))

        n_players = len(preds)
        u_draws = rng.choice(u, size=(n_players, n_sims), replace=True)
        ppg_sims = np.maximum(0.0, preds[:, None] + u_draws * (a + b * preds)[:, None])

        g_sims = rng.normal(games[:, None], g_sd_frac * L, size=(n_players, n_sims))
        g_sims = np.clip(g_sims, 0, L)

        totals = ppg_sims * g_sims
        out.loc[mask, "season_p10"] = np.quantile(totals, 0.10, axis=1)
        out.loc[mask, "season_p25"] = np.quantile(totals, 0.25, axis=1)
        out.loc[mask, "season_p50"] = np.quantile(totals, 0.50, axis=1)
        out.loc[mask, "season_p75"] = np.quantile(totals, 0.75, axis=1)
        out.loc[mask, "season_p90"] = np.quantile(totals, 0.90, axis=1)

    return out
