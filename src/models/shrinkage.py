"""
Empirical-Bayes shrinkage for rate metrics, with sample-size-dependent weights.

Replaces the hand-set constants (TD_REGRESSION_WEIGHT=0.55 etc.) with priors
ESTIMATED from the data. The posterior for every metric has the same form:

    posterior_i = (k * prior_mean_i + n_i * observed_i) / (k + n_i)

where n_i is the player's trial count (targets/carries/dropbacks) and k is the
fitted prior strength ("equivalent prior sample size"). A 40-target rookie is
shrunk hard; a 150-target veteran keeps most of their observed rate — the flat
constants could not express this.

Two prior families, chosen per metric (config.EB_BINOMIAL_METRICS):

Beta-binomial (TD rates, catch rate) — method of moments:
    mu   = pooled success rate
    tau2 = Var(p_i) - mu(1-mu) * mean(1/n_i)     (between-player variance,
                                                  sampling noise removed)
    k    = mu(1-mu)/tau2 - 1

Normal-normal (yards per target, YPC, YPA) — 1/n variance regression:
    Var(obs_i - mu) ≈ tau2 + s2/n_i   →  regress squared deviations on 1/n_i:
    intercept = tau2 (true between-player variance), slope = s2 (per-trial
    noise), k = s2/tau2.

Player-specific prior means: for TD rates the prior can center on the player's
own geometry-implied rate (x_rec_td_rate etc. from features/expected_td.py)
instead of the flat positional mean — "how often SHOULD this usage mix have
scored" is a far better prior than "how often does the average WR score".
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    EB_BINOMIAL_METRICS,
    EB_MIN_TRIALS,
    EB_PRIOR_STRENGTH_BOUNDS,
    EB_TRIALS_COLUMNS,
    POSITIONS,
)


@dataclass
class RatePrior:
    """Fitted prior for one (position, metric): posterior = (k*prior + n*obs)/(k+n)."""
    mean: float          # prior mean (positional)
    strength: float      # k, equivalent prior sample size
    family: str          # 'beta-binomial' | 'normal'
    n_players: int = 0


# ---------------------------------------------------------------------------
# Prior fitting
# ---------------------------------------------------------------------------

def fit_beta_binomial_prior(
    rates: np.ndarray,
    trials: np.ndarray,
    strength_bounds: tuple[float, float] = EB_PRIOR_STRENGTH_BOUNDS,
) -> RatePrior | None:
    """Method-of-moments beta-binomial fit on player-season (rate, trials) pairs."""
    rates = np.asarray(rates, dtype=float)
    trials = np.asarray(trials, dtype=float)
    valid = ~np.isnan(rates) & ~np.isnan(trials) & (trials > 0)
    rates, trials = rates[valid], trials[valid]
    if len(rates) < 10:
        return None

    mu = float(np.sum(rates * trials) / np.sum(trials))
    mu = min(max(mu, 1e-6), 1 - 1e-6)

    raw_var = float(np.var(rates))
    sampling_var = mu * (1 - mu) * float(np.mean(1.0 / trials))
    tau2 = raw_var - sampling_var

    lo, hi = strength_bounds
    if tau2 <= 0:
        # No detectable between-player spread → shrink maximally
        k = hi
    else:
        k = mu * (1 - mu) / tau2 - 1.0
        k = float(np.clip(k, lo, hi))

    return RatePrior(mean=mu, strength=k, family="beta-binomial", n_players=len(rates))


def fit_normal_prior(
    rates: np.ndarray,
    trials: np.ndarray,
    strength_bounds: tuple[float, float] = EB_PRIOR_STRENGTH_BOUNDS,
) -> RatePrior | None:
    """
    Normal-normal fit: regress squared deviations on 1/n to separate true
    between-player variance (intercept) from per-trial sampling noise (slope).
    """
    rates = np.asarray(rates, dtype=float)
    trials = np.asarray(trials, dtype=float)
    valid = ~np.isnan(rates) & ~np.isnan(trials) & (trials > 0)
    rates, trials = rates[valid], trials[valid]
    if len(rates) < 10:
        return None

    mu = float(np.sum(rates * trials) / np.sum(trials))
    dev2 = (rates - mu) ** 2
    inv_n = 1.0 / trials

    lo, hi = strength_bounds
    try:
        s2, tau2 = np.polyfit(inv_n, dev2, 1)  # slope, intercept
    except Exception:
        return RatePrior(mean=mu, strength=lo, family="normal", n_players=len(rates))

    if tau2 <= 0 or s2 <= 0:
        # Degenerate fit: fall back to conservative mid-strength shrinkage
        k = float(np.sqrt(lo * hi)) if tau2 <= 0 else lo
    else:
        k = float(np.clip(s2 / tau2, lo, hi))

    return RatePrior(mean=mu, strength=k, family="normal", n_players=len(rates))


# ---------------------------------------------------------------------------
# Shrinker
# ---------------------------------------------------------------------------

class EmpiricalBayesShrinker:
    """
    Fits per-(position, metric) priors from a training frame, then produces
    shrunk rate estimates for any frame with the metric + trials columns.
    """

    def __init__(
        self,
        metrics: dict[str, str] | None = None,
        min_trials: int = EB_MIN_TRIALS,
    ):
        # {metric: trials_column}
        self.metrics = dict(metrics) if metrics is not None else dict(EB_TRIALS_COLUMNS)
        self.min_trials = min_trials
        self._priors: dict[tuple[str, str], RatePrior] = {}

    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "EmpiricalBayesShrinker":
        """Fit priors per position for every metric whose columns exist in df."""
        if "position" not in df.columns:
            warnings.warn("EmpiricalBayesShrinker.fit: no 'position' column; nothing fitted.")
            return self

        for pos in POSITIONS:
            pos_df = df[df["position"] == pos]
            if pos_df.empty:
                continue
            for metric, trials_col in self.metrics.items():
                if metric not in pos_df.columns or trials_col not in pos_df.columns:
                    continue
                sub = pos_df[[metric, trials_col]].dropna()
                sub = sub[sub[trials_col] >= self.min_trials]
                if len(sub) < 10:
                    continue
                fitter = (
                    fit_beta_binomial_prior
                    if metric in EB_BINOMIAL_METRICS
                    else fit_normal_prior
                )
                prior = fitter(sub[metric].values, sub[trials_col].values)
                if prior is not None:
                    self._priors[(pos, metric)] = prior
        return self

    # ------------------------------------------------------------------

    def has(self, position: str, metric: str) -> bool:
        return (position, metric) in self._priors

    def prior(self, position: str, metric: str) -> RatePrior | None:
        return self._priors.get((position, metric))

    def shrink(
        self,
        features_df: pd.DataFrame,
        position: str,
        metric: str,
        prior_mean: np.ndarray | pd.Series | None = None,
    ) -> np.ndarray | None:
        """
        Posterior rate estimates for each row of features_df.

        prior_mean: optional per-player prior means (e.g. geometry-implied TD
        rates). NaN entries fall back to the fitted positional mean. Rows with
        missing observed rate or zero trials return the prior mean.
        """
        p = self._priors.get((position, metric))
        if p is None:
            return None

        n_rows = len(features_df)
        trials_col = self.metrics.get(metric)

        obs = (
            features_df[metric].astype(float).values
            if metric in features_df.columns
            else np.full(n_rows, np.nan)
        )
        trials = (
            features_df[trials_col].astype(float).values
            if trials_col and trials_col in features_df.columns
            else np.zeros(n_rows)
        )
        trials = np.nan_to_num(trials, nan=0.0)

        if prior_mean is not None:
            m = np.asarray(pd.Series(prior_mean, dtype=float).values)
            m = np.where(np.isnan(m), p.mean, m)
        else:
            m = np.full(n_rows, p.mean)

        # Missing observation → pure prior (treat trials as 0)
        obs_filled = np.where(np.isnan(obs), m, obs)
        eff_trials = np.where(np.isnan(obs), 0.0, trials)

        post = (p.strength * m + eff_trials * obs_filled) / (p.strength + eff_trials)
        return np.maximum(0.0, post)

    # ------------------------------------------------------------------

    def shrinkage_table(self) -> pd.DataFrame:
        """Fitted priors for diagnostics: one row per (position, metric)."""
        rows = []
        for (pos, metric), p in sorted(self._priors.items()):
            rows.append({
                "position": pos,
                "metric": metric,
                "family": p.family,
                "prior_mean": p.mean,
                "prior_strength_k": p.strength,
                "n_players": p.n_players,
                # Shrinkage weight on the OBSERVED rate at representative trial counts
                "obs_weight_at_n50": 50 / (50 + p.strength),
                "obs_weight_at_n150": 150 / (150 + p.strength),
            })
        return pd.DataFrame(rows)
