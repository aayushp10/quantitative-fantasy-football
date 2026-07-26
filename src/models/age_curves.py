"""
Empirical aging curves by position.

Provides multiplicative adjustments to projections based on player age.
Starting point is hardcoded priors; fit_age_curves() refines them from data.

Adjustment formula (quadratic decay from peak):
  multiplier = max(0.5, 1.0 - decay_rate * (age - peak_age)^2)

Multiplier interpretation:
  1.0 = peak-age performance
  0.9 = 10% below peak
  0.5 = minimum floor (prevents projecting zero for old players)
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from config import AGE_DECAY_RATES, PEAK_AGES, POSITIONS


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def _quadratic_curve(age: np.ndarray, peak: float, decay: float) -> np.ndarray:
    """Quadratic aging curve normalized to 1.0 at peak."""
    return np.maximum(0.5, 1.0 - decay * (age - peak) ** 2)


def get_age_multiplier(position: str, age: float) -> float:
    """
    Return the aging multiplier for a player at the given age.

    Uses the hardcoded priors from config.py unless fit_age_curves()
    has been called and its results stored separately.

    Parameters
    ----------
    position : str
        One of 'QB', 'RB', 'WR', 'TE'.
    age : float
        Player age at the start of the season (September 1).

    Returns
    -------
    float
        Multiplicative adjustment relative to peak-age baseline.
        Range: [0.5, ~1.05] (slight overshoot possible for pre-peak ages).
    """
    if position not in PEAK_AGES:
        return 1.0

    peak = PEAK_AGES[position]
    decay = AGE_DECAY_RATES[position]
    return float(np.maximum(0.5, 1.0 - decay * (age - peak) ** 2))


# ---------------------------------------------------------------------------
# Empirical curve fitting
# ---------------------------------------------------------------------------

def _age_deltas(pos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Within-player year-over-year (age, delta fpts_per_game) pairs.

    Accepts either YoY pairs (has 'next_fpts': each row is already a
    transition) or a raw feature matrix (consecutive seasons are paired
    within player via shift).
    """
    if "next_fpts" in pos_df.columns:
        df = pos_df.dropna(subset=["age", "fpts_per_game", "next_fpts"]).copy()
        df["_delta"] = df["next_fpts"] - df["fpts_per_game"]
        return df[["age", "_delta"]]

    df = pos_df.dropna(subset=["age", "fpts_per_game"]).copy()
    df = df.sort_values(["player_id", "season"])
    grp = df.groupby("player_id", observed=True)
    df["_next_val"] = grp["fpts_per_game"].shift(-1)
    df["_next_season"] = grp["season"].shift(-1)
    df = df[(df["_next_season"] == df["season"] + 1) & df["_next_val"].notna()]
    df["_delta"] = df["_next_val"] - df["fpts_per_game"]
    return df[["age", "_delta"]]


def fit_age_curves(
    features_df: pd.DataFrame,
    min_pairs: int = 50,
    min_pairs_per_age: int = 8,
    age_range: tuple[int, int] = (21, 38),
) -> dict[str, dict[str, float]]:
    """
    Fit empirical aging curves via the DELTA METHOD.

    The old cross-sectional fit (level vs age) had textbook survivorship
    bias: bad old players retire out of the sample, so the survivors at 30+
    are disproportionately elite — dragging apparent peaks to absurd ages
    (WR 29.6, TE 30.2 on 2012-2024 data) with maxed-out decay to compensate.

    The delta method is immune to that selection: it uses only WITHIN-player
    year-over-year changes ("how did the same player change from age A to
    A+1"), averages the deltas per age, integrates them into a relative
    level curve, and fits the quadratic to THAT curve.

    Parameters
    ----------
    features_df : pd.DataFrame
        Either YoY pairs (preferred — each row is a transition with
        'fpts_per_game', 'next_fpts', 'age') or a feature matrix with
        'player_id', 'season', 'age', 'fpts_per_game'.
    min_pairs : int
        Minimum transitions per position; below this, hardcoded priors.
    min_pairs_per_age : int
        Age buckets with fewer transitions are dropped from the fit.
    age_range : tuple
        Ages considered.

    Returns
    -------
    dict
        {position: {'peak_age': float, 'decay_rate': float}}
    """
    required = {"position", "age", "fpts_per_game"}
    missing = required - set(features_df.columns)
    if missing:
        raise ValueError(f"features_df missing columns: {missing}")

    fitted = {}

    for pos in POSITIONS:
        prior = {"peak_age": PEAK_AGES[pos], "decay_rate": AGE_DECAY_RATES[pos]}
        pos_df = features_df[features_df["position"] == pos]

        deltas = _age_deltas(pos_df)
        deltas = deltas[deltas["age"].between(*age_range)]
        if len(deltas) < min_pairs:
            warnings.warn(
                f"Only {len(deltas)} age transitions for {pos} (need {min_pairs}). "
                "Using hardcoded priors."
            )
            fitted[pos] = prior
            continue

        # Mean delta per integer age (age at the START of the transition)
        deltas["_age_i"] = deltas["age"].round().astype(int)
        by_age = deltas.groupby("_age_i")["_delta"].agg(["mean", "count"])
        by_age = by_age[by_age["count"] >= min_pairs_per_age]
        if len(by_age) < 4:
            fitted[pos] = prior
            continue

        # Integrate deltas into a relative level curve: level at the youngest
        # age is arbitrary; cumulative deltas trace the shape from there.
        ages = np.append(by_age.index.values, by_age.index.values[-1] + 1)
        levels = np.concatenate([[0.0], np.cumsum(by_age["mean"].values)])
        levels = levels - levels.max()

        # Convert to multiplicative scale: peak level -> 1.0. Typical
        # positional per-game scale keeps proportions comparable.
        scale = max(pos_df["fpts_per_game"].median(), 1.0)
        mult = 1.0 + levels / scale
        weights = np.append(by_age["count"].values, by_age["count"].values[-1])

        try:
            popt, _ = curve_fit(
                _quadratic_curve,
                ages.astype(float),
                mult,
                p0=[PEAK_AGES[pos], AGE_DECAY_RATES[pos]],
                sigma=1.0 / np.sqrt(weights),
                bounds=([21, 0.001], [34, 0.1]),
                maxfev=5000,
            )
            peak_fit, decay_fit = popt
            fitted[pos] = {"peak_age": float(peak_fit), "decay_rate": float(decay_fit)}
            print(f"  {pos}: delta-method peak={peak_fit:.1f}, decay={decay_fit:.4f} "
                  f"(prior: peak={PEAK_AGES[pos]}, decay={AGE_DECAY_RATES[pos]}, "
                  f"n={len(deltas)} transitions)")
        except RuntimeError as e:
            warnings.warn(f"Curve fitting failed for {pos}: {e}. Using hardcoded priors.")
            fitted[pos] = prior

    return fitted


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_age_curves(
    fitted_params: dict[str, dict] | None = None,
    age_range: tuple[int, int] = (21, 38),
) -> plt.Figure:
    """
    Plot aging curves for all positions.

    Parameters
    ----------
    fitted_params : dict, optional
        Output from fit_age_curves(). If None, uses hardcoded priors.
    age_range : tuple
        (min_age, max_age) range to plot.
    """
    ages = np.arange(age_range[0], age_range[1] + 1, 0.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"QB": "#3498db", "RB": "#e74c3c", "WR": "#2ecc71", "TE": "#f39c12"}

    for pos in POSITIONS:
        if fitted_params and pos in fitted_params:
            peak = fitted_params[pos]["peak_age"]
            decay = fitted_params[pos]["decay_rate"]
        else:
            peak = PEAK_AGES[pos]
            decay = AGE_DECAY_RATES[pos]

        mults = [float(np.maximum(0.5, 1.0 - decay * (a - peak) ** 2)) for a in ages]
        label = f"{pos} (peak {peak:.0f})"
        linestyle = "--" if fitted_params else "-"
        ax.plot(ages, mults, label=label, color=colors[pos], linewidth=2, linestyle=linestyle)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(0.9, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Player Age", fontsize=12)
    ax.set_ylabel("Aging Multiplier (1.0 = peak)", fontsize=12)
    title = "Empirical Aging Curves" if fitted_params else "Aging Curves (Hardcoded Priors)"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0.45, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def apply_age_adjustments(
    projections: pd.DataFrame,
    fitted_params: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    Apply age multipliers to a projections DataFrame in-place.

    Expects columns: 'position', 'age', 'projected_fpts_pg'.
    Adds 'age_multiplier' and modifies 'projected_fpts_pg'.
    """
    if "age" not in projections.columns or "position" not in projections.columns:
        warnings.warn("projections missing 'age' or 'position' column — skipping age adjustment.")
        return projections

    def _mult(row):
        pos = row["position"]
        age = row["age"]
        # Guard against Series returned when duplicate column names exist
        if isinstance(pos, pd.Series):
            pos = pos.iloc[0]
        if isinstance(age, pd.Series):
            age = age.iloc[0]
        if pd.isna(age) or pos not in POSITIONS:
            return 1.0
        if fitted_params and pos in fitted_params:
            peak = fitted_params[pos]["peak_age"]
            decay = fitted_params[pos]["decay_rate"]
            return float(np.maximum(0.5, 1.0 - decay * (float(age) - peak) ** 2))
        return get_age_multiplier(pos, float(age))

    projections = projections.copy()
    projections["age_multiplier"] = projections.apply(_mult, axis=1)
    if "projected_fpts_pg" in projections.columns:
        projections["projected_fpts_pg"] = (
            projections["projected_fpts_pg"] * projections["age_multiplier"]
        )
    return projections
