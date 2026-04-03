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

def fit_age_curves(
    features_df: pd.DataFrame,
    min_player_seasons: int = 50,
) -> dict[str, dict[str, float]]:
    """
    Fit empirical aging curves from the feature matrix.

    For each position, fit a quadratic curve to the (age, fpts_per_game)
    data using scipy.optimize.curve_fit. Falls back to hardcoded priors
    when fewer than min_player_seasons data points are available.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output from assemble_feature_matrix(). Must have columns:
        'position', 'age', 'fpts_per_game'.
    min_player_seasons : int
        Minimum data points required to fit a curve for a position.

    Returns
    -------
    dict
        {position: {'peak_age': float, 'decay_rate': float}}
        These can be passed to get_age_multiplier() to override priors.
    """
    required = {"position", "age", "fpts_per_game"}
    missing = required - set(features_df.columns)
    if missing:
        raise ValueError(f"features_df missing columns: {missing}")

    fitted = {}

    for pos in POSITIONS:
        pos_df = features_df[
            (features_df["position"] == pos) &
            features_df["age"].notna() &
            features_df["fpts_per_game"].notna()
        ].copy()

        if len(pos_df) < min_player_seasons:
            warnings.warn(
                f"Only {len(pos_df)} data points for {pos} (need {min_player_seasons}). "
                "Using hardcoded priors."
            )
            fitted[pos] = {"peak_age": PEAK_AGES[pos], "decay_rate": AGE_DECAY_RATES[pos]}
            continue

        # Normalize fpts_per_game to [0, 1] range for fitting
        max_fpts = pos_df["fpts_per_game"].quantile(0.95)
        pos_df["fpts_norm"] = (pos_df["fpts_per_game"] / max_fpts).clip(0, 1.5)

        ages = pos_df["age"].values
        fpts = pos_df["fpts_norm"].values

        try:
            popt, _ = curve_fit(
                _quadratic_curve,
                ages,
                fpts,
                p0=[PEAK_AGES[pos], AGE_DECAY_RATES[pos]],
                bounds=([20, 0.001], [35, 0.1]),
                maxfev=5000,
            )
            peak_fit, decay_fit = popt
            fitted[pos] = {"peak_age": float(peak_fit), "decay_rate": float(decay_fit)}
            print(f"  {pos}: fitted peak={peak_fit:.1f}, decay={decay_fit:.4f} "
                  f"(prior: peak={PEAK_AGES[pos]}, decay={AGE_DECAY_RATES[pos]})")
        except RuntimeError as e:
            warnings.warn(f"Curve fitting failed for {pos}: {e}. Using hardcoded priors.")
            fitted[pos] = {"peak_age": PEAK_AGES[pos], "decay_rate": AGE_DECAY_RATES[pos]}

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
