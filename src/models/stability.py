"""
Year-over-year factor stability analysis.

The Information Coefficient (IC) here is the Spearman rank correlation
between a factor value in season N and fpts_per_game in season N+1.
This is the BARRA-style factor quality metric.

Factors with high IC (strong predictive signal) should be prioritized in
the Ridge regression model. Factors with IC IR < 0.5 are candidates for
exclusion.

Stability tiers:
  STRONG   — |mean IC| > 0.6
  MODERATE — 0.3 <= |mean IC| <= 0.6
  WEAK     — |mean IC| < 0.3
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

def compute_factor_stability(
    yoy_df: pd.DataFrame,
    factors: list[str],
    min_sample: int = 10,
) -> pd.DataFrame:
    """
    Compute per-season IC and summary statistics for each factor.

    Parameters
    ----------
    yoy_df : pd.DataFrame
        YoY pairs from assembler.build_yoy_pairs(). Must have columns:
        'season', 'next_fpts', and all factor columns.
    factors : list[str]
        Factor column names to evaluate.
    min_sample : int
        Minimum valid (non-NaN) pairs required to compute IC for a
        season-factor combination.

    Returns
    -------
    pd.DataFrame
        Index: factor names. Columns: mean_ic, std_ic, ic_ir, pearson_r,
        n_seasons, stability_tier.
        Also sets 'ic_by_season' as a sub-DataFrame attribute for the heatmap.
    """
    if "next_fpts" not in yoy_df.columns:
        raise ValueError("yoy_df must have a 'next_fpts' column (from build_yoy_pairs).")

    target = "next_fpts"
    seasons = sorted(yoy_df["season"].unique())

    rows = []
    season_ic_dict: dict[str, dict] = {f: {} for f in factors}

    for factor in factors:
        if factor not in yoy_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(yoy_df[factor]):
            continue
        for season in seasons:
            subset = yoy_df[yoy_df["season"] == season][[factor, target]].dropna()
            if len(subset) < min_sample:
                season_ic_dict[factor][season] = np.nan
                continue
            ic, _ = spearmanr(subset[factor], subset[target])
            season_ic_dict[factor][season] = ic

        ics = [v for v in season_ic_dict[factor].values() if not np.isnan(v)]
        if not ics:
            rows.append({
                "factor": factor,
                "mean_ic": np.nan,
                "std_ic": np.nan,
                "ic_ir": np.nan,
                "pearson_r": np.nan,
                "n_seasons": 0,
                "stability_tier": "INSUFFICIENT DATA",
            })
            continue

        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        ic_ir = mean_ic / std_ic if std_ic > 0 else np.nan

        # Pearson as robustness check
        all_valid = yoy_df[[factor, target]].dropna()
        if len(all_valid) >= min_sample:
            pearson_r, _ = pearsonr(all_valid[factor], all_valid[target])
        else:
            pearson_r = np.nan

        if abs(mean_ic) > 0.6:
            tier = "STRONG"
        elif abs(mean_ic) >= 0.3:
            tier = "MODERATE"
        else:
            tier = "WEAK"

        rows.append({
            "factor": factor,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "ic_ir": ic_ir,
            "pearson_r": pearson_r,
            "n_seasons": len(ics),
            "stability_tier": tier,
        })

    summary = pd.DataFrame(rows).set_index("factor")
    summary = summary.sort_values("mean_ic", ascending=False)

    # Attach season-level IC as a separate DataFrame (used for heatmap)
    ic_by_season = pd.DataFrame(season_ic_dict).T  # factors × seasons
    ic_by_season.columns = [str(c) for c in ic_by_season.columns]
    summary.attrs["ic_by_season"] = ic_by_season

    return summary


# ---------------------------------------------------------------------------
# YoY correlation (factor-to-factor, not factor-to-outcome)
# ---------------------------------------------------------------------------

def compute_yoy_factor_correlation(
    feature_matrix: pd.DataFrame,
    factors: list[str],
    min_sample: int = 50,
) -> pd.DataFrame:
    """
    Compute year-over-year Pearson/Spearman correlation for each factor
    (factor_year_N vs factor_year_N+1 for same player).

    This measures factor "stickiness" — how much a player's factor value
    persists from season to season.

    Returns
    -------
    pd.DataFrame
        One row per factor with pearson_yoy, spearman_yoy, n_pairs, tier.
    """
    rows = []
    fm = feature_matrix.sort_values(["player_id", "season"])

    for factor in factors:
        if factor not in fm.columns:
            continue
        if not pd.api.types.is_numeric_dtype(fm[factor]):
            continue

        # Build (factor_N, factor_N+1) pairs
        shifted = (
            fm.groupby("player_id", observed=True)[["season", factor]]
            .apply(lambda g: g.assign(
                next_val=g[factor].shift(-1),
                next_season=g["season"].shift(-1),
            ), include_groups=False)
            .reset_index(drop=True)
        )
        valid = shifted[
            (shifted["next_season"] == shifted["season"] + 1) &
            shifted[factor].notna() &
            shifted["next_val"].notna()
        ]

        if len(valid) < min_sample:
            rows.append({"factor": factor, "pearson_yoy": np.nan, "spearman_yoy": np.nan,
                         "n_pairs": len(valid), "yoy_tier": "INSUFFICIENT DATA"})
            continue

        pr, _ = pearsonr(valid[factor], valid["next_val"])
        sr, _ = spearmanr(valid[factor], valid["next_val"])

        if abs(pr) > 0.6:
            tier = "STRONG"
        elif abs(pr) >= 0.3:
            tier = "MODERATE"
        else:
            tier = "WEAK"

        rows.append({
            "factor": factor,
            "pearson_yoy": pr,
            "spearman_yoy": sr,
            "n_pairs": len(valid),
            "yoy_tier": tier,
        })

    return pd.DataFrame(rows).set_index("factor").sort_values("pearson_yoy", ascending=False)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_stability_heatmap(
    stability_df: pd.DataFrame,
    title: str = "Factor IC by Season",
) -> plt.Figure:
    """
    Plot a heatmap of IC values: factors (rows) × seasons (columns).

    Parameters
    ----------
    stability_df : pd.DataFrame
        Output from compute_factor_stability(). Must have
        stability_df.attrs['ic_by_season'].
    title : str
        Plot title.

    Returns
    -------
    matplotlib.Figure
    """
    ic_by_season = stability_df.attrs.get("ic_by_season")
    if ic_by_season is None:
        raise ValueError("stability_df does not have 'ic_by_season' attribute. "
                         "Run compute_factor_stability() first.")

    # Sort by mean IC
    sorted_factors = stability_df.sort_values("mean_ic", ascending=False).index
    ic_by_season = ic_by_season.reindex(sorted_factors)

    fig, ax = plt.subplots(figsize=(max(8, len(ic_by_season.columns) * 1.2),
                                    max(6, len(sorted_factors) * 0.4)))
    sns.heatmap(
        ic_by_season.astype(float),
        ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=-0.8,
        vmax=0.8,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Spearman IC"},
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Season", fontsize=11)
    ax.set_ylabel("Factor", fontsize=11)
    plt.tight_layout()
    return fig


def plot_factor_ic_bar(
    stability_df: pd.DataFrame,
    top_n: int | None = None,
    title: str = "Mean Factor IC (Spearman)",
) -> plt.Figure:
    """
    Bar chart of mean IC per factor, sorted descending.
    """
    df = stability_df.dropna(subset=["mean_ic"]).sort_values("mean_ic", ascending=True)
    if top_n:
        df = df.tail(top_n)

    colors = df["mean_ic"].apply(
        lambda x: "#2ecc71" if x > 0.3 else ("#f39c12" if x > 0 else "#e74c3c")
    )

    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.35)))
    ax.barh(df.index, df["mean_ic"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.3, color="green", linewidth=0.8, linestyle="--", label="Moderate threshold")
    ax.axvline(0.6, color="darkgreen", linewidth=0.8, linestyle="--", label="Strong threshold")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Mean Spearman IC")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig
