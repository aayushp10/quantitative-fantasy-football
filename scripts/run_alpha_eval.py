"""
Stage B evaluation: does the fundamental model predict market errors
out of sample?

Runs the full honest walk-forward from src/models/alpha.py and writes to
output/experiments/alpha_v1/:

    config.json         settings, feature list, seasons, join rates
    metrics.json        residual IC / incremental IC / lambda / hit rates
    predictions.parquet one row per (player, pair-season) with market_expected,
                        fund_pred, pred_residual, lam_wf, fair, residual

All ICs are Spearman among ADP-joined players. Bootstrap CIs are
percentile intervals over player-row resamples (paired within a resample
for the incremental IC); the pooled residual IC also gets a
season-cluster bootstrap since rows within a season share a cross-section.

Usage:
    .venv/bin/python scripts/run_alpha_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import POSITIONS, TRAINING_SEASONS
from data.adp import load_adp
from features.assembler import assemble_feature_matrix, build_yoy_pairs
from models.alpha import (
    ALPHA_FEATURES,
    DEFAULT_ALPHA_MIN_TRAIN_SEASONS,
    DEFAULT_ISO_MIN_TRAIN_SEASONS,
    DEFAULT_LAMBDA_MIN_TRAIN_SEASONS,
    attach_market,
    estimate_lambda,
    run_alpha_walkforward,
)

OUT_DIR = ROOT / "output" / "experiments" / "alpha_v1"
RIDGE_ALPHA = 10.0
N_BOOT = 2000
SEED = 7

# FFC archives on disk: 2013-2024 (no 2025 archive year -> the 2024
# pair-season is excluded from market evaluation, matching notebook 06).
ADP_YEARS = list(range(2013, 2025))


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def _sp(a, b) -> float:
    ic, _ = spearmanr(a, b)
    return float(ic)


def spearman_with_ci(a, b, rng, n_boot: int = N_BOOT) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 10:
        return {"ic": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": n}
    ic = _sp(a, b)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(a[idx]) == 0 or np.std(b[idx]) == 0:
            continue
        boots.append(_sp(a[idx], b[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return {"ic": round(ic, 4), "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4), "n": n}


def incremental_ic_with_ci(fair, market, actual, rng, n_boot: int = N_BOOT) -> dict:
    """Spearman(fair, actual) - Spearman(market, actual), paired bootstrap."""
    fair = np.asarray(fair, dtype=float)
    market = np.asarray(market, dtype=float)
    actual = np.asarray(actual, dtype=float)
    ok = np.isfinite(fair) & np.isfinite(market) & np.isfinite(actual)
    fair, market, actual = fair[ok], market[ok], actual[ok]
    n = len(fair)
    if n < 10:
        return {"fair_ic": np.nan, "market_ic": np.nan, "inc_ic": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "n": n}
    fair_ic, market_ic = _sp(fair, actual), _sp(market, actual)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(actual[idx]) == 0:
            continue
        boots.append(_sp(fair[idx], actual[idx]) - _sp(market[idx], actual[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return {"fair_ic": round(fair_ic, 4), "market_ic": round(market_ic, 4),
            "inc_ic": round(fair_ic - market_ic, 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4), "n": n}


def season_cluster_ic_ci(df, pred_col, real_col, rng, n_boot: int = N_BOOT) -> dict:
    """Pooled IC with a bootstrap that resamples SEASONS (clusters)."""
    seasons = sorted(df["season"].unique())
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(seasons, size=len(seasons), replace=True)
        sub = pd.concat([df[df["season"] == s] for s in pick])
        if sub[pred_col].std() == 0 or sub[real_col].std() == 0:
            continue
        boots.append(_sp(sub[pred_col], sub[real_col]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return {"cluster_ci_lo": round(float(lo), 4), "cluster_ci_hi": round(float(hi), 4),
            "n_seasons": len(seasons)}


def edge_hit_rate(sub: pd.DataFrame, top_frac: float = 0.10) -> dict:
    """Among top-decile |pred_residual|: sign(realized) == sign(pred)."""
    s = sub.dropna(subset=["pred_residual", "residual"])
    if len(s) < 10:
        return {"hit_rate": np.nan, "k": 0, "n": len(s)}
    k = max(5, int(round(top_frac * len(s))))
    top = s.reindex(s["pred_residual"].abs().sort_values(ascending=False).index).head(k)
    hits = float((np.sign(top["pred_residual"]) == np.sign(top["residual"])).mean())
    return {"hit_rate": round(hits, 4), "k": int(k), "n": int(len(s))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading feature matrix (cache) ...")
    fm = assemble_feature_matrix(TRAINING_SEASONS)
    pairs = build_yoy_pairs(fm)
    print(f"Pairs: {len(pairs)} rows, seasons {pairs['season'].min()}-{pairs['season'].max()}")

    adp = load_adp(ADP_YEARS)
    print(f"ADP: {len(adp)} rows, draft years {sorted(adp['season'].unique())}")

    # Join-rate diagnostics on the pair-seasons that CAN have ADP
    joined = attach_market(pairs, adp)
    joinable = joined[joined["season"].between(min(ADP_YEARS) - 1, max(ADP_YEARS) - 1)]
    join_rate = float(joinable["adp_matched"].mean())
    print(f"ADP join rate over pair-seasons {min(ADP_YEARS)-1}-{max(ADP_YEARS)-1}: "
          f"{join_rate:.1%}")

    preds = run_alpha_walkforward(
        joined, adp, ridge_alpha=RIDGE_ALPHA,
    )

    scored = preds[preds["pred_residual"].notna()].copy()
    alpha_seasons = sorted(scored["season"].unique())
    fair_df = preds[preds["fair"].notna()].copy()
    fair_seasons = sorted(fair_df["season"].unique())

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics: dict = {}

    # Residual IC per season + pooled
    per_season = {}
    for s in alpha_seasons:
        sub = scored[scored["season"] == s]
        row = spearman_with_ci(sub["pred_residual"], sub["residual"], rng)
        row.update(edge_hit_rate(sub))
        per_season[int(s)] = row
    pooled = spearman_with_ci(scored["pred_residual"], scored["residual"], rng)
    pooled.update(season_cluster_ic_ci(scored, "pred_residual", "residual", rng))
    pooled_hit = edge_hit_rate(scored)
    season_ics = [v["ic"] for v in per_season.values() if np.isfinite(v["ic"])]
    metrics["residual_ic"] = {
        "per_season": per_season,
        "pooled": pooled,
        "mean_of_seasons": round(float(np.mean(season_ics)), 4),
        "pct_seasons_positive": round(float(np.mean([ic > 0 for ic in season_ics])), 3),
        "pooled_edge_hit_rate": pooled_hit,
    }

    # Per-position residual IC (pooled across alpha seasons)
    metrics["residual_ic_by_position"] = {
        pos: spearman_with_ci(
            scored.loc[scored["position"] == pos, "pred_residual"],
            scored.loc[scored["position"] == pos, "residual"], rng)
        for pos in POSITIONS
    }

    # Incremental IC (fair vs market) — only seasons with a walk-forward lambda
    inc_per_season = {
        int(s): incremental_ic_with_ci(
            fair_df.loc[fair_df["season"] == s, "fair"],
            fair_df.loc[fair_df["season"] == s, "market_expected"],
            fair_df.loc[fair_df["season"] == s, "next_fpts"], rng)
        for s in fair_seasons
    }
    inc_pooled = incremental_ic_with_ci(
        fair_df["fair"], fair_df["market_expected"], fair_df["next_fpts"], rng)
    inc_ics = [v["inc_ic"] for v in inc_per_season.values() if np.isfinite(v["inc_ic"])]
    metrics["incremental_ic"] = {
        "per_season": inc_per_season,
        "pooled": inc_pooled,
        "mean_of_seasons": round(float(np.mean(inc_ics)), 4) if inc_ics else np.nan,
        "pct_seasons_positive": round(float(np.mean([x > 0 for x in inc_ics])), 3)
        if inc_ics else np.nan,
    }

    # ADP-only baseline IC on the alpha-scored universe
    metrics["adp_baseline_ic"] = {
        "per_season": {
            int(s): spearman_with_ci(
                scored.loc[scored["season"] == s, "market_expected"],
                scored.loc[scored["season"] == s, "next_fpts"], rng)
            for s in alpha_seasons
        },
        "pooled": spearman_with_ci(scored["market_expected"], scored["next_fpts"], rng),
    }

    # Lambda: pooled and per-position over all alpha-scored rows, plus the
    # walk-forward values actually used per season
    metrics["lambda"] = {
        "pooled": {k: (round(v, 4) if np.isfinite(v) else None)
                   for k, v in estimate_lambda(scored["pred_residual"],
                                               scored["residual"]).items()},
        "per_position": {
            pos: {k: (round(v, 4) if np.isfinite(v) else None)
                  for k, v in estimate_lambda(
                      scored.loc[scored["position"] == pos, "pred_residual"],
                      scored.loc[scored["position"] == pos, "residual"]).items()}
            for pos in POSITIONS
        },
        "walk_forward_used": {
            int(s): round(float(fair_df.loc[fair_df["season"] == s, "lam_wf"].iloc[0]), 4)
            for s in fair_seasons
        },
    }

    # Sample sizes per season / position
    metrics["sample_sizes"] = {
        "alpha_scored": {
            int(s): {pos: int(((scored["season"] == s) &
                               (scored["position"] == pos)).sum())
                     for pos in POSITIONS} | {
                "total": int((scored["season"] == s).sum())}
            for s in alpha_seasons
        },
        "market_oos_total_rows": int(len(preds)),
        "alpha_scored_rows": int(len(scored)),
        "fair_rows": int(len(fair_df)),
    }

    metrics["runtime_seconds"] = round(time.time() - t0, 1)

    # ------------------------------------------------------------------
    # Verdict gate: residual IC
    # ------------------------------------------------------------------
    m_ic, lo = pooled["ic"], pooled["ci_lo"]
    pct_pos = metrics["residual_ic"]["pct_seasons_positive"]
    if np.isfinite(m_ic) and m_ic > 0 and lo > 0 and pct_pos >= 0.7:
        verdict = "PASS"
    elif not np.isfinite(m_ic) or abs(m_ic) < 0.03 or (lo < 0 < pooled["ci_hi"]):
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS" if m_ic > 0 else "FAIL"
    metrics["verdict"] = {
        "residual_ic_gate": verdict,
        "rule": "PASS if pooled residual IC > 0 with 95% row-bootstrap CI "
                "excluding 0 and >=70% of seasons positive; FAIL if clearly "
                "negative; else INCONCLUSIVE.",
    }

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    config = {
        "experiment": "alpha_v1",
        "date": pd.Timestamp.now().isoformat(),
        "target": "next_fpts (PPR points per game, season N+1)",
        "residual_definition": "next_fpts - market_expected (OOS isotonic ADP map)",
        "alpha_model": {
            "type": "Ridge, pooled across positions with position dummies",
            "ridge_alpha": RIDGE_ALPHA,
            "features": ALPHA_FEATURES,
            "dummies": [p for p in POSITIONS if p != "QB"],
            "imputation": "train-median, then z-scored with train stats",
        },
        "protocol": {
            "iso_min_train_seasons": DEFAULT_ISO_MIN_TRAIN_SEASONS,
            "alpha_min_train_seasons": DEFAULT_ALPHA_MIN_TRAIN_SEASONS,
            "lambda_min_train_seasons": DEFAULT_LAMBDA_MIN_TRAIN_SEASONS,
            "fundamental": "HybridProjectionModel(age_adjust=False), trained per "
                           "test season on earlier pair-seasons only (fit_age=False)",
            "lambda": "walk-forward pooled OLS slope of realized on predicted "
                      "residual from earlier alpha seasons, clipped to [0,1]",
        },
        "seasons": {
            "adp_draft_years": ADP_YEARS,
            "market_oos_pair_seasons": sorted(int(s) for s in preds["season"].unique()),
            "alpha_scored_pair_seasons": [int(s) for s in alpha_seasons],
            "fair_pair_seasons": [int(s) for s in fair_seasons],
            "excluded": "2024 pair-season (no 2025 FFC ADP archive)",
        },
        "join_rate_adp_pairs": round(join_rate, 4),
        "n_boot": N_BOOT,
        "seed": SEED,
    }

    keep_cols = [c for c in [
        "player_id", "player_name", "position", "season", "adp", "adp_pos_rank",
        "adp_stdev", "next_fpts", "market_expected", "fund_pred", "fund_gap",
        "residual", "pred_residual", "lam_wf", "fair",
    ] + [f for f in ALPHA_FEATURES if f not in ("fund_gap", "log_adp", "adp_stdev")]
        if c in preds.columns]
    preds[keep_cols].to_parquet(OUT_DIR / "predictions.parquet", index=False)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        raise TypeError(f"not serializable: {type(o)}")

    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2, default=_default))
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_default))

    print(f"\nArtifacts written to {OUT_DIR}")
    print(f"Residual IC pooled: {pooled['ic']} "
          f"[{pooled['ci_lo']}, {pooled['ci_hi']}] (row bootstrap), "
          f"season-cluster CI [{pooled.get('cluster_ci_lo')}, {pooled.get('cluster_ci_hi')}]")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
