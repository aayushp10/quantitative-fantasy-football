"""
Alpha v2: market-residual model in SEASON-POINTS units on
survivor-complete outcomes.

Two fixes over models.alpha (v1), both to the target rather than the
model:

1. Survivor-complete outcomes (data.outcomes / data.training_pairs):
   the residual now includes players who fell out of the league or lost
   their role — the audit shows the old construction dropped ~5% of
   drafted players whose mean next season was ~18 points vs ~188 for the
   kept ones. v1 could not learn the market's largest error class.

2. Season-points residual: the market prices season value (games
   included), so the target is
       residual = next_season_points − isotonic(ADP → season points)
   and the fundamental leg is hybrid PPG × predicted games, making
   availability mispricing visible to the model.

Also new here: per-player shrinkage λᵢ (shrink harder where the market
is confident — early picks, low ADP stdev), conviction z-scores, and
candidate features (prior games missed, positional TD over-expectation,
Vegas week-1 context, coaching change) gated by add-one walk-forward IC.

Reuses from models.alpha: attach_market (ADP+stdev join), AlphaModel
(pooled Ridge machinery), market_expected_oos (generic in its target),
estimate_lambda.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import POSITIONS
from data.outcomes import build_outcomes, team_games
from data.training_pairs import build_training_pairs
from models.alpha import (
    AlphaModel,
    attach_market,
    estimate_lambda,
    market_expected_oos,
)

TARGET = "next_season_points"

BASE_FEATURES: list[str] = [
    "fund_gap",                   # fundamental season pts − market season pts
    "age",
    "log_adp",
    "adp_stdev",
    "team_vacated_target_share",
    "target_share_delta",
    "fpts_pg_yoy_change",
    "years_in_league",
]
CANDIDATE_FEATURES: list[str] = [
    "games_missed_prior",         # availability signal, era-adjusted
    "td_oe_pos",                  # positional TD over-expectation (regression candidate)
    "vegas_implied_delta_next",   # market's view of the offense's change
    "hc_changed_next",            # coaching regime change
]
MIN_KEEP_IC_GAIN = 0.005          # add-one pooled OOS IC gain to keep a candidate


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------

def build_survivor_frame(fm: pd.DataFrame, weekly: pd.DataFrame,
                         adp_df: pd.DataFrame) -> pd.DataFrame:
    """(season-N features) → (season-N+1 survivor-complete outcome), with
    ADP of the target-season draft and the v2 candidate features."""
    from features.schedule_context import add_schedule_context_features

    outcomes = build_outcomes(weekly)
    pairs = build_training_pairs(fm, outcomes)
    pairs = add_schedule_context_features(pairs)          # cache-through schedules
    pairs = attach_market(pairs, adp_df)                  # draft-year ADP + stdev

    pairs["log_adp"] = np.log(pairs["adp"].clip(lower=1.0))
    pairs["games_missed_prior"] = (
        pairs["season"].map(team_games) - pairs["games_played"]
    ).clip(lower=0)
    pairs["td_oe_pos"] = np.select(
        [pairs["position"] == "QB", pairs["position"] == "RB"],
        [pairs.get("pass_td_oe", np.nan), pairs.get("rush_td_oe", np.nan)],
        default=pairs.get("rec_td_oe", np.nan),
    )
    return pairs


def fundamental_season_oos(pairs_old: pd.DataFrame, frame: pd.DataFrame,
                           test_seasons: list[int], verbose: bool = True) -> pd.DataFrame:
    """Walk-forward fundamental SEASON prediction: hybrid PPG (trained on
    earlier played-player pairs) × availability-model games."""
    from models.availability import AvailabilityModel
    from models.hybrid import HybridProjectionModel

    rows = []
    for s in sorted(test_seasons):
        train = pairs_old[pairs_old["season"] < s]
        test = frame[frame["season"] == s]
        if test.empty or train["season"].nunique() < 2:
            continue
        if verbose:
            print(f"fundamental_season_oos: season {s} (train n={len(train)})")
        hy = HybridProjectionModel(age_adjust=False)
        hy.train(train, target="next_fpts", fit_age=False)
        try:
            avail = AvailabilityModel().train(train)
        except Exception:
            avail = None
        for pos in POSITIONS:
            pos_test = test[test["position"] == pos].reset_index(drop=True)
            if pos_test.empty:
                continue
            try:
                ppg = hy.predict_position(pos, pos_test)
            except Exception:
                ppg = None
            if ppg is None:
                continue
            if avail is not None:
                try:
                    games = avail.predict_games(pos_test, target_season=s + 1)
                except Exception:
                    games = np.full(len(pos_test), team_games(s + 1))
            else:
                games = np.full(len(pos_test), team_games(s + 1))
            rows.append(pd.DataFrame({
                "player_id": pos_test["player_id"].values,
                "season": s,
                "position": pos,
                "fund_season": np.asarray(ppg, float) * np.asarray(games, float),
            }))
    if not rows:
        return pd.DataFrame(columns=["player_id", "season", "position", "fund_season"])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-player shrinkage
# ---------------------------------------------------------------------------

class LambdaShrinkage:
    """λᵢ = clip(b0 + b1·log_adp + b2·adp_stdev, 0, 1): regress realized
    residual on predicted residual interacted with market-confidence
    covariates. Falls back to the scalar OLS slope when the interaction
    fit is unavailable or degenerate."""

    COVARIATES = ["log_adp", "adp_stdev"]

    def __init__(self):
        self.coef_: np.ndarray | None = None   # [a, b0, b1, b2]
        self.scalar_: float = np.nan
        self.cov_medians_: pd.Series | None = None

    def fit(self, df: pd.DataFrame, pred_col: str = "pred_residual",
            real_col: str = "residual") -> "LambdaShrinkage":
        sub = df[df[pred_col].notna() & df[real_col].notna()]
        self.scalar_ = estimate_lambda(sub[pred_col].values, sub[real_col].values)["lam"]
        if len(sub) < 60:
            return self
        self.cov_medians_ = sub[self.COVARIATES].median()
        C = sub[self.COVARIATES].fillna(self.cov_medians_).values
        p = sub[pred_col].values
        X = np.column_stack([np.ones(len(sub)), p, p * C[:, 0], p * C[:, 1]])
        try:
            beta, *_ = np.linalg.lstsq(X, sub[real_col].values, rcond=None)
            self.coef_ = beta
        except np.linalg.LinAlgError:
            self.coef_ = None
        return self

    def lam(self, df: pd.DataFrame) -> np.ndarray:
        base = self.scalar_ if np.isfinite(self.scalar_) else 0.0
        if self.coef_ is None:
            return np.full(len(df), np.clip(base, 0.0, 1.0))
        C = df[self.COVARIATES].fillna(self.cov_medians_).values
        raw = self.coef_[1] + self.coef_[2] * C[:, 0] + self.coef_[3] * C[:, 1]
        return np.clip(raw, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def run_alpha_v2_walkforward(
    frame: pd.DataFrame,
    pairs_old: pd.DataFrame,
    features: list[str] | None = None,
    ridge_alpha: float = 10.0,
    iso_min_train_seasons: int = 2,
    alpha_min_train_seasons: int = 3,
    lambda_min_train_seasons: int = 2,
    fund_df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Honest walk-forward of the v2 stack. Returns (wf, fund_df) so
    feature-subset reruns can reuse the expensive fundamental leg."""
    feats = list(BASE_FEATURES) if features is None else list(features)

    mkt = market_expected_oos(frame, adp_df=None, target=TARGET,
                              min_train_seasons=iso_min_train_seasons)
    mkt_seasons = sorted(mkt["season"].unique())
    if fund_df is None:
        fund_df = fundamental_season_oos(pairs_old, frame, mkt_seasons, verbose=verbose)

    df = mkt.merge(fund_df, on=["player_id", "season", "position"], how="left")
    df["residual"] = df[TARGET] - df["market_expected"]
    df["fund_gap"] = df["fund_season"] - df["market_expected"]
    df = df[df["fund_gap"].notna()].reset_index(drop=True)

    df["pred_residual"] = np.nan
    df["lam_i"] = np.nan
    alpha_seasons = [s for s in mkt_seasons
                     if len([t for t in mkt_seasons if t < s]) >= alpha_min_train_seasons]
    for s in alpha_seasons:
        train = df[df["season"] < s]
        mask = df["season"] == s
        if not mask.any() or len(train) < 30:
            continue
        model = AlphaModel(features=feats, ridge_alpha=ridge_alpha)
        model.fit(train, target="residual")
        df.loc[mask, "pred_residual"] = model.predict(df[mask])

        prior = df[(df["season"] < s) & df["pred_residual"].notna()]
        if prior["season"].nunique() >= lambda_min_train_seasons:
            lm = LambdaShrinkage().fit(prior)
            df.loc[mask, "lam_i"] = lm.lam(df[mask])

    df["fair"] = df["market_expected"] + df["lam_i"] * df["pred_residual"]
    if verbose:
        n = int(df["pred_residual"].notna().sum())
        print(f"alpha v2 walk-forward: {n} scored rows over "
              f"{df.loc[df['pred_residual'].notna(), 'season'].nunique()} seasons")
    return df, fund_df


def select_features(frame: pd.DataFrame, pairs_old: pd.DataFrame,
                    verbose: bool = True) -> tuple[list[str], dict, pd.DataFrame]:
    """Add-one walk-forward gate: keep a candidate only if it lifts pooled
    OOS residual IC by MIN_KEEP_IC_GAIN. The fundamental leg is computed
    once and shared, so each variant is cheap. Returns (features, report,
    fund_df) so callers can reuse the expensive fundamental predictions."""
    from scipy.stats import spearmanr

    def pooled_ic(wf: pd.DataFrame) -> float:
        s = wf[wf["pred_residual"].notna()]
        return float(spearmanr(s["pred_residual"], s["residual"]).statistic)

    base_wf, fund_df = run_alpha_v2_walkforward(frame, pairs_old,
                                                features=BASE_FEATURES, verbose=verbose)
    base_ic = pooled_ic(base_wf)
    report = {"base_ic": base_ic, "candidates": {}}
    kept = []
    for c in CANDIDATE_FEATURES:
        cov = frame[c].notna().mean() if c in frame.columns else 0.0
        if cov < 0.30:
            report["candidates"][c] = {"ic": None, "gain": None,
                                       "kept": False, "coverage": round(float(cov), 2)}
            continue
        wf, _ = run_alpha_v2_walkforward(frame, pairs_old,
                                         features=BASE_FEATURES + [c],
                                         fund_df=fund_df, verbose=False)
        ic = pooled_ic(wf)
        keep = (ic - base_ic) >= MIN_KEEP_IC_GAIN
        report["candidates"][c] = {"ic": round(ic, 4), "gain": round(ic - base_ic, 4),
                                   "kept": keep, "coverage": round(float(cov), 2)}
        if keep:
            kept.append(c)
        if verbose:
            print(f"  candidate {c:26s} IC {ic:+.4f} (gain {ic - base_ic:+.4f}) "
                  f"cov {cov:.0%} -> {'KEEP' if keep else 'drop'}")
    report["selected"] = BASE_FEATURES + kept
    return BASE_FEATURES + kept, report, fund_df


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

def build_serving_alpha_v2(
    fm: pd.DataFrame,
    weekly: pd.DataFrame,
    adp_history: pd.DataFrame,
    adp_now: pd.DataFrame,
    features_now: pd.DataFrame,
    pairs_old: pd.DataFrame,
    features: list[str],
    wf_df: pd.DataFrame | None = None,
    frame: pd.DataFrame | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Fit the v2 stack on everything; apply to the upcoming season.

    Returns (current, diagnostics, wf_df):
        current : [player_id, position, adp, adp_stdev, market_season,
                   pred_residual, lam_i, fair_season, alpha_z, alpha_source]
    """
    from sklearn.isotonic import IsotonicRegression
    from config import RECENCY_DECAY
    from features.schedule_context import add_schedule_context_features
    from models.alpha import MIN_ISO_ROWS
    from models.availability import AvailabilityModel
    from models.hybrid import HybridProjectionModel

    if frame is None:
        frame = build_survivor_frame(fm, weekly, adp_history)
    if wf_df is None:
        wf_df, _ = run_alpha_v2_walkforward(frame, pairs_old, features=features,
                                            verbose=verbose)

    # Final alpha + λᵢ on all OOS rows
    final_alpha = AlphaModel(features=features).fit(wf_df, target="residual")
    lam_model = LambdaShrinkage().fit(wf_df[wf_df["pred_residual"].notna()])

    # Per-position OOS residual dispersion (for conviction z-scores)
    resid_std = {
        pos: float(grp["residual"].std(ddof=1))
        for pos, grp in wf_df.groupby("position") if len(grp) > 20
    }

    # Final season-points market curve on all priced survivor pairs
    train = frame[frame["adp"].notna() & frame[TARGET].notna()]
    max_season = int(train["season"].max())
    iso_by_pos, iso_n = {}, {}
    for pos in POSITIONS:
        sub = train[train["position"] == pos]
        if len(sub) < MIN_ISO_ROWS:
            continue
        w = np.power(RECENCY_DECAY, max_season - sub["season"].values)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(sub["adp"].values, sub[TARGET].values, sample_weight=w)
        iso_by_pos[pos], iso_n[pos] = iso, len(sub)

    # Current season: features priced at the upcoming draft
    cur = add_schedule_context_features(features_now)
    cur = attach_market(cur, adp_now, season_offset=1)
    cur = cur[cur["adp"].notna() & cur["position"].isin(iso_by_pos)].copy()
    cur["log_adp"] = np.log(cur["adp"].clip(lower=1.0))
    cur["games_missed_prior"] = (
        cur["season"].map(team_games) - cur["games_played"]).clip(lower=0)
    cur["td_oe_pos"] = np.select(
        [cur["position"] == "QB", cur["position"] == "RB"],
        [cur.get("pass_td_oe", np.nan), cur.get("rush_td_oe", np.nan)],
        default=cur.get("rec_td_oe", np.nan),
    )
    cur["market_season"] = np.nan
    for pos, iso in iso_by_pos.items():
        m = cur["position"] == pos
        if m.any():
            cur.loc[m, "market_season"] = iso.predict(cur.loc[m, "adp"].values)

    # Fundamental season leg, mirroring the walk-forward protocol
    hy = HybridProjectionModel(age_adjust=False)
    hy.train(pairs_old, target="next_fpts", fit_age=False)
    try:
        avail = AvailabilityModel().train(pairs_old)
    except Exception:
        avail = None
    target_season = int(cur["season"].max()) + 1
    cur["fund_season"] = np.nan
    for pos in iso_by_pos:
        pos_df = cur[cur["position"] == pos].reset_index()
        if pos_df.empty:
            continue
        try:
            ppg = hy.predict_position(pos, pos_df)
        except Exception:
            ppg = None
        if ppg is None:
            continue
        if avail is not None:
            try:
                games = avail.predict_games(pos_df, target_season=target_season)
            except Exception:
                games = np.full(len(pos_df), team_games(target_season))
        else:
            games = np.full(len(pos_df), team_games(target_season))
        cur.loc[pos_df["index"], "fund_season"] = np.asarray(ppg, float) * np.asarray(games, float)

    cur["fund_gap"] = cur["fund_season"] - cur["market_season"]

    scorable = cur["fund_gap"].notna() & cur["market_season"].notna()
    cur["pred_residual"] = np.nan
    if scorable.any():
        cur.loc[scorable, "pred_residual"] = final_alpha.predict(cur[scorable])
    cur["lam_i"] = np.nan
    cur.loc[scorable, "lam_i"] = lam_model.lam(cur[scorable])
    cur["alpha_source"] = np.where(scorable, "model", "market_only")
    cur["fair_season"] = cur["market_season"] + (
        cur["lam_i"].fillna(0.0) * cur["pred_residual"].fillna(0.0))
    cur["alpha_z"] = cur.apply(
        lambda r: (r["pred_residual"] / resid_std[r["position"]])
        if np.isfinite(r["pred_residual"]) and r["position"] in resid_std else np.nan,
        axis=1,
    )

    diagnostics = {
        "features": features,
        "lambda_scalar": lam_model.scalar_,
        "lambda_coef": None if lam_model.coef_ is None else [float(b) for b in lam_model.coef_],
        "lam_i_range_now": [float(cur.loc[scorable, "lam_i"].min()),
                            float(cur.loc[scorable, "lam_i"].max())] if scorable.any() else None,
        "alpha_train_rows": final_alpha.n_train_,
        "coefficients": final_alpha.coefficients().to_dict(),
        "resid_std": resid_std,
        "iso_train_rows": iso_n,
        "iso_by_pos": iso_by_pos,
        "n_scored": int(scorable.sum()),
        "n_market_only": int((~scorable).sum()),
    }
    if verbose:
        print(f"serving alpha v2: λ̄={np.nanmean(cur.loc[scorable, 'lam_i']):.2f}, "
              f"{diagnostics['n_scored']} model-scored, "
              f"{diagnostics['n_market_only']} market-only")
    out_cols = ["player_id", "position", "adp", "adp_stdev", "market_season",
                "pred_residual", "lam_i", "fair_season", "alpha_z", "alpha_source"]
    return cur[out_cols].reset_index(drop=True), diagnostics, wf_df


# ---------------------------------------------------------------------------
# Evaluation (shared by the eval script and the trust artifact)
# ---------------------------------------------------------------------------

ADP_BUCKETS = [(1, 24, "1-24"), (25, 60, "25-60"), (61, 120, "61-120"),
               (121, 10_000, "121+")]


def evaluate_walkforward(wf: pd.DataFrame, n_boot: int = 500, seed: int = 7) -> dict:
    """Residual IC (per-season + pooled, bootstrap CIs), incremental IC of
    fair over the pure ADP map, IC by ADP bucket, and a long/short spread
    in positional-rank space (decision units: draft slots, not correlation)."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)

    def ic_ci(pred, real):
        pred, real = np.asarray(pred, float), np.asarray(real, float)
        ic = float(spearmanr(pred, real).statistic)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(pred), len(pred))
            if np.std(pred[i]) > 0 and np.std(real[i]) > 0:
                boots.append(spearmanr(pred[i], real[i]).statistic)
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
        return ic, float(lo), float(hi), len(pred)

    scored = wf[wf["pred_residual"].notna() & wf["residual"].notna()].copy()

    per_season = []
    for s, grp in scored.groupby("season"):
        ic, lo, hi, n = ic_ci(grp["pred_residual"], grp["residual"])
        k = max(1, int(0.1 * n))
        top = grp.reindex(grp["pred_residual"].abs().sort_values(ascending=False).index[:k])
        per_season.append({
            "season": int(s), "residual_ic": round(ic, 3), "ci_lo": round(lo, 3),
            "ci_hi": round(hi, 3), "n": n,
            "hit_rate": round(float((np.sign(top["pred_residual"])
                                     == np.sign(top["residual"])).mean()), 3),
        })
    p_ic, p_lo, p_hi, p_n = ic_ci(scored["pred_residual"], scored["residual"])

    fair = wf[wf["fair"].notna() & wf[TARGET].notna()]
    incremental = []
    for s, grp in fair.groupby("season"):
        m = float(spearmanr(grp["market_expected"], grp[TARGET]).statistic)
        f = float(spearmanr(grp["fair"], grp[TARGET]).statistic)
        incremental.append({"season": int(s), "market_ic": round(m, 3),
                            "fair_ic": round(f, 3), "inc_ic": round(f - m, 3),
                            "n": int(len(grp))})

    by_bucket = []
    for lo_b, hi_b, label in ADP_BUCKETS:
        sub = scored[(scored["adp"] >= lo_b) & (scored["adp"] <= hi_b)]
        if len(sub) < 40:
            continue
        ic, lo, hi, n = ic_ci(sub["pred_residual"], sub["residual"])
        by_bucket.append({"bucket": label, "residual_ic": round(ic, 3),
                          "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "n": n})

    # Long/short spread in positional finish ranks (decision units)
    ls_rows = []
    for s, grp in scored.groupby("season"):
        grp = grp.copy()
        grp["finish_rank"] = grp.groupby("position")[TARGET].rank(ascending=False)
        grp["price_rank"] = grp.groupby("position")["adp"].rank(ascending=True)
        grp["displacement"] = grp["price_rank"] - grp["finish_rank"]  # + = beat price
        k = max(2, int(0.1 * len(grp)))
        longs = grp.nlargest(k, "pred_residual")["displacement"].mean()
        shorts = grp.nsmallest(k, "pred_residual")["displacement"].mean()
        ls_rows.append({"season": int(s), "long": round(float(longs), 2),
                        "short": round(float(shorts), 2),
                        "spread": round(float(longs - shorts), 2), "k": k})

    return {
        "per_season": per_season,
        "pooled": {"residual_ic": round(p_ic, 3), "ci_lo": round(p_lo, 3),
                   "ci_hi": round(p_hi, 3), "n": p_n, "n_seasons": len(per_season),
                   "pct_seasons_positive": round(float(np.mean(
                       [r["residual_ic"] > 0 for r in per_season])), 2) if per_season else None},
        "incremental": incremental,
        "by_adp_bucket": by_bucket,
        "long_short": ls_rows,
        "ls_mean_spread": round(float(np.mean([r["spread"] for r in ls_rows])), 2)
        if ls_rows else None,
    }
