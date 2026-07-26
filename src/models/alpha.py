"""
Stage B: market-residual ("alpha") model with shrinkage.

The market ensemble treats ADP as the base rate. This module asks the
sharper question: can we predict the MARKET'S ERROR itself?

    residual   = next_fpts - market_expected          (per-game units)
    fair       = market_expected + lambda * predicted_residual

where market_expected is a strictly out-of-sample per-position isotonic
ADP -> next_fpts mapping (same construction as market_ensemble.py, but
walk-forward: each test season's mapping is fitted ONLY on earlier
pair-seasons), and lambda is a shrinkage slope estimated from realized
vs. predicted residuals on earlier out-of-sample seasons.

Everything used as a model input is out-of-sample:
  - market_expected: isotonic fitted on pair-seasons < S
  - fund_pred:       HybridProjectionModel trained on pair-seasons < S
  - predicted residual: AlphaModel (Ridge) trained on OOS residual rows
                        from pair-seasons < S
  - lambda (walk-forward variant): slope fitted on alpha seasons < S

Design choice — POOLED Ridge with position dummies rather than
per-position models: a single position-season has only ~40-90 ADP-joined
rows, and a per-position walk-forward alpha model would start from ~150
training rows — hopeless for a noisy residual target. Pooling roughly
quadruples the sample; residuals are already in comparable per-game
units across positions (the market prior removes the positional level),
and the dummies absorb what level difference remains.

Public API
----------
market_expected_oos(pairs_adp, ...)      strictly-OOS market expectation
fundamental_oos(pairs, test_seasons)     walk-forward hybrid predictions
AlphaModel                               pooled Ridge on the residual
estimate_lambda(pred, realized)          shrinkage slope + SE
run_alpha_walkforward(pairs, adp_df)     full pipeline -> predictions df
"""
from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

from config import POSITIONS, RECENCY_DECAY

# Base (non-dummy) features for the alpha model. All must exist in the
# residual frame produced by run_alpha_walkforward. Kept deliberately
# small (~8 + 3 position dummies): the target is a noisy market error.
ALPHA_FEATURES: list[str] = [
    "fund_gap",                  # hybrid OOS prediction - market expected (the core signal)
    "age",
    "log_adp",
    "adp_stdev",                 # market disagreement about the player
    "team_vacated_target_share", # opportunity opening up (v4 high-signal)
    "target_share_delta",        # late-season role trend
    "fpts_pg_yoy_change",        # momentum / regression candidate
    "years_in_league",
]

MIN_ISO_ROWS = 30          # per-position minimum to fit an isotonic prior
DEFAULT_ISO_MIN_TRAIN_SEASONS = 2
DEFAULT_ALPHA_MIN_TRAIN_SEASONS = 3
DEFAULT_LAMBDA_MIN_TRAIN_SEASONS = 2


# ---------------------------------------------------------------------------
# ADP attach (adds stdev on top of data.adp.attach_adp's columns)
# ---------------------------------------------------------------------------

def attach_market(
    pairs: pd.DataFrame,
    adp_df: pd.DataFrame,
    season_offset: int = 1,
) -> pd.DataFrame:
    """
    Join ADP (price, positional rank, and draft-position stdev) onto YoY
    pairs by (normalized name, position, season + offset).

    Same join key as data.adp.attach_adp; carried out here so the 'stdev'
    column (market disagreement) survives the merge.
    Adds: adp, adp_pos_rank, adp_stdev, adp_matched.
    """
    from data.adp import normalize_name

    out = pairs.copy()
    out["_name_norm"] = out["player_name"].map(normalize_name)
    out["_adp_season"] = out["season"] + season_offset

    right = adp_df.copy()
    if "name_norm" not in right.columns:
        right["name_norm"] = right["player_name"].map(normalize_name)
    keep = ["season", "name_norm", "position", "adp", "adp_pos_rank"]
    if "stdev" in right.columns:
        keep.append("stdev")
    right = right[keep].rename(columns={
        "season": "_adp_season", "position": "_adp_position", "stdev": "adp_stdev",
    })
    if "adp_stdev" not in right.columns:
        right["adp_stdev"] = np.nan

    out = out.merge(
        right,
        left_on=["_adp_season", "_name_norm", "position"],
        right_on=["_adp_season", "name_norm", "_adp_position"],
        how="left",
    ).drop(columns=["name_norm", "_adp_position", "_name_norm", "_adp_season"],
           errors="ignore")
    out["adp_stdev"] = pd.to_numeric(out["adp_stdev"], errors="coerce")
    out["adp_matched"] = out["adp"].notna()
    return out


# ---------------------------------------------------------------------------
# Strictly out-of-sample market expectation
# ---------------------------------------------------------------------------

def market_expected_oos(
    pairs: pd.DataFrame,
    adp_df: pd.DataFrame | None = None,
    target: str = "next_fpts",
    min_train_seasons: int = DEFAULT_ISO_MIN_TRAIN_SEASONS,
    recency: float = RECENCY_DECAY,
) -> pd.DataFrame:
    """
    Walk-forward market-implied points per game.

    For each pair-season S with ADP coverage and >= min_train_seasons
    earlier ADP-covered pair-seasons, fit the per-position isotonic
    ADP -> next_fpts mapping (monotone decreasing, recency-weighted —
    identical construction to market_ensemble.MarketEnsembleModel) on
    pair-seasons strictly earlier than S only, then predict season S.

    Parameters
    ----------
    pairs : YoY pairs. If 'adp' is absent, adp_df must be given and is
        joined at season_offset=1 (draft year = target season).
    adp_df : output of data.adp.load_adp(), or None if already joined.

    Returns
    -------
    ADP-covered rows of the OOS test seasons with columns from `pairs`
    plus 'market_expected'. Seasons without enough earlier ADP history
    are excluded (they can never be scored out-of-sample).
    """
    df = pairs if "adp" in pairs.columns else attach_market(pairs, adp_df)
    df = df[df["adp"].notna() & df[target].notna()].copy()
    if df.empty:
        return df.assign(market_expected=np.nan)

    adp_seasons = sorted(df["season"].unique())
    out_frames = []
    for s in adp_seasons:
        train_seasons = [t for t in adp_seasons if t < s]
        if len(train_seasons) < min_train_seasons:
            continue
        train = df[df["season"] < s]
        test = df[df["season"] == s].copy()
        test["market_expected"] = np.nan
        max_train_season = int(train["season"].max())

        for pos in POSITIONS:
            sub = train[train["position"] == pos]
            if len(sub) < MIN_ISO_ROWS:
                warnings.warn(
                    f"market_expected_oos {s} {pos}: only {len(sub)} training "
                    "rows; no market expectation for this position-season."
                )
                continue
            w = np.power(recency, max_train_season - sub["season"].values)
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            iso.fit(sub["adp"].values, sub[target].values, sample_weight=w)
            mask = (test["position"] == pos).values
            if mask.any():
                test.loc[mask, "market_expected"] = iso.predict(
                    test.loc[mask, "adp"].values.astype(float)
                )
        out_frames.append(test[test["market_expected"].notna()])

    if not out_frames:
        return df.head(0).assign(market_expected=np.nan)
    return pd.concat(out_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Walk-forward fundamental (hybrid) predictions
# ---------------------------------------------------------------------------

def fundamental_oos(
    pairs: pd.DataFrame,
    test_seasons: list[int],
    target: str = "next_fpts",
    model_factory: Callable[[], Any] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Out-of-sample fundamental predictions: for each test season, train the
    hybrid model on strictly earlier pair-seasons (same protocol as
    models.backtest.rolling_backtest — age_adjust=False, fit_age=False)
    and predict the test season per position.

    Returns
    -------
    DataFrame [player_id, season, position, fund_pred].
    """
    if model_factory is None:
        from models.hybrid import HybridProjectionModel

        model_factory = lambda: HybridProjectionModel(age_adjust=False)  # noqa: E731

    rows = []
    for s in sorted(test_seasons):
        train_df = pairs[pairs["season"] < s]
        test_df = pairs[pairs["season"] == s]
        if test_df.empty or len(train_df["season"].unique()) < 2:
            warnings.warn(f"fundamental_oos: skipping {s} (insufficient training data)")
            continue
        if verbose:
            print(f"fundamental_oos: training hybrid for test season {s} "
                  f"(train n={len(train_df)})")
        model = model_factory()
        try:
            model.train(train_df, target=target, fit_age=False)
        except Exception as e:
            warnings.warn(f"fundamental_oos: training failed for {s}: {e}")
            continue
        for pos in POSITIONS:
            pos_test = test_df[test_df["position"] == pos].reset_index(drop=True)
            if pos_test.empty:
                continue
            try:
                pred = model.predict_position(pos, pos_test)
            except Exception:
                pred = None
            if pred is None:
                continue
            rows.append(pd.DataFrame({
                "player_id": pos_test["player_id"].values,
                "season": s,
                "position": pos,
                "fund_pred": np.asarray(pred, dtype=float),
            }))

    if not rows:
        return pd.DataFrame(columns=["player_id", "season", "position", "fund_pred"])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Alpha model: pooled Ridge on the market residual
# ---------------------------------------------------------------------------

class AlphaModel:
    """
    Ridge regression predicting the market residual (next_fpts −
    market_expected), pooled across positions with position dummies.

    Pooled rather than per-position: see module docstring. Features are
    median-imputed and z-scored with statistics learned at fit time
    (train-only — no test-set statistics leak into prediction).
    """

    def __init__(self, features: list[str] | None = None, ridge_alpha: float = 10.0):
        self.features = list(ALPHA_FEATURES) if features is None else list(features)
        self.ridge_alpha = ridge_alpha
        self.dummy_positions_ = [p for p in POSITIONS if p != "QB"]  # QB = reference
        self._model: Ridge | None = None
        self._medians: pd.Series | None = None
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None
        self.feature_names_: list[str] = []
        self.n_train_: int = 0

    # ------------------------------------------------------------------

    def _design(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        X = df[self.features].apply(pd.to_numeric, errors="coerce")
        if fit:
            self._medians = X.median()
        X = X.fillna(self._medians).fillna(0.0)

        dummies = np.column_stack([
            (df["position"] == p).astype(float).values for p in self.dummy_positions_
        ])
        mat = np.column_stack([X.values, dummies])

        if fit:
            self._means = mat.mean(axis=0)
            self._stds = mat.std(axis=0)
            self._stds[self._stds == 0] = 1.0
            self.feature_names_ = self.features + [f"pos_{p}" for p in self.dummy_positions_]
        return (mat - self._means) / self._stds

    def fit(self, df: pd.DataFrame, target: str = "residual") -> "AlphaModel":
        sub = df[df[target].notna()].reset_index(drop=True)
        if len(sub) < 30:
            raise ValueError(f"AlphaModel: only {len(sub)} training rows (need >= 30).")
        X = self._design(sub, fit=True)
        self._model = Ridge(alpha=self.ridge_alpha)
        self._model.fit(X, sub[target].values)
        self.n_train_ = len(sub)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("AlphaModel.predict called before fit.")
        return self._model.predict(self._design(df, fit=False))

    def coefficients(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("AlphaModel not fitted.")
        return pd.Series(self._model.coef_, index=self.feature_names_)


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------

def estimate_lambda(
    predicted: np.ndarray,
    realized: np.ndarray,
    clip: tuple[float, float] = (0.0, 1.0),
) -> dict[str, float]:
    """
    Shrinkage slope: OLS of realized residual on predicted residual
    (with intercept). Returns the raw slope, its standard error, the
    clipped slope ('lam', in [0, 1]), and n.
    """
    p = np.asarray(predicted, dtype=float)
    r = np.asarray(realized, dtype=float)
    ok = np.isfinite(p) & np.isfinite(r)
    p, r = p[ok], r[ok]
    n = len(p)
    if n < 10 or np.std(p) == 0:
        return {"slope": np.nan, "se": np.nan, "lam": np.nan, "n": n}

    pc = p - p.mean()
    slope = float(np.dot(pc, r) / np.dot(pc, pc))
    resid = r - r.mean() - slope * pc
    dof = max(n - 2, 1)
    se = float(np.sqrt((np.dot(resid, resid) / dof) / np.dot(pc, pc)))
    lam = float(np.clip(slope, *clip))
    return {"slope": slope, "se": se, "lam": lam, "n": n}


# ---------------------------------------------------------------------------
# Full walk-forward pipeline
# ---------------------------------------------------------------------------

def run_alpha_walkforward(
    pairs: pd.DataFrame,
    adp_df: pd.DataFrame,
    target: str = "next_fpts",
    features: list[str] | None = None,
    ridge_alpha: float = 10.0,
    iso_min_train_seasons: int = DEFAULT_ISO_MIN_TRAIN_SEASONS,
    alpha_min_train_seasons: int = DEFAULT_ALPHA_MIN_TRAIN_SEASONS,
    lambda_min_train_seasons: int = DEFAULT_LAMBDA_MIN_TRAIN_SEASONS,
    fund_pred_df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    End-to-end honest walk-forward:

      1. market_expected  : isotonic fitted on pair-seasons < S
      2. fund_pred        : hybrid trained on pair-seasons < S
      3. pred_residual    : AlphaModel trained on OOS residual rows from
                            pair-seasons < S (needs alpha_min_train_seasons
                            of them)
      4. lam_wf           : pooled shrinkage slope from alpha seasons < S
                            (needs lambda_min_train_seasons of them);
                            fair = market_expected + lam_wf * pred_residual
                            (NaN where lam_wf unavailable)

    Returns one row per (player, season) with ADP + market_expected
    coverage, including feature columns, residual, pred_residual (NaN for
    seasons that only serve as alpha training), lam_wf, and fair.

    fund_pred_df may be passed to reuse precomputed fundamental_oos output.
    """
    feats = list(ALPHA_FEATURES) if features is None else list(features)

    mkt = market_expected_oos(
        pairs, adp_df, target=target, min_train_seasons=iso_min_train_seasons
    )
    if mkt.empty:
        raise ValueError("No out-of-sample market expectations could be built.")
    mkt_seasons = sorted(mkt["season"].unique())
    if verbose:
        print(f"Market OOS seasons: {mkt_seasons}")

    if fund_pred_df is None:
        fund_pred_df = fundamental_oos(pairs, mkt_seasons, target=target, verbose=verbose)
    df = mkt.merge(fund_pred_df, on=["player_id", "season", "position"], how="left")

    df["residual"] = df[target] - df["market_expected"]
    df["fund_gap"] = df["fund_pred"] - df["market_expected"]
    df["log_adp"] = np.log(df["adp"].clip(lower=1.0))
    # Rows without a fundamental prediction can't carry the core signal;
    # drop them rather than imputing the headline feature.
    df = df[df["fund_gap"].notna()].reset_index(drop=True)

    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise KeyError(f"Alpha features missing from the frame: {missing}")

    df["pred_residual"] = np.nan
    df["lam_wf"] = np.nan

    alpha_seasons = [
        s for s in mkt_seasons
        if len([t for t in mkt_seasons if t < s]) >= alpha_min_train_seasons
    ]
    for s in alpha_seasons:
        train = df[df["season"] < s]
        test_mask = df["season"] == s
        if not test_mask.any() or len(train) < 30:
            continue
        model = AlphaModel(features=feats, ridge_alpha=ridge_alpha)
        model.fit(train, target="residual")
        df.loc[test_mask, "pred_residual"] = model.predict(df[test_mask])

        # Walk-forward lambda: earlier ALPHA seasons only (they are the
        # ones with an OOS pred_residual to learn the slope from).
        prior = df[(df["season"] < s) & df["pred_residual"].notna()]
        if prior["season"].nunique() >= lambda_min_train_seasons:
            lam = estimate_lambda(prior["pred_residual"], prior["residual"])
            if np.isfinite(lam["lam"]):
                df.loc[test_mask, "lam_wf"] = lam["lam"]

    df["fair"] = df["market_expected"] + df["lam_wf"] * df["pred_residual"]
    if verbose:
        scored = df["pred_residual"].notna()
        print(f"Alpha seasons scored: {sorted(df.loc[scored, 'season'].unique())} "
              f"({int(scored.sum())} rows); fair available for "
              f"{int(df['fair'].notna().sum())} rows")
    return df


# ---------------------------------------------------------------------------
# Serving: fit on everything, apply to the upcoming season
# ---------------------------------------------------------------------------

def build_serving_alpha(
    pairs: pd.DataFrame,
    adp_history: pd.DataFrame,
    adp_now: pd.DataFrame,
    features_now: pd.DataFrame,
    target: str = "next_fpts",
    wf_df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Final serving-side alpha for the upcoming season.

    Fits the per-position isotonic market curve on ALL ADP-joined
    pair-seasons, the AlphaModel on all walk-forward OOS residual rows,
    and lambda as the pooled OOS shrinkage slope — then applies the stack
    to the current feature season priced at the current draft's ADP.

    Coverage policy: every player with a current ADP gets a market
    expectation. Players whose alpha inputs exist get
    fair = market + lambda * predicted_residual (alpha_source="model");
    players without them (rookies, thin histories) get fair = market
    (alpha_source="market_only") so a fair-value ladder over the priced
    universe stays complete instead of excluding whoever the model can't
    score.

    Returns
    -------
    (current, diagnostics, wf_df)
        current : [player_id, position, adp, adp_stdev, market_pg,
                   pred_residual_pg, fair_pg, alpha_source]
        diagnostics : lambda (+se, n), alpha train size, coefficients,
                      iso train sizes
        wf_df : the walk-forward frame (reusable for trust metrics)
    """
    if wf_df is None:
        wf_df = run_alpha_walkforward(pairs, adp_history, target=target, verbose=verbose)

    # -- lambda: pooled OOS slope; conservative fallback to 0 (fair=market)
    scored = wf_df[wf_df["pred_residual"].notna()]
    lam_info = estimate_lambda(scored["pred_residual"].values, scored["residual"].values)
    lam = lam_info["lam"] if np.isfinite(lam_info["lam"]) else 0.0

    # -- final alpha model on every OOS residual row
    final_alpha = AlphaModel().fit(wf_df, target="residual")

    # -- final per-position isotonic market curve on all ADP-joined pairs
    pairs_adp = pairs if "adp" in pairs.columns else attach_market(pairs, adp_history)
    train = pairs_adp[pairs_adp["adp"].notna() & pairs_adp[target].notna()]
    max_season = int(train["season"].max())
    iso_by_pos: dict[str, IsotonicRegression] = {}
    iso_n: dict[str, int] = {}
    for pos in POSITIONS:
        sub = train[train["position"] == pos]
        if len(sub) < MIN_ISO_ROWS:
            continue
        w = np.power(RECENCY_DECAY, max_season - sub["season"].values)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(sub["adp"].values, sub[target].values, sample_weight=w)
        iso_by_pos[pos] = iso
        iso_n[pos] = len(sub)

    # -- current season: features (season N) priced at the season N+1 draft
    cur = attach_market(features_now, adp_now, season_offset=1)
    cur = cur[cur["adp"].notna() & cur["position"].isin(iso_by_pos)].copy()
    cur["market_pg"] = np.nan
    for pos, iso in iso_by_pos.items():
        mask = cur["position"] == pos
        if mask.any():
            cur.loc[mask, "market_pg"] = iso.predict(cur.loc[mask, "adp"].values)

    # -- fundamental prediction from the hybrid trained on all pairs
    from models.hybrid import HybridProjectionModel

    hy = HybridProjectionModel(age_adjust=False)
    hy.train(pairs, target=target, fit_age=False)
    cur["fund_pred"] = np.nan
    for pos in iso_by_pos:
        pos_df = cur[cur["position"] == pos].reset_index()
        if pos_df.empty:
            continue
        try:
            pred = hy.predict_position(pos, pos_df)
        except Exception:
            pred = None
        if pred is not None:
            cur.loc[pos_df["index"], "fund_pred"] = np.asarray(pred, dtype=float)

    cur["fund_gap"] = cur["fund_pred"] - cur["market_pg"]
    cur["log_adp"] = np.log(cur["adp"].clip(lower=1.0))

    scorable = cur["fund_gap"].notna() & cur["market_pg"].notna()
    cur["pred_residual_pg"] = np.nan
    if scorable.any():
        cur.loc[scorable, "pred_residual_pg"] = final_alpha.predict(cur[scorable])
    cur["alpha_source"] = np.where(scorable, "model", "market_only")
    cur["fair_pg"] = cur["market_pg"] + lam * cur["pred_residual_pg"].fillna(0.0)

    diagnostics = {
        "lambda": lam,
        "lambda_slope": lam_info["slope"],
        "lambda_se": lam_info["se"],
        "lambda_n": lam_info["n"],
        "alpha_train_rows": final_alpha.n_train_,
        "coefficients": final_alpha.coefficients().to_dict(),
        "iso_train_rows": iso_n,
        "n_scored": int(scorable.sum()),
        "n_market_only": int((~scorable).sum()),
        # Fitted market curves, so callers can price players outside the
        # feature matrix (rookies) at fair = market.
        "iso_by_pos": iso_by_pos,
    }
    if verbose:
        print(f"serving alpha: lam={lam:.2f} (n={lam_info['n']}), "
              f"{diagnostics['n_scored']} model-scored, "
              f"{diagnostics['n_market_only']} market-only")
    out_cols = ["player_id", "position", "adp", "adp_stdev",
                "market_pg", "pred_residual_pg", "fair_pg", "alpha_source"]
    return cur[out_cols].reset_index(drop=True), diagnostics, wf_df
