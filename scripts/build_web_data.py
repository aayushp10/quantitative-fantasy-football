"""
Freeze everything the web app needs into static JSON artifacts.

Runs the full v4 pipeline (notebook 06, scripted) and writes:

    players.json     one record per player on the merged board, all formats
    meta.json        build provenance
    trust.json       backtest / market-relative / coverage / factor evidence
    adp_board.json   the live ADP ladder the draft bots price from

Usage:
    python scripts/build_web_data.py --season 2026 --out webapp/data/

Slow is OK (retrains the model many times for the trust artifact); the API
never imports any of this — it only reads the JSON.
Exits nonzero if any artifact fails validation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from config import (
    CACHE_DIR,
    POSITIONS,
    ROSTER_SPOTS,
    TRAINING_SEASONS,
    POSITION_FEATURES,
    season_length,
)

# ---------------------------------------------------------------------------
# Format definitions (PPR-only — see DECISIONS.md)
# ---------------------------------------------------------------------------

FORMATS: dict[str, dict] = {
    "10_ppr": {"label": "10-team PPR", "league_size": 10, "roster": ROSTER_SPOTS["10team"]},
    "12_ppr": {"label": "12-team PPR", "league_size": 12, "roster": ROSTER_SPOTS["12team"]},
    "14_ppr": {"label": "14-team PPR", "league_size": 14, "roster": ROSTER_SPOTS["14team"]},
    "12_superflex_ppr": {
        "label": "12-team Superflex PPR",
        "league_size": 12,
        "roster": ROSTER_SPOTS["12team_superflex"],
    },
}

TRUST_TEST_SEASONS = [2020, 2021, 2022, 2023]  # matches notebook 06

_BUCKET_LABELS = {0: "UDFA", 1: "R6-7", 2: "R4-5", 3: "R3", 4: "R2", 5: "R1"}


def _pick_to_bucket_label(pick: float) -> str | None:
    if pick is None or (isinstance(pick, float) and math.isnan(pick)):
        return None
    p = float(pick)
    if p <= 32:
        return "R1"
    if p <= 64:
        return "R2"
    if p <= 105:
        return "R3"
    if p <= 176:
        return "R4-5"
    if p <= 262:
        return "R6-7"
    return "UDFA"


# ---------------------------------------------------------------------------
# JSON hygiene
# ---------------------------------------------------------------------------

def jclean(obj):
    """Recursively convert numpy/pandas scalars, NaN -> None."""
    if isinstance(obj, dict):
        return {str(k): jclean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jclean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def rnd(x, nd=2):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(jclean(payload), indent=None, separators=(",", ":")))
    print(f"  wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def build_board(season: int):
    """Run the full v5 projection pipeline; returns (board, features_now, extras).

    The SERVED projection is the market ensemble (career-augmented hybrid
    blended toward an isotonic ADP prior). The PURE hybrid's projection is
    kept alongside (`pure_fpts_season`) because model-vs-market products
    (predicted_adp / adp_edge) are only meaningful against the model's
    market-independent opinion.
    """
    from features.assembler import assemble_feature_matrix, build_yoy_pairs
    from models.two_stage import ALL_RATE_TARGET_COLS
    from models.market_ensemble import MarketEnsembleModel
    from models.uncertainty import (
        ResidualQuantiles,
        simulate_season_totals,
        walk_forward_residuals,
    )
    from models.availability import AvailabilityModel
    from models.rookie import RookieModel, merge_rookie_projections
    from data.adp import load_adp
    from data.loader import load_draft_picks, load_weekly

    fm = assemble_feature_matrix(TRAINING_SEASONS)
    pairs = build_yoy_pairs(fm, extra_target_cols=ALL_RATE_TARGET_COLS)
    print(f"matrix: {fm.shape}, pairs: {pairs.shape}")

    input_season = max(TRAINING_SEASONS)
    features_now = fm[fm["season"] == input_season].copy()

    # Dedupe mid-season team-change stints: keep the highest-usage stint
    usage = (
        features_now.get("targets", 0).fillna(0)
        + features_now.get("carries", 0).fillna(0)
        + features_now.get("dropbacks", 0).fillna(0)
    )
    features_now = (
        features_now.assign(_usage=usage)
        .sort_values("_usage", ascending=False)
        .drop_duplicates(subset=["player_id"], keep="first")
        .drop(columns=["_usage"])
        .reset_index(drop=True)
    )
    print(f"features_now ({input_season}): {len(features_now)} unique players")

    adp_all = load_adp(list(range(2015, season + 1)))

    model = MarketEnsembleModel(adp_history=adp_all)
    model.train(pairs)

    # Intervals calibrated on the residuals of the model actually served
    resids = walk_forward_residuals(
        lambda: MarketEnsembleModel(adp_history=adp_all, age_adjust=False),
        pairs, min_train_seasons=4,
    )
    rq = ResidualQuantiles().fit(resids)
    model.set_uncertainty(rq)

    proj = model.project(features_now, season=season)

    avail = AvailabilityModel().train(pairs)
    proj = avail.attach_to_projections(proj, features_now, target_season=season)
    proj = simulate_season_totals(proj, rq, target_season=season, games_sd=avail.residual_std_)

    # Rookie class
    draft_picks = load_draft_picks()
    weekly = load_weekly(list(TRAINING_SEASONS))
    team_ctx_cols = [c for c in ["team", "season", "team_pace", "team_pass_rate",
                                 "team_offensive_epa"] if c in fm.columns]
    team_ctx = fm[team_ctx_cols].drop_duplicates(["team", "season"]) if len(team_ctx_cols) > 2 else None
    qb_env = (fm[["team", "season", "qb_epa_per_dropback"]].drop_duplicates(["team", "season"])
              if "qb_epa_per_dropback" in fm.columns else None)

    rookie_frame = RookieModel.build_training_frame(
        draft_picks, weekly, team_context=team_ctx, qb_coupling=qb_env,
        end_season=input_season,
    )
    rookie_model = RookieModel().train(rookie_frame)
    rookies = rookie_model.project_class(
        draft_picks, season, team_context=team_ctx, qb_coupling=qb_env
    )

    # Synthetic ids for rookies missing a GSIS id, and draft-capital labels
    missing = rookies["player_id"].isna()
    rookies.loc[missing, "player_id"] = [
        f"rookie_{season}_p{int(p)}" for p in rookies.loc[missing, "pick"]
    ]
    rookie_picks = rookies.set_index("player_id")["pick"].to_dict()

    # Rookie season-total quantiles: the rookie model has no residual-quantile
    # machinery; approximate with the veteran per-position quantiles applied
    # to the rookie point estimate (wider-uncertainty caveat rendered in UI).
    board = merge_rookie_projections(proj, rookies)
    rk = board["rookie"].fillna(False) & board["season_p50"].isna()
    for pos in POSITIONS:
        mask = rk & (board["position"] == pos)
        if not mask.any():
            continue
        preds = board.loc[mask, "projected_fpts_pg"].values.astype(float)
        games = board.loc[mask, "projected_games"].values.astype(float)
        for q, col in [(0.10, "season_p10"), (0.25, "season_p25"), (0.50, "season_p50"),
                       (0.75, "season_p75"), (0.90, "season_p90")]:
            board.loc[mask, col] = np.maximum(0, rq.interval(pos, preds, q)) * games

    print(f"board: {len(board)} players ({int(board['rookie'].fillna(False).sum())} rookies)")
    extras = {
        "pairs": pairs,
        "fm": fm,
        "weekly": weekly,
        "resids": resids,
        "rq": rq,
        "rookie_picks": rookie_picks,
        "input_season": input_season,
    }
    return board, features_now, extras


def attach_market(board: pd.DataFrame, season: int) -> pd.DataFrame:
    """ADP join + predicted_adp ladder + adp_edge (12-team PPR market).

    Unsplit (user decision): predicted_adp / adp_edge rank by the SERVED
    ensemble, the same model that orders the board. The ensemble contains
    ADP, so marginal disagreements compress toward the market; because the
    edge lives in rank-ladder space, disagreements strong enough to survive
    the blend keep their full displacement.
    """
    from data.adp import load_adp, attach_adp

    adp_now = load_adp([season])
    if adp_now.empty:
        raise RuntimeError(f"No ADP available for {season}")

    # The ensemble's project() already joined an ADP column; drop it so the
    # canonical join below owns the adp/adp_pos_rank columns.
    board = board.drop(columns=["adp", "adp_pos_rank", "adp_matched"], errors="ignore")

    board = board.sort_values("vorp_12_ppr", ascending=False).reset_index(drop=True)
    board["model_overall_rank"] = np.arange(1, len(board) + 1)
    board["projected_season"] = season
    board = attach_adp(board, adp_now, season_offset=0, season_col="projected_season")

    # Rank→ladder mapping is done WITHIN each position, on that position's
    # own ADP ladder. Mapping overall VORP rank onto the overall ladder
    # conflates model opinion with the market's positional-scarcity pricing
    # (a 1-QB market drafts QBs far below their VORP), which manufactured
    # huge "edges" for every mid-QB even when the model agreed with the
    # market about the player.
    board["predicted_adp"] = np.nan
    for pos, pos_idx in board.groupby("position", observed=True).groups.items():
        pos_ladder = np.sort(adp_now.loc[adp_now["position"] == pos, "adp"].values)
        if len(pos_ladder) == 0:
            continue
        sub = board.loc[pos_idx].sort_values("vorp_12_ppr", ascending=False)
        ranks = np.arange(1, len(sub) + 1)
        idx = np.minimum(ranks - 1, len(pos_ladder) - 1)
        pred = pos_ladder[idx]
        # Overflow past the drafted universe: extrapolate at the ladder's
        # typical late spacing (capped — the last rung's gap can be a
        # one-off outlier that explodes deep-tail edges).
        if len(pos_ladder) > 5:
            tail_gap = float(np.clip(np.median(np.diff(pos_ladder[-6:])), 1.0, 6.0))
        else:
            tail_gap = 1.0
        overflow = ranks > len(pos_ladder)
        pred = np.where(overflow, pos_ladder[-1] + (ranks - len(pos_ladder)) * tail_gap, pred)
        board.loc[sub.index, "predicted_adp"] = np.round(pred, 1)

    board["adp_edge"] = (board["adp"] - board["predicted_adp"]).round(1)
    return board


def attach_alpha(board: pd.DataFrame, features_now: pd.DataFrame,
                 extras: dict, season: int):
    """Fair-value overlay from the alpha v2 (market-residual) model:
    season-points residuals on survivor-complete outcomes, walk-forward
    feature selection, per-player shrinkage λᵢ, conviction z-scores.

    Does NOT change the board's ranking, VORP, projections, or bot
    behavior — the board's source of truth stays real ADP + the served
    ensemble (user decision). Alpha rides alongside as columns:

        market_points   iso(ADP → season points) — what the price implies
        fair_points     market + λᵢ·predicted residual
        alpha_points    fair − market (0 for market-only rows)
        alpha_z         predicted residual / typical market error at the
                        position (conviction)
        fair_adp        fair_points rank mapped onto the position's own
                        current ADP ladder (same mapping as predicted_adp)
        fair_adp_edge   adp − fair_adp (positive = market drafts him later
                        than his fair slot — value)

    Players with a price but no alpha inputs (rookies, thin histories) get
    fair = market ("market_only"), keeping the fair ladder complete over
    the priced universe instead of gifting veterans phantom edges.
    """
    from data.adp import load_adp
    from models.alpha2 import (
        build_serving_alpha_v2,
        build_survivor_frame,
        run_alpha_v2_walkforward,
        select_features,
    )

    adp_history = load_adp(list(range(2013, season)))
    adp_now = load_adp([season])
    fm, weekly, pairs_old = extras["fm"], extras["weekly"], extras["pairs"]

    frame = build_survivor_frame(fm, weekly, adp_history)
    features, selection, fund_df = select_features(frame, pairs_old)
    wf_df, _ = run_alpha_v2_walkforward(frame, pairs_old, features=features,
                                        fund_df=fund_df, verbose=False)
    cur, diag, wf_df = build_serving_alpha_v2(
        fm, weekly, adp_history, adp_now, features_now, pairs_old,
        features=features, wf_df=wf_df, frame=frame)
    diag["feature_selection"] = selection

    board = board.merge(
        cur[["player_id", "market_season", "pred_residual", "lam_i",
             "fair_season", "alpha_z", "alpha_source"]],
        on="player_id", how="left",
    )

    # Price rookies / unmatched at fair = market from their real ADP
    iso_by_pos = diag["iso_by_pos"]
    needs = board["market_season"].isna() & board["adp"].notna()
    for pos, iso in iso_by_pos.items():
        mask = needs & (board["position"] == pos)
        if mask.any():
            mp = iso.predict(board.loc[mask, "adp"].values)
            board.loc[mask, "market_season"] = mp
            board.loc[mask, "fair_season"] = mp
            board.loc[mask, "alpha_source"] = "market_only"

    board["market_points"] = board["market_season"].round(1)
    board["fair_points"] = board["fair_season"].round(1)
    board["alpha_points"] = (board["fair_points"] - board["market_points"]).round(1)
    board["alpha_z"] = board["alpha_z"].round(2)
    board["alpha_lam"] = board["lam_i"].round(2)

    # fair_points rank → the position's own current ADP ladder
    board["fair_adp"] = np.nan
    priced = board["fair_points"].notna() & board["adp"].notna()
    for pos in board.loc[priced, "position"].unique():
        pos_ladder = np.sort(adp_now.loc[adp_now["position"] == pos, "adp"].values)
        if len(pos_ladder) == 0:
            continue
        sub = board[priced & (board["position"] == pos)].sort_values(
            "fair_points", ascending=False)
        ranks = np.arange(1, len(sub) + 1)
        idx = np.minimum(ranks - 1, len(pos_ladder) - 1)
        board.loc[sub.index, "fair_adp"] = np.round(pos_ladder[idx], 1)
    board["fair_adp_edge"] = (board["adp"] - board["fair_adp"]).round(1)

    n_fair = int(board["fair_points"].notna().sum())
    lam_rng = diag.get("lam_i_range_now")
    print(f"alpha overlay v2: {n_fair} priced players "
          f"({diag['n_scored']} model-scored, "
          f"λᵢ∈[{lam_rng[0]:.2f}, {lam_rng[1]:.2f}])" if lam_rng else "")
    return board, diag, wf_df


def per_format_vorp(board: pd.DataFrame) -> pd.DataFrame:
    """Compute vorp/overall_rank/pos_rank/tier for every format key."""
    from models.vor import calculate_vor
    from ranking.ranker import generate_rankings
    from ranking.tiers import assign_tiers_all_positions

    out = board.copy()
    for key, spec in FORMATS.items():
        f = calculate_vor(board, league_size=spec["league_size"], roster_config=spec["roster"])
        f = assign_tiers_all_positions(f)
        f = generate_rankings(f)
        f = f.set_index("player_id")
        out[f"vorp_{key}"] = out["player_id"].map(f["vorp"])
        out[f"overall_rank_{key}"] = out["player_id"].map(f["overall_rank"])
        out[f"pos_rank_{key}"] = out["player_id"].map(f["pos_rank"])
        out[f"tier_{key}"] = out["player_id"].map(f["tier"].astype(float))
    return out


def build_players_json(board: pd.DataFrame, features_now: pd.DataFrame,
                       rookie_picks: dict) -> list[dict]:
    feats = features_now.set_index("player_id")

    def fv(pid, col):
        if pid not in feats.index or col not in feats.columns:
            return None
        v = feats.at[pid, col]
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return v if isinstance(v, str) else None

    players = []
    for _, r in board.iterrows():
        pid = r["player_id"]
        pos = r["position"]
        is_rookie = bool(r.get("rookie") or False)

        games_played = fv(pid, "games_played")
        routes = fv(pid, "routes")
        routes_pg = (routes / games_played) if routes and games_played else None

        if pos == "QB":
            epa = fv(pid, "epa_per_dropback")
            x_td, td_oe, rz = fv(pid, "x_pass_td_rate"), fv(pid, "pass_td_oe"), None
            vac = None
        elif pos == "RB":
            epa = fv(pid, "epa_per_carry")
            x_td, td_oe = fv(pid, "x_rush_td_rate"), fv(pid, "rush_td_oe")
            rz = fv(pid, "rz_rush_share")
            vac = fv(pid, "team_vacated_carry_share")
        else:
            epa = fv(pid, "epa_per_target")
            x_td, td_oe = fv(pid, "x_rec_td_rate"), fv(pid, "rec_td_oe")
            rz = fv(pid, "rz_target_share")
            vac = fv(pid, "team_vacated_target_share")

        if is_rookie:
            bucket = _pick_to_bucket_label(rookie_picks.get(pid))
        else:
            b = fv(pid, "draft_round_bucket")
            bucket = _BUCKET_LABELS.get(int(b)) if b is not None else None

        years = fv(pid, "years_in_league")
        team_changed = fv(pid, "team_changed")
        qb_changed = fv(pid, "qb_changed")

        vorp = {}
        for key in FORMATS:
            v = r.get(f"vorp_{key}")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            tier = r.get(f"tier_{key}")
            vorp[key] = {
                "vorp": rnd(v, 1),
                "overall_rank": int(r[f"overall_rank_{key}"]),
                "pos_rank": int(r[f"pos_rank_{key}"]),
                "tier": int(tier) if tier is not None and not math.isnan(float(tier)) else None,
            }

        players.append({
            "player_id": pid,
            "name": r.get("player_name"),
            "position": pos,
            "team": r.get("team"),
            "age": rnd(r.get("age"), 1),
            "years_in_league": int(years) if years is not None else (0 if is_rookie else None),
            "rookie": is_rookie,
            "draft_capital_bucket": bucket,
            "fpts_pg_p50": rnd(r.get("projected_fpts_pg"), 2),
            "season_p10": rnd(r.get("season_p10"), 1),
            "season_p25": rnd(r.get("season_p25"), 1),
            "season_p50": rnd(r.get("season_p50"), 1),
            "season_p75": rnd(r.get("season_p75"), 1),
            "season_p90": rnd(r.get("season_p90"), 1),
            "expected_games": rnd(r.get("projected_games"), 1),
            "vorp": vorp,
            "adp": rnd(r.get("adp"), 1),
            "predicted_adp": rnd(r.get("predicted_adp"), 1),
            "adp_edge": rnd(r.get("adp_edge"), 1),
            "market_points": rnd(r.get("market_points"), 1),
            "fair_points": rnd(r.get("fair_points"), 1),
            "alpha_points": rnd(r.get("alpha_points"), 1),
            "alpha_z": rnd(r.get("alpha_z"), 2),
            "alpha_lam": rnd(r.get("alpha_lam"), 2),
            "fair_adp": rnd(r.get("fair_adp"), 1),
            "fair_adp_edge": rnd(r.get("fair_adp_edge"), 1),
            "alpha_source": r.get("alpha_source") if isinstance(r.get("alpha_source"), str) else None,
            "features": {
                "target_share": rnd(fv(pid, "target_share"), 3) if pos != "QB" else None,
                "wopr": rnd(fv(pid, "wopr"), 3) if pos in ("WR", "TE") else None,
                "tprr": rnd(fv(pid, "tprr"), 3) if pos != "QB" else None,
                "routes_per_game": rnd(routes_pg, 1) if pos != "QB" else None,
                "epa_per_target_or_carry": rnd(epa, 3),
                "x_td_rate": rnd(x_td, 4),
                "td_oe": rnd(td_oe, 4),
                "red_zone_share": rnd(rz, 3),
                "breakout_flag": fv(pid, "trend_class"),
                "consistency_score": rnd(fv(pid, "consistency_score"), 2),
                "boom_rate": rnd(fv(pid, "boom_rate"), 3),
                "bust_rate": rnd(fv(pid, "bust_rate"), 3),
                "snap_pct": rnd(fv(pid, "snap_percentage"), 3),
                "team_change": bool(team_changed) if team_changed is not None else None,
                "qb_changed": (bool(qb_changed) if qb_changed is not None else None)
                              if pos != "QB" else None,
                "vacated_share_gained": rnd(vac, 3),
            },
        })
    return players


def build_alpha_trust(wf_df: pd.DataFrame, diag: dict) -> dict:
    """Walk-forward alpha v2 evidence for the trust page: the shared
    evaluate_walkforward metrics (residual IC + CIs, incremental IC,
    ADP-bucket ICs, long/short spread in positional ranks) plus serving
    diagnostics (λᵢ, coefficients, feature selection)."""
    from models.alpha2 import evaluate_walkforward

    metrics = evaluate_walkforward(wf_df)
    coefs = pd.Series(diag["coefficients"]).sort_values(key=np.abs, ascending=False)
    lam_rng = diag.get("lam_i_range_now")
    sel = diag.get("feature_selection", {})
    return {
        **metrics,
        "lambda_scalar": rnd(diag.get("lambda_scalar"), 3),
        "lam_i_range": [rnd(lam_rng[0], 2), rnd(lam_rng[1], 2)] if lam_rng else None,
        "n_scored_now": int(diag["n_scored"]),
        "n_market_only_now": int(diag["n_market_only"]),
        "features": diag.get("features", []),
        "feature_selection": {
            k: v for k, v in sel.get("candidates", {}).items()
        },
        "coefficients": [{"feature": f, "coef": rnd(c, 3)} for f, c in coefs.items()],
    }


def build_trust_json(pairs: pd.DataFrame, resids, rq) -> dict:
    """Evaluate the model the site actually serves (the market ensemble)."""
    from data.adp import load_adp, attach_adp
    from models.market import rolling_market_backtest, market_baseline
    from models.backtest import rolling_backtest
    from models.market_ensemble import MarketEnsembleModel
    from models.stability import compute_factor_stability

    adp = load_adp(list(range(2015, 2025)))

    print("trust: accuracy backtest...")
    bt = rolling_backtest(MarketEnsembleModel, pairs,
                          test_seasons=TRUST_TEST_SEASONS,
                          adp_history=adp, age_adjust=False)
    print("trust: market backtest...")
    mkt = rolling_market_backtest(MarketEnsembleModel, pairs, adp,
                                  test_seasons=TRUST_TEST_SEASONS,
                                  adp_history=adp, age_adjust=False)
    baseline = market_baseline(attach_adp(pairs, adp, season_offset=1))

    cov_wide = rq.coverage_report(resids, lo=0.10, hi=0.90)
    cov_mid = rq.coverage_report(resids, lo=0.25, hi=0.75)

    all_factors = sorted({f for fs in POSITION_FEATURES.values() for f in fs})
    stab = compute_factor_stability(pairs, factors=all_factors)
    top = stab.dropna(subset=["mean_ic"]).sort_values("mean_ic", ascending=False).head(10)

    return {
        "test_seasons": TRUST_TEST_SEASONS,
        "backtest": [
            {"test_season": r["test_season"], "position": r["position"],
             "mae": rnd(r["mae"], 3), "r2": rnd(r["r2"], 3),
             "rank_ic": rnd(r["rank_corr"], 3), "n": int(r["n"])}
            for _, r in bt.iterrows()
        ],
        "market_baseline": [
            {"season": r["season"], "position": r["position"],
             "adp_rank_ic": rnd(r["adp_rank_ic"], 3), "n": int(r["n"])}
            for _, r in baseline.iterrows()
        ],
        "vs_market": [
            {"test_season": r["test_season"], "position": r["position"], "n": int(r["n"]),
             "model_ic": rnd(r["model_ic"], 3), "adp_ic": rnd(r["adp_ic"], 3),
             "ic_edge": rnd(r["ic_edge"], 3), "ls_spread": rnd(r["ls_spread"], 2),
             "long_hit_rate": rnd(r["long_hit_rate"], 2),
             "short_hit_rate": rnd(r["short_hit_rate"], 2)}
            for _, r in mkt.iterrows()
        ],
        "coverage": (
            [{"band": "p10_p90", "position": r["position"], "nominal": 0.80,
              "empirical": rnd(r["empirical_coverage"], 3), "n": int(r["n"])}
             for _, r in cov_wide.iterrows()]
            + [{"band": "p25_p75", "position": r["position"], "nominal": 0.50,
                "empirical": rnd(r["empirical_coverage"], 3), "n": int(r["n"])}
               for _, r in cov_mid.iterrows()]
        ),
        "top_factors": [
            {"factor": f, "mean_ic": rnd(r["mean_ic"], 3), "std_ic": rnd(r["std_ic"], 3),
             "ic_ir": rnd(r["ic_ir"], 2), "pct_positive": rnd(r["pct_positive"], 2),
             "tier": r["stability_tier"], "n_seasons": int(r["n_seasons"])}
            for f, r in top.iterrows()
        ],
    }


_HISTORY_COLS = {
    "team": "team", "age": "age", "games_played": "games",
    "fpts_per_game": "fpts_pg", "fpts": "fpts_total",
    "snap_percentage": "snap_pct", "target_share": "target_share",
    "wopr": "wopr", "tprr": "tprr", "targets_per_game": "targets_pg",
    "epa_per_target": "epa_per_target", "rush_share": "rush_share",
    "carries_per_game": "carries_pg", "epa_per_carry": "epa_per_carry",
    "ypc": "ypc", "dropbacks_per_game": "dropbacks_pg",
    "epa_per_dropback": "epa_per_dropback", "cpoe": "cpoe",
    "boom_rate": "boom_rate", "bust_rate": "bust_rate",
    "trend_class": "trend_class",
    "x_rec_td_rate": "x_rec_td_rate", "rec_td_oe": "rec_td_oe",
    "x_rush_td_rate": "x_rush_td_rate", "rush_td_oe": "rush_td_oe",
    "x_pass_td_rate": "x_pass_td_rate", "pass_td_oe": "pass_td_oe",
}

WEEKLY_SEASONS_BACK = 3  # weekly game logs exported for the last N seasons


def build_history_json(players: list[dict], fm: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    """Per-board-player career: one row per season from the feature matrix,
    plus weekly PPR game logs for the last few seasons."""
    ids = {p["player_id"] for p in players}

    hist = fm[fm["player_id"].isin(ids)].copy()
    usage = (
        hist.get("targets", 0).fillna(0)
        + hist.get("carries", 0).fillna(0)
        + hist.get("dropbacks", 0).fillna(0)
    )
    hist = (
        hist.assign(_usage=usage)
        .sort_values("_usage", ascending=False)
        .drop_duplicates(subset=["player_id", "season"], keep="first")
        .sort_values(["player_id", "season"])
    )

    out: dict[str, dict] = {pid: {"seasons": [], "weekly": {}} for pid in ids}
    cols = [(src, dst) for src, dst in _HISTORY_COLS.items() if src in hist.columns]
    for _, r in hist.iterrows():
        row = {"season": int(r["season"])}
        for src, dst in cols:
            v = r[src]
            if isinstance(v, str):
                row[dst] = v
            else:
                try:
                    f = float(v)
                    row[dst] = None if math.isnan(f) else round(f, 4)
                except (TypeError, ValueError):
                    row[dst] = None
        out[r["player_id"]]["seasons"].append(row)

    if {"player_id", "season", "week"}.issubset(weekly.columns):
        fp_col = "fantasy_points_ppr" if "fantasy_points_ppr" in weekly.columns else "fantasy_points"
        recent = weekly[
            weekly["player_id"].isin(ids)
            & (weekly["season"] >= int(weekly["season"].max()) - WEEKLY_SEASONS_BACK + 1)
        ][["player_id", "season", "week", fp_col]].dropna(subset=[fp_col])
        for (pid, season), grp in recent.groupby(["player_id", "season"], observed=True):
            grp = grp.sort_values("week")
            out[pid]["weekly"][str(int(season))] = [
                {"week": int(w), "pts": round(float(p), 1)}
                for w, p in zip(grp["week"], grp[fp_col])
            ]
    return out


def build_adp_board(season: int, players: list[dict]) -> tuple[list[dict], dict]:
    """Full FFC ladder incl. K/DST, joined to model player_ids by name."""
    from data.adp import normalize_name
    from config import ADP_FORMAT, ADP_TEAMS

    raw_path = CACHE_DIR / f"adp_{ADP_FORMAT}_{ADP_TEAMS}_{season}.parquet"
    if not raw_path.exists():
        raise RuntimeError(f"ADP cache missing: {raw_path} (run the pipeline once)")
    raw = pd.read_parquet(raw_path)
    raw["position"] = raw["position"].replace({"PK": "K", "DEF": "DST"})
    raw["adp"] = pd.to_numeric(raw["adp"], errors="coerce")
    raw = raw.dropna(subset=["adp"]).sort_values("adp").reset_index(drop=True)

    by_key = {(normalize_name(p["name"] or ""), p["position"]): p["player_id"]
              for p in players}

    model_ids = {p["player_id"] for p in players}
    rows, skill_total, skill_joined = [], 0, 0
    top120_total, top120_joined, unjoined_names = 0, 0, []
    for _, r in raw.iterrows():
        pos = r["position"]
        key = (normalize_name(r["player_name"]), pos)
        pid = by_key.get(key)
        is_skill = pos in POSITIONS
        if is_skill:
            skill_total += 1
            if float(r["adp"]) <= 120:
                top120_total += 1
                top120_joined += 1 if pid else 0
            if pid:
                skill_joined += 1
            else:
                unjoined_names.append(f"{r['player_name']} ({pos}, ADP {r['adp']:.0f})")
        if pid is None:
            pid = f"adp_{season}_{key[0].replace(' ', '_')}_{pos}"
        rows.append({
            "player_id": pid,
            "name": r["player_name"],
            "position": pos,
            "team": r.get("team"),
            "adp": rnd(r["adp"], 1),
            "adp_stdev": rnd(r.get("stdev"), 2),
            "bye": int(r["bye"]) if not pd.isna(r.get("bye")) else None,
            "streamer": pos in ("K", "DST"),
            "has_projection": pid in model_ids,
        })

    stats = {"skill_total": skill_total, "skill_joined": skill_joined,
             "join_rate": skill_joined / max(1, skill_total),
             "top120_total": top120_total, "top120_joined": top120_joined,
             "top120_rate": top120_joined / max(1, top120_total),
             "unjoined": unjoined_names}
    return rows, stats


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(players, trust, adp_board, adp_stats) -> list[str]:
    errors = []
    if not players:
        errors.append("players.json is empty")
    if not adp_board:
        errors.append("adp_board.json is empty")
    if not trust.get("backtest"):
        errors.append("trust.json backtest block is empty")
    if not trust.get("vs_market"):
        errors.append("trust.json vs_market block is empty")
    if not trust.get("alpha", {}).get("per_season"):
        errors.append("trust.json alpha block is empty")
    n_fair = sum(1 for p in players if p.get("fair_adp") is not None)
    if n_fair < 100:
        errors.append(f"only {n_fair} players carry fair_adp (expected the priced universe)")
    n_model_alpha = sum(1 for p in players if p.get("alpha_source") == "model")
    if n_model_alpha < 50:
        errors.append(f"only {n_model_alpha} players are model-scored by the alpha overlay")
    # Name-matching regression gate: top-120-ADP players are always in the
    # model universe unless injured out of 2025 entirely; require >=95% there.
    # Deeper players legitimately miss feature thresholds — allow up to 15%
    # overall. (See DECISIONS.md.)
    if adp_stats["top120_rate"] < 0.95:
        errors.append(
            f"ADP top-120 join rate {adp_stats['top120_rate']:.1%} < 95% "
            f"({adp_stats['top120_joined']}/{adp_stats['top120_total']}) — "
            f"unjoined: {adp_stats['unjoined']}"
        )
    if adp_stats["join_rate"] < 0.85:
        errors.append(
            f"ADP overall join rate {adp_stats['join_rate']:.1%} < 85% "
            f"({adp_stats['skill_joined']}/{adp_stats['skill_total']})"
        )
    for p in players:
        missing = [k for k in FORMATS if k not in p["vorp"]]
        if missing:
            errors.append(f"{p['name']}: missing format keys {missing}")
            break
    # Quantile monotonicity
    bad_q = [p["name"] for p in players
             if all(p[f"season_p{q}"] is not None for q in (10, 25, 50, 75, 90))
             and not (p["season_p10"] <= p["season_p25"] <= p["season_p50"]
                      <= p["season_p75"] <= p["season_p90"])]
    if bad_q:
        errors.append(f"non-monotone quantiles for {len(bad_q)} players (e.g. {bad_q[:3]})")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=ROOT / "webapp" / "data")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    board, features_now, extras = build_board(args.season)
    board = per_format_vorp(board)
    board = attach_market(board, args.season)
    board, alpha_diag, alpha_wf = attach_alpha(
        board, features_now, extras, args.season)

    players = build_players_json(board, features_now, extras["rookie_picks"])
    trust = build_trust_json(extras["pairs"], extras["resids"], extras["rq"])
    trust["alpha"] = build_alpha_trust(alpha_wf, alpha_diag)
    adp_board, adp_stats = build_adp_board(args.season, players)
    history = build_history_json(players, extras["fm"], extras["weekly"])

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        sha = "unknown"

    from config import ADP_FORMAT, ADP_TEAMS
    adp_path = CACHE_DIR / f"adp_{ADP_FORMAT}_{ADP_TEAMS}_{args.season}.parquet"
    adp_snapshot = dt.date.fromtimestamp(adp_path.stat().st_mtime).isoformat()

    meta = {
        "build_timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "seasons_trained": [int(TRAINING_SEASONS[0]), int(TRAINING_SEASONS[-1])],
        "projection_season": args.season,
        "model_version": sha,
        "formats": [{"key": k, "label": v["label"], "league_size": v["league_size"],
                     "roster": {p: n for p, n in v["roster"].items() if p != "league_size"}}
                    for k, v in FORMATS.items()],
        "player_count": len(players),
        "rookie_count": sum(1 for p in players if p["rookie"]),
        "adp_source": "FantasyFootballCalculator",
        "adp_snapshot_date": adp_snapshot,
        "adp_format": f"{ADP_TEAMS}-team {ADP_FORMAT}",
        "stdev_synthetic": False,
        "model_stack": "v5 market ensemble (career-augmented hybrid × isotonic ADP prior); "
                       "unsplit — adp_edge ranked by the served ensemble; "
                       "alpha v2 overlay (season-points fair = market + λᵢ·residual, "
                       "survivor-complete targets) alongside, not reranking",
        "alpha_lambda": rnd(alpha_diag.get("lambda_scalar"), 2),
    }

    # Durable ADP snapshot: the missing-2025-archive problem must not recur
    archive = ROOT / "data" / "adp_archive"
    archive.mkdir(parents=True, exist_ok=True)
    if adp_path.exists():
        import shutil
        shutil.copy2(adp_path, archive / adp_path.name)

    errors = validate(players, trust, adp_board, adp_stats)
    if errors:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        return 1

    n_with_history = sum(1 for h in history.values() if h["seasons"])
    if n_with_history < 100:
        print(f"\nVALIDATION FAILED:\n  ✗ history.json: only {n_with_history} players "
              "have season history", file=sys.stderr)
        return 1

    write_json(out / "players.json", players)
    write_json(out / "meta.json", meta)
    write_json(out / "trust.json", trust)
    write_json(out / "adp_board.json", adp_board)
    write_json(out / "history.json", history)

    by_pos = pd.Series([p["position"] for p in players]).value_counts()
    print("\n" + "=" * 62)
    print(f"  build_web_data — season {args.season}  (model {sha})")
    print("=" * 62)
    print(f"  players.json    {len(players):>5} players "
          f"({meta['rookie_count']} rookies)   " +
          "  ".join(f"{p}:{n}" for p, n in by_pos.items()))
    print(f"  formats         {', '.join(FORMATS)}")
    print(f"  adp_board.json  {len(adp_board):>5} rows  "
          f"(join {adp_stats['skill_joined']}/{adp_stats['skill_total']} "
          f"= {adp_stats['join_rate']:.0%}, top-120 {adp_stats['top120_rate']:.0%}, "
          f"snapshot {adp_snapshot})")
    if adp_stats["unjoined"]:
        print(f"  no projection   {', '.join(adp_stats['unjoined'])}")
    print(f"  history.json    {n_with_history} players with season history, "
          f"{sum(len(h['weekly']) for h in history.values())} weekly logs")
    print(f"  trust.json      {len(trust['backtest'])} backtest rows, "
          f"{len(trust['vs_market'])} market rows, "
          f"{len(trust['coverage'])} coverage rows, "
          f"{len(trust['top_factors'])} factors")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
