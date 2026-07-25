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
    """Run the full projection pipeline; returns (board, features_now, extras)."""
    from features.assembler import assemble_feature_matrix, build_yoy_pairs
    from models.two_stage import ALL_RATE_TARGET_COLS
    from models.hybrid import HybridProjectionModel
    from models.uncertainty import (
        ResidualQuantiles,
        simulate_season_totals,
        walk_forward_residuals,
    )
    from models.availability import AvailabilityModel
    from models.rookie import RookieModel, merge_rookie_projections
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

    model = HybridProjectionModel()
    model.train(pairs)

    resids = walk_forward_residuals(
        lambda: HybridProjectionModel(age_adjust=False), pairs, min_train_seasons=4
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
        "resids": resids,
        "rq": rq,
        "rookie_picks": rookie_picks,
        "input_season": input_season,
    }
    return board, features_now, extras


def attach_market(board: pd.DataFrame, season: int) -> pd.DataFrame:
    """ADP join + predicted_adp ladder + adp_edge (12-team PPR market)."""
    from data.adp import load_adp, attach_adp

    adp_now = load_adp([season])
    if adp_now.empty:
        raise RuntimeError(f"No ADP available for {season}")

    board = board.sort_values("vorp_12_ppr", ascending=False).reset_index(drop=True)
    board["model_overall_rank"] = np.arange(1, len(board) + 1)
    board["projected_season"] = season
    board = attach_adp(board, adp_now, season_offset=0, season_col="projected_season")

    ladder = np.sort(adp_now["adp"].values)
    idx = np.minimum(board["model_overall_rank"].values - 1, len(ladder) - 1)
    pred = ladder[idx]
    overflow = board["model_overall_rank"].values > len(ladder)
    board["predicted_adp"] = np.round(np.where(
        overflow, ladder[-1] + (board["model_overall_rank"].values - len(ladder)), pred), 1)
    board["adp_edge"] = (board["adp"] - board["predicted_adp"]).round(1)
    return board


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


def build_trust_json(pairs: pd.DataFrame, resids, rq) -> dict:
    from data.adp import load_adp, attach_adp
    from models.market import rolling_market_backtest, market_baseline
    from models.backtest import rolling_backtest
    from models.hybrid import HybridProjectionModel
    from models.stability import compute_factor_stability

    adp = load_adp(list(range(2015, 2025)))

    print("trust: accuracy backtest...")
    bt = rolling_backtest(HybridProjectionModel, pairs,
                          test_seasons=TRUST_TEST_SEASONS, age_adjust=False)
    print("trust: market backtest...")
    mkt = rolling_market_backtest(HybridProjectionModel, pairs, adp,
                                  test_seasons=TRUST_TEST_SEASONS, age_adjust=False)
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

    players = build_players_json(board, features_now, extras["rookie_picks"])
    trust = build_trust_json(extras["pairs"], extras["resids"], extras["rq"])
    adp_board, adp_stats = build_adp_board(args.season, players)

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
    }

    errors = validate(players, trust, adp_board, adp_stats)
    if errors:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        return 1

    write_json(out / "players.json", players)
    write_json(out / "meta.json", meta)
    write_json(out / "trust.json", trust)
    write_json(out / "adp_board.json", adp_board)

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
    print(f"  trust.json      {len(trust['backtest'])} backtest rows, "
          f"{len(trust['vs_market'])} market rows, "
          f"{len(trust['coverage'])} coverage rows, "
          f"{len(trust['top_factors'])} factors")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
