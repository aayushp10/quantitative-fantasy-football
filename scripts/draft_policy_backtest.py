"""
Draft-policy backtest: which pick strategy builds the best completed roster?

Runs full mock drafts (bots from the frozen engine, user picks from a
policy) and scores the user's final roster with roster_utility. Policies
face identical conditions — the draft id (hence bot boards and reach rolls)
is shared across policies for each (seed, slot) cell, so policy deltas are
paired differences over the same simulated opponents.

Policies: adp, greedy_vorp, heuristic_v1 (pre-VONA production formula),
vona_v2 (current production formula), and optionally rollout (full draft
simulation per candidate, --with-rollout).

Usage:
    .venv/bin/python scripts/draft_policy_backtest.py
    .venv/bin/python scripts/draft_policy_backtest.py --with-rollout --seeds 4

Writes output/experiments/draft_policy_v1/results.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webapp.api.engine.draft import Draft, DraftConfig            # noqa: E402
from webapp.api.engine.policy import POLICIES, pick_vona_v2       # noqa: E402
from webapp.api.engine.rollout import pick_rollout                # noqa: E402
from webapp.api.engine.roster_utility import RosterScorer         # noqa: E402

DEFAULT_ROSTER = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "BN": 7, "K": 0, "DST": 0}
OUT_DIR = ROOT / "output" / "experiments" / "draft_policy_v1"


def run_draft(draft_id: str, cfg: DraftConfig, pick_fn) -> dict:
    d = Draft.create(draft_id, cfg)
    d.advance_bots()
    while not d.complete:
        idx = pick_fn(d)
        d.user_pick(d.pool.ids[idx])
        d.advance_bots()
    scorer = RosterScorer(d.pool, cfg)
    roster = [d.pool.index[p["player_id"]] for p in d.picks if p["is_user"]]
    score = scorer.score(roster)
    return {
        "draft_id": draft_id,
        "utility": score.utility,
        **score.as_dict(),
        "roster": [d.pool.names[i] for i in sorted(roster, key=lambda i: -np.nan_to_num(scorer.p50[i]))],
    }


def make_rollout_policy(n_candidates: int, n_sims: int, max_round: int):
    def pick(d: Draft):
        if d.on_the_clock()["round"] <= max_round:
            idx = pick_rollout(d, n_candidates=n_candidates, n_sims=n_sims)
            if idx is not None:
                return idx
        return pick_vona_v2(d)
    return pick


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8, help="draft seeds per slot")
    ap.add_argument("--slots", type=str, default="1,4,7,10,12")
    ap.add_argument("--teams", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--format", type=str, default="12_ppr")
    ap.add_argument("--with-rollout", action="store_true")
    ap.add_argument("--rollout-cands", type=int, default=8)
    ap.add_argument("--rollout-sims", type=int, default=24)
    ap.add_argument("--rollout-rounds", type=int, default=8,
                    help="use rollouts through this round, vona_v2 after")
    args = ap.parse_args()

    slots = [int(s) for s in args.slots.split(",")]
    cells = [(seed, slot) for seed in range(args.seeds) for slot in slots]

    policies = dict(POLICIES)
    if args.with_rollout:
        policies["rollout"] = make_rollout_policy(
            args.rollout_cands, args.rollout_sims, args.rollout_rounds)

    results: dict[str, list[dict]] = {}
    for name, fn in policies.items():
        t0 = time.time()
        rows = []
        for seed, slot in cells:
            cfg = DraftConfig(teams=args.teams, user_slot=slot, rounds=args.rounds,
                              format=args.format, roster=dict(DEFAULT_ROSTER))
            # Same draft_id across policies -> same bot boards (paired comparison)
            rows.append(run_draft(f"bt{seed}s{slot}", cfg, fn))
        results[name] = rows
        u = np.array([r["utility"] for r in rows])
        print(f"{name:14s} mean utility {u.mean():7.1f}  ({time.time()-t0:.1f}s, {len(rows)} drafts)")

    # --- Summary with paired deltas vs the ADP baseline
    base = np.array([r["utility"] for r in results["adp"]])
    summary = {}
    print(f"\n{'policy':14s} {'utility':>9s} {'starters':>9s} {'Δ vs adp':>9s} {'95% CI':>16s}")
    for name, rows in results.items():
        u = np.array([r["utility"] for r in rows])
        sp = np.array([r["starter_points"] for r in rows])
        d = u - base
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        summary[name] = {
            "mean_utility": round(float(u.mean()), 1),
            "mean_starter_points": round(float(sp.mean()), 1),
            "delta_vs_adp": round(float(d.mean()), 1),
            "delta_ci95": [round(float(lo), 1), round(float(hi), 1)],
            "n_drafts": len(rows),
        }
        print(f"{name:14s} {u.mean():9.1f} {sp.mean():9.1f} {d.mean():+9.1f} "
              f"[{lo:+6.1f}, {hi:+6.1f}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps({
        "config": vars(args), "summary": summary, "drafts": results,
    }, indent=2))
    print(f"\nwrote {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
