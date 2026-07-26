"""
Draft rollouts: rank candidate picks by the expected utility of the
*completed* roster they lead to, not by their standalone value.

For each of the top candidates at the user's current pick:
  1. Force the candidate as this pick.
  2. Simulate every remaining pick to the end of the draft — bots pick by
     their noisy ADP boards (one shared noise board per simulation, as in
     survival.py), the user's future picks follow a simple base policy.
  3. Score the completed user roster with roster_utility.
  4. Average over simulations.

Common random numbers: every candidate faces the same per-sim noise boards
(same rng stream, keyed by draft id and current pick), so candidate deltas
are paired differences — the noise largely cancels and small utility gaps
resolve with far fewer sims than independent draws would need.

The whole batch (candidates × sims) advances in lockstep through the pick
schedule as one (C·M, n) matrix — each step is a masked argmin (bots) or
argmax (user base policy), so a full rollout is ~150 vectorized steps.

Base policy for the user's future picks: eligible best-VORP with a flat
bonus for positions whose starter slots are unfilled. Deliberately cheap —
nested Monte Carlo (VONA inside a rollout) would be quadratic — and shared
by every candidate, so base-policy bias mostly cancels in comparisons.
"""
from __future__ import annotations

import numpy as np

from .draft import Draft, eligible_position_mask, seed_from_id, slot_for_pick
from .pool import POS_IDX, POS_ORDER
from .policy import scored_candidates
from .roster_utility import RosterScorer

N_CANDIDATES = 8
N_SIMS = 24
STARTER_BOOST = 25.0   # season-points bonus steering the base policy to unfilled starter slots
_RNG_TAG = 104729      # distinct stream from survival.py's 7919


def rollout_recommendations(
    draft: Draft,
    n_candidates: int = N_CANDIDATES,
    n_sims: int = N_SIMS,
    scorer: RosterScorer | None = None,
) -> dict:
    """Rank candidate picks by expected completed-roster utility.

    Requires the user to be on the clock. Returns {"candidates": [...]}
    ranked by mean utility, each with paired delta vs the top candidate.
    Empty candidate list when no eligible player has a projection.
    """
    otc = draft.on_the_clock()
    if otc is None or not otc["is_user"]:
        raise ValueError("Rollouts require the user to be on the clock.")

    cand_scored = scored_candidates(draft, use_vona=True, k=n_candidates)
    if not cand_scored:
        return {"candidates": [], "n_sims": n_sims}
    cand_idx = np.array([i for i, _ in cand_scored], dtype=np.int64)

    pool, cfg = draft.pool, draft.config
    n, C, M = pool.n, len(cand_idx), n_sims
    rows = C * M
    scorer = scorer or RosterScorer(pool, cfg)

    # --- Common random numbers: one noise board per sim, tiled across candidates
    rng = np.random.default_rng((seed_from_id(draft.draft_id), draft.next_overall, _RNG_TAG))
    values = pool.adp[None, :] + rng.normal(0.0, 1.0, size=(M, n)) * pool.stdev[None, :]
    values[:, ~pool.bot_draftable()] = np.inf
    values = np.tile(values, (C, 1))                       # (rows, n)

    # --- State: availability, per-team position counts, picks made
    alive = np.tile(draft.available_mask(), (rows, 1))
    cand_of_row = np.repeat(cand_idx, M)                   # (rows,)
    alive[np.arange(rows), cand_of_row] = False

    counts = {s: np.tile(draft.team_counts(s), (rows, 1)) for s in range(1, cfg.teams + 1)}
    made = {s: np.full(rows, int(counts[s][0].sum())) for s in counts}
    ucounts = counts[cfg.user_slot]
    ucounts[np.arange(rows), pool.pos_idx[cand_of_row]] += 1
    made[cfg.user_slot] += 1

    # --- User base-policy inputs
    vorp_base = np.where(np.isfinite(pool.vorp) & pool.has_projection, pool.vorp, -np.inf)
    starters_arr = np.array([cfg.starters(p) for p in POS_ORDER])
    user_future: list[np.ndarray] = []                     # chosen pool idx per row, per user step

    # --- Lockstep simulation of every remaining pick
    for overall in range(draft.next_overall + 1, cfg.total_picks + 1):
        round_no, slot = slot_for_pick(overall, cfg.teams)
        pos_ok = eligible_position_mask(cfg, counts[slot], round_no, made[slot])
        pos_ok_players = pos_ok[:, pool.pos_idx]           # (rows, n)

        if slot == cfg.user_slot:
            need = counts[slot] < starters_arr             # (rows, 6) unfilled starter slots
            score = vorp_base[None, :] + STARTER_BOOST * need[:, pool.pos_idx]
            score = np.where(alive & pos_ok_players, score, -np.inf)
            choice = np.argmax(score, axis=1)
            # Rows with no projected candidate (forced K/DST): lowest true ADP
            dead = ~np.isfinite(score[np.arange(rows), choice])
            if dead.any():
                adp = np.where(alive[dead] & pos_ok_players[dead], pool.adp[None, :], np.inf)
                choice[dead] = np.argmin(adp, axis=1)
            user_future.append(choice.copy())
        else:
            elig = alive & pos_ok_players & np.isfinite(values)
            v = np.where(elig, values, np.inf)
            none_elig = ~np.isfinite(v).any(axis=1)
            if none_elig.any():
                v[none_elig] = np.where(alive[none_elig], values[none_elig], np.inf)
            choice = np.argmin(v, axis=1)

        r = np.arange(rows)
        alive[r, choice] = False
        counts[slot][r, pool.pos_idx[choice]] += 1
        made[slot] += 1

    # --- Score completed user rosters
    current = [pool.index[p["player_id"]] for p in draft.picks if p["is_user"]]
    future = np.array(user_future).T if user_future else np.empty((rows, 0), dtype=np.int64)
    utilities = np.empty(rows)
    for row in range(rows):
        roster = current + [int(cand_of_row[row])] + [int(i) for i in future[row]]
        utilities[row] = scorer.score(roster).utility
    utilities = utilities.reshape(C, M)

    # --- Rank; paired deltas vs the best candidate (CRN makes these low-noise)
    means = utilities.mean(axis=1)
    order = np.argsort(-means)
    best = order[0]
    out = []
    for c in order:
        idx = int(cand_idx[c])
        diff = utilities[best] - utilities[c]              # paired per-sim
        out.append({
            "player_id": pool.ids[idx],
            "name": pool.names[idx],
            "position": pool.positions[idx],
            "team": pool.teams[idx],
            "vorp": round(float(pool.vorp[idx]), 1),
            "adp": float(pool.adp[idx]) if np.isfinite(pool.adp[idx]) else None,
            "mean_utility": round(float(means[c]), 1),
            "se_utility": round(float(utilities[c].std(ddof=1) / np.sqrt(M)), 2),
            "delta_vs_best": round(float(diff.mean()), 1),
            "se_delta": round(float(diff.std(ddof=1) / np.sqrt(M)), 2) if c != best else 0.0,
        })
    return {"candidates": out, "n_sims": n_sims}


def pick_rollout(draft: Draft, n_candidates: int = N_CANDIDATES, n_sims: int = N_SIMS) -> int | None:
    """Policy wrapper: best rollout candidate's pool index, or None when
    there are no projected candidates (caller should fall back)."""
    res = rollout_recommendations(draft, n_candidates=n_candidates, n_sims=n_sims)
    if not res["candidates"]:
        return None
    return draft.pool.index[res["candidates"][0]["player_id"]]
