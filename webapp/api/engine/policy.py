"""
User draft policies — the strategies compared by the draft-policy backtest
and used for the user's future picks inside rollouts.

Each policy maps a Draft (with the user on the clock) to a pool index.
All policies are constrained by the same position-eligibility rules the
bots follow (caps, early-QB/TE, forced K/DST fill), so every policy
produces a legal, comparable roster.

    adp            lowest ADP among eligible players
    greedy_vorp    highest VORP among eligible players
    heuristic_v1   vorp × need_multiplier + (1 − p_survive) × tier_drop
                   (the pre-VONA production formula, kept as a benchmark)
    vona_v2        vorp × need_multiplier + max(vona, 0)
                   (VONA without the alpha overlay, kept as a benchmark)
    vona_alpha     vona_v2 + ALPHA_REC_WEIGHT × alpha_points
                   (the current production formula)

Caveat on backtesting vona_alpha: the policy backtest scores rosters with
the model's own season_p50, which does not include the alpha correction —
so the alpha tilt can only look neutral-to-negative there. Its real test
is next season's realized points (the end-to-end backtest).
"""
from __future__ import annotations

import numpy as np

from .. import store
from .draft import Draft, eligible_position_mask
from .recommend import ALPHA_REC_WEIGHT, need_multiplier, need_weights, next_tier_best_for_player
from .survival import p_survive
from .vona import expected_next_best, vona_for

# Cap the scored candidate pool: scoring every player is pointless and the
# heuristics' Monte Carlo inputs are only meaningful near the top.
CANDIDATES = 40


def _eligible(draft: Draft) -> np.ndarray:
    cfg = draft.config
    slot = cfg.user_slot
    counts = draft.team_counts(slot)
    round_no = draft.on_the_clock()["round"]
    pos_mask = eligible_position_mask(cfg, counts, round_no, np.array(int(counts.sum())))
    return draft.available_mask() & pos_mask[draft.pool.pos_idx]


def _fallback(draft: Draft, elig: np.ndarray) -> int:
    """Any eligible player (K/DST forced-fill rounds have no VORP/ADP)."""
    order = np.argsort(draft.pool.adp)
    for i in order:
        if elig[i]:
            return int(i)
    raise RuntimeError("No eligible player to pick.")


def pick_adp(draft: Draft) -> int:
    elig = _eligible(draft)
    adp = np.where(elig, draft.pool.adp, np.inf)
    if np.isfinite(adp).any():
        return int(np.argmin(adp))
    return _fallback(draft, elig)


def pick_greedy_vorp(draft: Draft) -> int:
    elig = _eligible(draft)
    vorp = np.where(elig & np.isfinite(draft.pool.vorp), draft.pool.vorp, -np.inf)
    if np.isfinite(vorp).any():
        return int(np.argmax(vorp))
    return _fallback(draft, elig)


def scored_candidates(draft: Draft, use_vona: bool = True, use_alpha: bool = True,
                      k: int = CANDIDATES) -> list[tuple[int, float]]:
    """Eligible candidates as (pool index, score), best first. Empty when no
    eligible player has a projection (e.g. forced K/DST rounds)."""
    pool = draft.pool
    elig = _eligible(draft)
    cand = np.flatnonzero(elig & np.isfinite(pool.vorp) & pool.has_projection)
    if len(cand) == 0:
        return []
    cand = cand[np.argsort(-pool.vorp[cand])][:CANDIDATES]

    weights = need_weights(draft)
    surv = p_survive(draft)
    next_best = expected_next_best(draft) if use_vona else None
    avail = draft.available_mask()
    by_id = store.players_by_id() if use_alpha else {}

    scored = []
    for idx in cand:
        idx = int(idx)
        pos = pool.positions[idx]
        vorp = float(pool.vorp[idx])
        mult = need_multiplier(weights.get(pos, 0.0))
        if use_vona:
            v = vona_for(draft, idx, next_best)
            urgency = max(v, 0.0) if v is not None else 0.0
        else:
            nt_best = next_tier_best_for_player(draft, avail, idx)
            tier_drop = (vorp - nt_best) if nt_best is not None else 0.0
            urgency = (1.0 - float(surv[idx])) * max(tier_drop, 0.0)
        alpha_pts = (by_id.get(pool.ids[idx], {}).get("alpha_points") or 0.0) if use_alpha else 0.0
        scored.append((idx, vorp * mult + urgency + ALPHA_REC_WEIGHT * alpha_pts))
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


def _scored_pick(draft: Draft, use_vona: bool, use_alpha: bool = False) -> int:
    scored = scored_candidates(draft, use_vona=use_vona, use_alpha=use_alpha, k=1)
    if not scored:
        return _fallback(draft, _eligible(draft))
    return scored[0][0]


def pick_heuristic_v1(draft: Draft) -> int:
    return _scored_pick(draft, use_vona=False)


def pick_vona_v2(draft: Draft) -> int:
    return _scored_pick(draft, use_vona=True)


def pick_vona_alpha(draft: Draft) -> int:
    return _scored_pick(draft, use_vona=True, use_alpha=True)


POLICIES = {
    "adp": pick_adp,
    "greedy_vorp": pick_greedy_vorp,
    "heuristic_v1": pick_heuristic_v1,
    "vona_v2": pick_vona_v2,
    "vona_alpha": pick_vona_alpha,
}
