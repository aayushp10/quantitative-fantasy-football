"""
P_survive: probability each available player is still on the board at the
user's next pick, by Monte Carlo over the intervening bot picks.

Fully vectorized over sims: for each intervening pick step we take an
argmin over an (M, n_players) value matrix. ~20 steps × 300 sims runs in
tens of milliseconds. Results are cached per (draft, current pick) on the
Draft object; any state mutation clears the cache.
"""
from __future__ import annotations

import numpy as np

from .draft import Draft, eligible_position_mask, seed_from_id, slot_for_pick
from .pool import POS_ORDER

M_SIMS = 300


def p_survive(draft: Draft) -> np.ndarray:
    """
    Survival probability per pool index, at the user's current pick.
    Already-drafted players get 0; if the draft is over or there are no
    intervening bot picks, available players get 1.
    """
    key = draft.next_overall
    cached = draft._survival_cache.get(key)
    if cached is not None:
        return cached

    avail = draft.available_mask()
    out = avail.astype(float)

    next_user = draft.next_user_overall()
    otc = draft.on_the_clock()
    if otc is None or next_user is None:
        draft._survival_cache[key] = out
        return out

    start = otc["overall"] + 1 if otc["is_user"] else otc["overall"]
    intervening = [
        ov for ov in range(start, next_user)
        if slot_for_pick(ov, draft.config.teams)[1] != draft.config.user_slot
    ]
    if not intervening:
        draft._survival_cache[key] = out
        return out

    pool = draft.pool
    n = pool.n
    rng = np.random.default_rng((seed_from_id(draft.draft_id), key, 7919))

    # Fresh board noise per sim (bot model), shared across that sim's picks
    values = pool.adp[None, :] + rng.normal(0.0, 1.0, size=(M_SIMS, n)) * pool.stdev[None, :]
    values[:, ~pool.bot_draftable()] = np.inf

    alive = np.tile(avail, (M_SIMS, 1))

    # Per-sim roster counts for each bot team involved
    slots = sorted({slot_for_pick(ov, draft.config.teams)[1] for ov in intervening})
    counts = {s: np.tile(draft.team_counts(s), (M_SIMS, 1)) for s in slots}
    made = {s: np.full(M_SIMS, int(draft.team_counts(s).sum())) for s in slots}

    n_pos = len(POS_ORDER)
    for ov in intervening:
        round_no, slot = slot_for_pick(ov, draft.config.teams)
        pos_ok = eligible_position_mask(draft.config, counts[slot], round_no, made[slot])
        elig = alive & pos_ok[:, pool.pos_idx] & np.isfinite(values)
        v = np.where(elig, values, np.inf)
        # Sims with nothing eligible fall back to best available
        none_elig = ~np.isfinite(v).any(axis=1)
        if none_elig.any():
            v[none_elig] = np.where(alive[none_elig], values[none_elig], np.inf)
        choice = np.argmin(v, axis=1)
        valid = np.isfinite(v[np.arange(M_SIMS), choice])
        rows = np.flatnonzero(valid)
        alive[rows, choice[rows]] = False
        picked_pos = pool.pos_idx[choice[rows]]
        counts[slot][rows, picked_pos] += 1
        made[slot][rows] += 1

    out = np.where(avail, alive.mean(axis=0), 0.0)
    draft._survival_cache[key] = out
    return out
