"""
VONA — value over next available.

For each position, the survival Monte Carlo's joint alive states give the
distribution of "best VORP still on the board at the user's next pick".
A player's VONA is his VORP minus that expectation for his position:

    vona(p) = vorp(p) − E[ max vorp available at pos(p) at next user pick ]

The expectation includes p's own survival, so for the top player at a
position vona ≈ (1 − p_survive) × (vorp − next best) — the old tier-drop
urgency, but with the simulated next-best instead of tier boundaries.
Clipped at 0 it is the expected regret of deferring the position one round.

If the user has no future pick the alive matrix collapses to the current
board, vona ≤ 0 everywhere, and urgency is naturally zero.
"""
from __future__ import annotations

import numpy as np

from .draft import Draft
from .pool import POS_IDX
from .survival import survival_alive

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


def expected_next_best(draft: Draft) -> dict[str, float | None]:
    """E[best available VORP at the user's next pick], per skill position.

    None when a position has no projected players left in any sim.
    Cached per (draft, current pick) alongside the survival results.
    """
    key = ("next_best", draft.next_overall)
    cached = draft._survival_cache.get(key)
    if cached is not None:
        return cached

    pool = draft.pool
    alive = survival_alive(draft)
    out: dict[str, float | None] = {}
    for pos in SKILL_POSITIONS:
        mask = (pool.pos_idx == POS_IDX[pos]) & np.isfinite(pool.vorp)
        if not mask.any():
            out[pos] = None
            continue
        v = np.where(alive[:, mask], pool.vorp[mask][None, :], -np.inf)
        best = v.max(axis=1)                      # (M,) best per sim
        has = np.isfinite(best)                   # sims where anyone is left
        out[pos] = float(best[has].mean()) if has.any() else None

    draft._survival_cache[key] = out
    return out


def vona_for(draft: Draft, idx: int, next_best: dict[str, float | None] | None = None) -> float | None:
    """VONA for one pool index; None if the player or position has no VORP."""
    pool = draft.pool
    if not np.isfinite(pool.vorp[idx]):
        return None
    nb = (next_best or expected_next_best(draft)).get(pool.positions[idx])
    if nb is None:
        return None
    return float(pool.vorp[idx] - nb)
