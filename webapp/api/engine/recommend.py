"""
Pick recommendations: VORP × roster need + urgency, with every component
exposed so the UI can render "why this pick" verbatim.

    need_weight   remaining starter+flex slots at pos ÷ remaining picks,
                  normalized to [0, 1] across positions
    tier_drop     vorp(p) − vorp(best player in the next tier down at pos)
    urgency       (1 − p_survive) × max(tier_drop, 0)
    rec_score     vorp × need_multiplier(need_weight) + urgency

need_multiplier maps need_weight linearly into [0.85, 1.15].
"""
from __future__ import annotations

import math

import numpy as np

from .. import store
from .draft import Draft
from .pool import POS_IDX
from .survival import p_survive

POOL_SIZE = 40
TIER_CLIFF_VORP = 15.0
TIER_CLIFF_REMAINING = 2
SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


def need_weights(draft: Draft) -> dict[str, float]:
    """Normalized starter+flex need per skill position for the user's team."""
    cfg = draft.config
    counts = draft.team_counts(cfg.user_slot)
    remaining_picks = max(1, draft.user_picks_remaining())

    unfilled = {p: max(0.0, cfg.starters(p) - counts[POS_IDX[p]]) for p in SKILL_POSITIONS}

    # FLEX (RB/WR/TE) and SUPERFLEX (QB too) absorb overflow beyond starters
    flex_positions = ["RB", "WR", "TE"]
    overflow = sum(max(0, counts[POS_IDX[p]] - cfg.starters(p)) for p in flex_positions)
    flex_unfilled = max(0, cfg.roster.get("FLEX", 0) - overflow)
    for p in flex_positions:
        unfilled[p] += flex_unfilled / 3.0

    sf_unfilled = max(
        0,
        cfg.roster.get("SUPERFLEX", 0)
        - max(0, counts[POS_IDX["QB"]] - cfg.starters("QB")),
    )
    unfilled["QB"] += sf_unfilled  # superflex is nearly always a QB slot

    raw = {p: unfilled[p] / remaining_picks for p in SKILL_POSITIONS}
    mx = max(raw.values()) or 1.0
    return {p: raw[p] / mx for p in SKILL_POSITIONS}


def need_multiplier(w: float) -> float:
    return 0.85 + 0.30 * w


def tier_structure(draft: Draft, avail: np.ndarray) -> dict[str, dict]:
    """Per position: current best tier among available, vorp of its best
    player, count remaining in it, and the best vorp in the next tier down."""
    pool = draft.pool
    out = {}
    for pos in SKILL_POSITIONS:
        mask = avail & (pool.pos_idx == POS_IDX[pos]) & np.isfinite(pool.tier)
        if not mask.any():
            continue
        tiers = pool.tier[mask]
        vorps = pool.vorp[mask]
        cur = float(np.nanmin(tiers))
        in_cur = mask & (pool.tier == cur)
        below = mask & (pool.tier > cur)
        out[pos] = {
            "current_tier": int(cur),
            "remaining_in_tier": int(in_cur.sum()),
            "best_vorp": float(np.nanmax(pool.vorp[in_cur])),
            "next_tier_best_vorp": float(np.nanmax(pool.vorp[below])) if below.any() else None,
        }
    return out


def next_tier_best_for_player(draft: Draft, avail: np.ndarray, idx: int) -> float | None:
    """Best available vorp in the first tier strictly below the player's own."""
    pool = draft.pool
    if not math.isfinite(pool.tier[idx]):
        return None
    mask = (
        avail
        & (pool.pos_idx == pool.pos_idx[idx])
        & np.isfinite(pool.tier)
        & (pool.tier > pool.tier[idx])
    )
    if not mask.any():
        return None
    next_tier = float(np.min(pool.tier[mask]))
    in_next = mask & (pool.tier == next_tier)
    return float(np.nanmax(pool.vorp[in_next]))


def tier_cliff_alerts(structure: dict[str, dict]) -> list[dict]:
    alerts = []
    for pos, s in structure.items():
        if s["next_tier_best_vorp"] is None:
            continue
        drop = s["best_vorp"] - s["next_tier_best_vorp"]
        if s["remaining_in_tier"] <= TIER_CLIFF_REMAINING and drop > TIER_CLIFF_VORP:
            alerts.append({
                "position": pos,
                "tier": s["current_tier"],
                "remaining_in_tier": s["remaining_in_tier"],
                "drop_to_next_tier": round(drop, 1),
            })
    return alerts


def recommendations(draft: Draft, n: int = 6) -> dict:
    pool = draft.pool
    avail = draft.available_mask()
    surv = p_survive(draft)
    weights = need_weights(draft)
    structure = tier_structure(draft, avail)
    by_id = store.players_by_id()

    cand = np.flatnonzero(avail & np.isfinite(pool.vorp) & pool.has_projection)
    cand = cand[np.argsort(-pool.vorp[cand])][:POOL_SIZE]

    recs = []
    for idx in cand:
        pos = pool.positions[idx]
        vorp = float(pool.vorp[idx])
        nw = weights.get(pos, 0.0)
        mult = need_multiplier(nw)
        nt_best = next_tier_best_for_player(draft, avail, int(idx))
        tier_drop = (vorp - nt_best) if nt_best is not None else 0.0
        ps = float(surv[idx])
        urgency = (1.0 - ps) * max(tier_drop, 0.0)
        rec_score = vorp * mult + urgency

        player = by_id.get(pool.ids[idx], {})
        recs.append({
            "player_id": pool.ids[idx],
            "name": pool.names[idx],
            "position": pos,
            "team": pool.teams[idx],
            "vorp": round(vorp, 1),
            "pos_rank": int(pool.pos_rank[idx]) if math.isfinite(pool.pos_rank[idx]) else None,
            "tier": int(pool.tier[idx]) if math.isfinite(pool.tier[idx]) else None,
            "adp": float(pool.adp[idx]) if math.isfinite(pool.adp[idx]) else None,
            "adp_edge": player.get("adp_edge"),
            "season_p10": player.get("season_p10"),
            "season_p25": player.get("season_p25"),
            "season_p50": player.get("season_p50"),
            "season_p75": player.get("season_p75"),
            "season_p90": player.get("season_p90"),
            "p_survive": round(ps, 3),
            "need_weight": round(nw, 3),
            "need_multiplier": round(mult, 3),
            "tier_drop": round(tier_drop, 1),
            "urgency": round(urgency, 2),
            "rec_score": round(rec_score, 2),
        })

    recs.sort(key=lambda r: -r["rec_score"])
    return {
        "recommendations": recs[:n],
        "pool_size": len(recs),
        "need_weights": {p: round(w, 3) for p, w in weights.items()},
        "tier_structure": structure,
        "tier_cliff_alerts": tier_cliff_alerts(structure),
    }
