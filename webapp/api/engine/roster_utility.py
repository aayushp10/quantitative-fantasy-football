"""
Roster utility v1 — the objective the policy backtest and rollouts optimize.

    utility = starter_points − FRAG_W · starter_downside + BENCH_W · bench_points

    starter_points     Σ season_p50 over the optimal legal starting lineup
                       (greedy fill: dedicated slots by position, then FLEX
                       from RB/WR/TE, then SUPERFLEX from QB/RB/WR/TE —
                       greedy is optimal for this nested slot structure)
    starter_downside   Σ (season_p50 − season_p10) over starters: a fragility
                       penalty for thin distributions
    bench_points       Σ season_p50 over the top BENCH_TOP_N bench players

K/DST contribute zero — they are streamers without projections and every
policy fills them in the same forced late-round window, so they cancel in
policy comparisons. Unfilled starter slots contribute zero, which is the
implicit penalty for failing to draft a startable player at a position.

season_p50/p10 come from the frozen players.json artifacts; this module is
deliberately v1 — it should be replaced by playoff/championship probability
once the weekly team simulator exists.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import store
from .draft import DraftConfig
from .pool import PlayerPool

FRAG_W = 0.10
BENCH_W = 0.15
BENCH_TOP_N = 3

FLEX_POSITIONS = ("RB", "WR", "TE")
SUPERFLEX_POSITIONS = ("QB", "RB", "WR", "TE")


@dataclass
class RosterScore:
    starter_points: float
    fragility: float
    bench_points: float

    @property
    def utility(self) -> float:
        return self.starter_points - FRAG_W * self.fragility + BENCH_W * self.bench_points

    def as_dict(self) -> dict:
        return {
            "starter_points": round(self.starter_points, 1),
            "fragility": round(self.fragility, 1),
            "bench_points": round(self.bench_points, 1),
            "utility": round(self.utility, 1),
        }


class RosterScorer:
    """Season-quantile arrays aligned to a pool, reused across many rosters."""

    def __init__(self, pool: PlayerPool, config: DraftConfig):
        self.pool = pool
        self.config = config
        by_id = store.players_by_id()
        p50 = np.full(pool.n, np.nan)
        p10 = np.full(pool.n, np.nan)
        for i, pid in enumerate(pool.ids):
            p = by_id.get(pid)
            if p and p.get("season_p50") is not None:
                p50[i] = p["season_p50"]
                p10[i] = p["season_p10"] if p.get("season_p10") is not None else p["season_p50"]
        self.p50 = p50
        self.p10 = p10

    def score(self, roster_indices) -> RosterScore:
        """Score a roster given as an iterable of pool indices."""
        idxs = [int(i) for i in roster_indices]
        # Projected players only, best first; K/DST and unprojected drop out
        idxs = [i for i in idxs if np.isfinite(self.p50[i])
                and self.pool.positions[i] not in ("K", "DST")]
        idxs.sort(key=lambda i: -self.p50[i])

        r = self.config.roster
        open_slots = {p: r.get(p, 0) for p in SUPERFLEX_POSITIONS}
        flex_open = r.get("FLEX", 0)
        sflex_open = r.get("SUPERFLEX", 0)

        starters: list[int] = []
        bench: list[int] = []
        for i in idxs:
            pos = self.pool.positions[i]
            if open_slots.get(pos, 0) > 0:
                open_slots[pos] -= 1
                starters.append(i)
            elif pos in FLEX_POSITIONS and flex_open > 0:
                flex_open -= 1
                starters.append(i)
            elif pos in SUPERFLEX_POSITIONS and sflex_open > 0:
                sflex_open -= 1
                starters.append(i)
            else:
                bench.append(i)

        starter_points = float(sum(self.p50[i] for i in starters))
        fragility = float(sum(self.p50[i] - self.p10[i] for i in starters))
        bench_points = float(sum(self.p50[i] for i in bench[:BENCH_TOP_N]))
        return RosterScore(starter_points, fragility, bench_points)
