"""
The draftable player pool for one draft: ADP-board players (bot-draftable)
plus model-only players (user-draftable), with the chosen format's VORP
fields attached. Everything is held in numpy arrays for the Monte Carlo
survival simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import store

POS_ORDER = ["QB", "RB", "WR", "TE", "K", "DST"]
POS_IDX = {p: i for i, p in enumerate(POS_ORDER)}


@dataclass
class PlayerPool:
    ids: list[str]
    names: list[str]
    teams: list[str]
    positions: list[str]          # QB/RB/WR/TE/K/DST
    pos_idx: np.ndarray           # int per player
    adp: np.ndarray               # float, np.inf where no ADP
    stdev: np.ndarray             # float, 0 where no ADP
    vorp: np.ndarray              # float, nan where no projection
    tier: np.ndarray              # float, nan where none
    pos_rank: np.ndarray
    streamer: np.ndarray          # bool
    has_projection: np.ndarray    # bool
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.index = {pid: i for i, pid in enumerate(self.ids)}

    @property
    def n(self) -> int:
        return len(self.ids)

    def bot_draftable(self) -> np.ndarray:
        return np.isfinite(self.adp)


def build_pool(format_key: str) -> PlayerPool:
    board = store.adp_board()
    by_id = store.players_by_id()

    ids, names, teams, positions = [], [], [], []
    adp, stdev, vorp, tier, pos_rank = [], [], [], [], []
    streamer, has_proj = [], []

    seen = set()
    for row in board:
        pid = row["player_id"]
        if pid in seen:
            continue
        seen.add(pid)
        p = by_id.get(pid)
        v = (p or {}).get("vorp", {}).get(format_key) if p else None
        ids.append(pid)
        names.append(row["name"])
        teams.append(row.get("team"))
        positions.append(row["position"])
        adp.append(float(row["adp"]))
        sd = row.get("adp_stdev")
        stdev.append(max(0.5, float(sd)) if sd is not None else max(2.0, 0.15 * float(row["adp"])))
        vorp.append(v["vorp"] if v else np.nan)
        tier.append(v["tier"] if v and v["tier"] is not None else np.nan)
        pos_rank.append(v["pos_rank"] if v else np.nan)
        streamer.append(bool(row.get("streamer")))
        has_proj.append(p is not None)

    # Model players with no ADP: user-draftable deep sleepers.
    for p in store.players():
        pid = p["player_id"]
        if pid in seen:
            continue
        v = p["vorp"].get(format_key)
        if not v:
            continue
        seen.add(pid)
        ids.append(pid)
        names.append(p["name"])
        teams.append(p.get("team"))
        positions.append(p["position"])
        adp.append(np.inf)
        stdev.append(0.0)
        vorp.append(v["vorp"])
        tier.append(v["tier"] if v["tier"] is not None else np.nan)
        pos_rank.append(v["pos_rank"])
        streamer.append(False)
        has_proj.append(True)

    return PlayerPool(
        ids=ids, names=names, teams=teams, positions=positions,
        pos_idx=np.array([POS_IDX[p] for p in positions], dtype=np.int64),
        adp=np.array(adp, dtype=float),
        stdev=np.array(stdev, dtype=float),
        vorp=np.array(vorp, dtype=float),
        tier=np.array(tier, dtype=float),
        pos_rank=np.array(pos_rank, dtype=float),
        streamer=np.array(streamer, dtype=bool),
        has_projection=np.array(has_proj, dtype=bool),
    )
