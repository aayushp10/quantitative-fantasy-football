"""
Mock-draft state machine and the ADP bot model.

Bot model (per spec):
- At draft creation each bot team draws a private board:
      value_b(p) = adp(p) + Normal(0, stdev(p))
- Bots pick their lowest-value available player subject to roster
  constraints evaluated at pick time; constraint-blocked players are
  skipped down the board.
- 8% chance per pick of a "reach": uniform choice from the bot's top 5
  eligible players.
- All randomness is seeded from the draft id: boards from the creation
  seed, each pick's reach roll from (seed, overall). A draft therefore
  replays identically after undo or a server restart.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .pool import PlayerPool, POS_IDX, POS_ORDER, build_pool

REACH_PROB = 0.08
REACH_TOP_N = 5
EARLY_QB_TE_ROUND = 10  # no 2nd QB / 2nd TE before this round in 1-QB formats


def seed_from_id(draft_id: str) -> int:
    return int(hashlib.sha256(draft_id.encode()).hexdigest()[:8], 16)


def slot_for_pick(overall: int, teams: int) -> tuple[int, int]:
    """1-indexed (round, slot) for a 1-indexed overall pick, snake order."""
    r = (overall - 1) // teams          # 0-based round
    i = (overall - 1) % teams
    slot = i + 1 if r % 2 == 0 else teams - i
    return r + 1, slot


@dataclass
class DraftConfig:
    teams: int
    user_slot: int
    rounds: int
    format: str
    roster: dict[str, int]      # QB RB WR TE FLEX (SUPERFLEX) BN K DST

    @property
    def total_picks(self) -> int:
        return self.teams * self.rounds

    @property
    def is_superflex(self) -> bool:
        return self.roster.get("SUPERFLEX", 0) > 0

    def starters(self, pos: str) -> int:
        return self.roster.get(pos, 0)

    def position_caps(self) -> dict[str, int]:
        r = self.roster
        bench_share = math.ceil(r.get("BN", 6) * 0.5)
        return {
            "QB": r.get("QB", 1) + r.get("SUPERFLEX", 0) + 1,
            "TE": r.get("TE", 1) + 1,
            "RB": r.get("RB", 2) + r.get("FLEX", 1) + bench_share,
            "WR": r.get("WR", 2) + r.get("FLEX", 1) + bench_share,
            "K": r.get("K", 0),
            "DST": r.get("DST", 0),
        }

    @property
    def n_kdst_rounds(self) -> int:
        return self.roster.get("K", 0) + self.roster.get("DST", 0)


def eligible_position_mask(
    cfg: DraftConfig,
    counts: np.ndarray,          # (..., 6) picks so far per position (POS_ORDER)
    round_no: int,
    picks_made: np.ndarray,      # (...,) total picks made by the team
) -> np.ndarray:
    """
    Boolean (..., 6) mask of positions the team may draft right now.
    Vectorized over any leading shape (used per-sim in the Monte Carlo).
    """
    caps = cfg.position_caps()
    cap_arr = np.array([caps[p] for p in POS_ORDER])
    mask = counts < cap_arr

    # No 2nd QB / 2nd TE early in 1-QB formats
    if not cfg.is_superflex and round_no < EARLY_QB_TE_ROUND:
        mask[..., POS_IDX["QB"]] &= counts[..., POS_IDX["QB"]] < 1
        mask[..., POS_IDX["TE"]] &= counts[..., POS_IDX["TE"]] < 1

    # K/DST only in the last n_kdst rounds
    if round_no <= cfg.rounds - cfg.n_kdst_rounds:
        mask[..., POS_IDX["K"]] = False
        mask[..., POS_IDX["DST"]] = False

    # Forced fill: when remaining picks just cover unfilled K/DST slots,
    # only those positions are eligible.
    k_need = np.maximum(0, cfg.starters("K") - counts[..., POS_IDX["K"]])
    d_need = np.maximum(0, cfg.starters("DST") - counts[..., POS_IDX["DST"]])
    remaining = cfg.rounds - picks_made
    forced = (k_need + d_need) >= remaining
    if np.any(forced):
        forced_mask = np.zeros_like(mask)
        forced_mask[..., POS_IDX["K"]] = k_need > 0
        forced_mask[..., POS_IDX["DST"]] = d_need > 0
        mask = np.where(forced[..., None], forced_mask, mask)

    return mask


@dataclass
class Draft:
    draft_id: str
    config: DraftConfig
    pool: PlayerPool
    picks: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    boards: np.ndarray | None = None            # (teams, n) bot private values
    _survival_cache: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, draft_id: str, config: DraftConfig) -> "Draft":
        pool = build_pool(config.format)
        d = cls(draft_id=draft_id, config=config, pool=pool)
        d._draw_boards()
        return d

    def _draw_boards(self) -> None:
        rng = np.random.default_rng(seed_from_id(self.draft_id))
        noise = rng.normal(0.0, 1.0, size=(self.config.teams, self.pool.n))
        boards = self.pool.adp[None, :] + noise * self.pool.stdev[None, :]
        boards[:, ~self.pool.bot_draftable()] = np.inf
        self.boards = boards

    # ------------------------------------------------------------------
    # Derived state
    # ------------------------------------------------------------------

    @property
    def next_overall(self) -> int:
        return len(self.picks) + 1

    @property
    def complete(self) -> bool:
        return len(self.picks) >= self.config.total_picks

    def on_the_clock(self) -> dict | None:
        if self.complete:
            return None
        overall = self.next_overall
        rnd, slot = slot_for_pick(overall, self.config.teams)
        return {"overall": overall, "round": rnd, "slot": slot,
                "is_user": slot == self.config.user_slot}

    def picked_ids(self) -> set[str]:
        return {p["player_id"] for p in self.picks}

    def available_mask(self) -> np.ndarray:
        mask = np.ones(self.pool.n, dtype=bool)
        for p in self.picks:
            i = self.pool.index.get(p["player_id"])
            if i is not None:
                mask[i] = False
        return mask

    def team_counts(self, slot: int) -> np.ndarray:
        counts = np.zeros(len(POS_ORDER), dtype=np.int64)
        for p in self.picks:
            if p["slot"] == slot:
                counts[POS_IDX[self.pool.positions[self.pool.index[p["player_id"]]]]] += 1
        return counts

    def rosters(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {s: [] for s in range(1, self.config.teams + 1)}
        for p in self.picks:
            out[p["slot"]].append(p["player_id"])
        return out

    def user_picks_remaining(self) -> int:
        made = sum(1 for p in self.picks if p["is_user"])
        return self.config.rounds - made

    def next_user_overall(self) -> int | None:
        """Overall number of the user's next pick strictly after the current state."""
        start = self.next_overall
        otc = self.on_the_clock()
        if otc and otc["is_user"]:
            start = otc["overall"] + 1  # the one after the current user pick
        for overall in range(start, self.config.total_picks + 1):
            _, slot = slot_for_pick(overall, self.config.teams)
            if slot == self.config.user_slot:
                return overall
        return None

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def _record(self, player_idx: int, is_user: bool) -> dict:
        overall = self.next_overall
        rnd, slot = slot_for_pick(overall, self.config.teams)
        pick = {
            "overall": overall, "round": rnd, "slot": slot,
            "player_id": self.pool.ids[player_idx],
            "player_name": self.pool.names[player_idx],
            "position": self.pool.positions[player_idx],
            "team": self.pool.teams[player_idx],
            "is_user": is_user,
        }
        self.picks.append(pick)
        self._survival_cache.clear()
        return pick

    def user_pick(self, player_id: str) -> dict:
        otc = self.on_the_clock()
        if otc is None:
            raise ValueError("Draft is complete.")
        if not otc["is_user"]:
            raise ValueError("It is not your pick.")
        idx = self.pool.index.get(player_id)
        if idx is None:
            raise ValueError(f"Unknown player_id {player_id}")
        if player_id in self.picked_ids():
            raise ValueError(f"{self.pool.names[idx]} is already drafted.")
        return self._record(idx, is_user=True)

    def bot_pick(self) -> dict:
        otc = self.on_the_clock()
        assert otc is not None and not otc["is_user"]
        slot, overall, round_no = otc["slot"], otc["overall"], otc["round"]

        counts = self.team_counts(slot)
        picks_made = np.array(int(counts.sum()))
        pos_mask = eligible_position_mask(self.config, counts, round_no, picks_made)

        avail_all = self.available_mask()
        pos_ok = pos_mask[self.pool.pos_idx]
        eligible = avail_all & self.pool.bot_draftable() & pos_ok
        if not eligible.any():
            # ADP-priced pool exhausted for allowed positions: draft the best
            # remaining projection at an allowed position (deep sleepers),
            # never violating position eligibility.
            for cand_mask in (
                avail_all & pos_ok & np.isfinite(self.pool.vorp),
                avail_all & np.isfinite(self.pool.vorp),
                avail_all,
            ):
                if cand_mask.any():
                    vorp = np.where(cand_mask & np.isfinite(self.pool.vorp),
                                    self.pool.vorp, -np.inf)
                    if np.isfinite(vorp).any():
                        idx = int(np.argmax(vorp))
                    else:
                        idx = int(np.flatnonzero(cand_mask)[0])
                    return self._record(idx, is_user=False)
        values = np.where(eligible, self.boards[slot - 1], np.inf)

        rng = np.random.default_rng((seed_from_id(self.draft_id), overall))
        n_elig = int(np.isfinite(values).sum())
        if n_elig > 1 and rng.random() < REACH_PROB:
            top = np.argsort(values)[: min(REACH_TOP_N, n_elig)]
            idx = int(rng.choice(top))
        else:
            idx = int(np.argmin(values))
        return self._record(idx, is_user=False)

    def advance_bots(self) -> list[dict]:
        """Run bot picks until the user is on the clock or the draft ends."""
        events = []
        while True:
            otc = self.on_the_clock()
            if otc is None or otc["is_user"]:
                break
            events.append(self.bot_pick())
        return events

    def undo_to_before_last_user_pick(self) -> int:
        """Drop the user's last pick and everything after it. Returns #dropped."""
        last_user = max(
            (i for i, p in enumerate(self.picks) if p["is_user"]), default=None
        )
        if last_user is None:
            return 0
        dropped = len(self.picks) - last_user
        self.picks = self.picks[:last_user]
        self._survival_cache.clear()
        return dropped

    # ------------------------------------------------------------------
    # Persistence (config + picks; boards regenerate from the seed)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "draft_id": self.draft_id,
            "created_at": self.created_at,
            "config": {
                "teams": self.config.teams, "user_slot": self.config.user_slot,
                "rounds": self.config.rounds, "format": self.config.format,
                "roster": self.config.roster,
            },
            "picks": self.picks,
        }

    def save(self, drafts_dir: Path) -> None:
        drafts_dir.mkdir(parents=True, exist_ok=True)
        (drafts_dir / f"{self.draft_id}.json").write_text(json.dumps(self.snapshot()))

    @classmethod
    def load(cls, drafts_dir: Path, draft_id: str) -> "Draft | None":
        path = drafts_dir / f"{draft_id}.json"
        if not path.exists():
            return None
        snap = json.loads(path.read_text())
        cfg = DraftConfig(**snap["config"])
        d = cls.create(draft_id, cfg)
        d.picks = snap["picks"]
        d.created_at = snap.get("created_at", time.time())
        return d
