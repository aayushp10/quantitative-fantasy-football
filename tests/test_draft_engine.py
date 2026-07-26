"""
Mock-draft engine tests. These run against the real frozen artifacts in
webapp/data/ (built by scripts/build_web_data.py) — skipped if absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")

from webapp.api import store  # noqa: E402

if not (store.DATA_DIR / "players.json").exists():
    pytest.skip("webapp/data artifacts not built (run make web-data)", allow_module_level=True)

from webapp.api.engine.draft import (  # noqa: E402
    Draft,
    DraftConfig,
    slot_for_pick,
)
from webapp.api.engine.pool import POS_IDX, POS_ORDER  # noqa: E402
from webapp.api.engine.recommend import recommendations  # noqa: E402
from webapp.api.engine.survival import p_survive  # noqa: E402

ROSTER = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "BN": 7, "K": 0, "DST": 0}
ROSTER_KDST = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "BN": 6, "K": 1, "DST": 1}


def make_draft(draft_id="testdraft", teams=12, user_slot=7, rounds=16, roster=ROSTER):
    cfg = DraftConfig(teams=teams, user_slot=user_slot, rounds=rounds,
                      format="12_ppr", roster=roster)
    return Draft.create(draft_id, cfg)


def autopick_user(d: Draft) -> None:
    """User strategy for tests: best available VORP."""
    vorp = np.where(d.available_mask(), d.pool.vorp, -np.inf)
    idx = int(np.argmax(np.where(np.isfinite(vorp), vorp, -np.inf)))
    d.user_pick(d.pool.ids[idx])


def run_full_draft(d: Draft) -> Draft:
    d.advance_bots()
    while not d.complete:
        autopick_user(d)
        d.advance_bots()
    return d


# ---------------------------------------------------------------------------
# Snake order
# ---------------------------------------------------------------------------

def test_snake_order():
    assert slot_for_pick(1, 12) == (1, 1)
    assert slot_for_pick(12, 12) == (1, 12)
    assert slot_for_pick(13, 12) == (2, 12)
    assert slot_for_pick(24, 12) == (2, 1)
    assert slot_for_pick(25, 12) == (3, 1)
    # Every team gets exactly one pick per round
    for teams in (10, 12, 14):
        for rnd_start in range(0, 3):
            slots = [slot_for_pick(rnd_start * teams + i, teams)[1]
                     for i in range(1, teams + 1)]
            assert sorted(slots) == list(range(1, teams + 1))


def test_user_on_the_clock_at_their_slot():
    d = make_draft()
    d.advance_bots()
    otc = d.on_the_clock()
    assert otc["is_user"] and otc["slot"] == 7 and otc["overall"] == 7
    assert len(d.picks) == 6  # six bots picked first


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def _bot_position_counts(d: Draft) -> dict[int, dict[str, int]]:
    out = {}
    for slot in range(1, d.config.teams + 1):
        if slot == d.config.user_slot:
            continue
        counts = d.team_counts(slot)
        out[slot] = {p: int(counts[POS_IDX[p]]) for p in POS_ORDER}
    return out


def test_full_draft_completes_and_bots_respect_caps():
    d = run_full_draft(make_draft("capsdraft"))
    assert d.complete and len(d.picks) == 12 * 16
    caps = d.config.position_caps()
    for slot, counts in _bot_position_counts(d).items():
        for pos in ("QB", "RB", "WR", "TE"):
            assert counts[pos] <= caps[pos], f"slot {slot} exceeded {pos} cap"
        assert counts["K"] == 0 and counts["DST"] == 0  # no slots in this roster


def test_no_early_second_qb_or_te():
    d = run_full_draft(make_draft("earlyqb"))
    seen: dict[tuple[int, str], int] = {}
    for p in d.picks:
        if p["is_user"]:
            continue
        key = (p["slot"], p["position"])
        seen[key] = seen.get(key, 0) + 1
        if p["position"] in ("QB", "TE") and p["round"] < 10:
            assert seen[key] <= 1, f"bot {p['slot']} took 2nd {p['position']} in round {p['round']}"


def test_kdst_only_in_final_rounds_and_filled():
    d = run_full_draft(make_draft("kdst", roster=ROSTER_KDST))
    n_kdst = 2
    for p in d.picks:
        if p["is_user"]:
            continue
        if p["position"] in ("K", "DST"):
            assert p["round"] > d.config.rounds - n_kdst, \
                f"bot {p['slot']} drafted {p['position']} in round {p['round']}"
    for slot, counts in _bot_position_counts(d).items():
        assert counts["K"] == 1, f"bot {slot} K={counts['K']}"
        assert counts["DST"] == 1, f"bot {slot} DST={counts['DST']}"


def test_bots_adp_plausible_early():
    d = make_draft("plausible")
    d.advance_bots()
    for p in d.picks:  # first six picks
        i = d.pool.index[p["player_id"]]
        assert d.pool.adp[i] <= 25, f"pick {p['overall']} was ADP {d.pool.adp[i]}"


# ---------------------------------------------------------------------------
# Determinism / undo / persistence
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    a = run_full_draft(make_draft("seeddraft"))
    b = run_full_draft(make_draft("seeddraft"))
    assert [p["player_id"] for p in a.picks] == [p["player_id"] for p in b.picks]


def test_different_seed_differs():
    a = run_full_draft(make_draft("seed-a"))
    b = run_full_draft(make_draft("seed-b"))
    assert [p["player_id"] for p in a.picks] != [p["player_id"] for p in b.picks]


def test_undo_rewinds_to_before_last_user_pick():
    d = make_draft("undodraft")
    d.advance_bots()
    autopick_user(d)
    d.advance_bots()
    picks_before_second = [p["player_id"] for p in d.picks]
    autopick_user(d)
    d.advance_bots()
    dropped = d.undo_to_before_last_user_pick()
    assert dropped >= 1
    assert [p["player_id"] for p in d.picks] == picks_before_second
    assert d.on_the_clock()["is_user"]


def test_snapshot_roundtrip(tmp_path):
    d = make_draft("snapdraft")
    d.advance_bots()
    autopick_user(d)
    d.advance_bots()
    d.save(tmp_path)
    d2 = Draft.load(tmp_path, "snapdraft")
    assert [p["player_id"] for p in d2.picks] == [p["player_id"] for p in d.picks]
    # Replays identically from the restored state
    a, b = run_full_draft(d), run_full_draft(d2)
    assert [p["player_id"] for p in a.picks] == [p["player_id"] for p in b.picks]


# ---------------------------------------------------------------------------
# Survival model
# ---------------------------------------------------------------------------

def test_p_survive_decreasing_in_adp():
    d = make_draft("survdraft")
    d.advance_bots()
    surv = p_survive(d)
    pool = d.pool
    avail = d.available_mask()
    for pos in ("RB", "WR"):
        idx = np.flatnonzero(avail & (pool.pos_idx == POS_IDX[pos]) & np.isfinite(pool.adp))
        idx = idx[np.argsort(pool.adp[idx])][:25]
        s = surv[idx]
        # Earlier-ADP players must be less likely to survive. MC noise allows
        # tiny local wiggles; enforce no violation worse than 5 points.
        diffs = np.diff(s)
        assert (diffs >= -0.05).all(), f"{pos}: survival not increasing with ADP: {s}"
        assert s[0] < s[-1], f"{pos}: earliest ADP should be least likely to survive"


def test_p_survive_bounds_and_drafted_zero():
    d = make_draft("survbounds")
    d.advance_bots()
    surv = p_survive(d)
    assert ((surv >= 0) & (surv <= 1)).all()
    for p in d.picks:
        assert surv[d.pool.index[p["player_id"]]] == 0.0


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def test_recommendations_schema_and_score_composition():
    d = make_draft("recdraft")
    d.advance_bots()
    out = recommendations(d, n=6)
    assert len(out["recommendations"]) == 6
    assert set(out["need_weights"]) == {"QB", "RB", "WR", "TE"}
    required = {"player_id", "name", "position", "vorp", "pos_rank", "tier",
                "adp", "adp_edge", "p_survive", "need_weight", "need_multiplier",
                "tier_drop", "urgency", "rec_score",
                "season_p10", "season_p25", "season_p50", "season_p75", "season_p90"}
    from webapp.api.engine.recommend import ALPHA_REC_WEIGHT
    for r in out["recommendations"]:
        assert required <= set(r), f"missing {required - set(r)}"
        # Components sum to rec_score (rounding tolerance)
        expected = (r["vorp"] * r["need_multiplier"] + r["urgency"]
                    + ALPHA_REC_WEIGHT * (r.get("alpha_points") or 0.0))
        assert abs(expected - r["rec_score"]) < 0.15, r
        assert 0.85 <= r["need_multiplier"] <= 1.15
        assert 0.0 <= r["p_survive"] <= 1.0
    scores = [r["rec_score"] for r in out["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_recommendations_exclude_drafted_players():
    d = make_draft("recdraft2")
    d.advance_bots()
    drafted = d.picked_ids()
    out = recommendations(d, n=40)
    assert not ({r["player_id"] for r in out["recommendations"]} & drafted)
