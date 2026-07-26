"""
VONA, roster utility, policy, and rollout tests. Like test_draft_engine.py
these run against the real frozen artifacts in webapp/data/ — skipped if
absent.
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

from webapp.api.engine.draft import Draft, DraftConfig            # noqa: E402
from webapp.api.engine.policy import POLICIES                     # noqa: E402
from webapp.api.engine.pool import POS_IDX                        # noqa: E402
from webapp.api.engine.recommend import recommendations           # noqa: E402
from webapp.api.engine.rollout import rollout_recommendations     # noqa: E402
from webapp.api.engine.roster_utility import RosterScorer         # noqa: E402
from webapp.api.engine.survival import M_SIMS, p_survive, survival_alive  # noqa: E402
from webapp.api.engine.vona import expected_next_best, vona_for   # noqa: E402

ROSTER = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "BN": 7, "K": 0, "DST": 0}


def make_draft(draft_id="testdraft", teams=12, user_slot=7, rounds=16):
    cfg = DraftConfig(teams=teams, user_slot=user_slot, rounds=rounds,
                      format="12_ppr", roster=ROSTER)
    return Draft.create(draft_id, cfg)


# ---------------------------------------------------------------------------
# Survival alive matrix / VONA
# ---------------------------------------------------------------------------

def test_survival_alive_consistent_with_p_survive():
    d = make_draft("vonadraft")
    d.advance_bots()
    alive = survival_alive(d)
    assert alive.shape == (M_SIMS, d.pool.n)
    surv = p_survive(d)
    avail = d.available_mask()
    np.testing.assert_allclose(surv[avail], alive.mean(axis=0)[avail])
    assert (surv[~avail] == 0).all()


def test_expected_next_best_between_bounds():
    d = make_draft("vonadraft2")
    d.advance_bots()
    nb = expected_next_best(d)
    avail = d.available_mask()
    pool = d.pool
    for pos, val in nb.items():
        if val is None:
            continue
        mask = avail & (pool.pos_idx == POS_IDX[pos]) & np.isfinite(pool.vorp)
        if not mask.any():
            continue
        best_now = float(pool.vorp[mask].max())
        worst_now = float(pool.vorp[mask].min())
        # Expected best at next pick can't beat the best available now,
        # and can't be below the worst player currently on the board.
        assert worst_now - 1e-9 <= val <= best_now + 1e-9


def test_vona_nonpositive_for_survivable_backups():
    """A player worse than his position's expected next-best has vona <= 0."""
    d = make_draft("vonadraft3")
    d.advance_bots()
    nb = expected_next_best(d)
    pool = d.pool
    avail = d.available_mask()
    checked = 0
    for pos, val in nb.items():
        if val is None:
            continue
        mask = avail & (pool.pos_idx == POS_IDX[pos]) & np.isfinite(pool.vorp)
        for idx in np.flatnonzero(mask):
            v = vona_for(d, int(idx), nb)
            assert v == pytest.approx(float(pool.vorp[idx]) - val)
            checked += 1
    assert checked > 0


def test_recommendations_include_vona_fields():
    d = make_draft("vonadraft4")
    d.advance_bots()
    out = recommendations(d, n=6)
    assert "expected_next_best" in out
    for r in out["recommendations"]:
        assert "vona" in r
        if r["vona"] is not None:
            assert r["urgency"] == pytest.approx(max(r["vona"], 0.0), abs=0.11)


# ---------------------------------------------------------------------------
# Roster utility
# ---------------------------------------------------------------------------

def test_roster_scorer_lineup_and_monotonicity():
    d = make_draft("scorerdraft")
    cfg = d.config
    scorer = RosterScorer(d.pool, cfg)
    pool = d.pool

    proj = np.flatnonzero(np.isfinite(scorer.p50))
    top = proj[np.argsort(-scorer.p50[proj])]
    # A full roster of top players
    roster = list(top[:16])
    s = scorer.score(roster)
    assert s.starter_points > 0
    # Starter count is bounded by slots: QB+RB+WR+TE+FLEX = 8 starters max
    n_starters = cfg.starters("QB") + cfg.starters("RB") + cfg.starters("WR") \
        + cfg.starters("TE") + cfg.roster["FLEX"]
    assert s.starter_points <= float(np.sort(scorer.p50[proj])[::-1][:n_starters].sum()) + 1e-6
    # Adding a player never decreases utility
    extra = int(top[20])
    s2 = scorer.score(roster + [extra])
    assert s2.utility >= s.utility - 1e-9
    # Empty roster scores zero
    z = scorer.score([])
    assert z.utility == 0.0


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(POLICIES))
def test_policy_produces_legal_full_roster(name):
    d = make_draft(f"pol_{name}", rounds=14)
    fn = POLICIES[name]
    d.advance_bots()
    while not d.complete:
        idx = fn(d)
        d.user_pick(d.pool.ids[idx])
        d.advance_bots()
    counts = d.team_counts(d.config.user_slot)
    caps = d.config.position_caps()
    from webapp.api.engine.pool import POS_ORDER
    for pos in POS_ORDER:
        assert counts[POS_IDX[pos]] <= caps[pos], f"{name} broke {pos} cap"
    assert counts.sum() == d.config.rounds
    # Starter coverage is NOT asserted: leaving a starter slot empty (the
    # v1 heuristic can finish TE-less) is legal — the utility backtest is
    # what punishes it. Only require the roster to be scoreable.
    scorer = RosterScorer(d.pool, d.config)
    roster = [d.pool.index[p["player_id"]] for p in d.picks if p["is_user"]]
    assert scorer.score(roster).utility > 0


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

def test_rollout_schema_and_determinism():
    d = make_draft("rolldraft")
    d.advance_bots()
    out = rollout_recommendations(d, n_candidates=4, n_sims=8)
    cands = out["candidates"]
    assert 1 <= len(cands) <= 4
    for c in cands:
        assert {"player_id", "mean_utility", "se_utility", "delta_vs_best"} <= set(c)
    # Ranked by mean utility, best first; deltas are vs the best
    utils = [c["mean_utility"] for c in cands]
    assert utils == sorted(utils, reverse=True)
    assert cands[0]["delta_vs_best"] == pytest.approx(0.0, abs=1e-9)
    # Deterministic for a given draft state
    d2 = make_draft("rolldraft")
    d2.advance_bots()
    out2 = rollout_recommendations(d2, n_candidates=4, n_sims=8)
    assert [c["player_id"] for c in out2["candidates"]] == [c["player_id"] for c in cands]
    assert [c["mean_utility"] for c in out2["candidates"]] == utils


def test_rollout_requires_user_on_clock():
    d = make_draft("rolldraft2", user_slot=12)
    # Before advance_bots, pick 1 belongs to slot 1 (a bot)
    with pytest.raises(ValueError):
        rollout_recommendations(d)
