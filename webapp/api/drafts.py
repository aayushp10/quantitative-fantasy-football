"""
Mock-draft endpoints. State lives in memory keyed by draft_id, with JSON
snapshots in webapp/data/drafts/ so a server restart resumes an active
draft (boards regenerate deterministically from the draft-id seed).
Single-user local tool — no auth.
"""
from __future__ import annotations

import secrets

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import store
from .engine.draft import Draft, DraftConfig
from .engine.recommend import recommendations, tier_cliff_alerts, tier_structure
from .engine.rollout import rollout_recommendations
from .engine.survival import p_survive

router = APIRouter(prefix="/api/drafts", tags=["drafts"])

_drafts: dict[str, Draft] = {}

DEFAULT_ROSTER = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "BN": 7, "K": 0, "DST": 0}


class CreateDraft(BaseModel):
    teams: int = Field(12, ge=4, le=16)
    user_slot: int = Field(7, ge=1)
    rounds: int = Field(16, ge=4, le=25)
    format: str = "12_ppr"
    roster: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_ROSTER))
    # False -> bots don't run automatically; the client paces them via /step
    auto_advance: bool = True


class PickBody(BaseModel):
    player_id: str
    advance: bool = True


def _get(draft_id: str) -> Draft:
    d = _drafts.get(draft_id)
    if d is None:
        d = Draft.load(store.DRAFTS_DIR, draft_id)
        if d is None:
            raise HTTPException(404, f"draft {draft_id} not found")
        _drafts[draft_id] = d
    return d


def _state(d: Draft) -> dict:
    surv = p_survive(d) if not d.complete else None
    avail_idx = np.flatnonzero(d.available_mask())
    pool = d.pool
    return {
        "draft_id": d.draft_id,
        "config": d.snapshot()["config"],
        "picks": d.picks,
        "rosters": {str(s): ids for s, ids in d.rosters().items()},
        "on_the_clock": d.on_the_clock(),
        "complete": d.complete,
        "available": [
            {
                "player_id": pool.ids[i],
                "p_survive": round(float(surv[i]), 3) if surv is not None else None,
            }
            for i in avail_idx
        ],
    }


@router.post("")
def create_draft(body: CreateDraft):
    if body.format not in store.format_keys():
        raise HTTPException(422, f"unknown format {body.format!r}; see /api/meta")
    if body.user_slot > body.teams:
        raise HTTPException(422, "user_slot must be <= teams")
    cfg = DraftConfig(
        teams=body.teams, user_slot=body.user_slot, rounds=body.rounds,
        format=body.format, roster=body.roster,
    )
    draft_id = secrets.token_hex(4)
    d = Draft.create(draft_id, cfg)
    events = d.advance_bots() if body.auto_advance else []
    d.save(store.DRAFTS_DIR)
    _drafts[draft_id] = d
    return {"draft_id": draft_id, "events": events, "state": _state(d)}


@router.get("/{draft_id}")
def get_draft(draft_id: str):
    return _state(_get(draft_id))


@router.post("/{draft_id}/pick")
def make_pick(draft_id: str, body: PickBody):
    d = _get(draft_id)
    try:
        user_pick = d.user_pick(body.player_id)
    except ValueError as e:
        raise HTTPException(409, str(e))
    bot_events = d.advance_bots() if body.advance else []
    d.save(store.DRAFTS_DIR)
    alerts = []
    if not d.complete:
        alerts = tier_cliff_alerts(tier_structure(d, d.available_mask()))
    return {
        "state": _state(d),
        "events": {
            "user_pick": user_pick,
            "bot_picks": bot_events,
            "tier_cliff_alerts": alerts,
        },
    }


@router.post("/{draft_id}/undo")
def undo(draft_id: str):
    d = _get(draft_id)
    dropped = d.undo_to_before_last_user_pick()
    if dropped == 0:
        raise HTTPException(409, "nothing to undo")
    d.save(store.DRAFTS_DIR)
    return {"dropped": dropped, "state": _state(d)}


@router.get("/{draft_id}/recommendations")
def get_recommendations(draft_id: str, n: int = 6):
    d = _get(draft_id)
    if d.complete:
        return {"recommendations": [], "tier_cliff_alerts": [], "complete": True}
    otc = d.on_the_clock()
    if not otc["is_user"]:
        raise HTTPException(409, "recommendations are only available on your pick")
    return recommendations(d, n=n)


@router.post("/{draft_id}/step")
def step(draft_id: str):
    """Advance exactly one bot pick — the client paces the draft with this.

    No-op (event: null) when the user is on the clock or the draft is over,
    so the client can call it blindly without racing the state machine.
    """
    d = _get(draft_id)
    otc = d.on_the_clock()
    if otc is None or otc["is_user"]:
        return {"event": None, "on_the_clock": otc, "complete": d.complete}
    ev = d.bot_pick()
    d.save(store.DRAFTS_DIR)
    return {"event": ev, "on_the_clock": d.on_the_clock(), "complete": d.complete}


@router.get("/{draft_id}/rollout")
def get_rollout(draft_id: str, n: int = 8, sims: int = 24):
    """Rank candidate picks by simulated completed-roster utility."""
    d = _get(draft_id)
    if d.complete:
        return {"candidates": [], "complete": True}
    otc = d.on_the_clock()
    if not otc["is_user"]:
        raise HTTPException(409, "rollouts are only available on your pick")
    return rollout_recommendations(d, n_candidates=min(n, 12), n_sims=min(sims, 100))
