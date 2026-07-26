"""FastAPI endpoint tests against the frozen artifacts (skipped if absent)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")

from webapp.api import store  # noqa: E402

if not (store.DATA_DIR / "players.json").exists():
    pytest.skip("webapp/data artifacts not built (run make web-data)", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from webapp.api.main import app  # noqa: E402

client = TestClient(app)


def test_meta():
    r = client.get("/api/meta")
    assert r.status_code == 200
    m = r.json()
    assert m["projection_season"] == 2026
    assert {f["key"] for f in m["formats"]} >= {"10_ppr", "12_ppr", "14_ppr"}


def test_players_list_and_filters():
    r = client.get("/api/players", params={"format": "12_ppr"})
    assert r.status_code == 200
    players = r.json()
    assert len(players) > 300
    assert players[0]["overall_rank"] == 1
    assert "vorp" in players[0] and "season_p50" in players[0]

    r = client.get("/api/players", params={"format": "12_ppr", "position": "TE"})
    assert {p["position"] for p in r.json()} == {"TE"}

    r = client.get("/api/players", params={"format": "12_ppr", "search": "chase"})
    assert any("Chase" in p["name"] for p in r.json())

    assert client.get("/api/players", params={"format": "nope"}).status_code == 422


def test_player_detail():
    pid = client.get("/api/players").json()[0]["player_id"]
    r = client.get(f"/api/players/{pid}", params={"format": "14_ppr"})
    assert r.status_code == 200
    assert r.json()["player_id"] == pid
    assert "vorp_all_formats" in r.json()
    assert client.get("/api/players/nope").status_code == 404


def test_player_history():
    players = client.get("/api/players").json()
    vet = next(p for p in players if not p["rookie"])
    h = client.get(f"/api/players/{vet['player_id']}/history").json()
    assert h["seasons"], "veteran should have season history"
    row = h["seasons"][-1]
    assert {"season", "fpts_pg", "games"} <= set(row)
    assert all(s0["season"] < s1["season"] for s0, s1 in zip(h["seasons"], h["seasons"][1:]))
    for wk in h["weekly"].values():
        assert all({"week", "pts"} <= set(w) for w in wk)
    assert client.get("/api/players/nope/history").status_code == 404


def test_trust():
    t = client.get("/api/trust").json()
    assert t["backtest"] and t["vs_market"] and t["coverage"] and t["top_factors"]


def test_draft_flow_and_latency():
    r = client.post("/api/drafts", json={
        "teams": 12, "user_slot": 7, "rounds": 16, "format": "12_ppr",
        "roster": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "BN": 7, "K": 0, "DST": 0},
    })
    assert r.status_code == 200
    draft_id = r.json()["draft_id"]
    state = r.json()["state"]
    assert state["on_the_clock"]["is_user"] and state["on_the_clock"]["overall"] == 7

    t0 = time.monotonic()
    r = client.get(f"/api/drafts/{draft_id}/recommendations", params={"n": 6})
    elapsed = time.monotonic() - t0
    assert r.status_code == 200
    assert elapsed < 1.0, f"recommendations took {elapsed:.2f}s"
    recs = r.json()["recommendations"]
    assert len(recs) == 6

    # Pick the top recommendation; bots auto-advance to our next turn
    r = client.post(f"/api/drafts/{draft_id}/pick", json={"player_id": recs[0]["player_id"]})
    assert r.status_code == 200
    body = r.json()
    assert body["events"]["user_pick"]["player_id"] == recs[0]["player_id"]
    assert len(body["events"]["bot_picks"]) == 10  # overalls 8–17; user again at 18
    assert body["state"]["on_the_clock"]["is_user"]

    # Picking out of turn / picking a drafted player is a 409
    r = client.post(f"/api/drafts/{draft_id}/pick", json={"player_id": recs[0]["player_id"]})
    assert r.status_code == 409

    # Undo returns us to before our last pick
    r = client.post(f"/api/drafts/{draft_id}/undo")
    assert r.status_code == 200
    assert r.json()["state"]["on_the_clock"]["overall"] == 7

    assert client.get("/api/drafts/doesnotexist").status_code == 404


def test_paced_draft_step_flow():
    """auto_advance=False + /step: the client paces bot picks one at a time."""
    r = client.post("/api/drafts", json={
        "teams": 12, "user_slot": 3, "rounds": 15, "format": "12_ppr",
        "roster": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "BN": 7, "K": 0, "DST": 0},
        "auto_advance": False,
    })
    assert r.status_code == 200
    body = r.json()
    did = body["draft_id"]
    try:
        assert body["events"] == []
        assert body["state"]["on_the_clock"] == {
            "overall": 1, "round": 1, "slot": 1, "is_user": False}

        # Two bot steps land the user on the clock (slot 3)
        for expected in (1, 2):
            s = client.post(f"/api/drafts/{did}/step").json()
            assert s["event"]["overall"] == expected
        s = client.post(f"/api/drafts/{did}/step").json()
        assert s["event"] is None                       # user on clock -> no-op
        assert s["on_the_clock"]["is_user"]

        # advance=False leaves the next bot un-stepped
        top = client.get(f"/api/drafts/{did}/recommendations?n=1").json()["recommendations"][0]
        p = client.post(f"/api/drafts/{did}/pick",
                        json={"player_id": top["player_id"], "advance": False}).json()
        assert p["events"]["bot_picks"] == []
        assert p["state"]["on_the_clock"] == {
            "overall": 4, "round": 1, "slot": 4, "is_user": False}
        s = client.post(f"/api/drafts/{did}/step").json()
        assert s["event"]["overall"] == 4 and not s["event"]["is_user"]
    finally:
        (store.DRAFTS_DIR / f"{did}.json").unlink(missing_ok=True)
