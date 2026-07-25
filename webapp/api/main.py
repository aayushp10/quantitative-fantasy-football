"""
FastAPI app serving the frozen artifacts plus the mock-draft engine.

    uvicorn webapp.api.main:app --port 8000

The app never imports the model — it only reads webapp/data/.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import store
from .drafts import router as drafts_router

app = FastAPI(title="ff-factor-model web API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(drafts_router)


@app.exception_handler(store.ArtifactsMissing)
async def artifacts_missing(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/api/meta")
def meta():
    return store.meta()


@app.get("/api/trust")
def trust():
    return store.trust()


@app.get("/api/players")
def players(format: str = "12_ppr", position: str | None = None, search: str | None = None):
    if format not in store.format_keys():
        raise HTTPException(422, f"unknown format {format!r}; see /api/meta")
    out = [store.flatten_for_format(p, format) for p in store.players()]
    if position:
        out = [p for p in out if p["position"] == position.upper()]
    if search:
        q = search.lower()
        out = [p for p in out if q in (p["name"] or "").lower()]
    out.sort(key=lambda p: p["overall_rank"] if p["overall_rank"] is not None else 10**6)
    return out


@app.get("/api/players/{player_id}")
def player_detail(player_id: str, format: str = "12_ppr"):
    p = store.players_by_id().get(player_id)
    if p is None:
        raise HTTPException(404, f"player {player_id} not found")
    out = store.flatten_for_format(p, format)
    out["vorp_all_formats"] = p["vorp"]
    return out


@app.get("/api/adp_board")
def adp_board():
    return store.adp_board()
