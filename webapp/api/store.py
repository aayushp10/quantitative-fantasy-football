"""
Read-only access to the frozen JSON artifacts in webapp/data/.

The API never imports the model or touches parquet caches — this module is
the only data access layer. Artifacts are loaded once at startup.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DRAFTS_DIR = DATA_DIR / "drafts"


class ArtifactsMissing(RuntimeError):
    pass


def _load(name: str):
    path = DATA_DIR / name
    if not path.exists():
        raise ArtifactsMissing(
            f"{path} not found — run `make web-data` to build the artifacts."
        )
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def meta() -> dict:
    return _load("meta.json")


@lru_cache(maxsize=1)
def trust() -> dict:
    return _load("trust.json")


@lru_cache(maxsize=1)
def players() -> list[dict]:
    return _load("players.json")


@lru_cache(maxsize=1)
def adp_board() -> list[dict]:
    return _load("adp_board.json")


@lru_cache(maxsize=1)
def history() -> dict:
    return _load("history.json")


@lru_cache(maxsize=1)
def players_by_id() -> dict[str, dict]:
    return {p["player_id"]: p for p in players()}


def format_keys() -> list[str]:
    return [f["key"] for f in meta()["formats"]]


def flatten_for_format(p: dict, format_key: str) -> dict:
    """Player record with the chosen format's VORP fields lifted to top level."""
    v = p["vorp"].get(format_key) or {}
    out = {k: p[k] for k in p if k != "vorp"}
    out["vorp"] = v.get("vorp")
    out["overall_rank"] = v.get("overall_rank")
    out["pos_rank"] = v.get("pos_rank")
    out["tier"] = v.get("tier")
    return out
