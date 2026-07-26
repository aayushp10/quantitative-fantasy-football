"""
Historical ADP (Average Draft Position) from FantasyFootballCalculator.

ADP is the MARKET PRICE of a player. It is both:
  1. the benchmark every projection must beat (ADP alone rank-correlates
     ~0.55-0.65 with end-of-season finish — a model below that has negative
     alpha and you'd be better off indexing the market), and
  2. an information source (the market aggregates camp reports, injuries,
     coaching intel the feature matrix cannot see).

Source: free, keyless JSON API
    https://fantasyfootballcalculator.com/api/v1/adp/{fmt}?teams={n}&year={year}
with archives back to ~2010. Cached per (format, teams, year) to
data/cache/adp_{fmt}_{teams}_{year}.parquet.

Season semantics: ADP for draft-year Y prices season Y. A YoY row with
features from season N targets season N+1, so it joins ADP year N+1 —
attach_adp(..., season_offset=1). Projections for season Y join offset 0.
"""
from __future__ import annotations

import json
import re
import urllib.request
import warnings

import numpy as np
import pandas as pd

from config import ADP_FORMAT, ADP_SOURCE_URL, ADP_TEAMS, CACHE_DIR

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# FFC position labels map directly; defense/kicker rows are dropped
_KEEP_POSITIONS = {"QB", "RB", "WR", "TE"}


def normalize_name(name: str) -> str:
    """
    Normalize a player name for cross-source matching:
    lowercase, strip punctuation and generational suffixes.
    'Odell Beckham Jr.' -> 'odell beckham', 'D.J. Moore' -> 'dj moore'
    """
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = re.sub(r"[.'’]", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    tokens = [t for t in s.split() if t not in _SUFFIXES]
    return " ".join(tokens)


def _adp_cache_path(fmt: str, teams: int, year: int):
    return CACHE_DIR / f"adp_{fmt}_{teams}_{year}.parquet"


def _payload_to_frame(payload: dict, year: int, fmt: str) -> pd.DataFrame | None:
    players = payload.get("players", [])
    if not players:
        return None
    df = pd.DataFrame(players)
    keep = [c for c in ["name", "position", "team", "adp", "times_drafted",
                        "high", "low", "stdev", "bye"] if c in df.columns]
    df = df[keep].rename(columns={"name": "player_name"})
    df["season"] = year          # the season this draft year prices
    df["adp_format"] = fmt
    return df


def _get_json(url: str, timeout: int = 30) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ff-factor-model/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        warnings.warn(f"ADP fetch failed ({url}): {e}")
        return None


def _fetch_adp_year(year: int, fmt: str, teams: int, timeout: int = 30) -> pd.DataFrame | None:
    """
    Fetch one draft year from the FFC API. Returns None on failure.

    The year-keyed archive lags for the IN-PROGRESS draft season, so when it
    returns nothing we fall back to the live (no-year) endpoint and accept it
    only if its meta end_date confirms it prices the requested year.
    """
    url = ADP_SOURCE_URL.format(fmt=fmt, teams=teams, year=year)
    payload = _get_json(url, timeout)
    if payload is not None:
        df = _payload_to_frame(payload, year, fmt)
        if df is not None:
            return df

    # Live fallback for the current draft season
    live_url = url.split("&year=")[0]
    payload = _get_json(live_url, timeout)
    if payload is None:
        return None
    end_date = str(payload.get("meta", {}).get("end_date", ""))
    if not end_date.startswith(str(year)):
        return None
    warnings.warn(
        f"ADP {year}: archive empty; using LIVE drafts through {end_date} "
        "(snapshot cached — delete the parquet to refresh)."
    )
    return _payload_to_frame(payload, year, fmt)


def load_adp(
    seasons: list[int],
    fmt: str = ADP_FORMAT,
    teams: int = ADP_TEAMS,
) -> pd.DataFrame:
    """
    Load ADP for the given draft years (cache-through).

    Returns one row per (season, player) with columns:
    season, player_name, name_norm, position, team, adp, adp_pos_rank.
    Years that cannot be fetched are skipped with a warning.
    """
    frames = []
    for year in seasons:
        path = _adp_cache_path(fmt, teams, year)
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue
        df = _fetch_adp_year(year, fmt, teams)
        if df is not None:
            df.to_parquet(path, index=False)
            frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["season", "player_name", "name_norm", "position",
                     "team", "adp", "adp_pos_rank"]
        )

    adp = pd.concat(frames, ignore_index=True)
    adp = adp[adp["position"].isin(_KEEP_POSITIONS)].copy()
    adp["adp"] = pd.to_numeric(adp["adp"], errors="coerce")
    adp = adp.dropna(subset=["adp"])
    adp["name_norm"] = adp["player_name"].map(normalize_name)

    # Positional ADP rank within each draft year (1 = drafted earliest)
    adp["adp_pos_rank"] = (
        adp.groupby(["season", "position"], observed=True)["adp"]
        .rank(method="min")
        .astype(int)
    )

    # One row per (season, name, position): keep the earliest-drafted duplicate
    adp = (
        adp.sort_values("adp")
        .drop_duplicates(subset=["season", "name_norm", "position"])
        .reset_index(drop=True)
    )
    return adp


def attach_adp(
    df: pd.DataFrame,
    adp_df: pd.DataFrame,
    season_offset: int = 0,
    name_col: str = "player_name",
    position_col: str = "position",
    season_col: str = "season",
) -> pd.DataFrame:
    """
    Join ADP onto any frame by (normalized name, position, season + offset).

    season_offset=1 for YoY pairs (features season N -> drafted before season
    N+1); 0 for projection frames already keyed by the draft season.

    Adds: adp, adp_pos_rank, adp_matched (bool). Match rate is reported —
    drafted players missing from the frame or vice versa are expected (the
    ADP universe is ~200 players deep).
    """
    out = df.copy()
    out["_name_norm"] = out[name_col].map(normalize_name)
    out["_adp_season"] = out[season_col] + season_offset

    right = adp_df.rename(columns={
        "season": "_adp_season", "position": "_adp_position",
    })[["_adp_season", "name_norm", "_adp_position", "adp", "adp_pos_rank"]]

    out = out.merge(
        right,
        left_on=["_adp_season", "_name_norm", position_col],
        right_on=["_adp_season", "name_norm", "_adp_position"],
        how="left",
    ).drop(columns=["name_norm", "_adp_position", "_name_norm"], errors="ignore")

    out["adp_matched"] = out["adp"].notna()
    n = len(out)
    if n:
        print(f"  ADP match: {out['adp_matched'].sum()}/{n} rows "
              f"({100 * out['adp_matched'].mean():.0f}%)")
    return out.drop(columns=["_adp_season"], errors="ignore")
