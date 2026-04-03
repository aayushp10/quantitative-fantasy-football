"""
Cache-through data loader wrapping nfl_data_py.

Every public function:
1. Checks for a local parquet file per season.
2. Downloads only missing seasons from nfl_data_py.
3. Caches each season to its own parquet file.
4. Applies nfl.clean_nfl_data() to standardize team abbreviations.

Important nfl_data_py notes:
- Always pass downcast=False to import_pbp_data() — float32 loses EPA/CPOE precision.
- nfl_data_py was archived Sep 2025; load_ngs() is wrapped defensively.
- Call clean_nfl_data() after every load to normalize OAK→LV, SD→LAC, etc.
- Python 3.11 or 3.12 required (3.13 has install failures).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

try:
    import nfl_data_py as nfl
except ImportError as e:
    raise ImportError(
        "nfl_data_py is required. Install with: pip install nfl_data_py\n"
        "Note: use Python 3.11 or 3.12 — 3.13 has known install failures."
    ) from e

from config import CACHE_DIR, PROJECTION_SEASON


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(name: str, season: int) -> Path:
    return CACHE_DIR / f"{name}_{season}.parquet"


def _load_cached(name: str, seasons: list[int]) -> tuple[list[pd.DataFrame], list[int]]:
    """Return (cached_frames, missing_seasons)."""
    cached, missing = [], []
    for s in seasons:
        p = _cache_path(name, s)
        if p.exists():
            cached.append(pd.read_parquet(p))
        else:
            missing.append(s)
    return cached, missing


def _save_by_season(df: pd.DataFrame, name: str, season_col: str = "season") -> None:
    """Persist each season's slice to its own parquet file."""
    for season, group in df.groupby(season_col):
        group.to_parquet(_cache_path(name, int(season)), index=False)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize team abbreviations via nfl_data_py."""
    try:
        return nfl.clean_nfl_data(df)
    except Exception:
        return df


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_pbp(seasons: list[int]) -> pd.DataFrame:
    """
    Load play-by-play data (~339 columns, ~50k rows/season).

    Always uses downcast=False to preserve float64 precision for EPA/CPOE.
    Filtered to play_type in ['pass', 'run'] happens in cleaning.py, not here,
    so the raw DataFrame is returned intact.
    """
    frames, missing = _load_cached("pbp", seasons)

    if missing:
        print(f"Downloading PBP for seasons: {missing}")
        df = nfl.import_pbp_data(missing, downcast=False)
        df = _clean(df)
        _save_by_season(df, "pbp")
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    return _clean(result)


def load_weekly(seasons: list[int]) -> pd.DataFrame:
    """
    Load pre-aggregated weekly player stats.

    The 'fantasy_points_ppr' column in this dataset is used as the model
    target variable — it handles edge cases (laterals, 2pt conversions)
    more reliably than PBP recomputation.
    """
    frames, missing = _load_cached("weekly", seasons)

    if missing:
        print(f"Downloading weekly stats for seasons: {missing}")
        df = nfl.import_weekly_data(missing)
        df = _clean(df)
        _save_by_season(df, "weekly")
        frames.append(df)

    return _clean(pd.concat(frames, ignore_index=True))


def load_seasonal(seasons: list[int]) -> pd.DataFrame:
    """
    Load season-level aggregates.

    Note: import_seasonal_data() includes pre-computed market share metrics
    (tgt_sh, ay_sh, wopr). Use these for validation only — factor engineering
    computes its own values from PBP for consistency.
    """
    frames, missing = _load_cached("seasonal", seasons)

    if missing:
        print(f"Downloading seasonal stats for seasons: {missing}")
        df = nfl.import_seasonal_data(missing)
        df = _clean(df)
        _save_by_season(df, "seasonal")
        frames.append(df)

    return _clean(pd.concat(frames, ignore_index=True))


def load_rosters(seasons: list[int]) -> pd.DataFrame:
    """
    Load player bios: position, team, age, draft capital.

    nfl_data_py renames gsis_id → player_id internally in seasonal rosters.
    The player_id column here matches receiver_player_id / rusher_player_id /
    passer_player_id in PBP.
    """
    frames, missing = _load_cached("rosters", seasons)

    if missing:
        print(f"Downloading rosters for seasons: {missing}")
        df = nfl.import_seasonal_rosters(missing)
        df = _clean(df)
        _save_by_season(df, "rosters")
        frames.append(df)

    result = _clean(pd.concat(frames, ignore_index=True))

    # Standardize the player ID column name
    if "gsis_id" in result.columns and "player_id" not in result.columns:
        result = result.rename(columns={"gsis_id": "player_id"})

    return result


def load_snap_counts(seasons: list[int]) -> pd.DataFrame:
    """
    Load snap participation data (available from 2012 onward).

    nfl_data_py returns pfr_player_id (Pro Football Reference IDs) here,
    not gsis_id. We bridge to gsis_id using the roster data, which contains
    both pfr_id and player_id (gsis_id) columns.
    """
    frames, missing = _load_cached("snaps", seasons)

    if missing:
        print(f"Downloading snap counts for seasons: {missing}")
        df = nfl.import_snap_counts(missing)
        df = _clean(df)

        # Bridge pfr_player_id → gsis_id via roster data
        pfr_col = next((c for c in ["pfr_player_id", "pfr_id"] if c in df.columns), None)
        if pfr_col:
            try:
                rosters = nfl.import_seasonal_rosters(missing)
                rosters = _clean(rosters)
                # Roster has player_id (gsis_id) and pfr_id
                roster_pfr_col = next(
                    (c for c in ["pfr_id", "pfr_player_id"] if c in rosters.columns), None
                )
                gsis_col = next(
                    (c for c in ["player_id", "gsis_id"] if c in rosters.columns), None
                )
                if roster_pfr_col and gsis_col:
                    id_map = (
                        rosters[[gsis_col, roster_pfr_col]]
                        .dropna(subset=[roster_pfr_col, gsis_col])
                        .drop_duplicates(subset=[roster_pfr_col])
                        .rename(columns={gsis_col: "player_id", roster_pfr_col: pfr_col})
                    )
                    df = df.merge(id_map, on=pfr_col, how="left")
            except Exception as e:
                warnings.warn(f"Could not bridge snap count IDs to gsis_id: {e}")

        _save_by_season(df, "snaps")
        frames.append(df)

    return _clean(pd.concat(frames, ignore_index=True))


def load_ngs(stat_type: str, seasons: list[int]) -> pd.DataFrame | None:
    """
    Load Next Gen Stats for the given stat_type ('passing', 'rushing', 'receiving').

    Returns None if unavailable — NGS coverage starts ~2022 for some stat types.
    Wrapped in try/except because nfl_data_py was archived and NGS endpoints
    may return errors for older seasons.
    """
    cache_name = f"ngs_{stat_type}"
    frames, missing = _load_cached(cache_name, seasons)

    if missing:
        try:
            print(f"Downloading NGS {stat_type} for seasons: {missing}")
            df = nfl.import_ngs_data(stat_type, missing)
            df = _clean(df)
            _save_by_season(df, cache_name)
            frames.append(df)
        except Exception as e:
            warnings.warn(
                f"NGS {stat_type} data unavailable for seasons {missing}: {e}. "
                "Returning cached seasons only (may be empty).",
                stacklevel=2,
            )

    if not frames:
        return None

    return _clean(pd.concat(frames, ignore_index=True))


def load_injuries(seasons: list[int]) -> pd.DataFrame | None:
    """
    Load injury report data if available.

    Note: as of 2025 the archived nfl_data_py may return 404 for
    PROJECTION_SEASON injury data. Returns None on failure.
    """
    frames, missing = _load_cached("injuries", seasons)

    if missing:
        try:
            print(f"Downloading injury data for seasons: {missing}")
            df = nfl.import_injuries(missing)
            df = _clean(df)
            _save_by_season(df, "injuries")
            frames.append(df)
        except Exception as e:
            warnings.warn(
                f"Injury data unavailable for seasons {missing}: {e}",
                stacklevel=2,
            )

    if not frames:
        return None

    return _clean(pd.concat(frames, ignore_index=True))


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    """
    Load game schedules (used for strength-of-schedule calculation in context.py).
    """
    frames, missing = _load_cached("schedules", seasons)

    if missing:
        print(f"Downloading schedules for seasons: {missing}")
        df = nfl.import_schedules(missing)
        df = _clean(df)
        _save_by_season(df, "schedules", season_col="season")
        frames.append(df)

    return _clean(pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """Wipe all local parquet cache files to force a full re-download."""
    removed = 0
    for p in CACHE_DIR.glob("*.parquet"):
        p.unlink()
        removed += 1
    print(f"Cleared {removed} parquet files from {CACHE_DIR}")


def refresh_current_season() -> None:
    """
    Re-download only the current PROJECTION_SEASON cache files.
    Use this for in-season updates without invalidating the training cache.
    """
    removed = 0
    for p in CACHE_DIR.glob(f"*_{PROJECTION_SEASON}.parquet"):
        p.unlink()
        removed += 1
    print(f"Removed {removed} cache files for season {PROJECTION_SEASON}. "
          "Next load will re-download from nfl_data_py.")
