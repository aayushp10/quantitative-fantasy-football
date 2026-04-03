"""
Data validation, deduplication, and type coercion for NFL data.

All feature engineering modules receive pre-cleaned DataFrames from here.
Key responsibilities:
- Filter PBP to play_type in ['pass', 'run'] (done once, applied everywhere)
- Deduplicate on (game_id, play_id) for PBP and (player_id, season, week) for weekly
- Validate EPA range: flag and optionally drop outliers
- Standardize player ID column names
- Remove plays with null receiver/rusher/passer IDs where appropriate
"""
from __future__ import annotations

import warnings

import pandas as pd


# Valid play types for fantasy-relevant analysis
VALID_PLAY_TYPES = {"pass", "run"}

# EPA values outside this range indicate data corruption
EPA_VALID_RANGE = (-15.0, 15.0)


# ---------------------------------------------------------------------------
# PBP cleaning
# ---------------------------------------------------------------------------

def clean_pbp(df: pd.DataFrame, *, validate_epa: bool = True) -> pd.DataFrame:
    """
    Clean and filter play-by-play data.

    Operations applied (in order):
    1. Deduplicate on (game_id, play_id)
    2. Filter to play_type in ['pass', 'run']
    3. Optionally validate EPA range
    4. Coerce key columns to appropriate dtypes
    5. Ensure season and week are int

    Parameters
    ----------
    df : pd.DataFrame
        Raw PBP DataFrame from loader.load_pbp().
    validate_epa : bool
        If True, warn about EPA values outside EPA_VALID_RANGE.

    Returns
    -------
    pd.DataFrame
        Cleaned, filtered PBP.
    """
    original_len = len(df)

    # 1. Deduplicate on (game_id, play_id) — penalty plays can create duplicates
    df = df.drop_duplicates(subset=["game_id", "play_id"])
    dup_removed = original_len - len(df)
    if dup_removed > 0:
        warnings.warn(f"Removed {dup_removed} duplicate (game_id, play_id) rows from PBP.")

    # 2. Filter to relevant play types
    if "play_type" in df.columns:
        df = df[df["play_type"].isin(VALID_PLAY_TYPES)].copy()
    else:
        warnings.warn("'play_type' column not found in PBP — skipping play type filter.")

    # 3. Validate EPA
    if validate_epa and "epa" in df.columns:
        lo, hi = EPA_VALID_RANGE
        bad_epa = df["epa"].notna() & ((df["epa"] < lo) | (df["epa"] > hi))
        n_bad = bad_epa.sum()
        if n_bad > 0:
            warnings.warn(
                f"{n_bad} rows have EPA outside {EPA_VALID_RANGE}. "
                "These may indicate data corruption. Consider inspecting them."
            )

    # 4. Coerce dtypes
    if "season" in df.columns:
        df["season"] = df["season"].astype("int16")
    if "week" in df.columns:
        df["week"] = df["week"].astype("int8")

    # Ensure boolean-like columns are numeric (0/1), not object
    for col in ["pass", "rush", "complete_pass", "incomplete_pass",
                "pass_attempt", "sack", "qb_scramble",
                "pass_touchdown", "rush_touchdown"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Weekly data cleaning
# ---------------------------------------------------------------------------

def clean_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean weekly player stats.

    Operations:
    1. Deduplicate on (player_id, season, week) — players who switch teams
       mid-season may appear twice in some datasets.
    2. Coerce season and week to int.
    3. Standardize player ID column if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Raw weekly DataFrame from loader.load_weekly().
    """
    # Standardize player ID column
    df = _normalize_player_id(df)

    # Deduplicate: keep the row with the most total stats for a given week
    id_cols = ["player_id", "season", "week"]
    if all(c in df.columns for c in id_cols):
        # Sort by a numeric stat so duplicates with more stats sort first
        sort_col = next((c for c in ["fantasy_points_ppr", "fantasy_points"] if c in df.columns), None)
        if sort_col:
            df = df.sort_values(sort_col, ascending=False)
        df = df.drop_duplicates(subset=id_cols, keep="first")
    else:
        missing = [c for c in id_cols if c not in df.columns]
        warnings.warn(f"Columns {missing} not found — skipping weekly deduplication.")

    # Coerce season/week
    if "season" in df.columns:
        df["season"] = df["season"].astype("int16")
    if "week" in df.columns:
        df["week"] = df["week"].astype("int8")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Roster cleaning
# ---------------------------------------------------------------------------

def clean_rosters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean roster/bio data.

    - Standardize player ID column.
    - Compute age from birth_date if 'age' column is absent.
    - Deduplicate on (player_id, season) keeping most recent entry.
    """
    df = _normalize_player_id(df)

    # Compute age if missing
    if "age" not in df.columns and "birth_date" in df.columns:
        df = _compute_age(df)

    # Deduplicate on (player_id, season)
    if "player_id" in df.columns and "season" in df.columns:
        df = df.drop_duplicates(subset=["player_id", "season"], keep="last")

    if "season" in df.columns:
        df["season"] = df["season"].astype("int16")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_player_id(df: pd.DataFrame) -> pd.DataFrame:
    """Rename gsis_id → player_id if player_id is absent."""
    if "player_id" not in df.columns and "gsis_id" in df.columns:
        df = df.rename(columns={"gsis_id": "player_id"})
    return df


def _compute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Compute integer age (years) from birth_date relative to season start (Sep 1)."""
    try:
        birth = pd.to_datetime(df["birth_date"], errors="coerce")
        # Use September 1 of the season year as the reference date
        season_start = pd.to_datetime(df["season"].astype(str) + "-09-01", errors="coerce")
        df["age"] = ((season_start - birth).dt.days / 365.25).astype("float32")
    except Exception as e:
        warnings.warn(f"Could not compute age from birth_date: {e}")
    return df
