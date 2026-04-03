"""
Draft capital and career trajectory features.

nfl_data_py's import_seasonal_rosters() includes:
  - draft_number (overall pick 1-262; NaN for UDFAs)
  - draft_round  (round number; NaN for UDFAs)
  - years_exp    (years of NFL experience, may be present)
  - entry_year   (first year in NFL, used to derive years_in_league if years_exp absent)

Draft capital is a strong predictor of volume opportunity, especially for RBs.
First-round picks receive more carries/targets than UDFAs at the same skill level
because coaches trust them with high-leverage situations.

Sophomore breakouts (years_in_league == 1) are a documented phenomenon for WRs/RBs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Pick boundaries for round buckets
_ROUND_BOUNDARIES = [
    (32, 5),    # Round 1 → 5
    (64, 4),    # Round 2 → 4
    (100, 3),   # Round 3 → 3
    (176, 2),   # Rounds 4-5 → 2
    (262, 1),   # Rounds 6-7 → 1
]


def _pick_to_bucket(pick: float) -> int:
    """Convert overall draft pick number to ordinal bucket (0–5)."""
    if pd.isna(pick):
        return 0  # UDFA
    for cutoff, bucket in _ROUND_BOUNDARIES:
        if pick <= cutoff:
            return bucket
    return 1  # beyond 262 (rare edge case)


def build_pedigree_features(rosters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build draft capital and career experience features per (player_id, season).

    Parameters
    ----------
    rosters_df : pd.DataFrame
        Cleaned roster data. Expected columns (subset used):
            player_id, season, draft_number, draft_round, years_exp, entry_year.
        All draft/experience columns are optional — missing ones are handled
        gracefully (features default to 0 or NaN).

    Returns
    -------
    pd.DataFrame
        Columns: player_id, season, draft_round_bucket (int 0–5),
                 draft_capital_score (float 0–1), years_in_league (int),
                 is_rookie (int 0/1), sophomore_flag (int 0/1).
    """
    required = {"player_id", "season"}
    if not required.issubset(rosters_df.columns):
        raise ValueError(f"rosters_df missing required columns: {required - set(rosters_df.columns)}")

    df = rosters_df[list(required | (required & set(rosters_df.columns)))].copy()

    # Keep only the columns we need (gracefully tolerate missing ones)
    keep_cols = ["player_id", "season"]
    for col in ["draft_number", "draft_round", "years_exp", "entry_year"]:
        if col in rosters_df.columns:
            df[col] = rosters_df[col].values

    df = df.drop_duplicates(subset=["player_id", "season"]).copy()

    # -----------------------------------------------------------------------
    # Draft capital
    # -----------------------------------------------------------------------
    if "draft_number" in df.columns:
        df["draft_round_bucket"] = df["draft_number"].apply(_pick_to_bucket).astype(int)
        # Continuous score: scales linearly from 0 (UDFA / pick 262) to 1 (pick 1)
        df["draft_capital_score"] = np.where(
            df["draft_number"].isna(),
            0.0,
            ((260.0 - df["draft_number"].clip(upper=260)) / 260.0).clip(lower=0.0),
        )
    else:
        df["draft_round_bucket"] = 0
        df["draft_capital_score"] = 0.0

    # -----------------------------------------------------------------------
    # Years in league
    # -----------------------------------------------------------------------
    if "years_exp" in df.columns:
        yil = df["years_exp"].fillna(0).clip(upper=15).astype(int)
    elif "entry_year" in df.columns:
        yil = (df["season"] - df["entry_year"]).fillna(0).clip(lower=0, upper=15).astype(int)
    else:
        yil = pd.Series(0, index=df.index)

    df["years_in_league"] = yil
    df["is_rookie"] = (df["years_in_league"] == 0).astype(int)
    df["sophomore_flag"] = (df["years_in_league"] == 1).astype(int)

    return df[
        ["player_id", "season", "draft_round_bucket", "draft_capital_score",
         "years_in_league", "is_rookie", "sophomore_flag"]
    ].reset_index(drop=True)
