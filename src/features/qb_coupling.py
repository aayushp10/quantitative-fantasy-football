"""
QB quality features attributed to each team-season.

A receiver's ceiling is capped by their QB. This module identifies the primary
QB for each team-season (by dropback count), computes their quality metrics,
and detects when the primary QB changed between seasons.

These features are joined to ALL skill position players on the same team,
not just to the QB row. A WR/TE on a team with an elite QB should project
higher than the same player on a team with a poor one.

Public API
----------
build_qb_coupling_features(pbp_df) -> pd.DataFrame
    One row per (team, season) with qb_epa_per_dropback, qb_cpoe,
    qb_deep_ball_rate, qb_td_rate, qb_changed (0/1), qb_upgrade_delta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dropback_mask(pbp: pd.DataFrame) -> pd.Series:
    """True for plays that are a QB dropback (pass attempt, sack, or scramble)."""
    mask = pd.Series(False, index=pbp.index)
    if "pass_attempt" in pbp.columns:
        mask |= pbp["pass_attempt"].fillna(0).astype(bool)
    if "sack" in pbp.columns:
        mask |= pbp["sack"].fillna(0).astype(bool)
    if "qb_scramble" in pbp.columns:
        mask |= pbp["qb_scramble"].fillna(0).astype(bool)
    return mask


# ---------------------------------------------------------------------------
# Step 1: Primary QB quality per team-season
# ---------------------------------------------------------------------------

def build_qb_quality_by_team(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the primary QB's quality metrics for each (team, season).

    The "primary QB" is the passer with the most dropbacks on that team.
    Using the primary QB avoids diluting metrics with garbage-time backup stats.

    Parameters
    ----------
    pbp_df : pd.DataFrame
        Cleaned play-by-play data.

    Returns
    -------
    pd.DataFrame
        Columns: team, season, primary_qb_id,
                 qb_epa_per_dropback, qb_cpoe, qb_deep_ball_rate, qb_td_rate.
    """
    required = {"posteam", "season"}
    if not required.issubset(pbp_df.columns):
        return pd.DataFrame(columns=["team", "season", "qb_epa_per_dropback"])

    passer_col = next(
        (c for c in ["passer_player_id", "passer_id"] if c in pbp_df.columns), None
    )
    if passer_col is None:
        return pd.DataFrame(columns=["team", "season", "qb_epa_per_dropback"])

    db_mask = _dropback_mask(pbp_df)
    pbp_db = pbp_df[db_mask & pbp_df[passer_col].notna()].copy()

    if pbp_db.empty:
        return pd.DataFrame(columns=["team", "season", "qb_epa_per_dropback"])

    pbp_db = pbp_db.rename(columns={"posteam": "team"})

    # Identify primary QB (most dropbacks) per team-season
    db_counts = (
        pbp_db.groupby(["team", "season", passer_col], observed=True)
        .size()
        .reset_index(name="n_dropbacks")
    )
    primary_qb = (
        db_counts.sort_values("n_dropbacks", ascending=False)
        .drop_duplicates(subset=["team", "season"])
        .rename(columns={passer_col: "primary_qb_id"})
    )

    # Compute quality metrics for each passer's dropbacks
    agg_cols: dict = {"n_dropbacks": ("season", "count")}
    if "epa" in pbp_db.columns:
        agg_cols["qb_epa_per_dropback"] = ("epa", "mean")
    if "cpoe" in pbp_db.columns:
        agg_cols["qb_cpoe"] = ("cpoe", "mean")
    if "pass_touchdown" in pbp_db.columns:
        agg_cols["_td_sum"] = ("pass_touchdown", "sum")

    deep_available = "air_yards" in pbp_db.columns
    if deep_available:
        pbp_db["_is_deep"] = (pbp_db["air_yards"].fillna(0) >= 20).astype(int)

    passer_agg = (
        pbp_db.groupby(["team", "season", passer_col], observed=True)
        .agg(
            n_dropbacks=(passer_col, "count"),
            **{
                k: v for k, v in {
                    "qb_epa_per_dropback": ("epa", "mean") if "epa" in pbp_db.columns else None,
                    "qb_cpoe": ("cpoe", "mean") if "cpoe" in pbp_db.columns else None,
                    "_td_sum": ("pass_touchdown", "sum") if "pass_touchdown" in pbp_db.columns else None,
                    "_deep_sum": ("_is_deep", "sum") if deep_available else None,
                }.items() if v is not None
            },
        )
        .reset_index()
        .rename(columns={passer_col: "primary_qb_id"})
    )

    if "_td_sum" in passer_agg.columns:
        passer_agg["qb_td_rate"] = np.where(
            passer_agg["n_dropbacks"] > 0,
            passer_agg["_td_sum"] / passer_agg["n_dropbacks"],
            np.nan,
        )
        passer_agg = passer_agg.drop(columns=["_td_sum"], errors="ignore")

    if "_deep_sum" in passer_agg.columns:
        passer_agg["qb_deep_ball_rate"] = np.where(
            passer_agg["n_dropbacks"] > 0,
            passer_agg["_deep_sum"] / passer_agg["n_dropbacks"],
            np.nan,
        )
        passer_agg = passer_agg.drop(columns=["_deep_sum"], errors="ignore")

    # Join to primary QB selection
    result = primary_qb[["team", "season", "primary_qb_id"]].merge(
        passer_agg.drop(columns=["n_dropbacks"], errors="ignore"),
        on=["team", "season", "primary_qb_id"],
        how="left",
    )

    keep_cols = ["team", "season", "primary_qb_id"] + [
        c for c in ["qb_epa_per_dropback", "qb_cpoe", "qb_deep_ball_rate", "qb_td_rate"]
        if c in result.columns
    ]
    return result[keep_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Detect QB changes between seasons
# ---------------------------------------------------------------------------

def detect_qb_changes(qb_by_team: pd.DataFrame) -> pd.DataFrame:
    """
    Detect teams where the primary QB changed between season N and N+1.

    Uses the same shift(-1) pattern as detect_team_changes() in situation.py.

    Parameters
    ----------
    qb_by_team : pd.DataFrame
        Output from build_qb_quality_by_team(): team, season, primary_qb_id,
        qb_epa_per_dropback.

    Returns
    -------
    pd.DataFrame
        Columns: team, season (= N), qb_changed (0/1),
                 old_qb_epa, new_qb_epa, qb_upgrade_delta.
        One row per (team, season) for ALL teams.
    """
    if qb_by_team.empty or "primary_qb_id" not in qb_by_team.columns:
        return pd.DataFrame(columns=["team", "season", "qb_changed"])

    df = (
        qb_by_team.sort_values(["team", "season"])
        .copy()
    )

    df["next_qb_id"] = df.groupby("team", observed=True)["primary_qb_id"].shift(-1)
    df["next_season"] = df.groupby("team", observed=True)["season"].shift(-1)
    df["next_qb_epa"] = df.groupby("team", observed=True)["qb_epa_per_dropback"].shift(-1)

    # qb_changed = 1 if consecutive seasons AND different primary QB
    df["qb_changed"] = (
        (df["next_season"] == df["season"] + 1)
        & df["next_qb_id"].notna()
        & (df["next_qb_id"] != df["primary_qb_id"])
    ).astype(int)

    # For unchanged QBs, set delta to 0; for changed, compute upgrade/downgrade
    epa_col = "qb_epa_per_dropback" if "qb_epa_per_dropback" in df.columns else None
    if epa_col:
        df["old_qb_epa"] = df[epa_col]
        df["new_qb_epa"] = df["next_qb_epa"]
        df["qb_upgrade_delta"] = np.where(
            df["qb_changed"] == 1,
            df["new_qb_epa"] - df["old_qb_epa"],
            0.0,
        )
    else:
        df["qb_upgrade_delta"] = 0.0

    keep = ["team", "season", "qb_changed"]
    for col in ["old_qb_epa", "new_qb_epa", "qb_upgrade_delta"]:
        if col in df.columns:
            keep.append(col)

    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_qb_coupling_features(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build QB quality features attributed to each team-season.

    In the assembler, these are joined on (team, season) so that all
    WR/TE/RB players on the same team see the same QB quality metrics.

    Parameters
    ----------
    pbp_df : pd.DataFrame
        Cleaned play-by-play data (multiple seasons).

    Returns
    -------
    pd.DataFrame
        Columns: team, season, qb_epa_per_dropback, qb_cpoe,
                 qb_deep_ball_rate, qb_td_rate, qb_changed (0/1),
                 qb_upgrade_delta.
    """
    qb_quality = build_qb_quality_by_team(pbp_df)
    if qb_quality.empty:
        return pd.DataFrame(columns=["team", "season"])

    qb_changes = detect_qb_changes(qb_quality)

    result = qb_quality.drop(columns=["primary_qb_id"], errors="ignore")
    if not qb_changes.empty:
        change_cols = ["team", "season", "qb_changed", "qb_upgrade_delta"]
        change_cols = [c for c in change_cols if c in qb_changes.columns]
        result = result.merge(qb_changes[change_cols], on=["team", "season"], how="left")
        if "qb_changed" in result.columns:
            result["qb_changed"] = result["qb_changed"].fillna(0).astype(int)

    return result.reset_index(drop=True)
