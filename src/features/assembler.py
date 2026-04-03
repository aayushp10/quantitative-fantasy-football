"""
Merge all factor DataFrames into a unified feature matrix.

This is the integration point for the entire feature engineering pipeline.
It produces two outputs:
1. feature_matrix  — one row per (player_id, season), with all factors
2. yoy_pairs       — one row per (player_id, season_N) with season_N+1 target

Design decisions:
- Target variable: use fantasy_points_ppr from import_weekly_data(), NOT
  recomputed from PBP. The official weekly stats handle edge cases
  (laterals, 2pt conversions) more reliably.
- YoY pair construction: shift fpts_per_game by -1 within each player group.
- Minimum thresholds: QB >= 100 dropbacks, RB >= 50 (carries+targets),
  WR/TE >= 30 targets. Players below threshold are excluded from training.
"""
from __future__ import annotations

from functools import reduce

import numpy as np
import pandas as pd

from config import (
    _FEATURE_VERSION,
    CACHE_DIR,
    MIN_DROPBACKS_QB,
    MIN_TARGETS_TE,
    MIN_TARGETS_WR,
    MIN_TOUCHES_RB,
    POSITIONS,
    PPR_SCORING,
)
from data.loader import (
    load_pbp,
    load_rosters,
    load_seasonal,
    load_snap_counts,
    load_weekly,
)
from data.cleaning import clean_pbp, clean_rosters, clean_weekly
from features.consistency import build_consistency_features
from features.context import build_context_factors, get_team_context
from features.efficiency import build_efficiency_factors
from features.opportunity import build_opportunity_factors
from features.pedigree import build_pedigree_features
from features.qb_coupling import build_qb_coupling_features
from features.situation import build_situation_features
from features.trend import detect_trends
from features.vacated import assign_vacated_shares_to_players, compute_vacated_shares


# ---------------------------------------------------------------------------
# Fantasy points per game from weekly data
# ---------------------------------------------------------------------------

def _compute_fpts_per_game(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly PPR points into season-level fpts and fpts_per_game.

    Uses fantasy_points_ppr column if available; falls back to
    fantasy_points for other scoring formats.
    """
    fpts_col = "fantasy_points_ppr" if "fantasy_points_ppr" in weekly.columns else "fantasy_points"

    season_agg = (
        weekly.groupby(["player_id", "season"], observed=True)
        .agg(
            fpts=pd.NamedAgg(column=fpts_col, aggfunc="sum"),
            games_played_wk=pd.NamedAgg(column="week", aggfunc="nunique"),
        )
        .reset_index()
    )
    season_agg["fpts_per_game"] = np.where(
        season_agg["games_played_wk"] > 0,
        season_agg["fpts"] / season_agg["games_played_wk"],
        np.nan,
    )
    return season_agg


# ---------------------------------------------------------------------------
# Position assignment from rosters
# ---------------------------------------------------------------------------

def _get_player_meta(rosters: pd.DataFrame) -> pd.DataFrame:
    """Extract player metadata from rosters, including draft capital columns for pedigree features."""
    keep_cols = ["player_id", "season"]
    optional = [
        "position", "age", "player_name", "full_name", "team",
        "depth_chart_position",
        # Draft/experience columns needed by pedigree module
        "draft_number", "draft_round", "years_exp", "entry_year",
    ]

    cols = keep_cols + [c for c in optional if c in rosters.columns]
    meta = rosters[cols].drop_duplicates(subset=["player_id", "season"]).copy()

    # Normalize name column
    if "player_name" not in meta.columns and "full_name" in meta.columns:
        meta = meta.rename(columns={"full_name": "player_name"})

    # Use depth_chart_position as fallback
    if "position" not in meta.columns and "depth_chart_position" in meta.columns:
        meta = meta.rename(columns={"depth_chart_position": "position"})

    return meta


# ---------------------------------------------------------------------------
# Minimum threshold filtering
# ---------------------------------------------------------------------------

def _apply_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude players who didn't meet minimum play thresholds.

    Thresholds ensure the factor estimates are statistically meaningful.
    """
    mask = pd.Series(False, index=df.index)

    if "position" not in df.columns:
        return df

    qb_mask = (df["position"] == "QB") & (df.get("dropbacks", pd.Series(0, index=df.index)) >= MIN_DROPBACKS_QB)
    rb_touches = df.get("carries", pd.Series(0, index=df.index)).fillna(0) + \
                 df.get("targets", pd.Series(0, index=df.index)).fillna(0)
    rb_mask = (df["position"] == "RB") & (rb_touches >= MIN_TOUCHES_RB)
    wr_mask = (df["position"] == "WR") & (df.get("targets", pd.Series(0, index=df.index)).fillna(0) >= MIN_TARGETS_WR)
    te_mask = (df["position"] == "TE") & (df.get("targets", pd.Series(0, index=df.index)).fillna(0) >= MIN_TARGETS_TE)

    mask = qb_mask | rb_mask | wr_mask | te_mask
    return df[mask].copy()


# ---------------------------------------------------------------------------
# YoY pair construction
# ---------------------------------------------------------------------------

def build_yoy_pairs(
    feature_matrix: pd.DataFrame,
    extra_target_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Create (season N features) → (season N+1 target) pairs per player.

    The most recent season in the matrix has features but no target;
    it's used for projection only, not training.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Unified feature matrix with player_id, season, fpts_per_game columns.
    extra_target_cols : list[str] | None
        Additional columns to shift forward as targets (e.g. rate stats for
        the two-stage model). Each col creates a 'next_{col}' column.
        Columns missing from feature_matrix are silently skipped.

    Returns
    -------
    pd.DataFrame
        feature_matrix rows with 'next_fpts' (and optionally 'next_{col}' for
        each extra_target_cols entry). Rows where next_fpts is NaN are dropped.
    """
    if "player_id" not in feature_matrix.columns or "season" not in feature_matrix.columns:
        raise ValueError("feature_matrix must have 'player_id' and 'season' columns.")

    # Sort by player then season so shift(-1) gives next season's value
    fm = feature_matrix.sort_values(["player_id", "season"]).copy()

    # Shift fpts_per_game within each player group
    fm["next_fpts"] = fm.groupby("player_id", observed=True)["fpts_per_game"].shift(-1)
    fm["next_season"] = fm.groupby("player_id", observed=True)["season"].shift(-1)

    # Shift any additional target columns requested by two-stage model
    if extra_target_cols:
        for col in extra_target_cols:
            if col in fm.columns:
                fm[f"next_{col}"] = fm.groupby("player_id", observed=True)[col].shift(-1)

    # Only keep rows where the next season is exactly current + 1
    # (avoids bridging gaps when a player misses a season)
    valid = fm["next_season"] == fm["season"] + 1
    pairs = fm[valid & fm["next_fpts"].notna()].copy()
    pairs = pairs.drop(columns=["next_season"])

    return pairs.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def _coalesce_suffixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    After an outer merge, pandas renames colliding columns to col_x / col_y.
    This coalesces each pair back into a single column (first non-null wins)
    and drops the suffixed duplicates.

    Without this, count columns like `dropbacks`, `carries`, and `targets`
    that appear in both opportunity and efficiency outputs become `dropbacks_x`
    / `dropbacks_y`, causing the threshold filter in _apply_thresholds to
    find neither and default to zero — silently dropping all QBs.
    """
    x_cols = {c[:-2] for c in df.columns if c.endswith("_x")}
    for base in x_cols:
        col_x, col_y = f"{base}_x", f"{base}_y"
        if col_y in df.columns:
            df[base] = df[col_x].combine_first(df[col_y])
            df = df.drop(columns=[col_x, col_y])
    return df


def _feature_matrix_cache_path(seasons: list[int]):
    seasons_key = "_".join(str(s) for s in sorted(seasons))
    return CACHE_DIR / f"feature_matrix_{seasons_key}_{_FEATURE_VERSION}.parquet"


def assemble_feature_matrix(
    seasons: list[int],
    snap_df: pd.DataFrame | None = None,
    schedules: pd.DataFrame | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Build the unified feature matrix for the given seasons.

    The result is cached to data/cache/feature_matrix_<seasons>.parquet.
    On subsequent calls with the same seasons, the cached version is loaded
    instantly instead of re-running all feature engineering.

    Pass force_recompute=True to ignore the cache and rebuild from scratch
    (e.g. after changing feature engineering code or adding new data).

    Parameters
    ----------
    seasons : list[int]
        Seasons to include in the matrix.
    force_recompute : bool
        If True, ignore any existing cache and recompute. Default False.
    snap_df : pd.DataFrame, optional
        Pre-loaded snap count data. If None, loaded automatically.
    schedules : pd.DataFrame, optional
        Schedule data for SOS calculation.

    Returns
    -------
    pd.DataFrame
        Unified feature matrix, one row per (player_id, season).
        Schema: player_id, player_name, position, team, season, age,
                fpts, fpts_per_game, games_played, [all factors]
    """
    cache_path = _feature_matrix_cache_path(seasons)

    if not force_recompute and cache_path.exists():
        print(f"Loading feature matrix from cache: {cache_path.name}")
        return pd.read_parquet(cache_path)

    print(f"Assembling feature matrix for seasons: {seasons}")

    # --- Load raw data ---
    pbp_raw = load_pbp(seasons)
    pbp = clean_pbp(pbp_raw)

    weekly_raw = load_weekly(seasons)
    weekly = clean_weekly(weekly_raw)

    rosters_raw = load_rosters(seasons)
    rosters = clean_rosters(rosters_raw)

    if snap_df is None:
        try:
            snap_df = load_snap_counts(seasons)
        except Exception:
            snap_df = None

    # --- Build factor sets ---
    print("  Computing opportunity factors...")
    opp = build_opportunity_factors(pbp)

    print("  Computing efficiency factors...")
    eff = build_efficiency_factors(pbp)

    print("  Computing context factors...")
    ctx = build_context_factors(pbp, weekly, snap_df, schedules)

    print("  Computing trend factors...")
    trend_frames = []
    for s in seasons:
        season_weekly = weekly[weekly["season"] == s]
        if len(season_weekly) > 0:
            try:
                t = detect_trends(pbp, weekly, season=s, snap_df=snap_df)
                trend_frames.append(t)
            except Exception as e:
                print(f"    Trend detection failed for {s}: {e}")
    trend = pd.concat(trend_frames, ignore_index=True) if trend_frames else pd.DataFrame()

    # --- Merge all factor DataFrames ---
    print("  Merging factors...")
    join_keys_full = ["player_id", "team", "season"]
    join_keys_player = ["player_id", "season"]

    factor_frames = [f for f in [opp, eff, ctx] if not f.empty]
    if factor_frames:
        merged = reduce(
            lambda a, b: a.merge(b, on=join_keys_full, how="outer"),
            factor_frames
        )
        merged = _coalesce_suffixed_columns(merged)
    else:
        return pd.DataFrame()

    if not trend.empty:
        trend_keys = [k for k in join_keys_full if k in trend.columns]
        merged = merged.merge(trend, on=trend_keys, how="left")
        merged = _coalesce_suffixed_columns(merged)

    # --- QB coupling features (team-level; must precede top-down) ---
    print("  Computing QB coupling features...")
    try:
        qb_coupling = build_qb_coupling_features(pbp)
        merged = merged.merge(qb_coupling, on=["team", "season"], how="left")
        merged = _coalesce_suffixed_columns(merged)
    except Exception as e:
        print(f"    QB coupling features failed: {e}")

    # --- Situation change features ---
    print("  Computing situation change features...")
    try:
        team_ctx = get_team_context(pbp)
        situation = build_situation_features(rosters, team_ctx)
        merged = merged.merge(situation, on=join_keys_player, how="left")
        merged = _coalesce_suffixed_columns(merged)

        # For team-change players, replace old team context with destination team context
        # so the Ridge model sees the environment the player is entering, not leaving
        if "team_changed" in merged.columns:
            tc_mask = merged["team_changed"] == 1
            for old_col, new_col in [
                ("team_pace", "new_team_pace"),
                ("team_pass_rate", "new_team_pass_rate"),
                ("team_offensive_epa", "new_team_offensive_epa"),
            ]:
                if new_col in merged.columns and old_col in merged.columns:
                    merged.loc[tc_mask, old_col] = merged.loc[tc_mask, new_col]
    except Exception as e:
        print(f"    Situation features failed: {e}")

    # --- Vacated share features (reads target_share/rush_share from merged) ---
    print("  Computing vacated share features...")
    try:
        vacated_team = compute_vacated_shares(merged, rosters)
        vacated_player = assign_vacated_shares_to_players(vacated_team, rosters, merged)
        merged = merged.merge(vacated_player, on=join_keys_player, how="left")
        merged = _coalesce_suffixed_columns(merged)
    except Exception as e:
        print(f"    Vacated share features failed: {e}")

    # --- Pedigree features (draft capital + experience) ---
    print("  Computing pedigree features...")
    try:
        pedigree = build_pedigree_features(rosters)
        merged = merged.merge(pedigree, on=join_keys_player, how="left")
        merged = _coalesce_suffixed_columns(merged)
    except Exception as e:
        print(f"    Pedigree features failed: {e}")

    # --- Consistency features (weekly scoring variance) ---
    print("  Computing consistency features...")
    try:
        consistency = build_consistency_features(weekly)
        merged = merged.merge(consistency, on=join_keys_player, how="left")
        merged = _coalesce_suffixed_columns(merged)
    except Exception as e:
        print(f"    Consistency features failed: {e}")

    # --- Fantasy points target variable ---
    fpts = _compute_fpts_per_game(weekly)
    merged = merged.merge(fpts, on=join_keys_player, how="left")

    # games_played: prefer from context (pbp-derived), fallback to weekly
    if "games_played" not in merged.columns:
        if "games_played_wk" in merged.columns:
            merged = merged.rename(columns={"games_played_wk": "games_played"})
    else:
        merged["games_played"] = merged["games_played"].fillna(merged.get("games_played_wk", np.nan))

    # --- Rate stats (targets for two-stage efficiency models) ---
    # Derived from weekly aggregates; used as shift targets in build_yoy_pairs
    _weekly_agg_cols = {
        "targets": "total_targets",
        "receptions": "total_receptions",
        "receiving_yards": "total_receiving_yards",
        "receiving_tds": "total_receiving_tds",
        "carries": "total_carries",
        "rushing_yards": "total_rushing_yards",
        "rushing_tds": "total_rushing_tds",
        "passing_yards": "total_passing_yards",
        "passing_tds": "total_passing_tds",
        "attempts": "total_pass_attempts",
    }
    available_weekly_cols = {k: v for k, v in _weekly_agg_cols.items() if k in weekly.columns}
    if available_weekly_cols:
        rate_agg = (
            weekly.groupby(["player_id", "season"], observed=True)
            .agg(**{v: pd.NamedAgg(column=k, aggfunc="sum")
                    for k, v in available_weekly_cols.items()})
            .reset_index()
        )
        t = rate_agg.get("total_targets", pd.Series(1, index=rate_agg.index)).clip(lower=1)
        c = rate_agg.get("total_carries", pd.Series(1, index=rate_agg.index)).clip(lower=1)
        a = rate_agg.get("total_pass_attempts", pd.Series(1, index=rate_agg.index)).clip(lower=1)
        if "total_receiving_yards" in rate_agg.columns:
            rate_agg["yards_per_target"] = rate_agg["total_receiving_yards"] / t
        if "total_receiving_tds" in rate_agg.columns:
            rate_agg["rec_td_rate"] = rate_agg["total_receiving_tds"] / t
        if "total_rushing_tds" in rate_agg.columns:
            rate_agg["rush_td_rate"] = rate_agg["total_rushing_tds"] / c
        if "total_passing_yards" in rate_agg.columns:
            rate_agg["pass_yards_per_attempt"] = rate_agg["total_passing_yards"] / a
        if "total_passing_tds" in rate_agg.columns:
            rate_agg["pass_td_rate"] = rate_agg["total_passing_tds"] / a
        # Per-game volume cols for two-stage volume stage targets
        if "total_targets" in rate_agg.columns and "games_played_wk" not in merged.columns:
            pass  # games_played derived later; skip per-game volume here
        rate_cols = ["player_id", "season"] + [
            c for c in ["yards_per_target", "rec_td_rate", "rush_td_rate",
                        "pass_yards_per_attempt", "pass_td_rate"]
            if c in rate_agg.columns
        ]
        merged = merged.merge(rate_agg[rate_cols], on=join_keys_player, how="left")
        merged = _coalesce_suffixed_columns(merged)

    # --- Top-down team constraint features (requires qb_coupling in matrix) ---
    print("  Computing top-down team constraint features...")
    try:
        from models.team_constraint import build_topdown_features
        topdown_frames = []
        for s in seasons:
            td = build_topdown_features(merged[merged["season"] == s], target_season=s + 1)
            if td is not None and not td.empty:
                topdown_frames.append(td)
        if topdown_frames:
            topdown = pd.concat(topdown_frames, ignore_index=True)
            merged = merged.merge(topdown, on=join_keys_player, how="left")
            merged = _coalesce_suffixed_columns(merged)
    except Exception as e:
        print(f"    Top-down features failed: {e}")

    # --- Player metadata ---
    meta = _get_player_meta(rosters)
    merged = merged.merge(meta, on=join_keys_player, how="left", suffixes=("", "_roster"))

    # Reconcile team column (use roster team if PBP team is missing)
    if "team_roster" in merged.columns:
        merged["team"] = merged["team"].fillna(merged["team_roster"])
        merged = merged.drop(columns=["team_roster"], errors="ignore")

    # --- Filter to fantasy-relevant positions and apply thresholds ---
    if "position" in merged.columns:
        merged = merged[merged["position"].isin(POSITIONS)]
    merged = _apply_thresholds(merged)

    # --- Final column ordering ---
    priority_cols = [
        "player_id", "player_name", "position", "team", "season", "age",
        "fpts", "fpts_per_game", "games_played",
    ]
    other_cols = [c for c in merged.columns if c not in priority_cols]
    col_order = [c for c in priority_cols if c in merged.columns] + other_cols
    merged = merged[col_order]

    result = merged.reset_index(drop=True)
    print(f"  Feature matrix: {len(result)} rows, {len(result.columns)} columns")
    print(f"  Saving to cache: {cache_path.name}")
    result.to_parquet(cache_path, index=False)
    return result
