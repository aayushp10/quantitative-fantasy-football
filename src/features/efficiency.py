"""
Skill / efficiency factor construction from play-by-play data.

These are the "alpha" factors — performance above or below what opportunity
alone would predict. All factors computed at the (player_id, season) level.

Key implementation notes:
- Dropback = pass_attempt OR sack OR qb_scramble (not just play_type == 'pass')
- Division by zero: use np.where(denom > 0, num/denom, np.nan) throughout
- CPOE: mean of non-null values only (dropna() before aggregation)
- Success rate: % of plays with positive EPA
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    return np.where(denom > 0, num / denom, np.nan)


# ---------------------------------------------------------------------------
# QB efficiency
# ---------------------------------------------------------------------------

def _qb_efficiency(pbp: pd.DataFrame) -> pd.DataFrame:
    pass_plays = pbp[pbp["pass"] == 1]

    # Dropbacks: pass attempts + sacks + scrambles
    dropback_mask = (
        (pass_plays.get("pass_attempt", pd.Series(0, index=pass_plays.index)) == 1) |
        (pass_plays.get("sack", pd.Series(0, index=pass_plays.index)) == 1) |
        (pass_plays.get("qb_scramble", pd.Series(0, index=pass_plays.index)) == 1)
    )
    db = pass_plays[dropback_mask].copy()
    db["player_id"] = db["passer_player_id"]
    db = db.dropna(subset=["player_id"])

    def _agg_qb(g):
        n = len(g)
        if n == 0:
            return pd.Series(dtype=float)

        epa = g["epa"] if "epa" in g else pd.Series(np.nan, index=g.index)
        cpoe = g["cpoe"].dropna() if "cpoe" in g else pd.Series(dtype=float)
        air = g["air_yards"] if "air_yards" in g else pd.Series(np.nan, index=g.index)
        comp = g["complete_pass"] if "complete_pass" in g else pd.Series(0, index=g.index)
        td = g["pass_touchdown"] if "pass_touchdown" in g else pd.Series(0, index=g.index)
        interception = g["interception"] if "interception" in g else pd.Series(0, index=g.index)
        sacks = g["sack"] if "sack" in g else pd.Series(0, index=g.index)

        deep_mask = air >= 20
        pass_att = g.get("pass_attempt", pd.Series(1, index=g.index))

        return pd.Series({
            "epa_per_dropback": epa.mean(),
            "cpoe": cpoe.mean() if len(cpoe) > 0 else np.nan,
            "success_rate": (epa > 0).mean(),
            "deep_ball_rate": deep_mask.mean(),
            "deep_ball_completion_rate": (comp[deep_mask].mean() if deep_mask.any() else np.nan),
            "td_rate": _safe_divide(pd.Series([td.sum()]), pd.Series([n]))[0],
            "int_rate": _safe_divide(pd.Series([interception.sum()]), pd.Series([n]))[0],
            "sack_rate": _safe_divide(pd.Series([sacks.sum()]), pd.Series([n]))[0],
            "dropbacks": n,
        })

    result = (
        db.groupby(["player_id", "posteam", "season"], observed=True)
        .apply(_agg_qb, include_groups=False)
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    return result


# ---------------------------------------------------------------------------
# WR / TE efficiency
# ---------------------------------------------------------------------------

def _receiver_efficiency(pbp: pd.DataFrame) -> pd.DataFrame:
    tgt = pbp[(pbp["pass"] == 1) & pbp["receiver_player_id"].notna()].copy()
    tgt["player_id"] = tgt["receiver_player_id"]

    def _agg_recv(g):
        n = len(g)
        if n == 0:
            return pd.Series(dtype=float)

        epa = g["epa"] if "epa" in g else pd.Series(np.nan, index=g.index)
        comp = g["complete_pass"] if "complete_pass" in g else pd.Series(0, index=g.index)
        yac = g["yards_after_catch"] if "yards_after_catch" in g else pd.Series(np.nan, index=g.index)
        yac_epa = g["yac_epa"] if "yac_epa" in g else pd.Series(np.nan, index=g.index)
        air = g["air_yards"] if "air_yards" in g else pd.Series(np.nan, index=g.index)
        td = g["pass_touchdown"] if "pass_touchdown" in g else pd.Series(0, index=g.index)
        rec_yards = g["yards_gained"] if "yards_gained" in g else pd.Series(np.nan, index=g.index)

        completions = comp.sum()
        deep_mask = (air >= 15) & comp.notna()

        # Separation from NGS (if column present)
        sep = g["avg_separation"].mean() if "avg_separation" in g.columns else np.nan

        return pd.Series({
            "epa_per_target": epa.mean(),
            "yac_per_rec": (
                yac[comp == 1].mean() if completions > 0 else np.nan
            ),
            "yac_epa": yac_epa.mean(),
            "catch_rate": _safe_divide(pd.Series([completions]), pd.Series([n]))[0],
            "avg_depth_of_target": air.mean(),
            "explosive_play_rate": (
                ((rec_yards[comp == 1] >= 20).mean()) if completions > 0 else np.nan
            ),
            "first_down_rate": (
                (g["first_down_pass"].sum() / completions if "first_down_pass" in g.columns and completions > 0 else np.nan)
            ),
            "contested_catch_rate_proxy": (
                comp[deep_mask].mean() if deep_mask.any() else np.nan
            ),
            "separation_proxy": sep,
            "targets": n,
        })

    result = (
        tgt.groupby(["player_id", "posteam", "season"], observed=True)
        .apply(_agg_recv, include_groups=False)
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    return result


# ---------------------------------------------------------------------------
# RB efficiency
# ---------------------------------------------------------------------------

def _rb_efficiency(pbp: pd.DataFrame) -> pd.DataFrame:
    rush = pbp[pbp["rush"] == 1].copy()
    rush["player_id"] = rush["rusher_player_id"]
    rush = rush.dropna(subset=["player_id"])

    def _agg_rb(g):
        n = len(g)
        if n == 0:
            return pd.Series(dtype=float)

        epa = g["epa"] if "epa" in g else pd.Series(np.nan, index=g.index)
        yards = g["rushing_yards"] if "rushing_yards" in g else g.get("yards_gained", pd.Series(np.nan, index=g.index))

        return pd.Series({
            "epa_per_carry": epa.mean(),
            "rush_success_rate": (epa > 0).mean(),
            "ypc": yards.mean(),
            "explosive_run_rate": (yards >= 10).mean(),
            "breakaway_run_rate": (yards >= 15).mean(),
            "stuff_rate": (yards <= 0).mean(),
            "carries": n,
        })

    result = (
        rush.groupby(["player_id", "posteam", "season"], observed=True)
        .apply(_agg_rb, include_groups=False)
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # RB receiving efficiency (as a pass-catcher)
    recv = _receiver_efficiency(pbp)
    recv_rb = recv.rename(columns={
        "epa_per_target": "receiving_efficiency",
        "catch_rate": "rb_catch_rate",
        "yac_per_rec": "rb_yac_per_rec",
    })[["player_id", "team", "season", "receiving_efficiency", "rb_catch_rate", "rb_yac_per_rec", "targets"]]

    result = result.merge(recv_rb, on=["player_id", "team", "season"], how="left")
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_efficiency_factors(
    pbp: pd.DataFrame,
    position: str = "ALL",
) -> pd.DataFrame:
    """
    Compute skill/efficiency factors from play-by-play data.

    Parameters
    ----------
    pbp : pd.DataFrame
        Cleaned PBP DataFrame (play_type already filtered to pass/run).
    position : str
        'QB', 'RB', 'WR', 'TE', or 'ALL'. When 'ALL', all factor sets
        are outer-merged.

    Returns
    -------
    pd.DataFrame
        One row per (player_id, team, season) with efficiency factor columns.
    """
    if position == "QB":
        return _qb_efficiency(pbp)
    if position == "RB":
        return _rb_efficiency(pbp)
    if position in ("WR", "TE"):
        return _receiver_efficiency(pbp)

    # ALL: outer merge all three
    qb = _qb_efficiency(pbp)
    rb = _rb_efficiency(pbp)
    recv = _receiver_efficiency(pbp)

    result = qb.merge(rb, on=["player_id", "team", "season"], how="outer")
    result = result.merge(recv, on=["player_id", "team", "season"], how="outer",
                          suffixes=("_rb", "_recv"))
    return result
