"""
Expected touchdowns from usage geometry.

TDs are mostly a function of WHERE usage happens: a carry from the 2-yard line
converts ~45% of the time, one from the 40 almost never. A player's realized
TD rate therefore splits into:

    x_*_td_rate  — the rate their usage MIX implies at league-average
                   conversion (persistent: it follows role/geometry)
    *_td_oe      — realized minus expected, per opportunity (mostly luck:
                   mean-reverts hard year over year)

"Fantasy points over expected regresses; expected points from usage persists."
The x-rates feed the two-stage model as player-specific EB prior means
(config.EB_GEOMETRY_PRIORS) and both columns enter the single-stage Ridge.

Bucket conversion rates are league-wide structural constants pooled across all
seasons in the provided PBP — they are extremely stable across the modern era,
which is what justifies treating them as a fixed map rather than re-estimating
per season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# yardline_100 bucket edges: (0,2], (2,5], (5,10], (10,20], (20,40], (40,100]
RUSH_TD_BUCKET_EDGES: list[int] = [0, 2, 5, 10, 20, 40, 100]
# Receiving/passing: (0,5], (5,10], (10,20], (20,40], (40,100]
PASS_TD_BUCKET_EDGES: list[int] = [0, 5, 10, 20, 40, 100]


def _expected_td_per_play(
    plays: pd.DataFrame,
    td_col: str,
    edges: list[int],
) -> pd.Series:
    """League-average TD probability of each play's field-position bucket."""
    buckets = pd.cut(plays["yardline_100"], bins=edges)
    return plays.groupby(buckets, observed=False)[td_col].transform("mean")


def _player_expected_td(
    plays: pd.DataFrame,
    player_col: str,
    td_col: str,
    edges: list[int],
    prefix: str,
) -> pd.DataFrame:
    """
    Aggregate per-play expected TDs to (player_id, team, season).

    Returns columns: x_{prefix}_td, {prefix}_tds, {prefix}_att_geo,
                     x_{prefix}_td_rate, {prefix}_td_oe.
    """
    df = plays.dropna(subset=[player_col, "yardline_100"]).copy()
    if df.empty or td_col not in df.columns:
        return pd.DataFrame(columns=["player_id", "team", "season"])

    df["_x_td"] = _expected_td_per_play(df, td_col, edges)
    df["player_id"] = df[player_col]
    df["team"] = df["posteam"]

    agg = (
        df.groupby(["player_id", "team", "season"], observed=True)
        .agg(
            **{
                f"x_{prefix}_td": pd.NamedAgg(column="_x_td", aggfunc="sum"),
                f"{prefix}_tds": pd.NamedAgg(column=td_col, aggfunc="sum"),
                f"{prefix}_att_geo": pd.NamedAgg(column="_x_td", aggfunc="count"),
            }
        )
        .reset_index()
    )

    att = agg[f"{prefix}_att_geo"].replace(0, np.nan)
    agg[f"x_{prefix}_td_rate"] = agg[f"x_{prefix}_td"] / att
    agg[f"{prefix}_td_oe"] = (agg[f"{prefix}_tds"] - agg[f"x_{prefix}_td"]) / att
    return agg


def build_expected_td_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute expected-TD features for rushers, receivers, and passers.

    Parameters
    ----------
    pbp : pd.DataFrame
        Cleaned PBP (pass/run plays) with rush/pass indicator columns,
        yardline_100, rush_touchdown / pass_touchdown, and player id columns.

    Returns
    -------
    pd.DataFrame
        One row per (player_id, team, season) with:
        x_rush_td_rate, rush_td_oe, x_rec_td_rate, rec_td_oe,
        x_pass_td_rate, pass_td_oe (NaN where the player has no such usage).
    """
    frames = []

    if "rush_touchdown" in pbp.columns:
        rush = pbp[pbp["rush"] == 1]
        frames.append(_player_expected_td(
            rush, "rusher_player_id", "rush_touchdown", RUSH_TD_BUCKET_EDGES, "rush"
        ))

    if "pass_touchdown" in pbp.columns:
        targeted = pbp[(pbp["pass"] == 1) & pbp["receiver_player_id"].notna()]
        frames.append(_player_expected_td(
            targeted, "receiver_player_id", "pass_touchdown", PASS_TD_BUCKET_EDGES, "rec"
        ))
        frames.append(_player_expected_td(
            targeted, "passer_player_id", "pass_touchdown", PASS_TD_BUCKET_EDGES, "pass"
        ))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["player_id", "team", "season"])

    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["player_id", "team", "season"], how="outer")

    keep = ["player_id", "team", "season"] + [
        c for c in [
            "x_rush_td", "x_rush_td_rate", "rush_td_oe",
            "x_rec_td", "x_rec_td_rate", "rec_td_oe",
            "x_pass_td", "x_pass_td_rate", "pass_td_oe",
        ]
        if c in result.columns
    ]
    return result[keep]
