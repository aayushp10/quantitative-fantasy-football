"""
Forward-looking context from the league schedule: Vegas week-1 lines and
head-coach changes.

Every other team-context feature (pace, pass rate, offensive EPA) is
backward-looking — last season's value carried forward. The schedule file
carries two forward-looking signals for the season being PREDICTED:

  Vegas week-1 lines   spread_line (home margin, home-favored positive) and
                       total_line give each team's implied points in its
                       opener — the betting market's read on the offense
                       entering season N+1.
  Head coaches         home_coach / away_coach per game → coach per
                       (team, season) → regime-change flags. The pace /
                       pass-rate carryover assumption breaks exactly when
                       the coach changes; the flag lets the models learn
                       that discount.

Features added to each (player, season N) row, keyed by the season-N team:

  vegas_implied_pts_next    team implied points in the season-N+1 opener
  vegas_total_next          total line of the season-N+1 opener
  vegas_implied_delta_next  N+1 opener implied points − N opener implied
  hc_changed_next           head coach entering N+1 differs from N (0/1)
  hc_changed                head coach entering N differed from N−1 (0/1)

Honest caveats (documented, accepted):
  - Closing week-1 lines finalize in early September — a fantasy draft in
    July/August sees near-final but not identical numbers. Opening lines
    aren't in the data; treat this as a tight proxy for draft-time market
    expectations, not a literally tradeable timestamp.
  - Team relocations (STL→LA, SD→LAC, OAK→LV) break the same-code join in
    the relocation year; those team-seasons get NaN and are imputed.
  - Players who change teams get their OLD team's forward context — the
    same limitation every team-context feature already has.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

SCHEDULE_CONTEXT_FEATURES: list[str] = [
    "vegas_implied_pts_next",
    "vegas_total_next",
    "vegas_implied_delta_next",
    "hc_changed_next",
    "hc_changed",
]


def build_schedule_context(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, season): opener implied points/total + head coach."""
    required = {"season", "week", "game_type", "home_team", "away_team"}
    if schedules is None or schedules.empty or not required.issubset(schedules.columns):
        return pd.DataFrame(columns=["team", "season", "implied_pts_wk1",
                                     "total_wk1", "head_coach"])

    reg = schedules[schedules["game_type"] == "REG"].copy()
    wk1 = reg[reg["week"] == 1]

    frames = []
    for side, opp_sign in [("home", +1), ("away", -1)]:
        f = wk1[["season", f"{side}_team", "total_line", "spread_line"]].copy()
        f.columns = ["season", "team", "total_wk1", "spread"]
        # spread_line = home margin → home implied = (total+spread)/2
        f["implied_pts_wk1"] = (f["total_wk1"] + opp_sign * f["spread"]) / 2.0
        frames.append(f.drop(columns=["spread"]))
    lines = pd.concat(frames, ignore_index=True)

    # Head coach per (team, season): modal coach across the team's REG games
    coach_frames = []
    for side in ["home", "away"]:
        col = f"{side}_coach"
        if col not in reg.columns:
            continue
        c = reg[["season", f"{side}_team", col]].copy()
        c.columns = ["season", "team", "head_coach"]
        coach_frames.append(c)
    if coach_frames:
        coaches = (
            pd.concat(coach_frames, ignore_index=True)
            .dropna(subset=["head_coach"])
            .groupby(["team", "season"], observed=True)["head_coach"]
            .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
            .reset_index()
        )
    else:
        coaches = pd.DataFrame(columns=["team", "season", "head_coach"])

    out = lines.merge(coaches, on=["team", "season"], how="outer")
    return out.drop_duplicates(subset=["team", "season"]).reset_index(drop=True)


def add_schedule_context_features(
    feature_matrix: pd.DataFrame,
    schedules: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join forward-looking schedule context onto the feature matrix.

    Loads schedules (cache-through) for all matrix seasons plus one — the
    N+1 lines are the point. Degrades to NaN columns on any failure.
    """
    fm = feature_matrix.drop(
        columns=[c for c in SCHEDULE_CONTEXT_FEATURES if c in feature_matrix.columns]
    )
    if "team" not in fm.columns or "season" not in fm.columns:
        return fm

    if schedules is None:
        try:
            from data.loader import load_schedules

            seasons = sorted(fm["season"].unique())
            schedules = load_schedules([int(s) for s in seasons] + [int(max(seasons)) + 1])
        except Exception as e:
            warnings.warn(f"schedule context unavailable ({e}); features NaN")
            out = fm.copy()
            for c in SCHEDULE_CONTEXT_FEATURES:
                out[c] = np.nan
            return out

    ctx = build_schedule_context(schedules)
    if ctx.empty:
        out = fm.copy()
        for c in SCHEDULE_CONTEXT_FEATURES:
            out[c] = np.nan
        return out

    cur = ctx.rename(columns={
        "implied_pts_wk1": "_implied_cur", "total_wk1": "_total_cur",
        "head_coach": "_hc_cur",
    })
    nxt = ctx.copy()
    nxt["season"] = nxt["season"] - 1  # season-N rows join season-N+1 context
    nxt = nxt.rename(columns={
        "implied_pts_wk1": "vegas_implied_pts_next", "total_wk1": "vegas_total_next",
        "head_coach": "_hc_next",
    })
    prv = ctx[["team", "season", "head_coach"]].copy()
    prv["season"] = prv["season"] + 1  # season-N rows join season-N−1 coach
    prv = prv.rename(columns={"head_coach": "_hc_prev"})

    out = (
        fm.merge(cur, on=["team", "season"], how="left")
        .merge(nxt, on=["team", "season"], how="left")
        .merge(prv, on=["team", "season"], how="left")
    )
    out["vegas_implied_delta_next"] = out["vegas_implied_pts_next"] - out["_implied_cur"]

    def _changed(a: pd.Series, b: pd.Series) -> pd.Series:
        both = a.notna() & b.notna()
        return pd.Series(np.where(both, (a != b).astype(float), np.nan), index=a.index)

    out["hc_changed_next"] = _changed(out["_hc_cur"], out["_hc_next"])
    out["hc_changed"] = _changed(out["_hc_prev"], out["_hc_cur"])

    return out.drop(columns=["_implied_cur", "_total_cur", "_hc_cur", "_hc_next", "_hc_prev"])
