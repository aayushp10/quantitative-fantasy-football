"""
Rookie projection model: draft capital + landing spot.

The main pipeline requires a prior NFL season of features, so every rookie —
a large fraction of real draft capital — previously got NO projection at all.
This model projects rookie-season fantasy output from information available
on draft day:

  Draft capital   — log(overall pick), round. The strongest public prior on
                    rookie volume: teams force-feed players they paid for.
  Landing spot    — destination team's prior-season pace / pass rate /
                    offensive EPA, QB quality (qb_epa_per_dropback), and the
                    target/carry share VACATED on that roster heading into
                    the rookie's first season.
  Age             — younger entrants at the same pick outperform.

Target: rookie-season total PPR points divided by the TEAM's games
(season_length), not games played — availability risk is priced in rather
than inflating 3-game cameos.

Training universe: all drafted QB/RB/WR/TE from ROOKIE_TRAIN_START on,
INCLUDING players who never recorded a stat (target = 0). Excluding them
would truncate the left tail and inflate every projection (survivorship).

Usage
-----
>>> frame = RookieModel.build_training_frame(draft_picks, weekly, team_ctx, qb, vac)
>>> model = RookieModel().train(frame)
>>> rookies_2025 = model.project_class(draft_picks, 2025, team_ctx, qb, vac)
>>> board = merge_rookie_projections(veteran_projections, rookies_2025)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    POSITIONS,
    ROOKIE_MAX_DRAFT_PICK,
    ROOKIE_TRAIN_START,
    season_length,
)

_FEATURES = [
    "log_pick", "draft_round", "entry_age",
    "team_pace_prev", "team_pass_rate_prev", "team_off_epa_prev",
    "qb_epa_prev",
    "vacated_target_share", "vacated_carry_share",
]


def _rookie_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])


class RookieModel:
    """Per-position Ridge on draft capital + landing spot."""

    def __init__(self):
        self._models: dict[str, Pipeline] = {}
        self._fallback_means: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Frame construction (shared by train and project)
    # ------------------------------------------------------------------

    @staticmethod
    def _engineer_features(
        picks: pd.DataFrame,
        team_context: pd.DataFrame | None = None,
        qb_coupling: pd.DataFrame | None = None,
        vacated_team: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Draft-pick rows → feature rows. Landing-spot frames are keyed by
        (team, season); the PRIOR season (draft_year - 1) describes the
        environment the rookie is entering.
        """
        df = picks.copy()

        # Column normalization across nfl_data_py versions
        if "gsis_id" in df.columns and "player_id" not in df.columns:
            df = df.rename(columns={"gsis_id": "player_id"})
        name_col = next(
            (c for c in ["pfr_player_name", "player_name", "name"] if c in df.columns),
            None,
        )
        df["player_name"] = df[name_col] if name_col else np.nan

        df = df[df["position"].isin(POSITIONS)]
        df = df[pd.to_numeric(df["pick"], errors="coerce").notna()]
        df["pick"] = df["pick"].astype(float)
        df = df[df["pick"] <= ROOKIE_MAX_DRAFT_PICK].copy()

        df["log_pick"] = np.log(df["pick"])
        df["draft_round"] = pd.to_numeric(df.get("round", np.nan), errors="coerce")
        df["entry_age"] = pd.to_numeric(df.get("age", np.nan), errors="coerce")
        df["_prev_season"] = df["season"] - 1

        def _join_env(env: pd.DataFrame | None, cols: dict[str, str]) -> None:
            nonlocal df
            if env is None or env.empty:
                return
            keep = ["team", "season"] + [c for c in cols if c in env.columns]
            if len(keep) <= 2:
                return
            right = env[keep].rename(
                columns={**cols, "season": "_prev_season"}
            ).drop_duplicates(subset=["team", "_prev_season"])
            df = df.merge(right, on=["team", "_prev_season"], how="left")

        _join_env(team_context, {
            "team_pace": "team_pace_prev",
            "team_pass_rate": "team_pass_rate_prev",
            "team_offensive_epa": "team_off_epa_prev",
        })
        _join_env(qb_coupling, {"qb_epa_per_dropback": "qb_epa_prev"})
        # Vacated shares keyed to (team, season N) describe shares leaving
        # AFTER season N — i.e. available in season N+1, the rookie year.
        _join_env(vacated_team, {
            "vacated_target_share": "vacated_target_share",
            "vacated_carry_share": "vacated_carry_share",
        })

        for f in _FEATURES:
            if f not in df.columns:
                df[f] = np.nan

        keep = ["player_id", "player_name", "position", "team", "season", "pick"] + _FEATURES
        return df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    @staticmethod
    def build_training_frame(
        draft_picks: pd.DataFrame,
        weekly: pd.DataFrame,
        team_context: pd.DataFrame | None = None,
        qb_coupling: pd.DataFrame | None = None,
        vacated_team: pd.DataFrame | None = None,
        train_start: int = ROOKIE_TRAIN_START,
        end_season: int | None = None,
    ) -> pd.DataFrame:
        """
        One row per drafted skill player with features and the target
        'rookie_fpts_per_team_game'. Players with no NFL stats get target 0.
        """
        picks = draft_picks[draft_picks["season"] >= train_start].copy()
        if end_season is not None:
            picks = picks[picks["season"] <= end_season]

        frame = RookieModel._engineer_features(
            picks, team_context, qb_coupling, vacated_team
        )
        if frame.empty:
            return frame

        fpts_col = (
            "fantasy_points_ppr"
            if "fantasy_points_ppr" in weekly.columns
            else "fantasy_points"
        )
        rookie_fpts = (
            weekly.groupby(["player_id", "season"], observed=True)[fpts_col]
            .sum()
            .reset_index()
            .rename(columns={fpts_col: "_rookie_fpts"})
        )
        frame = frame.merge(rookie_fpts, on=["player_id", "season"], how="left")
        # Drafted but never recorded a stat → 0 points (NOT dropped: dropping
        # them is survivorship bias and inflates every rookie projection)
        frame["_rookie_fpts"] = frame["_rookie_fpts"].fillna(0.0)
        frame["rookie_fpts_per_team_game"] = frame.apply(
            lambda r: r["_rookie_fpts"] / season_length(int(r["season"])), axis=1
        )
        return frame.drop(columns=["_rookie_fpts"])

    # ------------------------------------------------------------------
    # Train / project
    # ------------------------------------------------------------------

    def train(
        self,
        training_frame: pd.DataFrame,
        target: str = "rookie_fpts_per_team_game",
    ) -> "RookieModel":
        if target not in training_frame.columns:
            raise ValueError(f"'{target}' missing — use build_training_frame().")

        for pos in POSITIONS:
            pos_df = training_frame[training_frame["position"] == pos].dropna(subset=[target])
            self._fallback_means[pos] = (
                float(pos_df[target].mean()) if len(pos_df) else 0.0
            )
            if len(pos_df) < 20:
                warnings.warn(f"RookieModel {pos}: only {len(pos_df)} rows; using positional mean.")
                continue
            pipe = _rookie_pipeline()
            pipe.fit(pos_df[_FEATURES].values, pos_df[target].values)
            self._models[pos] = pipe
            print(f"  Rookie {pos}: n={len(pos_df)}")
        return self

    def project_class(
        self,
        draft_picks: pd.DataFrame,
        draft_year: int,
        team_context: pd.DataFrame | None = None,
        qb_coupling: pd.DataFrame | None = None,
        vacated_team: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Project one draft class. Output schema matches the veteran
        projections (projected_fpts_pg is per TEAM game), plus rookie=True.
        """
        picks = draft_picks[draft_picks["season"] == draft_year]
        frame = self._engineer_features(picks, team_context, qb_coupling, vacated_team)
        if frame.empty:
            return pd.DataFrame()

        games = season_length(draft_year)
        preds = np.full(len(frame), np.nan)
        for pos in POSITIONS:
            mask = (frame["position"] == pos).values
            if not mask.any():
                continue
            if pos in self._models:
                raw = self._models[pos].predict(frame.loc[mask, _FEATURES].values)
            else:
                raw = np.full(mask.sum(), self._fallback_means.get(pos, 0.0))
            preds[mask] = np.maximum(0.0, raw)

        out = frame[["player_id", "player_name", "position", "team", "pick"]].copy()
        out["projected_fpts_pg"] = preds
        out["projected_games"] = games
        out["projected_fpts_season"] = out["projected_fpts_pg"] * games
        out["projected_season"] = draft_year
        out["rookie"] = True
        return (
            out.sort_values("projected_fpts_season", ascending=False)
            .reset_index(drop=True)
        )


def merge_rookie_projections(
    veteran_proj: pd.DataFrame,
    rookie_proj: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append rookie projections to the veteran board (VOR/tiers run after this
    so replacement levels see the full player universe). Veterans win any
    player_id collision.
    """
    if rookie_proj is None or rookie_proj.empty:
        return veteran_proj
    vet = veteran_proj.copy()
    if "rookie" not in vet.columns:
        vet["rookie"] = False

    rook = rookie_proj.copy()
    if "player_id" in vet.columns and "player_id" in rook.columns:
        rook = rook[~rook["player_id"].isin(vet["player_id"].dropna())]

    combined = pd.concat([vet, rook], ignore_index=True)
    return (
        combined.sort_values("projected_fpts_season", ascending=False)
        .reset_index(drop=True)
    )
