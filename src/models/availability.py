"""
Games-played (availability) model.

The projection pipeline previously assumed EVERY player plays 17 games —
the single largest unmodeled input to season totals. Season value is
E[games] × E[pts/game], and expected games varies systematically with age,
position, workload, and durability history.

Model: pooled Ridge across positions (position one-hots) predicting
    next_games_frac = next_games_played / season_length(next season)

from: age, current games fraction, prior-season games fraction (durability
history), touches per game (workload), and position. Season lengths are
era-adjusted (16 games through 2020, 17 after), so the target is comparable
across the 2012+ training window.

Usage
-----
>>> avail = AvailabilityModel().train(yoy_pairs)          # needs next_games_played
>>> games = avail.predict_games(feature_matrix, target_season=2025)
>>> proj = avail.attach_to_projections(projections, feature_matrix, target_season=2025)
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
    AVAILABILITY_MAX_GAMES_FRAC,
    AVAILABILITY_MIN_GAMES_FRAC,
    POSITIONS,
    RECENCY_DECAY,
    season_length,
)

_FEATURES = [
    "age", "games_frac", "games_frac_prev", "touches_per_game",
    "pos_QB", "pos_RB", "pos_WR", "pos_TE",
]


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Build availability features for every row of a (multi-season) frame."""
    out = df.copy()

    season_len = out["season"].map(season_length)
    out["games_frac"] = (out.get("games_played", np.nan) / season_len).clip(0, 1)

    # Prior-season games fraction (durability history) within the given frame
    out = out.sort_values(["player_id", "season"])
    out["games_frac_prev"] = out.groupby("player_id", observed=True)["games_frac"].shift(1)

    # Workload: touches (carries + targets) or dropbacks per game
    carries = out.get("carries", pd.Series(0, index=out.index)).fillna(0)
    targets = out.get("targets", pd.Series(0, index=out.index)).fillna(0)
    dropbacks = out.get("dropbacks", pd.Series(0, index=out.index)).fillna(0)
    touches = carries + targets + dropbacks
    games = out.get("games_played", pd.Series(np.nan, index=out.index)).replace(0, np.nan)
    out["touches_per_game"] = touches / games

    for pos in POSITIONS:
        out[f"pos_{pos}"] = (out.get("position", "") == pos).astype(float)

    return out


class AvailabilityModel:
    """Ridge model for expected games played next season."""

    def __init__(self):
        self._pipe: Pipeline | None = None
        self.residual_std_: dict[str, float] = {}   # per-position games-frac residual std
        self.n_train_: int = 0

    # ------------------------------------------------------------------

    def train(self, yoy_df: pd.DataFrame, target: str = "next_games_played") -> "AvailabilityModel":
        """
        Fit on YoY pairs. Requires 'next_games_played' — created by
        build_yoy_pairs(fm, extra_target_cols=ALL_RATE_TARGET_COLS), which
        includes games_played in the shift list.
        """
        if target not in yoy_df.columns:
            raise ValueError(
                f"'{target}' not in yoy_df. Build pairs with "
                "extra_target_cols=ALL_RATE_TARGET_COLS (includes games_played)."
            )

        df = _engineer(yoy_df)
        next_len = (df["season"] + 1).map(season_length)
        df["_target_frac"] = (df[target] / next_len).clip(0, 1)

        df = df.dropna(subset=["_target_frac"])
        if len(df) < 50:
            raise ValueError(f"Too few availability training rows: {len(df)}")

        X = df[_FEATURES].values
        y = df["_target_frac"].values
        max_season = int(df["season"].max())
        weights = np.power(RECENCY_DECAY, (max_season - df["season"].values)).astype(float)

        self._pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ])
        self._pipe.fit(X, y, ridge__sample_weight=weights)
        self.n_train_ = len(df)

        # Per-position residual scale (games-frac units) for season simulation.
        # In-sample — a scale estimate, not a performance claim.
        resid = y - self._pipe.predict(X)
        for pos in POSITIONS:
            mask = df["position"] == pos if "position" in df.columns else np.zeros(len(df), bool)
            if mask.sum() >= 20:
                self.residual_std_[pos] = float(np.std(resid[np.asarray(mask)]))
        self.residual_std_["overall"] = float(np.std(resid))

        print(f"  Availability model: n={self.n_train_}, "
              f"resid std={self.residual_std_['overall']:.3f} (games frac)")
        return self

    # ------------------------------------------------------------------

    def predict_games_frac(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predicted games FRACTION for each row of features_df."""
        if self._pipe is None:
            raise RuntimeError("AvailabilityModel not trained.")
        df = _engineer(features_df)
        raw = self._pipe.predict(df[_FEATURES].values)
        clipped = np.clip(raw, AVAILABILITY_MIN_GAMES_FRAC, AVAILABILITY_MAX_GAMES_FRAC)
        # _engineer sorts rows — restore the caller's order
        return pd.Series(clipped, index=df.index).reindex(features_df.index).values

    def predict_games(self, features_df: pd.DataFrame, target_season: int) -> np.ndarray:
        """Expected games played in target_season for each row of features_df."""
        return self.predict_games_frac(features_df) * season_length(target_season)

    # ------------------------------------------------------------------

    def attach_to_projections(
        self,
        projections: pd.DataFrame,
        features_df: pd.DataFrame,
        target_season: int,
    ) -> pd.DataFrame:
        """
        Replace the flat 17-game assumption in a projections frame with
        model-expected games, and recompute season totals.

        Players missing from features_df keep their existing projected_games.
        """
        if "player_id" not in projections.columns or "player_id" not in features_df.columns:
            warnings.warn("attach_to_projections: missing player_id; returning unchanged.")
            return projections

        games = pd.DataFrame({
            "player_id": features_df["player_id"].values,
            "expected_games": self.predict_games(features_df, target_season),
        }).drop_duplicates(subset=["player_id"])

        out = projections.merge(games, on="player_id", how="left")
        out["projected_games"] = out["expected_games"].fillna(out["projected_games"])
        out = out.drop(columns=["expected_games"])
        out["projected_fpts_season"] = out["projected_fpts_pg"] * out["projected_games"]

        # Rescale interval columns to the new games counts if present
        for col in ["confidence_interval_low", "confidence_interval_high"]:
            if col in out.columns:
                out[f"{col}_season"] = out[col] * out["projected_games"]

        return out.sort_values("projected_fpts_season", ascending=False).reset_index(drop=True)
