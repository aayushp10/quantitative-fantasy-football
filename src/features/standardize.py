"""
BARRA-style per-season cross-sectional factor standardization.

Each continuous factor is z-scored WITHIN its season cross-section:
    z = (x - mean_season) / std_season

Why: factor levels drift across eras (league pass rate, pace, 16 -> 17 game
schedules, EPA model recalibrations). Standardizing within each season makes a
2012 target share directly comparable to a 2024 target share, which is what
lets the training window extend back to 2012 without crushing old seasons via
aggressive recency decay.

Behavior details:
- Near-binary columns (<= 2 distinct values at fit time, e.g. team_changed,
  sophomore_flag) are recorded at fit and left raw.
- NaN inputs stay NaN (downstream SimpleImputer handles them; the median of a
  z-scored column is ~0, i.e. "average player", which is the right imputation).
- A column constant within a season maps to 0 for its non-null entries.
- transform() computes stats on the frame it is GIVEN — cross-sectional by
  construction, so a projection-season slice is standardized against its own
  cross-section exactly like each training season was.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonStandardizer:
    """
    Records which columns to standardize (and which to skip as binary) at fit
    time; z-scores within season groups at transform time.
    """

    def __init__(self, season_col: str = "season"):
        self.season_col = season_col
        self.cols_: list[str] = []
        self.binary_cols_: set[str] = set()

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "SeasonStandardizer":
        self.cols_ = [c for c in cols if c in df.columns]
        self.binary_cols_ = set()
        for c in self.cols_:
            if df[c].dropna().nunique() <= 2:
                self.binary_cols_.add(c)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        z_cols = [
            c for c in self.cols_
            if c not in self.binary_cols_ and c in out.columns
        ]
        if not z_cols:
            return out

        if self.season_col in out.columns:
            grouped = out.groupby(self.season_col, observed=True)[z_cols]
            mean = grouped.transform("mean")
            std = grouped.transform("std")
        else:
            # No season column: treat the whole frame as one cross-section
            mean = pd.DataFrame(
                np.tile(out[z_cols].mean().values, (len(out), 1)),
                index=out.index, columns=z_cols,
            )
            std = pd.DataFrame(
                np.tile(out[z_cols].std().values, (len(out), 1)),
                index=out.index, columns=z_cols,
            )

        centered = out[z_cols] - mean
        z = centered / std.where(std > 0)
        # Constant-within-season (std 0 or single row): non-null values -> 0
        degenerate = (std == 0) | std.isna()
        z = z.mask(degenerate & centered.notna(), 0.0)
        out[z_cols] = z
        return out

    def fit_transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return self.fit(df, cols).transform(df)
