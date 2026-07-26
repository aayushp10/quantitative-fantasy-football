"""
Season-grouped walk-forward cross-validation.

The previous TimeSeriesSplit-on-rows approach cut folds MID-SEASON: train and
validation folds shared the same season's cross-section, and season effects
(league-wide pass-rate drift, scoring environment, 16 vs 17 game schedules)
are strongly shared within a season — so the alpha grid search was tuned on
leaky folds. gap=1 gapped one ROW, not one season, and did nothing.

Here a fold is: validation = one entire season, train = ALL strictly earlier
seasons. Because the target is next-season output, a training row from season
S has its target realized in S+1 <= validation feature season, so no future
information crosses the boundary and no embargo is needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def season_walk_forward_folds(
    seasons: np.ndarray | pd.Series | list[int],
    min_train_seasons: int = 3,
    min_fold_size: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build walk-forward (train_idx, val_idx) folds grouped by season.

    Parameters
    ----------
    seasons : array-like of int
        Per-row season labels, aligned with the X matrix rows.
    min_train_seasons : int
        A season only becomes a validation fold once at least this many
        earlier seasons exist to train on.
    min_fold_size : int
        Skip folds whose train or validation side has fewer rows than this.

    Returns
    -------
    list of (train_idx, val_idx)
        Positional index arrays, directly usable as the `cv` argument of
        GridSearchCV. Empty list if no valid fold exists.
    """
    s = np.asarray(seasons)
    unique_seasons = np.sort(np.unique(s))

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i, val_season in enumerate(unique_seasons):
        train_seasons = unique_seasons[:i]
        if len(train_seasons) < min_train_seasons:
            continue
        train_idx = np.flatnonzero(np.isin(s, train_seasons))
        val_idx = np.flatnonzero(s == val_season)
        if len(train_idx) < min_fold_size or len(val_idx) < min_fold_size:
            continue
        folds.append((train_idx, val_idx))
    return folds
