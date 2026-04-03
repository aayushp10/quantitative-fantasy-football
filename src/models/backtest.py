"""
Rolling multi-season backtesting framework.

Walk-forward validation: train on all seasons before the test season,
evaluate on the held-out test season. Repeating this across multiple
test seasons gives a much more reliable estimate of model quality than
a single held-out season.

Default test seasons: [2021, 2022, 2023] — each has 4+ prior training
years with the expanded 2017-2024 training window.

Public API
----------
rolling_backtest(model_class, yoy_df, test_seasons, ...) -> pd.DataFrame
    Walk-forward validation for a single model class.

compare_models(yoy_df, test_seasons, models_dict) -> pd.DataFrame
    Run rolling_backtest for multiple model configurations and return
    a comparison table with model_name, test_season, position, metrics.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------------

def rolling_backtest(
    model_class: type,
    yoy_df: pd.DataFrame,
    test_seasons: list[int] | None = None,
    target: str = "next_fpts",
    positions: list[str] | None = None,
    **model_kwargs: Any,
) -> pd.DataFrame:
    """
    Walk-forward backtest across multiple held-out seasons.

    For each test_season:
      - train on yoy_df[season < test_season]
      - evaluate on yoy_df[season == test_season]

    Parameters
    ----------
    model_class : type
        A model class with .train(yoy_df, target, fit_age) and
        .project(features_df, season) or .backtest(yoy_df, test_season) methods.
        Compatible with FantasyProjectionModel, TwoStageProjectionModel,
        and HybridProjectionModel.
    yoy_df : pd.DataFrame
        Full YoY pairs dataset.
    test_seasons : list[int] | None
        Seasons to hold out. If None, defaults to [2021, 2022, 2023]
        (suitable for the 2017-2024 training window, each having 4+ training years).
        Seasons without enough training data are automatically skipped.
    target : str
        Column name for the actual target (default 'next_fpts').
    positions : list[str] | None
        Positions to evaluate. Default: all positions present in yoy_df.
    **model_kwargs
        Passed to model_class constructor (e.g. age_adjust=False, blend_weight=0.5).

    Returns
    -------
    pd.DataFrame
        Columns: test_season, position, mae, r2, rank_corr, n.
        Includes rows for each (test_season, position), each (test_season, 'overall'),
        and summary rows ('average', position) and ('average', 'overall').
    """
    if test_seasons is None:
        test_seasons = [2021, 2022, 2023]

    available_seasons = sorted(yoy_df["season"].unique())
    if positions is None:
        positions = sorted(yoy_df["position"].unique()) if "position" in yoy_df.columns else []

    rows = []

    for test_season in test_seasons:
        train_df = yoy_df[yoy_df["season"] < test_season].copy()
        test_df = yoy_df[yoy_df["season"] == test_season].copy()

        if train_df.empty:
            warnings.warn(f"No training data before {test_season}. Skipping.")
            continue
        if test_df.empty:
            warnings.warn(f"No test data for {test_season}. Skipping.")
            continue
        if len(train_df["season"].unique()) < 2:
            warnings.warn(f"Only 1 training season before {test_season}. Skipping.")
            continue

        # Instantiate and train
        model = model_class(**model_kwargs)
        try:
            model.train(train_df, target=target, fit_age=False)
        except Exception as e:
            warnings.warn(f"Training failed for test_season={test_season}: {e}")
            continue

        # Generate predictions by position
        all_pred, all_actual = [], []

        for pos in positions:
            pos_test = test_df[test_df["position"] == pos].copy().reset_index(drop=True) \
                if "position" in test_df.columns else test_df.copy().reset_index(drop=True)
            if pos_test.empty:
                continue

            y_actual = pos_test[target].values if target in pos_test.columns else None
            if y_actual is None:
                continue

            # Get predictions: use project() if available, else use backtest internals
            y_pred = _predict_for_backtest(model, model_class, pos_test, pos, target)
            if y_pred is None:
                continue

            valid = ~np.isnan(y_actual) & ~np.isnan(y_pred)
            if valid.sum() < 3:
                continue

            mae = mean_absolute_error(y_actual[valid], y_pred[valid])
            r2 = r2_score(y_actual[valid], y_pred[valid])
            rank_corr, _ = spearmanr(y_actual[valid], y_pred[valid])

            rows.append({
                "test_season": test_season,
                "position": pos,
                "mae": round(float(mae), 3),
                "r2": round(float(r2), 3),
                "rank_corr": round(float(rank_corr), 3),
                "n": int(valid.sum()),
            })

            all_pred.extend(y_pred[valid].tolist())
            all_actual.extend(y_actual[valid].tolist())

        # Overall row for this test season
        if all_pred:
            overall_mae = mean_absolute_error(all_actual, all_pred)
            overall_r2 = r2_score(all_actual, all_pred)
            overall_rc, _ = spearmanr(all_actual, all_pred)
            rows.append({
                "test_season": test_season,
                "position": "overall",
                "mae": round(float(overall_mae), 3),
                "r2": round(float(overall_r2), 3),
                "rank_corr": round(float(overall_rc), 3),
                "n": len(all_pred),
            })

    if not rows:
        return pd.DataFrame(columns=["test_season", "position", "mae", "r2", "rank_corr", "n"])

    result = pd.DataFrame(rows)

    # Add average rows across test seasons
    avg_rows = []
    for pos in result["position"].unique():
        pos_data = result[result["position"] == pos]
        avg_rows.append({
            "test_season": "average",
            "position": pos,
            "mae": round(float(pos_data["mae"].mean()), 3),
            "r2": round(float(pos_data["r2"].mean()), 3),
            "rank_corr": round(float(pos_data["rank_corr"].mean()), 3),
            "n": int(pos_data["n"].sum()),
        })

    result = pd.concat([result, pd.DataFrame(avg_rows)], ignore_index=True)
    return result


def _predict_for_backtest(
    model: Any,
    model_class: type,
    pos_test: pd.DataFrame,
    pos: str,
    target: str,
) -> np.ndarray | None:
    """
    Extract position predictions from a trained model.

    Uses the model's internal predict path without calling project()
    to avoid needing a full season snapshot.
    """
    from config import POSITION_FEATURES, POSITIONS

    # FantasyProjectionModel path
    if hasattr(model, "_models") and pos in getattr(model, "_models", {}):
        features = [f for f in POSITION_FEATURES.get(pos, []) if f in pos_test.columns]
        if not features:
            return None
        try:
            return model._models[pos].predict(pos_test[features].values)
        except Exception:
            return None

    # TwoStageProjectionModel path
    if hasattr(model, "_volume_models") and hasattr(model, "_regressed_efficiency"):
        try:
            n = len(pos_test)
            vol_preds = model._predict_volume(pos, pos_test)
            eff_preds = model._regressed_efficiency(pos, pos_test)
            if not vol_preds and not eff_preds:
                return None
            catch_rate = model._get_catch_rate(pos, pos_test)
            return model._combine_to_fpts(pos, vol_preds, eff_preds, catch_rate, n)
        except Exception:
            return None

    # HybridProjectionModel path — delegate to internal models
    if hasattr(model, "_single") and hasattr(model, "_two_stage"):
        try:
            single_pred = _predict_for_backtest(model._single, type(model._single), pos_test, pos, target)
            two_pred = _predict_for_backtest(model._two_stage, type(model._two_stage), pos_test, pos, target)
            if single_pred is None and two_pred is None:
                return None
            if single_pred is None:
                return two_pred
            if two_pred is None:
                return single_pred
            w = model.blend_weight
            return w * single_pred + (1 - w) * two_pred
        except Exception:
            return None

    return None


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    yoy_df: pd.DataFrame,
    test_seasons: list[int],
    models_dict: dict[str, tuple[type, dict]],
    target: str = "next_fpts",
) -> pd.DataFrame:
    """
    Run rolling_backtest for multiple model configurations.

    Parameters
    ----------
    yoy_df : pd.DataFrame
        Full YoY pairs dataset.
    test_seasons : list[int]
        Seasons to hold out (passed to rolling_backtest).
    models_dict : dict
        {model_name: (model_class, model_kwargs)}
        Example:
            {
                "single_stage": (FantasyProjectionModel, {"age_adjust": False}),
                "two_stage":    (TwoStageProjectionModel, {"age_adjust": False}),
                "hybrid_55":    (HybridProjectionModel, {"blend_weight": 0.55}),
            }
    target : str
        Target column name.

    Returns
    -------
    pd.DataFrame
        Columns: model_name, test_season, position, mae, r2, rank_corr, n.
    """
    frames = []
    for model_name, (model_class, model_kwargs) in models_dict.items():
        print(f"\n=== Rolling backtest: {model_name} ===")
        try:
            result = rolling_backtest(
                model_class=model_class,
                yoy_df=yoy_df,
                test_seasons=test_seasons,
                target=target,
                **model_kwargs,
            )
            result.insert(0, "model_name", model_name)
            frames.append(result)
        except Exception as e:
            warnings.warn(f"Rolling backtest failed for {model_name}: {e}")

    if not frames:
        return pd.DataFrame(columns=["model_name", "test_season", "position",
                                     "mae", "r2", "rank_corr", "n"])

    return pd.concat(frames, ignore_index=True)
