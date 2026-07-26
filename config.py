"""
Central configuration for the Fantasy Football Factor Model.
All constants live here — every other module imports from this file.
"""
from pathlib import Path
from typing import TypedDict


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

class ScoringWeights(TypedDict):
    pass_yd: float
    pass_td: float
    interception: float
    rush_yd: float
    rush_td: float
    reception: float
    rec_yd: float
    rec_td: float
    fumble_lost: float
    pass_2pt: float
    rush_2pt: float
    rec_2pt: float


PPR_SCORING: ScoringWeights = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "interception": -2.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "reception": 1.0,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "fumble_lost": -2.0,
    "pass_2pt": 2.0,
    "rush_2pt": 2.0,
    "rec_2pt": 2.0,
}

HALF_PPR_SCORING: ScoringWeights = {
    **PPR_SCORING,
    "reception": 0.5,
}

STANDARD_SCORING: ScoringWeights = {
    **PPR_SCORING,
    "reception": 0.0,
}

# ---------------------------------------------------------------------------
# Season configuration
# ---------------------------------------------------------------------------

TRAINING_SEASONS: list[int] = list(range(2012, 2026))  # 2012–2025 inclusive
PROJECTION_SEASON: int = 2026


def season_length(season: int) -> int:
    """Regular-season games: 16 through 2020, 17 from 2021 onward."""
    return 17 if season >= 2021 else 16

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Feature matrix schema version — bump when adding/removing columns to auto-invalidate cache
_FEATURE_VERSION: str = "v4"

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output" / "projections"

# Ensure cache and output dirs exist at import time
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

POSITIONS: list[str] = ["QB", "RB", "WR", "TE"]

# ---------------------------------------------------------------------------
# League roster configurations (for VOR calculation)
# ---------------------------------------------------------------------------
# Format: {position: starters_count}. FLEX is counted separately.

ROSTER_SPOTS: dict[str, dict] = {
    "10team": {
        "league_size": 10,
        "QB": 1,
        "RB": 2,
        "WR": 2,
        "TE": 1,
        "FLEX": 1,   # RB/WR/TE eligible
        "K": 1,
        "DST": 1,
        "bench": 6,
    },
    "12team": {
        "league_size": 12,
        "QB": 1,
        "RB": 2,
        "WR": 2,
        "TE": 1,
        "FLEX": 1,
        "K": 1,
        "DST": 1,
        "bench": 6,
    },
    "14team": {
        "league_size": 14,
        "QB": 1,
        "RB": 2,
        "WR": 3,
        "TE": 1,
        "FLEX": 1,
        "K": 1,
        "DST": 1,
        "bench": 6,
    },
    "12team_superflex": {
        "league_size": 12,
        "QB": 1,
        "RB": 2,
        "WR": 2,
        "TE": 1,
        "FLEX": 1,
        "SUPERFLEX": 1,  # QB/RB/WR/TE eligible
        "K": 1,
        "DST": 1,
        "bench": 6,
    },
}

# ---------------------------------------------------------------------------
# Feature engineering thresholds
# ---------------------------------------------------------------------------

# Minimum play thresholds for including a player in the feature matrix
MIN_DROPBACKS_QB: int = 100
MIN_TOUCHES_RB: int = 50   # carries + targets combined
MIN_TARGETS_WR: int = 30
MIN_TARGETS_TE: int = 30

# Trend detection: late-season window
TREND_WEEK_START: int = 13
MIN_GAMES_FOR_TREND: int = 4

# Trend classification thresholds (delta in target_share or rush_share)
BREAKOUT_THRESHOLD: float = 0.03
DECLINING_THRESHOLD: float = -0.03

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Exponential recency decay base: weight = RECENCY_DECAY^(seasons_ago).
# Relaxed from 0.7 → 0.85: era drift is now handled by per-season cross-sectional
# z-scoring (STANDARDIZE_BY_SEASON), so old seasons no longer need to be crushed.
RECENCY_DECAY: float = 0.85

# Ridge alpha candidates for grid search
RIDGE_ALPHA_GRID: list[float] = [0.1, 1.0, 10.0, 100.0, 1000.0]

# Cross-validation: walk-forward by SEASON GROUPS (never split within a season —
# rows in the same season share league-wide environment and would leak).
# A validation fold is a single season; training folds are all strictly earlier seasons.
MIN_TRAIN_SEASONS_CV: int = 3

# BARRA-style factor standardization: z-score each continuous factor within its
# season cross-section before fitting. Removes era drift (pass-rate inflation,
# 16→17 game schedules) without discarding old sample.
STANDARDIZE_BY_SEASON: bool = True

# Legacy TimeSeriesSplit parameters (deprecated — kept for backward compat imports)
CV_N_SPLITS: int = 4
CV_GAP: int = 1

# Position-specific projection caps (season total fantasy points)
PROJECTION_CAPS: dict[str, float] = {
    "QB": 450.0,
    "RB": 350.0,
    "WR": 350.0,
    "TE": 280.0,
}

# Feature sets used in the Ridge model, per position
POSITION_FEATURES: dict[str, list[str]] = {
    "QB": [
        "epa_per_dropback",
        "cpoe",
        "deep_ball_rate",
        "td_rate",
        "int_rate",
        "rush_attempt_share",
        "team_pace",
        "sack_rate",
        "games_played",
        # Trend factors
        "target_share_delta",
        # Situation / pedigree / consistency
        "team_changed",
        "years_in_league",
        "weekly_fpts_cv",
        # Expected-TD geometry (v4)
        "x_pass_td_rate",
        "pass_td_oe",
        # Career-to-date (v5)
        "seasons_played_todate",
        "career_fpts_pg",
        "peak_fpts_pg",
        "fpts_pg_prev",
        "fpts_pg_yoy_change",
        "durability_todate",
        # Forward-looking schedule context (v5)
        "vegas_implied_pts_next",
        "vegas_implied_delta_next",
        "hc_changed_next",
    ],
    "RB": [
        "rush_share",
        "target_share",
        "rz_rush_share",
        "rz_target_share",
        "epa_per_carry",
        "rush_success_rate",
        "ypc",
        "catch_rate",
        "explosive_run_rate",
        "snap_percentage",
        "games_played",
        "team_pace",
        # Trend factors
        "rush_share_delta",
        "target_share_delta",
        "snap_trend",
        # Situation / pedigree / consistency
        "team_changed",
        "context_delta_pace",
        "draft_round_bucket",
        "years_in_league",
        "sophomore_flag",
        "weekly_fpts_cv",
        "boom_rate",
        # QB coupling + vacated shares
        "qb_epa_per_dropback",
        "team_vacated_carry_share",
        "team_vacated_target_share",
        "top_departed_target_share",
        # Expected-TD geometry + routes (v4)
        "x_rush_td_rate",
        "rush_td_oe",
        "x_rec_td_rate",
        "rec_td_oe",
        "tprr",
        "route_participation",
        # Career-to-date (v5)
        "seasons_played_todate",
        "career_fpts_pg",
        "peak_fpts_pg",
        "fpts_pg_prev",
        "peak_rush_share",
        "peak_target_share",
        "rush_share_prev",
        "fpts_pg_yoy_change",
        "durability_todate",
        # Forward-looking schedule context (v5)
        "vegas_implied_pts_next",
        "vegas_implied_delta_next",
        "hc_changed_next",
    ],
    "WR": [
        "target_share",
        "air_yard_share",
        "wopr",
        "rz_target_share",
        "catch_rate",
        "avg_depth_of_target",
        "epa_per_target",
        "yac_per_rec",
        "explosive_play_rate",
        "snap_percentage",
        "games_played",
        "team_pass_rate",
        "team_pace",
        # Trend factors
        "target_share_delta",
        "wopr_delta",
        "snap_trend",
        # Situation / pedigree / consistency
        "team_changed",
        "context_delta_pace",
        "draft_round_bucket",
        "years_in_league",
        "sophomore_flag",
        "weekly_fpts_cv",
        "consistency_score",
        # QB coupling + vacated shares + top-down
        "qb_epa_per_dropback",
        "qb_cpoe",
        "qb_changed",
        "team_vacated_target_share",
        "top_departed_target_share",
        "topdown_fpts_pg",
        # Expected-TD geometry + routes (v4)
        "x_rec_td_rate",
        "rec_td_oe",
        "tprr",
        "route_participation",
        # Career-to-date (v5)
        "seasons_played_todate",
        "career_fpts_pg",
        "peak_fpts_pg",
        "fpts_pg_prev",
        "peak_target_share",
        "target_share_prev",
        "fpts_pg_yoy_change",
        "durability_todate",
        # Forward-looking schedule context (v5)
        "vegas_implied_pts_next",
        "vegas_implied_delta_next",
        "hc_changed_next",
    ],
    "TE": [
        "target_share",
        "air_yard_share",
        "rz_target_share",
        "catch_rate",
        "epa_per_target",
        "yac_per_rec",
        "snap_percentage",
        "games_played",
        "team_pass_rate",
        "team_pace",
        # Trend factors
        "target_share_delta",
        "snap_trend",
        # Situation / pedigree / consistency
        "team_changed",
        "context_delta_pace",
        "years_in_league",
        "weekly_fpts_cv",
        # QB coupling + vacated shares + top-down
        "qb_epa_per_dropback",
        "qb_changed",
        "team_vacated_target_share",
        "top_departed_target_share",
        "topdown_fpts_pg",
        # Expected-TD geometry + routes (v4)
        "x_rec_td_rate",
        "rec_td_oe",
        "tprr",
        "route_participation",
        # Career-to-date (v5)
        "seasons_played_todate",
        "career_fpts_pg",
        "peak_fpts_pg",
        "fpts_pg_prev",
        "peak_target_share",
        "target_share_prev",
        "fpts_pg_yoy_change",
        "durability_todate",
        # Forward-looking schedule context (v5)
        "vegas_implied_pts_next",
        "vegas_implied_delta_next",
        "hc_changed_next",
    ],
}

# ---------------------------------------------------------------------------
# Two-stage volume × efficiency model configuration
# ---------------------------------------------------------------------------

# Separate alpha grids: volume signals are persistent (low alpha), efficiency noisy (high alpha)
VOLUME_RIDGE_ALPHA_GRID: list[float] = [0.1, 1.0, 10.0]
EFFICIENCY_RIDGE_ALPHA_GRID: list[float] = [10.0, 100.0, 1000.0]

# --- Legacy fixed shrinkage weights -----------------------------------------
# These are now FALLBACKS only. The two-stage model fits empirical-Bayes priors
# (models/shrinkage.py) whose shrinkage is sample-size dependent; the constants
# below are used when EB fitting fails (missing trials columns, tiny samples).

# TD rate mean reversion weight (0 = pure model, 1 = pure positional mean)
TD_REGRESSION_WEIGHT: float = 0.55

# Catch rate mean reversion weight for combination stage
CATCH_RATE_REGRESSION_WEIGHT: float = 0.30

# Per-metric regression weights for the two-stage regressed efficiency approach
# Higher weight = stronger pull toward positional mean (for noisier metrics)
EFFICIENCY_REGRESSION_WEIGHTS: dict[str, float] = {
    "yards_per_target":       0.30,
    "ypc":                    0.30,
    "pass_yards_per_attempt": 0.25,
    "rec_td_rate":            0.55,
    "rush_td_rate":           0.55,
    "pass_td_rate":           0.50,
}

# Default blend weight for HybridProjectionModel (single-stage share)
DEFAULT_BLEND_WEIGHT: float = 0.55

# Volume stage feature sets (predicts per-game opportunity volume)
VOLUME_FEATURES: dict[str, list[str]] = {
    "QB": [
        "epa_per_dropback", "cpoe", "team_pace", "rush_attempt_share",
        "games_played", "team_changed", "context_delta_pace", "target_share_delta",
        "seasons_played_todate", "durability_todate",
        "vegas_implied_pts_next", "vegas_implied_delta_next", "hc_changed_next",
    ],
    "RB": [
        "rush_share", "target_share", "snap_percentage", "team_pace",
        "team_changed", "context_delta_pace", "rush_share_delta",
        "target_share_delta", "snap_trend", "games_played",
        "draft_round_bucket", "years_in_league",
        "qb_epa_per_dropback", "team_vacated_carry_share", "team_vacated_target_share",
        "peak_rush_share", "rush_share_prev", "seasons_played_todate", "durability_todate",
        "vegas_implied_pts_next", "vegas_implied_delta_next", "hc_changed_next",
    ],
    "WR": [
        "target_share", "air_yard_share", "wopr", "snap_percentage",
        "team_pace", "team_pass_rate", "team_changed", "context_delta_pace",
        "target_share_delta", "wopr_delta", "snap_trend", "games_played",
        "draft_round_bucket", "years_in_league",
        "qb_epa_per_dropback", "qb_changed", "team_vacated_target_share", "top_departed_target_share",
        "tprr", "route_participation",
        "peak_target_share", "target_share_prev", "seasons_played_todate", "durability_todate",
        "vegas_implied_pts_next", "vegas_implied_delta_next", "hc_changed_next",
    ],
    "TE": [
        "target_share", "air_yard_share", "snap_percentage", "team_pace",
        "team_pass_rate", "team_changed", "context_delta_pace",
        "target_share_delta", "snap_trend", "games_played",
        "draft_round_bucket", "years_in_league",
        "qb_epa_per_dropback", "qb_changed", "team_vacated_target_share",
        "tprr", "route_participation",
        "peak_target_share", "target_share_prev", "seasons_played_todate", "durability_todate",
        "vegas_implied_pts_next", "vegas_implied_delta_next", "hc_changed_next",
    ],
}

# Efficiency stage feature sets (predicts per-play output rates)
EFFICIENCY_FEATURES: dict[str, list[str]] = {
    "QB":  ["epa_per_dropback", "cpoe", "deep_ball_rate", "sack_rate", "int_rate"],
    "RB":  ["epa_per_carry", "rush_success_rate", "ypc", "explosive_run_rate",
            "stuff_rate", "catch_rate"],
    "WR":  ["epa_per_target", "catch_rate", "avg_depth_of_target",
            "yac_per_rec", "explosive_play_rate"],
    "TE":  ["epa_per_target", "catch_rate", "yac_per_rec"],
}

# ---------------------------------------------------------------------------
# Aging curve priors (hardcoded; fit_age_curves() refines with data)
# ---------------------------------------------------------------------------

PEAK_AGES: dict[str, int] = {"QB": 28, "RB": 24, "WR": 26, "TE": 27}

AGE_DECAY_RATES: dict[str, float] = {
    "QB": 0.015,
    "RB": 0.025,
    "WR": 0.018,
    "TE": 0.016,
}

# ---------------------------------------------------------------------------
# Empirical-Bayes shrinkage (models/shrinkage.py)
# ---------------------------------------------------------------------------

# Minimum trials (targets/carries/dropbacks) for a player-season to inform priors
EB_MIN_TRIALS: int = 10

# Bounds on the fitted prior strength (equivalent prior sample size).
# Prevents degenerate fits: k below → almost no shrinkage; k above → everyone
# collapses to the positional mean.
EB_PRIOR_STRENGTH_BOUNDS: tuple[float, float] = (10.0, 5000.0)

# Trials column in the feature matrix backing each rate metric
EB_TRIALS_COLUMNS: dict[str, str] = {
    "rec_td_rate":            "targets",
    "rush_td_rate":           "carries",
    "pass_td_rate":           "dropbacks",
    "catch_rate":             "targets",
    "yards_per_target":       "targets",
    "ypc":                    "carries",
    "pass_yards_per_attempt": "dropbacks",
}

# Metrics treated as binomial successes/trials (beta-binomial prior);
# the rest use a normal-normal hierarchical prior.
EB_BINOMIAL_METRICS: set[str] = {
    "rec_td_rate", "rush_td_rate", "pass_td_rate", "catch_rate",
}

# Player-specific prior means from usage geometry (expected-TD model):
# when the x-column exists, the EB prior centers on the player's own
# field-position-implied rate instead of the flat positional mean.
EB_GEOMETRY_PRIORS: dict[str, str] = {
    "rec_td_rate":  "x_rec_td_rate",
    "rush_td_rate": "x_rush_td_rate",
    "pass_td_rate": "x_pass_td_rate",
}

# ---------------------------------------------------------------------------
# Availability (games played) model — models/availability.py
# ---------------------------------------------------------------------------

# Clip projected games to this fraction range of the season length
AVAILABILITY_MIN_GAMES_FRAC: float = 0.25
AVAILABILITY_MAX_GAMES_FRAC: float = 1.0

# ---------------------------------------------------------------------------
# Market data (ADP) — data/adp.py, models/market.py
# ---------------------------------------------------------------------------

ADP_FORMAT: str = "ppr"          # ppr | half-ppr | standard | 2qb
ADP_TEAMS: int = 12
ADP_SOURCE_URL: str = "https://fantasyfootballcalculator.com/api/v1/adp/{fmt}?teams={teams}&year={year}"

# Long/short residual-vs-market test: fraction of each position taken per side
MARKET_LS_TOP_FRAC: float = 0.2

# ---------------------------------------------------------------------------
# Rookie model — models/rookie.py
# ---------------------------------------------------------------------------

# First draft class with reliable supporting data in the pipeline
ROOKIE_TRAIN_START: int = 2013

# Rookies below this draft position get floored volume expectations anyway;
# beyond pick ~260 is UDFA territory (excluded)
ROOKIE_MAX_DRAFT_PICK: int = 262
