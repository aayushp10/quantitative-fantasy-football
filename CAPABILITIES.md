# Capabilities

What this codebase can currently do, end to end. Setup and run order live in
[README.md](README.md); this is the capability inventory. Current state: trains
on 2012–2025, projects the **2026** season, and evaluates itself against the
live 2026 draft market.

---

## 1. Data acquisition (`src/data/`)

All loaders are cache-through: first call downloads and writes per-season
parquet to `data/cache/`, later calls read locally.

| Data | Coverage | Notes |
|---|---|---|
| Play-by-play | 1999+ (2012–2025 in use) | Season-by-season download, resumable; ~380 columns incl. EPA/CPOE/win prob |
| Weekly player stats | 2012–2025 | Target variable source (`fantasy_points_ppr`) |
| Seasonal rosters | 2012–2025 | Position, age, draft capital; old-era string dtypes coerced |
| Snap counts | 2012–2025 | PFR→GSIS id bridge built in |
| Participation (routes) | ~2016–2023 | Read directly from nflverse releases; snap-share proxy fills the gap |
| Draft picks | 1980–2026 | Full history incl. the 2026 class |
| Schedules, injuries, NGS | best effort | Wrapped defensively |
| **ADP (market prices)** | 2013–2024 archive + live current year | FantasyFootballCalculator API, free/keyless; PPR/half/standard, 10–14 team |

**Post-archive resilience**: `nfl_data_py` (archived Sep 2025) 404s on 2025+
assets; every loader falls back to reading nflverse release parquets directly,
including the renamed weekly-stats release (`loader.py::_download_season`).

## 2. Feature engineering (`src/features/`)

One row per (player, season): ~150 columns, ~4,100 player-seasons at current
thresholds (QB ≥100 dropbacks, RB ≥50 touches, WR/TE ≥30 targets).

- **Opportunity**: target/rush/air-yard shares, WOPR, red-zone & end-zone
  shares, goal-line/two-minute/designed carry shares, per-game volumes
- **Efficiency**: EPA per dropback/target/carry, CPOE, success rates, YPC,
  catch rate, aDOT, YAC, deep-ball and explosive-play rates
- **Expected TDs from usage geometry** (`expected_td.py`): league conversion
  by field-position bucket × the player's usage mix → `x_*_td_rate`
  (persistent TD equity) and `*_td_oe` (luck component that mean-reverts)
- **Routes/TPRR** (`routes.py`): routes run, route participation, targets per
  route run — participation-based with snap-share fallback
- **Context**: team pace/pass rate/offensive EPA, garbage-time share,
  neutral-script EPA, games played, snap %
- **Trend**: late-season (wk 13+) vs early-season role deltas →
  BREAKOUT / STABLE / DECLINING
- **Situation**: team-change detection with destination-team context swap
- **QB coupling**: primary-QB quality attached to every pass catcher,
  `qb_changed`, upgrade delta
- **Vacated shares**: departed teammates' target/carry share on the player's
  next-season roster
- **Pedigree**: draft capital buckets, years in league, sophomore flag
- **Consistency**: weekly CV, boom/bust rates, composite score
- **Top-down team constraint**: mean-reverted team passing projection
  distributed by shares → `topdown_fpts_pg`
- **Career-to-date memory** (`career.py`, v5): games-weighted career PPG,
  peak season PPG / target share / rush share, strict previous-season lags
  (NaN across missed seasons), YoY change, durability, seasons played —
  expanding windows over seasons ≤ N, recomputed on every matrix load (no
  cache invalidation). Standalone ICs: career_fpts_pg 0.66 (100% sign-
  consistent), peak_fpts_pg 0.61, fpts_pg_prev 0.60
- **Forward-looking schedule context** (`schedule_context.py`, v5): Vegas
  week-1 implied points / total for the season being predicted (from
  nflverse schedule lines; mean IC +0.11, positive 92% of seasons) and
  head-coach-change flags (consistently negative IC — regime change
  predicts decline). Individually predictive; largely redundant with the
  market ensemble's ADP prior at the model level (kept for the pure-model
  edge view). 2026 opener lines and coaches fully covered
- **Per-season cross-sectional z-scoring** (`standardize.py`): BARRA-style;
  removes era drift, makes 2012 comparable to 2025, preserves NaN and binaries

`build_yoy_pairs` produces (season N features → season N+1 target) rows and can
shift any extra column forward (rate targets, games played).

## 3. Projection models (`src/models/`)

All three share one API (`train / project / backtest / predict_position /
feature_importance / set_uncertainty`):

- **Single-stage Ridge** (`projection.py`): per-position, season-grouped
  walk-forward CV (never splits within a season), recency-weighted
  (0.85^years), alpha grid-searched, intercept exposed in importances
- **Two-stage volume × efficiency** (`two_stage.py`): low-alpha Ridge predicts
  volume (targets/carries/dropbacks per game); efficiency uses
  **empirical-Bayes shrinkage** (`shrinkage.py`) — `(k·prior + n·obs)/(k+n)`
  with k fitted per (position, metric), so a 40-target rookie shrinks harder
  than a 150-target veteran; TD priors center on each player's geometry-implied
  x-rate; recombined via the PPR scoring formula
- **Hybrid blend** (`hybrid.py`): weighted blend (default 0.55/0.45) with
  grid-search blend-weight optimization over held-out seasons
- **Market ensemble** (`market_ensemble.py`, v5): the headline model.
  `w·isotonic(ADP) + (1−w)·hybrid` per position — a recency-weighted
  monotone-decreasing map from ADP to next-season PPG, fitted on training
  pairs only, blended with per-position weights (QB 0.80, RB 0.65, WR/TE
  0.50; every w in [0.5, 0.8] beat the pure model). Order-preserving:
  disagreements with the market shrink but never flip sign, so the L/S
  alpha survives. Players without an ADP fall back to the pure hybrid.
  2020–2023 walk-forward: IC edge vs ADP positive at ALL positions
  (was −0.10 mean), L/S spreads intact or better (RB +20.8, WR +26.0)
- **GBM benchmark** (`gbm.py`, v5): per-position LightGBM in the identical
  harness (same features, per-season standardization, recency weights,
  tree count via temporal holdout on the latest training season).
  Competitive, not dominant: beats the hybrid at RB (MAE 2.99 vs 3.14),
  trails elsewhere; top gain feature at every position is career_fpts_pg.
  Kept as a benchmark / future blend member, not in the headline stack

Post-adjustments and add-ons:

- **Age curves via the delta method** (`age_curves.py`): fitted on
  within-player year-over-year changes — immune to retirement survivorship.
  Real-data peaks: RB 23.9 < WR 25.0 < TE 25.9 < QB 26.1, QB decay ~flat
- **Availability model** (`availability.py`): expected games from age,
  workload, durability history (era-adjusted 16/17-game seasons) — replaces
  the flat 17-game assumption
- **Rookie model** (`rookie.py`): projects draft classes from draft capital +
  landing spot (prior-season team context, QB quality, vacated shares);
  trains on all drafted players including never-played (no survivorship
  trimming); merges into the veteran board before VOR
- **Calibrated uncertainty** (`uncertainty.py`): prediction intervals from
  walk-forward out-of-sample residual quantiles with heteroscedastic scale,
  empirical coverage validation, and a season-total Monte Carlo composing
  per-game and games-played uncertainty → `season_p10/p25/p50/p75/p90`

## 4. Evaluation (`src/models/`)

- **Factor stability** (`stability.py`): per-season Spearman IC, mean IC,
  IC volatility, IC IR, **pct_positive** (sign consistency), STRONG/MODERATE/
  WEAK tiers, factor YoY persistence, heatmaps and bar charts
- **Rolling backtests** (`backtest.py`): walk-forward across multiple held-out
  seasons; MAE / R² / rank correlation per position; multi-model comparison
- **Market-relative evaluation** (`market.py`) — the alpha test:
  - ADP's own predictive power as the baseline (`market_baseline`)
  - model rank IC vs ADP rank IC per test season (`ic_edge`)
  - **long/short residual test**: do model-vs-market disagreements predict
    outcomes? (`ls_spread` > 0 consistently = real edge; ≈ 0 = the model is
    repricing consensus and you should draft by ADP)
  - **v5 state of play** (2020–2023 walk-forward averages): pure hybrid
    with career features — overall MAE 2.69, R² 0.54, IC 0.736 (v4: 2.79 /
    0.50 / 0.716); market ensemble on top — IC edge vs ADP +0.02/+0.01/
    +0.00/+0.03 (QB/RB/WR/TE) with L/S spreads +8.1/+20.8/+26.0/+5.8

## 5. Draft products (`src/models/vor.py`, `src/ranking/`)

- VORP with league-aware replacement levels (10/12/14-team, superflex, FLEX
  as best-of-overflow), across all formats at once
- K-means tiers per position (value cliffs), overall + positional rankings
- **Rankings vs market**: live-ADP comparison with `predicted_adp` (model's
  VORP rank mapped onto the market's ADP ladder) and `adp_edge` (positive =
  market lets you draft the player later than the model values them)
- CSV exports to `output/projections/` (draft board, rankings-vs-ADP)

## 6. Workflow

| Notebook | What it does |
|---|---|
| `01_data_exploration` | Pipeline sanity checks |
| `02_factor_stability` | IC analysis incl. v3/v4 factors, sign consistency, factor persistence |
| `03_model_training` | Train + backtest + feature importances |
| `04_projections` | Legacy projection export (superseded by 06) |
| `05_draft_strategy` | VOR analysis, tier board |
| `06_market_and_v4_pipeline` | **The full current pipeline**: market backtest vs ADP, EB shrinkage diagnostics, calibrated uncertainty + coverage, availability-adjusted projections, season P10/P50/P90, rookies, rankings-vs-ADP tables, board export |

Tests: 99 passing (`tests/`) — feature aggregation on synthetic PBP, scoring,
and the full v4 surface (CV folds, standardization, EB shrinkage math, xTD,
ADP joins, long/short test, availability, uncertainty calibration, routes,
rookie model, age-curve survivorship resistance).

## Known limitations (deliberate scope edges)

- **No in-season weekly model** — projections are season-long; no waiver/
  start-sit support, no opponent-adjusted weekly numbers
- **No draft simulator** — VOR/tiers are static; no pick-by-pick optimizer,
  opponent modeling, or stacking/correlation logic
- **Rookies get no availability discount** (projected at full season) and ride
  a small draft-capital model — wider error bars than veterans
- **Routes data gap**: participation ends ~2023; later seasons use the snap
  proxy
- **ADP archive gap**: FFC has no 2025 archive year; live endpoint covers the
  current draft season only
- **No props/Vegas data, no PFF grades, no coaching-change features** — the
  highest-value next data sources
- Bucket rates for xTD and the age curves are pooled across eras (stable by
  construction, but not regime-aware)
