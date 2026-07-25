# Fantasy Football Factor Model

Season-long PPR fantasy football projections using a BARRA-style cross-sectional factor model, evaluated AGAINST THE MARKET (ADP), with calibrated uncertainty.

## Architecture

```
Raw NFL Data (nfl_data_py + nflverse releases)      Market Data (FFC ADP API)
    ↓                                                   ↓
Feature Engineering                                 data/adp.py (cached, name-matched)
  opportunity + efficiency + context + trend            ↓
  + expected-TD geometry (xTD) + routes/TPRR        models/market.py
    ↓                                                 model IC vs ADP IC
Per-season cross-sectional z-scoring                  long/short residual test
    ↓
Ridge Regression (position-specific,
  season-grouped walk-forward CV, recency-weighted)
  + Two-stage volume × EB-shrunk efficiency
  + Hybrid blend
    ↓
Age Curve (multiplicative) + Availability model (expected games)
    ↓
Residual-quantile intervals + season Monte Carlo (P10/P50/P90)
    ↓
Rookie model (draft capital + landing spot) merged in
    ↓
VOR Rankings + K-means Tiers → Draft Board / CSV Export
```

## Setup

**Requires Python 3.11 or 3.12** (nfl_data_py has known install failures on 3.13).

```bash
pip install -r requirements.txt
```

## Usage

Run notebooks in order:

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration` | Verify data pipeline, sanity-check scoring |
| `02_factor_stability` | IC analysis — which factors are predictive? |
| `03_model_training` | Train Ridge models, backtest, feature importances |
| `04_projections` | Generate 2025 projections, export CSVs |
| `05_draft_strategy` | VOR analysis, tier board, ADP comparison |

## Key Design Decisions

- **Target variable**: `fantasy_points_ppr` from `import_weekly_data()`, not recomputed from PBP (handles edge cases more reliably)
- **YoY pairs**: Season N factors → Season N+1 fpts/game (lose most recent season as a target, gain it as a feature-only projection input)
- **Market baseline (v4)**: ADP is the price. `models/market.py::rolling_market_backtest` reports model rank IC vs ADP rank IC per test season, plus a long/short test of model-vs-market disagreements (`ls_spread` > 0 across seasons = actual alpha). Historical ADP from FantasyFootballCalculator, cached per year.
- **Era handling (v4)**: factors are z-scored within each season cross-section (`features/standardize.py`) — this is what lets training extend back to 2012 (`TRAINING_SEASONS`). Recency decay relaxed 0.7 → 0.85 accordingly.
- **CV (v4)**: walk-forward by whole seasons (`models/validation.py`). The old TimeSeriesSplit cut folds mid-season and leaked shared season effects into alpha selection.
- **Efficiency shrinkage (v4)**: empirical Bayes (`models/shrinkage.py`) — `posterior = (k·prior + n·obs)/(k+n)` with k fitted per (position, metric). Sample-size dependent: 40-target rookies shrink harder than 150-target veterans. TD-rate priors center on the player's own geometry-implied rate from `features/expected_td.py` (xTD: league conversion by field-position bucket × the player's usage mix; `*_td_oe` = luck component that mean-reverts).
- **Availability (v4)**: `models/availability.py` predicts expected games from age/workload/durability history — replaces the flat 17-games assumption. Season lengths era-adjusted (16 pre-2021).
- **Uncertainty (v4)**: `models/uncertainty.py` — intervals from walk-forward out-of-sample residual quantiles with heteroscedastic scale, coverage-validated; season P10/P50/P90 via Monte Carlo composing per-game and games-played uncertainty.
- **Rookies (v4)**: `models/rookie.py` projects draft classes from draft capital + landing spot (prior-season team context, QB quality, vacated shares). Never-played rookies stay in training at 0 (no survivorship trimming). `merge_rookie_projections` appends them before VOR/tiers.
- **Routes/TPRR (v4)**: targets per route run from nflverse participation data (~2016–2023), with a snap-share proxy fallback for uncovered seasons.
- **Sample weights**: `0.85^years_ago` — recent seasons weighted more
- **Age adjustment**: Quadratic decay from position-specific peak ages (multiplicative post-adjustment, not a feature)
- **VOR**: Replacement rank = starters + 1 per position. FLEX replacement = best of RB/WR/TE overflow.

## Market-relative evaluation (the test that matters)

```python
from data.adp import load_adp
from models.market import rolling_market_backtest
from models.hybrid import HybridProjectionModel

adp = load_adp(list(range(2015, 2025)))          # cached after first fetch
result = rolling_market_backtest(
    HybridProjectionModel, yoy_pairs, adp,
    test_seasons=[2020, 2021, 2022, 2023],
    age_adjust=False,
)
# columns: model_ic, adp_ic, ic_edge, ls_spread, long/short hit rates
```

A model whose `ic_edge` ≤ 0 and `ls_spread` ≈ 0 is repricing the consensus — draft by ADP instead.

## Full v4 pipeline sketch

```python
from features.assembler import assemble_feature_matrix, build_yoy_pairs
from models.two_stage import ALL_RATE_TARGET_COLS
from models.hybrid import HybridProjectionModel
from models.availability import AvailabilityModel
from models.uncertainty import walk_forward_residuals, ResidualQuantiles, simulate_season_totals
from models.rookie import RookieModel, merge_rookie_projections
from data.loader import load_draft_picks, load_weekly

fm = assemble_feature_matrix(TRAINING_SEASONS)                 # v4 cache (force_recompute after upgrades)
pairs = build_yoy_pairs(fm, extra_target_cols=ALL_RATE_TARGET_COLS)  # includes next_games_played

model = HybridProjectionModel().train(pairs)
rq = ResidualQuantiles().fit(walk_forward_residuals(
    lambda: HybridProjectionModel(age_adjust=False), pairs))
model.set_uncertainty(rq)

proj = model.project(fm[fm["season"] == 2024], season=2025)
avail = AvailabilityModel().train(pairs)
proj = avail.attach_to_projections(proj, fm[fm["season"] == 2024], target_season=2025)
proj = simulate_season_totals(proj, rq, target_season=2025, games_sd=avail.residual_std_)

rookies = RookieModel()
rookies.train(RookieModel.build_training_frame(load_draft_picks(), load_weekly(list(TRAINING_SEASONS))))
proj = merge_rookie_projections(proj, rookies.project_class(load_draft_picks(), 2025))
```

## Factor Tiers (Expected)

| Factor | Stability |
|--------|-----------|
| target_share, rush_share, wopr, air_yard_share | STRONG |
| catch_rate, aDOT, epa_per_dropback, cpoe | MODERATE |
| TD rate, explosive_play_rate, stuff_rate | WEAK |

## ADP Integration

`nfl_data_py` does not include ADP. For ADP comparison in notebook 05, save a CSV to `data/adp_2025.csv` with columns: `player_name, position, team, adp`.

Sources: FantasyPros consensus ADP, Underdog, Sleeper API.

## Data Notes

- nfl_data_py was archived September 2025. Works for historical data. 2025 in-season data may be incomplete. Participation (routes) data is read directly from nflverse-data GitHub releases (no wrapper needed).
- **v4 upgrade note**: `TRAINING_SEASONS` now starts at 2012 and `_FEATURE_VERSION` is v4 — the next `assemble_feature_matrix()` call will download 2012–2016 PBP (~1.5 GB extra) and rebuild the matrix. ADP and draft-pick data are small and already cached on first use.
- First run downloads ~2–3 GB of PBP data. Subsequent runs use the local parquet cache in `data/cache/`.
- Use `loader.refresh_current_season()` for in-season updates without invalidating the training cache.
- Use `loader.clear_cache()` to force a full re-download.

## Web app (research terminal + mock draft room)

A local web layer on top of the pipeline — rankings, model-vs-market,
model-trust reporting, and a mock draft room with ADP-driven bots and live
pick recommendations (VORP × roster need + survival odds via Monte Carlo).

```bash
make web-data   # freeze pipeline outputs to webapp/data/*.json
                # (needs the parquet cache; first ever run downloads ~2-3 GB
                #  and retrains — takes a while. Re-run after data refreshes.)
make web        # FastAPI on :8000 + Vite dev server on :5173
make web-test   # engine/API tests + TypeScript check
```

Open http://localhost:5173. The API never imports the model — it serves the
frozen JSON artifacts plus the in-memory draft engine (drafts snapshot to
`webapp/data/drafts/` and survive restarts). Product decisions made while
building this layer are recorded in `DECISIONS.md`.
