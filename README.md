# Fantasy Football Factor Model

Season-long PPR fantasy football projections using a BARRA-style cross-sectional factor model. More analytically rigorous than tools like WalterPicks.

## Architecture

```
Raw NFL Data (nfl_data_py)
    ↓
Feature Engineering (opportunity + efficiency + context + trend factors)
    ↓
Ridge Regression (position-specific, TimeSeriesSplit CV, recency-weighted)
    ↓
Age Curve Adjustment (multiplicative)
    ↓
VOR Rankings + K-means Tiers
    ↓
Draft Board / CSV Export
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
- **Sample weights**: `0.7^years_ago` — recent seasons weighted more
- **Age adjustment**: Quadratic decay from position-specific peak ages (multiplicative post-adjustment, not a feature)
- **VOR**: Replacement rank = starters + 1 per position. FLEX replacement = best of RB/WR/TE overflow.

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

- nfl_data_py was archived September 2025. Works for historical data (2020–2024). 2025 in-season data may be incomplete.
- First run downloads ~2–3 GB of PBP data. Subsequent runs use the local parquet cache in `data/cache/`.
- Use `loader.refresh_current_season()` for in-season updates without invalidating the training cache.
- Use `loader.clear_cache()` to force a full re-download.
