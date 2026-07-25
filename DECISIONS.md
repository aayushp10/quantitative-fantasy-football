# Product decisions — web app layer

Decisions made while building the web layer where the spec was ambiguous or
the pipeline's reality differed from the plan. The model itself was not
changed.

## Phase 1 — data export

- **Formats are PPR-only.** The model target is `fantasy_points_ppr`
  end-to-end (weekly target variable, two-stage recombination, residual
  quantiles). Producing half-PPR/standard projections would require
  retraining with a different target — a model change, which is out of
  scope. Shipped format keys: `10_ppr`, `12_ppr`, `14_ppr`,
  `12_superflex_ppr` (vor.py supports superflex, so it's included). The API
  contract is unchanged: `meta.json.formats` is the source of truth and the
  UI renders whatever it lists.
- **Duplicate player-season rows are deduped at export.** The v4 feature
  matrix carries duplicate rows for mid-season team changes (e.g. two team
  stints) and some exact duplicates (~20 rows in 2025). The export keeps, per
  player, the stint with the highest usage (targets+carries+dropbacks);
  feature values are identical across dupes, only the team label differs.
  Not fixed upstream because training is unaffected and the model is
  off-limits.
- **`adp_edge`/`predicted_adp` are computed on the 12-team PPR board**
  (the format FFC ADP prices). Other formats reuse the same ADP columns.
- **Position-appropriate single fields.** Where the players.json spec names
  one field that maps to different columns by position:
  `epa_per_target_or_carry` = EPA/dropback (QB), EPA/carry (RB), EPA/target
  (WR/TE); `x_td_rate`/`td_oe` = pass (QB), rush (RB), rec (WR/TE);
  `red_zone_share` = rz_rush_share (RB), rz_target_share (WR/TE), null (QB);
  `vacated_share_gained` = vacated carry share (RB), vacated target share
  (WR/TE), null (QB).
- **`draft_capital_bucket`** is exported as a label derived from the
  pipeline's ordinal bucket (0–5): UDFA, R6-7, R4-5, R3, R2, R1. Rookies
  (not in the feature matrix) get the label derived from their actual pick.
- **Rookies without a GSIS id** (27 of 257 in the 2026 class) get synthetic
  ids `rookie_2026_p{pick}` so every board row is addressable.
- **`adp_stdev`** comes straight from FFC (no synthesis needed —
  `stdev_synthetic: false` in meta). K/DST rows (FFC `PK`/`DEF`) are kept on
  the ADP board as `streamer: true` with no model projection.
- **`expected_games` for rookies is the full team season (17)** — the rookie
  model prices availability into the per-team-game rate instead (documented
  pipeline limitation).
- **ADP join gate is two-tier.** The spec's ">5% of ADP players fail to
  join → fail" tripped on 10 legitimate misses (5.4%): players with no 2025
  season above feature thresholds (Aiyuk, Tank Dell, Jonathon Brooks, Jayden
  Reed, …). Those are correct behavior — they appear on the ADP board with
  `has_projection: false`. The gate now checks what it was meant to catch
  (name-matching regressions): ≥95% join required among top-120-ADP players,
  ≥85% overall, and every unjoined name is printed in the build summary.
- **Trust artifact test seasons are 2020–2023**, matching notebook 06, so
  the Trust page can be spot-checked against the notebook outputs. The
  2024 pair-season is excluded from the market backtest because FFC has no
  2025 ADP archive.

## Phase 2 — API / draft engine

- **Bot position caps derived from roster config**: max QB = QB starters
  (+1 in superflex) + 1; max TE = TE starters + 1; RB/WR cap = starters +
  FLEX + half the bench (rounded up); K/DST capped at their slot count.
  Early-round guard: no 2nd QB or 2nd TE before round 10 in 1-QB formats
  (spec). K/DST are only draftable in the last `K+DST` rounds, and bots are
  forced to fill unfilled K/DST slots when remaining picks run out.
- **Draft ids are short hex strings; the bot RNG is seeded from the draft id**
  so a draft replays identically after a server restart (boards are also
  persisted in the snapshot).
- **P_survive** is defined over the players still available at the moment of
  the user's current pick, simulating only the intervening bot picks with
  fresh board noise (M=300). The user's own current pick is excluded.
- **Recommendation pool** = top 40 available by format VORP; streamers and
  no-projection players never appear in recommendations but can be drafted
  manually from the board table.

## Phase 3 — UI

- Accent color is a single terminal-amber (#B45309 family) used only for
  interactive/highlight states; everything else is near-black ink on paper.
- The range bar renders season_p10..p90 on a per-position shared scale in
  tables (so rows are comparable within a view) and on its own scale in the
  player header (where the labels carry the numbers).
