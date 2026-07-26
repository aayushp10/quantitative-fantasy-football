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

## v5 model serving (unsplit)

- **One model everywhere.** The exported projections (rankings, VORP,
  tiers, season quantiles, draft recommendations) AND
  `predicted_adp`/`adp_edge` all come from `MarketEnsembleModel`
  (career-augmented hybrid blended toward an isotonic ADP prior).
  Originally the edges were ranked by the pre-blend hybrid to keep them
  at full strength; the user chose the unsplit version for coherence
  (one board order, one edge story). Consequence, documented: the
  ensemble contains ADP, so *marginal* disagreements compress toward
  the market — but because the edge is computed in rank-ladder space,
  disagreements strong enough to survive the blend (a player the model
  keeps ranked far from their price) retain their full pick
  displacement. Expect fewer, stronger edges rather than many shrunken
  ones.
- **Prediction intervals are calibrated on the served model's residuals**
  (walk-forward residuals of the ensemble, not the hybrid), and
  `trust.json` backtests the ensemble — the trust page evaluates exactly
  what the board serves.

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

- **v2 redesign (user request): Sleeper × Apple liquid glass.** The original
  light "paper terminal" was replaced with a dark glass system: deep-navy
  base with fixed ambient color glows, translucent blurred cards with
  hairline light borders, pill controls and Apple-style segmented controls,
  one teal accent for interactive states. All data color re-validated on
  the dark card surface (#141B2F): position badges QB #D6408B / WR #3B76E0 /
  RB #0FA372 / TE #C98110 (declared adjacency order keeps confusable hues
  apart; worst adjacent ΔE 9.0, and the badge text label always carries
  identity), edge polarity pair #12A878 / #C0334D (deutan ΔE 9.8, signs
  always shown). Chart series use blue #3B76E0 primary / orange #C98110
  secondary (CVD-safe pair) with legends on two-series charts.
- **history.json** (build script) freezes per-player career rows from the
  feature matrix (one per season, team-stint deduped by usage) plus weekly
  PPR game logs for the last 3 seasons; served at
  `/api/players/{id}/history`. Player pages render career PPG (with the
  2026 projection as a dashed outline bar), usage trend, efficiency trend
  (EPA vs zero line), a weekly game log with boom (≥20) / bust (<8) tinting
  and season switcher, and percentile bars vs veteran position cohort
  (computed client-side from the players list).
- The range bar renders season_p10..p90 on a shared scale per table view
  (so rows are comparable) and on its own scale in the player header
  (where the labels carry the numbers).
- **Vs-market page is restricted to ADP ≤ 150.** Deep-tail players get
  ladder-overflow model ADPs (edges of ±200) that carry no market signal
  and drowned both leaderboards; the cutoff is stated in the page caption.
- **Rookie player pages collapse the feature sections** (rookies have no
  prior-season feature row — every field would be a dash) into a
  what-the-projection-is-built-from block, plus a wider-uncertainty caveat.
- **`vacated_share_gained` gets an in-context caveat when > 50%.** With
  next-season rosters unsettled, unsigned teammates count as departed and
  the feature saturates league-wide (2025 mean 0.84); the UI says to read
  it relatively.
- The Trust verdict sentence is generated from the numbers and explicitly
  reconciles the two results (negative IC edge, positive L/S spread)
  rather than cherry-picking the favorable one.

## Decision layer — VONA, roster utility, policy backtest, rollouts

- **Urgency in the pick score is now VONA** (`engine/vona.py`): vorp minus
  the expected best VORP still available at the position at the user's
  next pick, computed from the survival Monte Carlo's *joint* alive states
  (survival.py now exposes the per-sim alive matrix; p_survive is its
  marginal mean). Replaces `(1 − p_survive) × tier_drop` — same structure,
  but the "next best" comes from simulated boards instead of tier
  boundaries. `tier_drop` stays exposed for the UI's tier-cliff framing.
- **Roster utility v1** (`engine/roster_utility.py`):
  `starter_p50 − 0.10 × Σ(p50−p10 of starters) + 0.15 × top-3 bench p50`,
  greedy lineup fill (optimal for the nested slot structure). K/DST and
  unfilled slots contribute zero — the zero for empty starter slots is the
  penalty that punishes TE-less rosters. To be replaced by playoff /
  championship probability once a weekly team simulator exists.
- **Policy backtest** (`scripts/draft_policy_backtest.py`): full drafts
  with the user played by a policy, paired across policies by sharing the
  draft id (identical bot boards + reach rolls). On 40 paired drafts
  (12-team PPR, 16 rounds, slots 1/4/7/10/12): adp +0, greedy_vorp +120,
  heuristic_v1 +150, vona_v2 +157 utility vs the ADP baseline;
  vona − heuristic paired delta +5.5 [−2.0, +12.9]. The v1 heuristic can
  finish a draft with zero TEs (nothing in the formula ever forces the
  position); tests assert legality only and let utility punish it.
- **Rollouts** (`engine/rollout.py`, `GET /drafts/{id}/rollout`): rank the
  top ~8 candidates by mean completed-roster utility over ~24 simulated
  draft completions, common random numbers across candidates (paired
  deltas), user's future picks by a cheap base policy (eligible best-VORP
  + 25-pt unfilled-starter bonus), bots one shared noise board per sim as
  in survival.py. Backtest (20 paired drafts, rollouts through round 8):
  rollout − vona_v2 = +31.5 [+24.1, +38.8], 19/20 drafts.
- **Alpha model gate PASSED** (`src/models/alpha.py`,
  `output/experiments/alpha_v1/`): walk-forward residual IC pooled 0.161
  (95% CIs exclude 0 under row and season-cluster bootstrap; 7/7 seasons
  positive), incremental IC of shrunk fair values over the pure ADP map
  +0.019 [+0.009, +0.029], λ ≈ 0.84 ± 0.16. Clears the way to serve
  fair-value edges from market + λ·predicted-residual instead of fixed
  blend weights.
- **Rollouts in the draft room are on-demand** ("simulate rollouts" button
  in the rec panel), not auto-fetched: a rollout costs ~1s vs ~50ms for
  recommendations, and its result is cleared the moment the board changes.
  The panel flags when the rollout winner disagrees with the heuristic top
  pick, and Δ-vs-best is shown with its paired (common-random-numbers)
  standard error. The "why this pick" table now explains the VONA formula.
- **Sleeper-style draft room.** Bots no longer auto-advance: drafts are
  created with `auto_advance: false` and the client paces one bot pick at a
  time via `POST /drafts/{id}/step` (no-op when the user is on the clock,
  so it can be called blindly; defaults unchanged for API back-compat).
  The room adds a snake draft board grid (teams × rounds, position-colored
  cards, auto-scroll, pulsing on-the-clock cell), a bot speed toggle
  (slow/fast/instant), a pick clock chosen at setup (client-side, default
  1:00) that autodrafts on expiry — queue first, then top recommendation,
  then best VORP — and a player queue (localStorage per draft id, pruned
  as players get drafted). /step returns a lite payload (event + clock);
  the client merges it locally and only fetches full state (with survival
  odds) when the user comes on the clock.
- **Alpha serving is an overlay, not a re-rank (user decision).** Bots
  draft real FFC ADP; the board ranks by the served ensemble projections.
  The alpha model (fair = isotonic(ADP) + λ·predicted residual, λ from the
  pooled OOS slope — 0.65 in this build) adds columns: market/fair/alpha
  points, `fair_adp` (fair-points rank mapped onto the position's own ADP
  ladder), `fair_adp_edge`. Priced players without alpha inputs (rookies,
  thin histories) are carried at fair = market ("market_only") so the fair
  ladder covers the whole priced universe; their small ladder-artifact
  edges are not displayed (UI shows the edge chip only for model-scored
  rows). Recommendations add ALPHA_REC_WEIGHT (0.5) × alpha_points —
  conservative because the ensemble already carries some of the same
  signal; the policy backtest keeps `vona_v2` (no alpha) alongside
  `vona_alpha` since roster utility scores with the model's own p50 and
  can't credit the alpha tilt. Trust page gains the walk-forward alpha
  evidence (per-season residual IC + CIs, incremental IC of fair over
  ADP, λ) computed fresh at build time — served numbers, not notebook
  numbers.
