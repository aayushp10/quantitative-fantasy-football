export interface FormatInfo {
  key: string;
  label: string;
  league_size: number;
  roster: Record<string, number>;
}

export interface Meta {
  build_timestamp: string;
  seasons_trained: [number, number];
  projection_season: number;
  model_version: string;
  formats: FormatInfo[];
  player_count: number;
  rookie_count: number;
  adp_source: string;
  adp_snapshot_date: string;
  adp_format: string;
}

export interface PlayerFeatures {
  target_share: number | null;
  wopr: number | null;
  tprr: number | null;
  routes_per_game: number | null;
  epa_per_target_or_carry: number | null;
  x_td_rate: number | null;
  td_oe: number | null;
  red_zone_share: number | null;
  breakout_flag: string | null;
  consistency_score: number | null;
  boom_rate: number | null;
  bust_rate: number | null;
  snap_pct: number | null;
  team_change: boolean | null;
  qb_changed: boolean | null;
  vacated_share_gained: number | null;
}

export interface Player {
  player_id: string;
  name: string;
  position: string;
  team: string | null;
  age: number | null;
  years_in_league: number | null;
  rookie: boolean;
  draft_capital_bucket: string | null;
  fpts_pg_p50: number | null;
  season_p10: number | null;
  season_p25: number | null;
  season_p50: number | null;
  season_p75: number | null;
  season_p90: number | null;
  expected_games: number | null;
  vorp: number | null;
  overall_rank: number | null;
  pos_rank: number | null;
  tier: number | null;
  adp: number | null;
  predicted_adp: number | null;
  adp_edge: number | null;
  features: PlayerFeatures;
}

export interface PlayerDetail extends Player {
  vorp_all_formats: Record<
    string,
    { vorp: number; overall_rank: number; pos_rank: number; tier: number | null }
  >;
}

export interface TrustData {
  test_seasons: number[];
  backtest: {
    test_season: number | string;
    position: string;
    mae: number | null;
    r2: number | null;
    rank_ic: number | null;
    n: number;
  }[];
  market_baseline: { season: number | string; position: string; adp_rank_ic: number; n: number }[];
  vs_market: {
    test_season: number | string;
    position: string;
    n: number;
    model_ic: number | null;
    adp_ic: number | null;
    ic_edge: number | null;
    ls_spread: number | null;
    long_hit_rate: number | null;
    short_hit_rate: number | null;
  }[];
  coverage: { band: string; position: string; nominal: number; empirical: number; n: number }[];
  top_factors: {
    factor: string;
    mean_ic: number;
    std_ic: number;
    ic_ir: number;
    pct_positive: number;
    tier: string;
    n_seasons: number;
  }[];
}

export interface Pick {
  overall: number;
  round: number;
  slot: number;
  player_id: string;
  player_name: string;
  position: string;
  team: string | null;
  is_user: boolean;
}

export interface DraftState {
  draft_id: string;
  config: {
    teams: number;
    user_slot: number;
    rounds: number;
    format: string;
    roster: Record<string, number>;
  };
  picks: Pick[];
  rosters: Record<string, string[]>;
  on_the_clock: { overall: number; round: number; slot: number; is_user: boolean } | null;
  complete: boolean;
  available: { player_id: string; p_survive: number | null }[];
}

export interface TierCliffAlert {
  position: string;
  tier: number;
  remaining_in_tier: number;
  drop_to_next_tier: number;
}

export interface Recommendation {
  player_id: string;
  name: string;
  position: string;
  team: string | null;
  vorp: number;
  pos_rank: number | null;
  tier: number | null;
  adp: number | null;
  adp_edge: number | null;
  season_p10: number | null;
  season_p25: number | null;
  season_p50: number | null;
  season_p75: number | null;
  season_p90: number | null;
  p_survive: number;
  need_weight: number;
  need_multiplier: number;
  tier_drop: number;
  urgency: number;
  rec_score: number;
}

export interface Recommendations {
  recommendations: Recommendation[];
  pool_size: number;
  need_weights: Record<string, number>;
  tier_structure: Record<
    string,
    {
      current_tier: number;
      remaining_in_tier: number;
      best_vorp: number;
      next_tier_best_vorp: number | null;
    }
  >;
  tier_cliff_alerts: TierCliffAlert[];
}

export interface AdpBoardRow {
  player_id: string;
  name: string;
  position: string;
  team: string | null;
  adp: number;
  adp_stdev: number | null;
  bye: number | null;
  streamer: boolean;
  has_projection: boolean;
}

export interface SeasonRow {
  season: number;
  team: string | null;
  age: number | null;
  games: number | null;
  fpts_pg: number | null;
  fpts_total: number | null;
  snap_pct: number | null;
  target_share: number | null;
  wopr: number | null;
  tprr: number | null;
  targets_pg: number | null;
  epa_per_target: number | null;
  rush_share: number | null;
  carries_pg: number | null;
  epa_per_carry: number | null;
  ypc: number | null;
  dropbacks_pg: number | null;
  epa_per_dropback: number | null;
  cpoe: number | null;
  boom_rate: number | null;
  bust_rate: number | null;
  trend_class: string | null;
  x_rec_td_rate: number | null;
  rec_td_oe: number | null;
  x_rush_td_rate: number | null;
  rush_td_oe: number | null;
  x_pass_td_rate: number | null;
  pass_td_oe: number | null;
}

export interface WeeklyPoint {
  week: number;
  pts: number;
}

export interface PlayerHistory {
  seasons: SeasonRow[];
  weekly: Record<string, WeeklyPoint[]>;
}
