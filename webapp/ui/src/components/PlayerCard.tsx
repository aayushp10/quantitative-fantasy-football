import type { PlayerDetail } from "../lib/types";
import { RangeBarLarge } from "./RangeBar";
import { SectionTitle } from "./ui";
import {
  edgeClass,
  fmtAdp,
  fmtDec,
  fmtGames,
  fmtPct,
  fmtSigned,
  fmtVorp,
} from "../lib/format";

function Stat({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div className="flex justify-between gap-4 py-0.5 border-b border-rule last:border-b-0" title={hint}>
      <span className="text-ink-soft">{label}</span>
      <span className="num">{value}</span>
    </div>
  );
}

export function PlayerCard({ p }: { p: PlayerDetail }) {
  const f = p.features;
  const isQB = p.position === "QB";
  const isRB = p.position === "RB";
  const epaLabel = isQB ? "EPA / dropback" : isRB ? "EPA / carry" : "EPA / target";
  const tdKind = isQB ? "pass" : isRB ? "rush" : "receiving";

  return (
    <div>
      {/* Header */}
      <div className="flex items-baseline gap-3 flex-wrap">
        <h1 className="text-xl font-medium tracking-tight">{p.name}</h1>
        <span className="num text-ink-soft">
          {p.position}
          {p.pos_rank} · {p.team ?? "FA"} · age {fmtDec(p.age, 0)}
        </span>
        {p.rookie ? (
          <span className="text-[11px] uppercase tracking-wide text-accent border border-accent px-1.5">
            rookie · {p.draft_capital_bucket ?? "?"}
          </span>
        ) : (
          p.draft_capital_bucket && (
            <span className="text-[11px] uppercase tracking-wide text-ink-mute border border-rule-strong px-1.5">
              {p.draft_capital_bucket}
              {p.years_in_league != null && ` · yr ${p.years_in_league + 1}`}
            </span>
          )
        )}
        <span className="num text-ink-mute ml-auto">
          overall #{p.overall_rank ?? "–"} · tier {p.tier ?? "–"} · VORP {fmtVorp(p.vorp)}
        </span>
      </div>

      {/* Distribution */}
      <SectionTitle>Season outcome distribution (PPR points)</SectionTitle>
      <RangeBarLarge
        p10={p.season_p10}
        p25={p.season_p25}
        p50={p.season_p50}
        p75={p.season_p75}
        p90={p.season_p90}
      />
      <div className="flex gap-6 mt-1 text-[12px] text-ink-soft num">
        <span>expected games {fmtGames(p.expected_games)}</span>
        <span>{fmtDec(p.fpts_pg_p50, 1)} pts/gm median</span>
      </div>
      {p.rookie && (
        <p className="mt-2 text-[12px] text-ink-soft max-w-lg">
          Rookie projection from draft capital + landing spot only — the distribution
          borrows veteran error scales and understates true rookie variance. Treat the
          band as a floor on the uncertainty, not the ceiling.
        </p>
      )}

      {/* Model vs market */}
      <SectionTitle>Model vs market</SectionTitle>
      <div className="grid grid-cols-3 gap-x-8 max-w-lg text-[13px]">
        <Stat label="market ADP" value={fmtAdp(p.adp)} />
        <Stat label="model-implied ADP" value={fmtAdp(p.predicted_adp)} />
        <Stat
          label="edge"
          value={<span className={edgeClass(p.adp_edge)}>{fmtSigned(p.adp_edge)}</span>}
          hint="positive: market lets you draft later than the model values them"
        />
      </div>

      {/* Features — rookies have no prior-season feature row; show the
          draft-capital framing the projection is actually built from. */}
      {p.rookie ? (
        <div className="max-w-lg text-[13px]">
          <SectionTitle>What the projection is built from</SectionTitle>
          <Stat label="draft capital" value={p.draft_capital_bucket ?? "unknown"} />
          <Stat label="landing spot" value={p.team ?? "–"} hint="prior-season team pace / pass rate / offensive EPA, QB quality, vacated shares" />
          <p className="text-[11px] text-ink-mute mt-2">
            Rookie projections use draft capital + landing-spot context only. No
            per-route or efficiency data exists yet — those sections appear after
            their first NFL season.
          </p>
        </div>
      ) : (
      <div className="grid grid-cols-2 gap-x-10 max-w-2xl text-[13px]">
        <div>
          <SectionTitle>Opportunity</SectionTitle>
          {!isQB && <Stat label="target share" value={fmtPct(f.target_share)} />}
          {(p.position === "WR" || p.position === "TE") && (
            <Stat label="WOPR" value={fmtDec(f.wopr, 2)} />
          )}
          {!isQB && <Stat label="targets / route run" value={fmtDec(f.tprr, 2)} />}
          {!isQB && <Stat label="routes / game" value={fmtDec(f.routes_per_game, 1)} />}
          {f.red_zone_share != null && (
            <Stat label={isRB ? "RZ rush share" : "RZ target share"} value={fmtPct(f.red_zone_share)} />
          )}
          <Stat label="snap %" value={fmtPct(f.snap_pct)} />
        </div>
        <div>
          <SectionTitle>Efficiency</SectionTitle>
          <Stat label={epaLabel} value={fmtDec(f.epa_per_target_or_carry, 3)} />
          <Stat label="consistency score" value={fmtDec(f.consistency_score, 2)} />
          <Stat label="boom rate" value={fmtPct(f.boom_rate)} />
          <Stat label="bust rate" value={fmtPct(f.bust_rate)} />
        </div>
        <div>
          <SectionTitle>TD equity</SectionTitle>
          <Stat
            label={`expected ${tdKind} TD rate`}
            value={fmtDec(f.x_td_rate, 3)}
            hint="from usage geometry: league conversion by field position × usage mix"
          />
          <Stat
            label="TD rate over expected"
            value={<span className={edgeClass(f.td_oe != null ? -f.td_oe : null)}>{fmtSigned(f.td_oe, 3)}</span>}
            hint="luck component — mean-reverts"
          />
          <p className="text-[11px] text-ink-mute mt-1">
            Over-expected TD production is luck and tends to revert; the expected rate is
            the persistent part.
          </p>
        </div>
        <div>
          <SectionTitle>Trend &amp; situation</SectionTitle>
          <Stat
            label="late-season trend"
            value={
              <span
                className={
                  f.breakout_flag === "BREAKOUT"
                    ? "text-edge-pos"
                    : f.breakout_flag === "DECLINING"
                      ? "text-edge-neg"
                      : ""
                }
              >
                {f.breakout_flag ?? "–"}
              </span>
            }
          />
          <Stat label="changed team" value={f.team_change == null ? "–" : f.team_change ? "yes" : "no"} />
          {!isQB && (
            <Stat label="new QB" value={f.qb_changed == null ? "–" : f.qb_changed ? "yes" : "no"} />
          )}
          <Stat
            label="vacated share gained"
            value={fmtPct(f.vacated_share_gained)}
            hint={isRB ? "carry share departed from roster" : "target share departed from roster"}
          />
          {(f.vacated_share_gained ?? 0) > 0.5 && (
            <p className="text-[11px] text-ink-mute mt-1">
              Next-season rosters aren't settled yet, so unsigned teammates count as
              departed and this saturates league-wide. Read it relative to other
              players, not as a literal share.
            </p>
          )}
        </div>
      </div>
      )}
    </div>
  );
}
