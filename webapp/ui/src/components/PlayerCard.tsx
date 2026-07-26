import { useMemo, useState } from "react";
import { useApi } from "../lib/api";
import type { Player, PlayerDetail, PlayerHistory, SeasonRow } from "../lib/types";
import { RangeBarLarge } from "./RangeBar";
import { POS_COLORS, PosBadge, Segmented } from "./ui";
import {
  ChartCard,
  PercentileBar,
  SeasonBars,
  TrendLines,
  WeeklyBars,
} from "./Charts";
import {
  edgeClass,
  fmtAdp,
  fmtDec,
  fmtGames,
  fmtPct,
  fmtSigned,
  fmtVorp,
} from "../lib/format";

function pctile(cohort: number[], v: number | null | undefined): number | null {
  if (v == null || !cohort.length) return null;
  return cohort.filter((x) => x <= v).length / cohort.length;
}

function Tile({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div className="glass-soft px-3 py-2" title={hint}>
      <div className="text-[10px] uppercase tracking-[0.08em] text-ink-mute">{label}</div>
      <div className="num text-[15px] mt-0.5">{value}</div>
    </div>
  );
}

export function PlayerCard({ p, format }: { p: PlayerDetail; format: string }) {
  const { data: history } = useApi<PlayerHistory>(`/api/players/${p.player_id}/history`);
  const { data: cohortPlayers } = useApi<Player[]>(
    `/api/players?format=${format}&position=${p.position}`,
  );

  const f = p.features;
  const isQB = p.position === "QB";
  const isRB = p.position === "RB";
  const posColor = POS_COLORS[p.position] ?? "var(--color-ink-mute)";
  const tdKind = isQB ? "pass" : isRB ? "rush" : "receiving";

  const seasons = history?.seasons ?? [];
  const careerData = useMemo(() => {
    const rows = seasons
      .filter((s) => s.fpts_pg != null)
      .map((s) => ({ label: `'${String(s.season).slice(2)}`, fpts_pg: s.fpts_pg, proj: null }));
    if (p.fpts_pg_p50 != null) {
      rows.push({ label: "'26*", fpts_pg: null as unknown as number, proj: p.fpts_pg_p50 as unknown as null });
    }
    return rows;
  }, [seasons, p.fpts_pg_p50]);

  const label = (s: SeasonRow) => `'${String(s.season).slice(2)}`;
  const usage = useMemo(() => {
    if (isQB)
      return {
        data: seasons.map((s) => ({ label: label(s), a: s.dropbacks_pg })),
        series: [{ key: "a", label: "dropbacks / game" }],
        fmt: (v: number) => v.toFixed(0),
        zero: false,
      };
    if (isRB)
      return {
        data: seasons.map((s) => ({ label: label(s), a: s.rush_share, b: s.target_share })),
        series: [
          { key: "a", label: "rush share" },
          { key: "b", label: "target share" },
        ],
        fmt: (v: number) => `${Math.round(v * 100)}%`,
        zero: false,
      };
    return {
      data: seasons.map((s) => ({ label: label(s), a: s.target_share, b: s.wopr })),
      series: [
        { key: "a", label: "target share" },
        { key: "b", label: "WOPR" },
      ],
      fmt: (v: number) => v.toFixed(2),
      zero: false,
    };
  }, [seasons, isQB, isRB]);

  const effKey = isQB ? "epa_per_dropback" : isRB ? "epa_per_carry" : "epa_per_target";
  const effLabel = isQB ? "EPA / dropback" : isRB ? "EPA / carry" : "EPA / target";
  const effData = seasons.map((s) => ({ label: label(s), a: s[effKey] as number | null }));

  const weeklySeasons = Object.keys(history?.weekly ?? {}).sort();
  const [weeklySeason, setWeeklySeason] = useState<string | null>(null);
  const activeWeekly = weeklySeason ?? weeklySeasons[weeklySeasons.length - 1] ?? null;
  const weeklyData = activeWeekly ? (history?.weekly[activeWeekly] ?? []) : [];
  const weeklyMedian = useMemo(() => {
    const pts = weeklyData.map((w) => w.pts).sort((a, b) => a - b);
    if (!pts.length) return 0;
    const m = Math.floor(pts.length / 2);
    return pts.length % 2 ? pts[m] : (pts[m - 1] + pts[m]) / 2;
  }, [weeklyData]);

  const percentiles = useMemo(() => {
    const cohort = (cohortPlayers ?? []).filter((c) => !c.rookie);
    const col = (key: keyof typeof f) =>
      cohort.map((c) => c.features[key]).filter((x): x is number => typeof x === "number");
    const rows: { label: string; value: string; pct: number | null }[] = [];
    const add = (key: keyof typeof f, lab: string, fmt: (v: number) => string) => {
      const v = f[key];
      if (typeof v !== "number") return;
      rows.push({ label: lab, value: fmt(v), pct: pctile(col(key), v) });
    };
    if (!isQB) add("target_share", "target share", (v) => `${Math.round(v * 100)}%`);
    if (!isQB && !isRB) add("wopr", "WOPR", (v) => v.toFixed(2));
    if (!isQB) add("tprr", "targets / route", (v) => v.toFixed(2));
    add("epa_per_target_or_carry", effLabel, (v) => v.toFixed(3));
    add("red_zone_share", isRB ? "RZ rush share" : "RZ target share", (v) => `${Math.round(v * 100)}%`);
    add("x_td_rate", `expected ${tdKind} TD rate`, (v) => v.toFixed(3));
    add("snap_pct", "snap %", (v) => `${Math.round(v * 100)}%`);
    add("boom_rate", "boom rate", (v) => `${Math.round(v * 100)}%`);
    return rows;
  }, [cohortPlayers, f, isQB, isRB, effLabel, tdKind]);

  return (
    <div className="float-in">
      {/* Hero */}
      <div className="glass p-5">
        <div className="flex items-center gap-3 flex-wrap">
          <PosBadge pos={p.position} rank={p.pos_rank} />
          <h1 className="text-[22px] font-bold tracking-tight">{p.name}</h1>
          <span className="num text-ink-soft">
            {p.team ?? "FA"} · age {fmtDec(p.age, 0)}
          </span>
          {p.rookie ? (
            <span className="pill px-2.5 py-0.5 text-[10.5px] uppercase tracking-wide font-semibold bg-accent-soft text-accent">
              rookie · {p.draft_capital_bucket ?? "?"}
            </span>
          ) : (
            p.draft_capital_bucket && (
              <span className="pill glass-soft px-2.5 py-0.5 text-[10.5px] uppercase tracking-wide text-ink-soft">
                {p.draft_capital_bucket}
                {p.years_in_league != null && ` · yr ${p.years_in_league + 1}`}
              </span>
            )
          )}
        </div>

        <div className="grid grid-cols-6 gap-2 mt-4">
          <Tile label="VORP" value={<span className="font-semibold">{fmtVorp(p.vorp)}</span>} />
          <Tile label="overall" value={`#${p.overall_rank ?? "–"}`} />
          <Tile label="tier" value={p.tier ?? "–"} />
          <Tile label="ADP" value={fmtAdp(p.adp)} />
          <Tile
            label="model ADP"
            value={<span className="text-accent">{fmtAdp(p.predicted_adp)}</span>}
            hint="pre-market model: the ranking before the ADP blend"
          />
          <Tile
            label="edge"
            value={<span className={edgeClass(p.adp_edge)}>{fmtSigned(p.adp_edge)}</span>}
            hint="positive: market lets you draft later than the pre-market model values them"
          />
        </div>

        <div className="mt-5">
          <div className="text-[10.5px] uppercase tracking-[0.1em] text-ink-mute mb-1">
            2026 season outcome distribution (PPR points)
          </div>
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
              borrows veteran error scales and understates true rookie variance.
            </p>
          )}
        </div>
      </div>

      {/* History & trends */}
      {p.rookie ? (
        <div className="glass-soft p-4 mt-4 max-w-xl">
          <h3 className="text-[10.5px] uppercase tracking-[0.1em] text-ink-soft font-semibold mb-2">
            What the projection is built from
          </h3>
          <div className="grid grid-cols-2 gap-2">
            <Tile label="draft capital" value={p.draft_capital_bucket ?? "unknown"} />
            <Tile label="landing spot" value={p.team ?? "–"} hint="prior-season team pace / pass rate / offensive EPA, QB quality, vacated shares" />
          </div>
          <p className="text-[11.5px] text-ink-mute mt-3">
            No NFL history yet — career and trend charts appear after their first season.
          </p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 mt-4">
            <ChartCard title="Career fantasy PPG" note="dashed = 2026 model projection">
              {careerData.length > 0 ? (
                <SeasonBars
                  data={careerData}
                  dataKey="fpts_pg"
                  projKey="proj"
                  format={(v) => v.toFixed(1)}
                  unit="pts/gm"
                />
              ) : (
                <div className="text-ink-mute text-[12px]">no season history</div>
              )}
            </ChartCard>

            <ChartCard
              title={`Weekly game log${activeWeekly ? ` · ${activeWeekly}` : ""}`}
              note="dashed line = season median"
            >
              {weeklyData.length > 0 ? (
                <div className="h-full flex flex-col">
                  {weeklySeasons.length > 1 && (
                    <div className="mb-1 self-end">
                      <Segmented
                        options={weeklySeasons}
                        value={activeWeekly!}
                        onChange={setWeeklySeason}
                      />
                    </div>
                  )}
                  <div className="flex-1 min-h-0">
                    <WeeklyBars data={weeklyData} median={weeklyMedian} />
                  </div>
                </div>
              ) : (
                <div className="text-ink-mute text-[12px]">no recent game logs</div>
              )}
            </ChartCard>

            <ChartCard title="Usage trend">
              {seasons.length >= 2 ? (
                <TrendLines data={usage.data} series={usage.series} format={usage.fmt} />
              ) : (
                <div className="text-ink-mute text-[12px]">needs two seasons of history</div>
              )}
            </ChartCard>

            <ChartCard title={`Efficiency trend · ${effLabel}`} note="0 = league-average play value">
              {seasons.length >= 2 ? (
                <TrendLines
                  data={effData}
                  series={[{ key: "a", label: effLabel }]}
                  format={(v) => v.toFixed(2)}
                  zeroLine
                />
              ) : (
                <div className="text-ink-mute text-[12px]">needs two seasons of history</div>
              )}
            </ChartCard>
          </div>

          {/* Percentiles vs position */}
          {percentiles.length > 0 && (
            <div className="glass-soft p-4 mt-3">
              <h3 className="text-[10.5px] uppercase tracking-[0.1em] text-ink-soft font-semibold mb-1.5">
                2025 profile vs {p.position}s
                <span className="text-ink-mute font-normal normal-case tracking-normal ml-2">
                  bar = percentile among veteran {p.position}s on the board
                </span>
              </h3>
              {percentiles.map((row) => (
                <PercentileBar key={row.label} {...row} color={posColor} />
              ))}
            </div>
          )}

          {/* Situation tiles */}
          <div className="grid grid-cols-6 gap-2 mt-3">
            <Tile
              label={`expected ${tdKind} TD rate`}
              value={fmtDec(f.x_td_rate, 3)}
              hint="from usage geometry: league conversion by field position × usage mix"
            />
            <Tile
              label="TD over expected"
              value={<span className={edgeClass(f.td_oe != null ? -f.td_oe : null)}>{fmtSigned(f.td_oe, 3)}</span>}
              hint="luck component — mean-reverts"
            />
            <Tile
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
            <Tile label="consistency" value={fmtDec(f.consistency_score, 2)} />
            <Tile
              label="situation"
              value={
                f.team_change == null && f.qb_changed == null
                  ? "–"
                  : [f.team_change ? "new team" : null, f.qb_changed ? "new QB" : null]
                      .filter(Boolean)
                      .join(" · ") || "stable"
              }
            />
            <Tile
              label="vacated share"
              value={fmtPct(f.vacated_share_gained)}
              hint="Next-season rosters aren't settled: unsigned teammates count as departed, so this saturates league-wide — read it relatively."
            />
          </div>
        </>
      )}
    </div>
  );
}
