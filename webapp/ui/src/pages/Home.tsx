import { useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useFormat } from "../App";
import { useApi } from "../lib/api";
import type { Player, PlayerHistory, TrustData } from "../lib/types";
import { RangeBar, RangeBarLarge } from "../components/RangeBar";
import { PosBadge } from "../components/ui";
import { edgeClass, fmtAdp, fmtPct, fmtSigned, fmtVorp } from "../lib/format";

function Card({
  to,
  className = "",
  delay = 0,
  children,
}: {
  to?: string;
  className?: string;
  delay?: number;
  children: React.ReactNode;
}) {
  const nav = useNavigate();
  return (
    <div
      className={`glass p-5 float-in flex flex-col ${to ? "cursor-pointer transition-transform hover:-translate-y-0.5" : ""} ${className}`}
      style={{ animationDelay: `${delay}ms`, animationFillMode: "backwards" }}
      onClick={to ? () => nav(to) : undefined}
      tabIndex={to ? 0 : undefined}
      onKeyDown={to ? (e) => e.key === "Enter" && nav(to) : undefined}
      role={to ? "link" : undefined}
    >
      {children}
    </div>
  );
}

function CardTitle({ kicker, title }: { kicker: string; title: string }) {
  return (
    <div className="mb-3">
      <div className="text-[10.5px] uppercase tracking-[0.14em] text-accent font-semibold">{kicker}</div>
      <div className="text-[17px] font-bold tracking-tight mt-0.5">{title}</div>
    </div>
  );
}

/** Tiny inline career sparkbars (no chart lib — homepage stays light). */
function CareerSpark({ history }: { history: PlayerHistory | null }) {
  const rows = (history?.seasons ?? []).filter((s) => s.fpts_pg != null);
  if (!rows.length) return null;
  const max = Math.max(...rows.map((s) => s.fpts_pg!));
  const w = 26;
  return (
    <svg width={rows.length * (w + 8)} height={84} className="mt-2" role="img" aria-label="career points per game">
      {rows.map((s, i) => {
        const h = Math.max(4, (s.fpts_pg! / max) * 64);
        return (
          <g key={s.season}>
            <rect
              x={i * (w + 8)}
              y={68 - h}
              width={w}
              height={h}
              rx={5}
              fill="#3B76E0"
              opacity={0.55 + 0.45 * (i / Math.max(1, rows.length - 1))}
            />
            <text
              x={i * (w + 8) + w / 2}
              y={81}
              textAnchor="middle"
              fontSize={9.5}
              fill="var(--color-ink-mute)"
              fontFamily="var(--font-mono)"
            >
              '{String(s.season).slice(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function Home() {
  const { format, meta } = useFormat();
  const { data: players } = useApi<Player[]>(`/api/players?format=${format}`);
  const { data: trust } = useApi<TrustData>("/api/trust");

  const ranked = useMemo(
    () => (players ?? []).filter((p) => p.overall_rank != null),
    [players],
  );
  const top5 = ranked.slice(0, 5);
  const no1 = ranked[0] ?? null;
  const showcase = ranked.find((p) => !p.rookie) ?? null;
  const { data: showcaseHistory } = useApi<PlayerHistory>(
    showcase ? `/api/players/${showcase.player_id}/history` : null,
  );

  const priced = useMemo(
    () =>
      ranked.filter(
        (p) => p.adp != null && p.adp <= 150 && p.adp_edge != null && p.predicted_adp != null && p.predicted_adp <= 210,
      ),
    [ranked],
  );
  const loves = [...priced].sort((a, b) => b.adp_edge! - a.adp_edge!).slice(0, 3);
  const fades = [...priced].sort((a, b) => a.adp_edge! - b.adp_edge!).slice(0, 3);

  const metrics = useMemo(() => {
    if (!trust) return null;
    const avg = trust.vs_market.filter((r) => r.test_season === "average");
    const ls = avg.reduce((s, r) => s + (r.ls_spread ?? 0), 0) / (avg.length || 1);
    const cov = trust.coverage.filter((c) => c.band === "p10_p90");
    const coverage = cov.reduce((s, c) => s + c.empirical, 0) / (cov.length || 1);
    const perSeason = trust.vs_market.filter((r) => r.test_season !== "average");
    const seasons = [...new Set(perSeason.map((r) => r.test_season))];
    const positive = seasons.filter((s) => {
      const rows = perSeason.filter((r) => r.test_season === s);
      return rows.reduce((a, r) => a + (r.ls_spread ?? 0), 0) / (rows.length || 1) > 0;
    }).length;
    return { ls, coverage, seasons: seasons.length, positive };
  }, [trust]);

  const domain5: [number, number] = useMemo(
    () => [0, Math.max(1, ...top5.map((p) => p.season_p90 ?? 0))],
    [top5],
  );

  const miniRow = (p: Player) => (
    <div key={p.player_id} className="flex items-center gap-2.5 py-[5px] border-b border-rule last:border-b-0">
      <span className="num text-ink-mute w-4 text-right text-[11px]">{p.overall_rank}</span>
      <PosBadge pos={p.position} rank={p.pos_rank} />
      <span className="font-medium text-[13px] truncate">{p.name}</span>
      <span className="ml-auto num text-[12px] font-semibold">{fmtVorp(p.vorp)}</span>
      <RangeBar
        p10={p.season_p10}
        p25={p.season_p25}
        p50={p.season_p50}
        p75={p.season_p75}
        p90={p.season_p90}
        domain={domain5}
        width={72}
      />
    </div>
  );

  const edgeRow = (p: Player) => (
    <div key={p.player_id} className="flex items-center gap-2.5 py-[5px] border-b border-rule last:border-b-0">
      <PosBadge pos={p.position} rank={p.pos_rank} />
      <span className="font-medium text-[13px] truncate">{p.name}</span>
      <span className="ml-auto num text-[12px] text-ink-mute">adp {fmtAdp(p.adp)}</span>
      <span className={`num text-[12.5px] font-semibold ${edgeClass(p.adp_edge)}`}>{fmtSigned(p.adp_edge)}</span>
    </div>
  );

  return (
    <div>
      {/* Hero */}
      <section className="text-center pt-16 pb-14 float-in">
        <div className="num text-[12px] uppercase tracking-[0.22em] text-ink-soft">
          {meta ? `${meta.projection_season} season · trained ${meta.seasons_trained[0]}–${meta.seasons_trained[1]}` : "…"}
        </div>
        <h1 className="text-[64px] leading-[1.04] font-bold tracking-[-0.03em] mt-4">
          Draft like the
          <br />
          <span
            className="bg-clip-text text-transparent"
            style={{
              backgroundImage: "linear-gradient(92deg, #2DD4BF 5%, #5B8DEF 45%, #D6408B 95%)",
            }}
          >
            market can't see you.
          </span>
        </h1>
        <p className="text-[17px] text-ink-soft max-w-xl mx-auto mt-5 leading-relaxed">
          A quantitative projection engine for fantasy football. Fourteen seasons of
          play-by-play, calibrated uncertainty on every player, and a draft room that
          prices each pick against the crowd.
        </p>
        <div className="flex items-center justify-center gap-3 mt-8">
          <Link to="/draft" className="btn-primary px-6 py-2.5 text-[14px]">
            Start a mock draft
          </Link>
          <Link to="/rankings" className="btn-ghost px-6 py-2.5 text-[14px]">
            Explore the board →
          </Link>
        </div>
        <div className="num flex items-center justify-center gap-8 mt-10 text-[12px] text-ink-mute">
          <span>{meta ? `${meta.player_count} players priced` : "…"}</span>
          <span>·</span>
          <span>{meta ? `${meta.formats.length} league formats` : "…"}</span>
          <span>·</span>
          <span>P10–P90 on every projection</span>
          <span>·</span>
          <span>{metrics ? `+${metrics.ls.toFixed(0)} rank L/S edge vs ADP` : "…"}</span>
        </div>
      </section>

      {/* Bento */}
      <section className="grid grid-cols-6 gap-4">
        <Card to="/rankings" className="col-span-3" delay={60}>
          <CardTitle kicker="Rankings" title="The board, by value over replacement." />
          <div>{top5.map(miniRow)}</div>
          <div className="text-[12px] text-accent mt-3">Full 360-player board →</div>
        </Card>

        <Card className="col-span-3" delay={120}>
          <CardTitle kicker="Uncertainty" title="Distributions, not points." />
          {no1 && (
            <>
              <div className="flex items-center gap-2 mb-1">
                <PosBadge pos={no1.position} rank={no1.pos_rank} />
                <span className="font-medium text-[13px]">{no1.name}</span>
                <span className="num text-[12px] text-ink-mute">2026 season outcomes</span>
              </div>
              <RangeBarLarge
                p10={no1.season_p10}
                p25={no1.season_p25}
                p50={no1.season_p50}
                p75={no1.season_p75}
                p90={no1.season_p90}
              />
            </>
          )}
          <p className="text-[12.5px] text-ink-soft mt-2 leading-relaxed">
            Every projection ships as a calibrated P10–P90 range from walk-forward
            residuals — the same bar, everywhere, so you always see the risk you're
            drafting.
          </p>
        </Card>

        <Card to="/market" className="col-span-2" delay={180}>
          <CardTitle kicker="Vs market" title="Priced against the crowd." />
          <div className="text-[10.5px] uppercase tracking-[0.1em] text-ink-mute mb-1">model loves</div>
          <div>{loves.map(edgeRow)}</div>
          <div className="text-[10.5px] uppercase tracking-[0.1em] text-ink-mute mb-1 mt-3">market loves</div>
          <div>{fades.map(edgeRow)}</div>
          <div className="text-[12px] text-accent mt-auto pt-3">Edge leaderboard →</div>
        </Card>

        <Card to="/draft" className="col-span-2" delay={240}>
          <CardTitle kicker="Draft room" title="Every pick, scored live." />
          {no1 && (
            <div
              className="glass-soft p-3"
              style={{ borderColor: "rgba(45,212,191,0.4)" }}
            >
              <div className="flex items-center gap-2">
                <PosBadge pos={no1.position} rank={no1.pos_rank} />
                <span className="font-semibold text-[13.5px]">{no1.name}</span>
                <span className="btn-primary ml-auto px-3 py-0.5 text-[11px]">Draft</span>
              </div>
              <div className="num text-[11.5px] text-ink-soft mt-2 leading-relaxed">
                VORP {fmtVorp(no1.vorp)} · ADP {fmtAdp(no1.adp)} · edge{" "}
                <span className={edgeClass(no1.adp_edge)}>{fmtSigned(no1.adp_edge)}</span>
                <br />
                <span className="text-ink-mute">survival odds computed live on your pick</span>
              </div>
            </div>
          )}
          <p className="text-[12.5px] text-ink-soft mt-3 leading-relaxed">
            ADP-driven bots, Monte-Carlo survival odds, tier-cliff alerts — and a
            "why this pick" breakdown on every card.
          </p>
          <div className="text-[12px] text-accent mt-auto pt-3">Enter the room →</div>
        </Card>

        <Card to="/trust" className="col-span-2" delay={300}>
          <CardTitle kicker="Trust" title="Honest about its edge." />
          <div className="flex gap-6 mt-1">
            <div>
              <div className="num text-[30px] font-bold text-edge-pos leading-none">
                {metrics ? `+${metrics.ls.toFixed(1)}` : "…"}
              </div>
              <div className="text-[11px] text-ink-soft mt-1 leading-snug">
                rank L/S spread vs ADP,
                <br />
                positive in {metrics?.positive ?? "…"} of {metrics?.seasons ?? "…"} held-out seasons
              </div>
            </div>
            <div>
              <div className="num text-[30px] font-bold leading-none">
                {metrics ? fmtPct(metrics.coverage) : "…"}
              </div>
              <div className="text-[11px] text-ink-soft mt-1 leading-snug">
                empirical coverage of the
                <br />
                80% prediction interval
              </div>
            </div>
          </div>
          <p className="text-[12.5px] text-ink-soft mt-3 leading-relaxed">
            Backtests, market baselines, and calibration — including the numbers
            that don't flatter the model.
          </p>
          <div className="text-[12px] text-accent mt-auto pt-3">Read the evidence →</div>
        </Card>

        <Card
          to={showcase ? `/players/${showcase.player_id}` : undefined}
          className="col-span-6"
          delay={360}
        >
          <div className="flex items-start gap-10">
            <div className="max-w-sm">
              <CardTitle kicker="Player pages" title="Careers, trends, and game logs." />
              <p className="text-[12.5px] text-ink-soft leading-relaxed">
                Every player page charts career scoring, usage and efficiency trends,
                weekly boom/bust logs, and a percentile profile against their position —
                all from the same data the model trains on.
              </p>
              <div className="text-[12px] text-accent mt-3">
                {showcase ? `See ${showcase.name} →` : ""}
              </div>
            </div>
            {showcase && (
              <div className="flex items-end gap-10">
                <div>
                  <div className="flex items-center gap-2">
                    <PosBadge pos={showcase.position} rank={showcase.pos_rank} />
                    <span className="font-semibold">{showcase.name}</span>
                    <span className="num text-[11.5px] text-ink-mute">career pts/game</span>
                  </div>
                  <CareerSpark history={showcaseHistory ?? null} />
                </div>
              </div>
            )}
          </div>
        </Card>
      </section>
    </div>
  );
}
