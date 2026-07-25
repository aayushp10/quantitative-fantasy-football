import { useMemo } from "react";
import { useApi } from "../lib/api";
import type { TrustData } from "../lib/types";
import { ErrorPanel, Loading, SectionTitle } from "../components/ui";
import { edgeClass, fmtDec, fmtPct, fmtSigned } from "../lib/format";

const POS_ORDER = ["QB", "RB", "WR", "TE", "overall"];

export default function Trust() {
  const { data, error, loading } = useApi<TrustData>("/api/trust");

  const verdict = useMemo(() => {
    if (!data) return null;
    const avg = data.vs_market.filter((r) => r.test_season === "average");
    const meanEdge = avg.reduce((s, r) => s + (r.ic_edge ?? 0), 0) / (avg.length || 1);
    const meanLs = avg.reduce((s, r) => s + (r.ls_spread ?? 0), 0) / (avg.length || 1);
    const perSeason = data.vs_market.filter((r) => r.test_season !== "average");
    const seasons = [...new Set(perSeason.map((r) => r.test_season))];
    const posLsSeasons = seasons.filter((s) => {
      const rows = perSeason.filter((r) => r.test_season === s);
      return rows.reduce((a, r) => a + (r.ls_spread ?? 0), 0) / (rows.length || 1) > 0;
    });
    const consistent = posLsSeasons.length / (seasons.length || 1);

    let text: string;
    const icCaveat =
      meanEdge < -0.02
        ? ` Note the raw ranking test goes the other way (IC edge ${meanEdge.toFixed(2)}): ` +
          `ADP alone orders players better than the model does. The edge lives in the ` +
          `disagreements, not in wholesale re-ranking — draft near ADP and deviate where ` +
          `the model's conviction is specific.`
        : "";
    if (meanLs > 1 && consistent >= 0.75) {
      text =
        `Where the model disagrees with the market, the disagreement has predicted outcomes: ` +
        `the long/short spread averages ${meanLs.toFixed(1)} positional ranks and was positive in ` +
        `${posLsSeasons.length} of ${seasons.length} held-out seasons.` + icCaveat;
    } else if (meanLs > 0) {
      text =
        `Model-vs-market disagreements have been mildly predictive (average long/short spread ` +
        `${meanLs.toFixed(1)} ranks, positive in ${posLsSeasons.length} of ${seasons.length} seasons). ` +
        `Treat the model as a tiebreaker on top of ADP rather than a replacement for it.`;
    } else {
      text =
        `Disagreements with the market have not predicted outcomes (average long/short spread ` +
        `${meanLs.toFixed(1)}). The model is repricing consensus — draft by ADP and use the model ` +
        `only for tier structure and roster planning.`;
    }
    return { meanEdge, meanLs, text };
  }, [data]);

  if (error) return <ErrorPanel error={error} />;
  if (loading || !data || !verdict) return <Loading label="trust report" />;

  const seasons = [...new Set(data.backtest.map((r) => r.test_season))];

  const cell = (v: number | null | undefined, dp = 3, signed = false) => (
    <td className={`num ${signed ? edgeClass(v) : ""}`}>
      {v == null ? "–" : signed ? fmtSigned(v, dp) : fmtDec(v, dp)}
    </td>
  );

  return (
    <div className="max-w-5xl float-in">
      <div className="glass p-5 my-2 max-w-3xl">
        <div className="text-[11px] uppercase tracking-[0.1em] text-accent mb-1 font-semibold">verdict</div>
        <p className="leading-relaxed">{verdict.text}</p>
        <p className="num text-[12px] text-ink-mute mt-2">
          mean IC edge vs ADP {fmtSigned(verdict.meanEdge, 3)} · mean L/S spread{" "}
          {fmtSigned(verdict.meanLs, 1)} positional ranks · held-out seasons{" "}
          {data.test_seasons.join(", ")}
        </p>
      </div>

      <div className="glass px-5 py-4 mt-4 overflow-x-auto">
      <SectionTitle>Accuracy per held-out season (points per game)</SectionTitle>
      <table className="data max-w-3xl">
        <thead>
          <tr>
            <th>position</th>
            {seasons.map((s) => (
              <th key={String(s)} colSpan={3} className="num !text-center">
                {String(s)}
              </th>
            ))}
          </tr>
          <tr>
            <th></th>
            {seasons.map((s) => (
              <FragmentHeader key={String(s)} />
            ))}
          </tr>
        </thead>
        <tbody>
          {POS_ORDER.map((pos) => {
            const rows = data.backtest.filter((r) => r.position === pos);
            if (!rows.length) return null;
            return (
              <tr key={pos}>
                <td className="num">{pos}</td>
                {seasons.map((s) => {
                  const r = rows.find((x) => x.test_season === s);
                  return (
                    <FragmentCells
                      key={String(s)}
                      mae={r?.mae ?? null}
                      r2={r?.r2 ?? null}
                      ic={r?.rank_ic ?? null}
                    />
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>

      <SectionTitle>Model vs the market (ADP baseline)</SectionTitle>
      <table className="data max-w-3xl">
        <thead>
          <tr>
            <th>season</th>
            <th>pos</th>
            <th className="num">model IC</th>
            <th className="num">ADP IC</th>
            <th className="num">IC edge</th>
            <th className="num">L/S spread</th>
            <th className="num">long hit</th>
            <th className="num">short hit</th>
            <th className="num">n</th>
          </tr>
        </thead>
        <tbody>
          {data.vs_market.map((r, i) => (
            <tr key={i} className={r.test_season === "average" ? "bg-paper-raised font-medium" : ""}>
              <td className="num">{String(r.test_season)}</td>
              <td className="num">{r.position}</td>
              {cell(r.model_ic)}
              {cell(r.adp_ic)}
              {cell(r.ic_edge, 3, true)}
              {cell(r.ls_spread, 1, true)}
              {cell(r.long_hit_rate, 2)}
              {cell(r.short_hit_rate, 2)}
              <td className="num text-ink-mute">{r.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[11px] text-ink-mute mt-1 max-w-3xl">
        L/S spread: take the players the model likes most vs the market (longs) and least
        (shorts); the spread is how many positional ranks longs beat their price by, minus the
        same for shorts. Consistently positive = the disagreement carries information.
      </p>

      <SectionTitle>Uncertainty calibration</SectionTitle>
      <table className="data max-w-lg">
        <thead>
          <tr>
            <th>interval</th>
            <th>pos</th>
            <th className="num">nominal</th>
            <th className="num">empirical</th>
            <th className="num">n</th>
          </tr>
        </thead>
        <tbody>
          {data.coverage.map((r, i) => (
            <tr key={i}>
              <td className="num">{r.band === "p10_p90" ? "p10–p90" : "p25–p75"}</td>
              <td className="num">{r.position}</td>
              <td className="num">{fmtPct(r.nominal)}</td>
              <td className="num">{fmtPct(r.empirical)}</td>
              <td className="num text-ink-mute">{r.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[11px] text-ink-mute mt-1 max-w-3xl">
        Empirical coverage close to nominal means the range bars can be taken literally.
      </p>

      <SectionTitle>Top factors by information coefficient</SectionTitle>
      <table className="data max-w-2xl">
        <thead>
          <tr>
            <th>factor</th>
            <th className="num">mean IC</th>
            <th className="num">IC IR</th>
            <th className="num">% seasons positive</th>
            <th>tier</th>
          </tr>
        </thead>
        <tbody>
          {data.top_factors.map((f) => (
            <tr key={f.factor}>
              <td className="num">{f.factor}</td>
              <td className="num">{fmtDec(f.mean_ic, 3)}</td>
              <td className="num">{fmtDec(f.ic_ir, 2)}</td>
              <td className="num">{fmtPct(f.pct_positive)}</td>
              <td className="text-[11px] uppercase tracking-wide text-ink-soft">{f.tier}</td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </div>
  );
}

function FragmentHeader() {
  return (
    <>
      <th className="num">mae</th>
      <th className="num">r²</th>
      <th className="num">ic</th>
    </>
  );
}

function FragmentCells({ mae, r2, ic }: { mae: number | null; r2: number | null; ic: number | null }) {
  return (
    <>
      <td className="num">{mae == null ? "–" : fmtDec(mae, 2)}</td>
      <td className="num">{r2 == null ? "–" : fmtDec(r2, 2)}</td>
      <td className="num">{ic == null ? "–" : fmtDec(ic, 2)}</td>
    </>
  );
}
