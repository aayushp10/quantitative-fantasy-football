import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useFormat } from "../App";
import { useApi } from "../lib/api";
import type { Player } from "../lib/types";
import { ErrorPanel, Loading, PosBadge, SectionTitle, Segmented } from "../components/ui";
import { edgeClass, fmtAdp, fmtDec, fmtSigned, fmtVorp } from "../lib/format";

const POSITIONS = ["ALL", "QB", "RB", "WR", "TE"];
const SOURCES = ["ensemble", "alpha"];

type Row = Player & { m_adp: number; m_edge: number };

export default function Market() {
  const { format } = useFormat();
  const [pos, setPos] = useState("ALL");
  const [source, setSource] = useState("ensemble");
  const nav = useNavigate();
  const { data, error, loading } = useApi<Player[]>(`/api/players?format=${format}`);

  const alpha = source === "alpha";

  // Draft-relevant universe only: deep-tail players get ladder-overflow
  // model ADPs (edge ±200) that carry no market signal and drown the boards.
  // Alpha mode additionally requires a model-scored row — market-only
  // players have fair = market by construction (no edge to report).
  const MAX_ADP = 150;
  const priced = useMemo<Row[]>(() => {
    const out: Row[] = [];
    for (const p of data ?? []) {
      if (p.adp == null || p.adp > MAX_ADP) continue;
      if (pos !== "ALL" && p.position !== pos) continue;
      const mAdp = alpha ? p.fair_adp : p.predicted_adp;
      const mEdge = alpha ? p.fair_adp_edge : p.adp_edge;
      if (mAdp == null || mEdge == null || mAdp > MAX_ADP * 1.4) continue;
      if (alpha && p.alpha_source !== "model") continue;
      out.push({ ...p, m_adp: mAdp, m_edge: mEdge });
    }
    return out;
  }, [data, pos, alpha]);

  const modelLoves = useMemo(
    () => [...priced].sort((a, b) => b.m_edge - a.m_edge).slice(0, 15),
    [priced],
  );
  const marketLoves = useMemo(
    () => [...priced].sort((a, b) => a.m_edge - b.m_edge).slice(0, 15),
    [priced],
  );
  const maxAxis = useMemo(
    () => Math.ceil(Math.max(...priced.map((p) => Math.max(p.adp!, p.m_adp)), 10) / 10) * 10,
    [priced],
  );

  if (error) return <ErrorPanel error={error} />;
  if (loading || !data) return <Loading label="market comparison" />;

  const adpLabel = alpha ? "fair adp" : "model adp";

  const board = (rows: Row[], title: string, note: string) => (
    <div className="flex-1 min-w-0 glass px-4 py-3">
      <SectionTitle>{title}</SectionTitle>
      <p className="text-[11px] text-ink-mute -mt-1 mb-2">{note}</p>
      <table className="data">
        <thead>
          <tr>
            <th>player</th>
            <th>pos</th>
            <th className="num">adp</th>
            <th className="num">{adpLabel}</th>
            <th className="num">edge</th>
            <th className="num">{alpha ? "z" : "vorp"}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr
              key={p.player_id}
              className="cursor-pointer"
              tabIndex={0}
              onClick={() => nav(`/players/${p.player_id}`)}
              onKeyDown={(e) => e.key === "Enter" && nav(`/players/${p.player_id}`)}
            >
              <td className="font-medium">{p.name}</td>
              <td>
                <PosBadge pos={p.position} rank={p.pos_rank} />
              </td>
              <td className="num">{fmtAdp(p.adp)}</td>
              <td className="num text-accent">{fmtAdp(p.m_adp)}</td>
              <td className={`num ${edgeClass(p.m_edge)}`}>{fmtSigned(p.m_edge)}</td>
              <td className="num">
                {alpha ? (
                  <span className={Math.abs(p.alpha_z ?? 0) >= 1 ? "text-accent font-semibold" : ""}>
                    {fmtDec(p.alpha_z, 2)}
                  </span>
                ) : (
                  fmtVorp(p.vorp)
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return (
    <div>
      <div className="flex items-center gap-3 mb-3 flex-wrap">
        <Segmented options={POSITIONS} value={pos} onChange={setPos} />
        <span className="text-[10px] uppercase tracking-wide text-ink-mute ml-2">lens</span>
        <Segmented options={SOURCES} value={source} onChange={setSource} />
        <span className="text-[12px] text-ink-mute">
          {alpha ? (
            <>
              edge = ADP − fair ADP (market + λᵢ·predicted market error, survivor-complete
              backtest — see Trust). z = conviction.{" "}
            </>
          ) : (
            <>
              edge = ADP − model-implied ADP (the served model&apos;s board order priced on the
              market&apos;s ladder).{" "}
            </>
          )}
          <span className="text-edge-pos">+ market lets you wait</span> ·{" "}
          <span className="text-edge-neg">− you must reach</span> · draft-relevant universe (ADP ≤{" "}
          {MAX_ADP})
        </span>
      </div>

      <div className="flex gap-4 items-start">
        {board(
          modelLoves,
          alpha ? "Alpha longs (market late)" : "Model loves (market late)",
          alpha
            ? "biggest positive fair-value gap — the residual model backs them to beat their price"
            : "biggest positive edge — value at cost",
        )}
        {board(
          marketLoves,
          alpha ? "Alpha shorts (market pays up)" : "Market loves (model out)",
          alpha
            ? "biggest negative fair-value gap — the residual model expects them to miss their price"
            : "biggest negative edge — the market pays more than the model would",
        )}
      </div>

      <div className="glass px-5 py-4 mt-4 max-w-3xl">
      <SectionTitle>{alpha ? "Fair ADP vs market ADP" : "Model-implied ADP vs market ADP"}</SectionTitle>
      <p className="text-[11px] text-ink-mute -mt-1 mb-2">
        below the diagonal: {alpha ? "the alpha model" : "the model"} prices the player earlier
        than the market drafts them
      </p>
      <div className="h-[440px]">
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 8, right: 16, bottom: 26, left: 8 }}>
            <CartesianGrid stroke="var(--color-rule)" strokeDasharray="2 4" />
            <XAxis
              type="number"
              dataKey="adp"
              domain={[0, maxAxis]}
              tick={{ fontSize: 11, fill: "var(--color-ink-soft)", fontFamily: "var(--font-mono)" }}
              stroke="var(--color-rule-strong)"
              label={{
                value: "market ADP",
                position: "insideBottom",
                offset: -16,
                fontSize: 11,
                fill: "var(--color-ink-mute)",
              }}
            />
            <YAxis
              type="number"
              dataKey="m_adp"
              domain={[0, maxAxis]}
              tick={{ fontSize: 11, fill: "var(--color-ink-soft)", fontFamily: "var(--font-mono)" }}
              stroke="var(--color-rule-strong)"
              label={{
                value: alpha ? "fair ADP" : "model-implied ADP",
                angle: -90,
                position: "insideLeft",
                fontSize: 11,
                fill: "var(--color-ink-mute)",
              }}
            />
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: maxAxis, y: maxAxis },
              ]}
              stroke="var(--color-accent)"
              strokeDasharray="4 4"
            />
            <Tooltip
              cursor={{ stroke: "var(--color-rule-strong)" }}
              content={({ payload }) => {
                const p = payload?.[0]?.payload as Row | undefined;
                if (!p) return null;
                return (
                  <div className="glass !rounded-xl px-2.5 py-1.5 text-[12px]">
                    <div className="font-medium">{p.name}</div>
                    <div className="num text-ink-soft">
                      {p.position}
                      {p.pos_rank} · adp {fmtAdp(p.adp)} · {alpha ? "fair" : "model"}{" "}
                      {fmtAdp(p.m_adp)} ·{" "}
                      <span className={edgeClass(p.m_edge)}>{fmtSigned(p.m_edge)}</span>
                      {alpha && p.alpha_z != null && <> · z {fmtDec(p.alpha_z, 2)}</>}
                    </div>
                  </div>
                );
              }}
            />
            <Scatter
              data={priced}
              fill="var(--color-ink-soft)"
              fillOpacity={0.8}
              onClick={(d: unknown) => {
                const p = d as Row;
                if (p?.player_id) nav(`/players/${p.player_id}`);
              }}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      </div>
    </div>
  );
}
