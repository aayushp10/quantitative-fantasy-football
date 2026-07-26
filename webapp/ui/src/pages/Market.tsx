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
import { edgeClass, fmtAdp, fmtSigned, fmtVorp } from "../lib/format";

const POSITIONS = ["ALL", "QB", "RB", "WR", "TE"];

export default function Market() {
  const { format } = useFormat();
  const [pos, setPos] = useState("ALL");
  const nav = useNavigate();
  const { data, error, loading } = useApi<Player[]>(`/api/players?format=${format}`);

  // Draft-relevant universe only: deep-tail players get ladder-overflow
  // model ADPs (edge ±200) that carry no market signal and drown the boards.
  const MAX_ADP = 150;
  const priced = useMemo(
    () =>
      (data ?? []).filter(
        (p) =>
          p.adp != null &&
          p.adp <= MAX_ADP &&
          p.predicted_adp != null &&
          p.predicted_adp <= MAX_ADP * 1.4 &&
          p.adp_edge != null &&
          (pos === "ALL" || p.position === pos),
      ),
    [data, pos],
  );

  const modelLoves = useMemo(
    () => [...priced].sort((a, b) => b.adp_edge! - a.adp_edge!).slice(0, 15),
    [priced],
  );
  const marketLoves = useMemo(
    () => [...priced].sort((a, b) => a.adp_edge! - b.adp_edge!).slice(0, 15),
    [priced],
  );
  const maxAxis = useMemo(
    () => Math.ceil(Math.max(...priced.map((p) => Math.max(p.adp!, p.predicted_adp!)), 10) / 10) * 10,
    [priced],
  );

  if (error) return <ErrorPanel error={error} />;
  if (loading || !data) return <Loading label="market comparison" />;

  const board = (rows: Player[], title: string, note: string) => (
    <div className="flex-1 min-w-0 glass px-4 py-3">
      <SectionTitle>{title}</SectionTitle>
      <p className="text-[11px] text-ink-mute -mt-1 mb-2">{note}</p>
      <table className="data">
        <thead>
          <tr>
            <th>player</th>
            <th>pos</th>
            <th className="num">adp</th>
            <th className="num">model adp</th>
            <th className="num">edge</th>
            <th className="num">vorp</th>
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
              <td className="num text-accent">{fmtAdp(p.predicted_adp)}</td>
              <td className={`num ${edgeClass(p.adp_edge)}`}>{fmtSigned(p.adp_edge)}</td>
              <td className="num">{fmtVorp(p.vorp)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return (
    <div>
      <div className="flex items-center gap-3 mb-3">
        <Segmented options={POSITIONS} value={pos} onChange={setPos} />
        <span className="text-[12px] text-ink-mute">
          edge = ADP − model-implied ADP (pre-market model — the ranking before the ADP blend).{" "}
          <span className="text-edge-pos">+ market lets you wait</span> ·{" "}
          <span className="text-edge-neg">− you must reach</span> · draft-relevant universe (ADP ≤ {MAX_ADP})
        </span>
      </div>

      <div className="flex gap-4 items-start">
        {board(modelLoves, "Model loves (market late)", "biggest positive edge — value at cost")}
        {board(marketLoves, "Market loves (model out)", "biggest negative edge — the market pays more than the model would")}
      </div>

      <div className="glass px-5 py-4 mt-4 max-w-3xl">
      <SectionTitle>Model-implied ADP vs market ADP</SectionTitle>
      <p className="text-[11px] text-ink-mute -mt-1 mb-2">
        below the diagonal: model prices the player earlier than the market drafts them
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
              dataKey="predicted_adp"
              domain={[0, maxAxis]}
              tick={{ fontSize: 11, fill: "var(--color-ink-soft)", fontFamily: "var(--font-mono)" }}
              stroke="var(--color-rule-strong)"
              label={{
                value: "model-implied ADP",
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
                const p = payload?.[0]?.payload as Player | undefined;
                if (!p) return null;
                return (
                  <div className="glass !rounded-xl px-2.5 py-1.5 text-[12px]">
                    <div className="font-medium">{p.name}</div>
                    <div className="num text-ink-soft">
                      {p.position}
                      {p.pos_rank} · adp {fmtAdp(p.adp)} · model {fmtAdp(p.predicted_adp)} ·{" "}
                      <span className={edgeClass(p.adp_edge)}>{fmtSigned(p.adp_edge)}</span>
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
                const p = d as Player;
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
