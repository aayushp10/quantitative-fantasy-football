import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useFormat } from "../App";
import { useApi } from "../lib/api";
import type { Player } from "../lib/types";
import { RangeBar } from "../components/RangeBar";
import { ErrorPanel, Loading, PosBadge, Segmented, TierBreak } from "../components/ui";
import { edgeClass, fmtAdp, fmtDec, fmtPts, fmtSigned, fmtVorp } from "../lib/format";

const POSITIONS = ["ALL", "QB", "RB", "WR", "TE"];

type SortKey = "overall_rank" | "vorp" | "season_p50" | "adp" | "adp_edge" | "age";

const COLS: { key: SortKey | null; label: string; num?: boolean }[] = [
  { key: "overall_rank", label: "rk", num: true },
  { key: null, label: "player" },
  { key: null, label: "pos" },
  { key: null, label: "team" },
  { key: "age", label: "age", num: true },
  { key: "vorp", label: "vorp", num: true },
  { key: "season_p50", label: "p50", num: true },
  { key: null, label: "p10–p90" },
  { key: "adp", label: "adp", num: true },
  { key: "adp_edge", label: "edge", num: true },
];

export default function Rankings() {
  const { format } = useFormat();
  const [pos, setPos] = useState("ALL");
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({
    key: "overall_rank",
    asc: true,
  });
  const nav = useNavigate();
  const { data, error, loading } = useApi<Player[]>(`/api/players?format=${format}`);

  const rows = useMemo(() => {
    if (!data) return [];
    let out = data.filter((p) => p.overall_rank != null);
    if (pos !== "ALL") out = out.filter((p) => p.position === pos);
    if (q.trim()) out = out.filter((p) => p.name?.toLowerCase().includes(q.trim().toLowerCase()));
    const dir = sort.asc ? 1 : -1;
    out.sort((a, b) => {
      const av = a[sort.key];
      const bv = b[sort.key];
      if (av == null) return 1;
      if (bv == null) return -1;
      return (av - bv) * dir;
    });
    return out;
  }, [data, pos, q, sort]);

  const domain = useMemo<[number, number]>(() => {
    const hi = Math.max(1, ...rows.map((p) => p.season_p90 ?? 0));
    return [0, hi];
  }, [rows]);

  const showTiers = pos !== "ALL" && sort.key === "overall_rank" && sort.asc;

  if (error) return <ErrorPanel error={error} />;
  if (loading || !data) return <Loading label="rankings" />;

  const header = (c: (typeof COLS)[number]) =>
    c.key ? (
      <button
        className={`uppercase tracking-[0.08em] ${sort.key === c.key ? "text-accent" : ""}`}
        onClick={() =>
          setSort((s) =>
            s.key === c.key
              ? { key: c.key!, asc: !s.asc }
              : { key: c.key!, asc: c.key === "overall_rank" || c.key === "adp" || c.key === "age" },
          )
        }
      >
        {c.label}
        {sort.key === c.key ? (sort.asc ? " ↑" : " ↓") : ""}
      </button>
    ) : (
      c.label
    );

  let lastTier: number | null = null;

  return (
    <div className="float-in">
      <div className="flex items-center gap-3 mb-3">
        <Segmented options={POSITIONS} value={pos} onChange={setPos} />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="search player…"
          aria-label="Search player"
          className="field px-3 py-1.5 text-[13px] w-56"
        />
        <span className="num text-ink-mute text-[12px] ml-auto">{rows.length} players</span>
      </div>

      <div className="glass px-4 py-2">
        <table className="data">
          <thead>
            <tr>
              {COLS.map((c, i) => (
                <th key={i} className={c.num ? "num" : ""}>
                  {header(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => {
              const tierBreak =
                showTiers && p.tier != null && p.tier !== lastTier ? (
                  <TierBreak key={`t${p.tier}`} tier={p.tier} colSpan={COLS.length} />
                ) : null;
              if (showTiers && p.tier != null) lastTier = p.tier;
              return (
                <FragmentRow key={p.player_id} tierBreak={tierBreak}>
                  <tr
                    className="cursor-pointer"
                    tabIndex={0}
                    onClick={() => nav(`/players/${p.player_id}`)}
                    onKeyDown={(e) => e.key === "Enter" && nav(`/players/${p.player_id}`)}
                  >
                    <td className="num text-ink-mute">{p.overall_rank}</td>
                    <td className="font-medium">
                      {p.name}
                      {p.rookie && (
                        <span className="ml-1.5 text-[10px] uppercase text-accent tracking-wide">R</span>
                      )}
                    </td>
                    <td>
                      <PosBadge pos={p.position} rank={p.pos_rank} />
                    </td>
                    <td className="text-ink-soft">{p.team ?? "–"}</td>
                    <td className="num">{fmtDec(p.age, 0)}</td>
                    <td className="num font-semibold">{fmtVorp(p.vorp)}</td>
                    <td className="num">{fmtPts(p.season_p50)}</td>
                    <td>
                      <RangeBar
                        p10={p.season_p10}
                        p25={p.season_p25}
                        p50={p.season_p50}
                        p75={p.season_p75}
                        p90={p.season_p90}
                        domain={domain}
                      />
                    </td>
                    <td className="num text-ink-soft">{fmtAdp(p.adp)}</td>
                    <td
                      className={`num ${edgeClass(p.adp_edge)}`}
                      title="positive: market lets you wait; negative: must reach"
                    >
                      {fmtSigned(p.adp_edge)}
                    </td>
                  </tr>
                </FragmentRow>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FragmentRow({
  tierBreak,
  children,
}: {
  tierBreak: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <>
      {tierBreak}
      {children}
    </>
  );
}
