import { useMemo, useState } from "react";
import type { AdpBoardRow, DraftState, Player } from "../../lib/types";
import { RangeBar } from "../RangeBar";
import { PosBadge, Segmented } from "../ui";
import { edgeClass, fmtAdp, fmtPct, fmtSigned, fmtVorp } from "../../lib/format";

const POSITIONS = ["ALL", "QB", "RB", "WR", "TE", "K", "DST"];

interface Row {
  player_id: string;
  name: string;
  position: string;
  team: string | null;
  player: Player | null;
  adp: number | null;
  p_survive: number | null;
  streamer: boolean;
}

export function AvailableTable({
  state,
  playersById,
  adpById,
  surviveById,
  canPick,
  onPick,
  onInspect,
}: {
  state: DraftState;
  playersById: Map<string, Player>;
  adpById: Map<string, AdpBoardRow>;
  surviveById: Map<string, number | null>;
  canPick: boolean;
  onPick: (id: string) => void;
  onInspect: (id: string) => void;
}) {
  const [pos, setPos] = useState("ALL");
  const [q, setQ] = useState("");

  const rows = useMemo<Row[]>(() => {
    const out: Row[] = [];
    for (const a of state.available) {
      const p = playersById.get(a.player_id) ?? null;
      const adp = adpById.get(a.player_id) ?? null;
      if (!p && !adp) continue;
      out.push({
        player_id: a.player_id,
        name: p?.name ?? adp?.name ?? a.player_id,
        position: p?.position ?? adp?.position ?? "?",
        team: p?.team ?? adp?.team ?? null,
        player: p,
        adp: p?.adp ?? adp?.adp ?? null,
        p_survive: surviveById.get(a.player_id) ?? null,
        streamer: adp?.streamer ?? false,
      });
    }
    out.sort((a, b) => {
      const av = a.player?.vorp;
      const bv = b.player?.vorp;
      if (av != null && bv != null) return bv - av;
      if (av != null) return -1;
      if (bv != null) return 1;
      return (a.adp ?? 999) - (b.adp ?? 999);
    });
    return out;
  }, [state.available, playersById, adpById, surviveById]);

  const filtered = useMemo(() => {
    let out = rows;
    if (pos !== "ALL") out = out.filter((r) => r.position === pos);
    if (q.trim()) out = out.filter((r) => r.name.toLowerCase().includes(q.trim().toLowerCase()));
    return out.slice(0, 200);
  }, [rows, pos, q]);

  const domain = useMemo<[number, number]>(() => {
    const hi = Math.max(1, ...filtered.map((r) => r.player?.season_p90 ?? 0));
    return [0, hi];
  }, [filtered]);

  return (
    <div>
      <div className="flex items-center gap-3 mb-2">
        <span className="text-[11px] uppercase tracking-[0.1em] text-ink-mute">available</span>
        <Segmented options={POSITIONS} value={pos} onChange={setPos} />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="search…"
          aria-label="Search available players"
          className="field px-3 py-1 text-[12px] w-44"
        />
      </div>
      <div className="glass px-3 py-1 max-h-[520px] overflow-y-auto">
        <table className="data">
          <thead className="sticky top-0" style={{ background: "rgba(16,23,42,0.92)", backdropFilter: "blur(8px)" }}>
            <tr>
              <th>player</th>
              <th>pos</th>
              <th>team</th>
              <th className="num">vorp</th>
              <th>p10–p90</th>
              <th className="num">adp</th>
              <th className="num">edge</th>
              <th className="num">surv</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr key={r.player_id}>
                <td>
                  <button
                    className="font-medium hover:text-accent text-left"
                    onClick={() => r.player && onInspect(r.player_id)}
                  >
                    {r.name}
                  </button>
                  {r.streamer && (
                    <span className="ml-1.5 text-[10px] uppercase text-ink-mute">streamer</span>
                  )}
                  {r.player == null && !r.streamer && (
                    <span className="ml-1.5 text-[10px] uppercase text-ink-mute">no proj</span>
                  )}
                </td>
                <td>
                  <PosBadge pos={r.position} rank={r.player?.pos_rank} />
                </td>
                <td className="text-ink-soft">{r.team ?? "–"}</td>
                <td className="num font-medium">{fmtVorp(r.player?.vorp ?? null)}</td>
                <td>
                  {r.player ? (
                    <RangeBar
                      p10={r.player.season_p10}
                      p25={r.player.season_p25}
                      p50={r.player.season_p50}
                      p75={r.player.season_p75}
                      p90={r.player.season_p90}
                      domain={domain}
                      width={80}
                    />
                  ) : (
                    <span className="text-ink-mute">–</span>
                  )}
                </td>
                <td className="num text-ink-soft">{fmtAdp(r.adp)}</td>
                <td className={`num ${edgeClass(r.player?.adp_edge)}`}>
                  {fmtSigned(r.player?.adp_edge ?? null)}
                </td>
                <td className="num">
                  {r.p_survive == null ? "–" : (
                    <span className={r.p_survive < 0.4 ? "text-edge-neg" : ""}>
                      {fmtPct(r.p_survive)}
                    </span>
                  )}
                </td>
                <td>
                  <button
                    onClick={() => onPick(r.player_id)}
                    disabled={!canPick}
                    className="btn-ghost px-2.5 py-0.5 text-[11px] disabled:opacity-30"
                  >
                    draft
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
