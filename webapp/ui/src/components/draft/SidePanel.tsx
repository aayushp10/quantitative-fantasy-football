import { useEffect, useMemo, useRef } from "react";
import type { AdpBoardRow, DraftState, Pick, Player } from "../../lib/types";

/** Assign a team's picks (in draft order) to roster slots greedily. */
export function assignSlots(
  picks: { position: string; player_id: string }[],
  roster: Record<string, number>,
): { slot: string; player_id: string | null }[] {
  const slots: { slot: string; player_id: string | null }[] = [];
  const order = ["QB", "RB", "WR", "TE", "FLEX", "SUPERFLEX", "K", "DST", "BN"];
  for (const s of order) {
    for (let i = 0; i < (roster[s] ?? 0); i++) slots.push({ slot: s, player_id: null });
  }
  const flexOk = new Set(["RB", "WR", "TE"]);
  const sfOk = new Set(["QB", "RB", "WR", "TE"]);
  for (const p of picks) {
    let placed =
      place(slots, p.position, p.player_id) ||
      (flexOk.has(p.position) && place(slots, "FLEX", p.player_id)) ||
      (sfOk.has(p.position) && place(slots, "SUPERFLEX", p.player_id)) ||
      place(slots, "BN", p.player_id);
    if (!placed) slots.push({ slot: "BN+", player_id: p.player_id });
  }
  return slots;
}

function place(
  slots: { slot: string; player_id: string | null }[],
  name: string,
  pid: string,
): boolean {
  const s = slots.find((x) => x.slot === name && x.player_id === null);
  if (s) {
    s.player_id = pid;
    return true;
  }
  return false;
}

export function SidePanel({
  state,
  playersById,
  adpById,
  needWeights,
  newPicksFrom,
  busy,
  onUndo,
  onExit,
}: {
  state: DraftState;
  playersById: Map<string, Player>;
  adpById: Map<string, AdpBoardRow>;
  needWeights: Record<string, number> | null;
  newPicksFrom: number;
  busy: boolean;
  onUndo: () => void;
  onExit: () => void;
}) {
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [state.picks.length]);

  const myPicks = useMemo(
    () => state.picks.filter((p) => p.is_user),
    [state.picks],
  );
  const slots = useMemo(
    () => assignSlots(myPicks, state.config.roster),
    [myPicks, state.config.roster],
  );

  const name = (pid: string | null) => {
    if (!pid) return null;
    const p = playersById.get(pid);
    if (p) return `${p.name} · ${p.position}${p.pos_rank ?? ""}`;
    const a = adpById.get(pid);
    return a ? `${a.name} · ${a.position}` : pid;
  };

  const otc = state.on_the_clock;

  return (
    <div className="flex-1 min-w-[300px] max-w-[380px]">
      <div className="flex items-center gap-3 border-b border-ink pb-1.5 mb-3">
        <span className="num text-[13px]">
          {otc
            ? `pick ${otc.overall}/${state.config.teams * state.config.rounds} · round ${otc.round}`
            : "draft complete"}
        </span>
        <button
          onClick={onUndo}
          disabled={busy || !state.picks.some((p) => p.is_user)}
          className="ml-auto border border-rule-strong px-2 py-0.5 text-[11px] hover:border-ink disabled:opacity-30"
        >
          undo my last pick
        </button>
        <button onClick={onExit} className="text-[11px] text-ink-mute hover:text-edge-neg">
          abandon
        </button>
      </div>

      <div className="text-[11px] uppercase tracking-[0.08em] text-ink-mute mb-1">
        your roster · slot {state.config.user_slot}
      </div>
      <table className="data mb-3">
        <tbody>
          {slots.map((s, i) => (
            <tr key={i}>
              <td className="num text-ink-mute w-14">{s.slot}</td>
              <td className={s.player_id ? "" : "text-ink-mute"}>
                {name(s.player_id) ?? "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {needWeights && (
        <>
          <div className="text-[11px] uppercase tracking-[0.08em] text-ink-mute mb-1">
            positional need
          </div>
          <div className="mb-3">
            {Object.entries(needWeights).map(([pos, w]) => (
              <div key={pos} className="flex items-center gap-2 py-0.5">
                <span className="num text-[11px] w-7 text-ink-soft">{pos}</span>
                <div className="h-[5px] flex-1 bg-rule">
                  <div
                    className="h-full bg-ink-soft"
                    style={{ width: `${Math.round(w * 100)}%` }}
                  />
                </div>
                <span className="num text-[11px] text-ink-mute w-8 text-right">
                  {Math.round(w * 100)}
                </span>
              </div>
            ))}
          </div>
        </>
      )}

      <div className="text-[11px] uppercase tracking-[0.08em] text-ink-mute mb-1">pick log</div>
      <div ref={logRef} className="max-h-[380px] overflow-y-auto border-t border-rule">
        <table className="data">
          <tbody>
            {state.picks.map((p: Pick, i: number) => (
              <tr
                key={p.overall}
                className={`${i >= newPicksFrom ? "pick-in" : ""} ${
                  p.is_user ? "bg-accent-soft" : ""
                }`}
              >
                <td className="num text-ink-mute w-12">
                  {p.round}.{String(p.overall - (p.round - 1) * state.config.teams).padStart(2, "0")}
                </td>
                <td className="num text-ink-mute w-8">t{p.slot}</td>
                <td className={p.is_user ? "font-medium" : ""}>{p.player_name}</td>
                <td className="num text-ink-soft">{p.position}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
