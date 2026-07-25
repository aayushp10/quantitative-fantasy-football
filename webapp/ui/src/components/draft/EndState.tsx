import { useMemo } from "react";
import type { AdpBoardRow, DraftState, Player } from "../../lib/types";
import { fmtPts } from "../../lib/format";

export function EndState({
  state,
  playersById,
  adpById,
  onExit,
  onUndo,
}: {
  state: DraftState;
  playersById: Map<string, Player>;
  adpById: Map<string, AdpBoardRow>;
  onExit: () => void;
  onUndo: () => void;
}) {
  const { teams, rounds, user_slot } = state.config;

  const grid = useMemo(() => {
    const g: (typeof state.picks)[number][][] = Array.from({ length: rounds }, () => []);
    for (const p of state.picks) g[p.round - 1][p.slot - 1] = p;
    return g;
  }, [state.picks, rounds]);

  const totals = useMemo(() => {
    const t = new Map<number, number>();
    for (const p of state.picks) {
      const pts = playersById.get(p.player_id)?.season_p50 ?? 0;
      t.set(p.slot, (t.get(p.slot) ?? 0) + pts);
    }
    return t;
  }, [state.picks, playersById]);

  const userTotal = totals.get(user_slot) ?? 0;
  const botTotals = [...totals.entries()]
    .filter(([s]) => s !== user_slot)
    .map(([, v]) => v)
    .sort((a, b) => a - b);
  const median =
    botTotals.length % 2
      ? botTotals[(botTotals.length - 1) / 2]
      : (botTotals[botTotals.length / 2 - 1] + botTotals[botTotals.length / 2]) / 2;
  const diff = userTotal - median;

  const exportCsv = () => {
    const lines = ["overall,round,slot,team_label,player,position,nfl_team,is_user"];
    for (const p of state.picks) {
      lines.push(
        [
          p.overall,
          p.round,
          p.slot,
          p.slot === user_slot ? "YOU" : `Bot ${p.slot}`,
          `"${p.player_name}"`,
          p.position,
          p.team ?? adpById.get(p.player_id)?.team ?? "",
          p.is_user,
        ].join(","),
      );
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `mock_draft_${state.draft_id}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="float-in">
      <div className="glass flex items-center gap-6 mb-4 px-5 py-3">
        <h1 className="text-lg font-semibold">Draft complete</h1>
        <span className="num text-[14px]">
          your projected points <span className="font-semibold">{fmtPts(userTotal)}</span> · league
          median {fmtPts(median)} ·{" "}
          <span className={diff >= 0 ? "text-edge-pos" : "text-edge-neg"}>
            {diff >= 0 ? "+" : ""}
            {fmtPts(diff)}
          </span>
        </span>
        <button onClick={exportCsv} className="btn-primary ml-auto px-4 py-1.5 text-[12px]">
          export CSV
        </button>
        <button onClick={onUndo} className="btn-ghost px-4 py-1.5 text-[12px]">
          undo last pick
        </button>
        <button onClick={onExit} className="text-[12px] text-ink-mute hover:text-edge-neg">
          new draft
        </button>
      </div>

      <div className="glass px-4 py-3 overflow-x-auto">
        <table className="data">
          <thead>
            <tr>
              <th className="num">rd</th>
              {Array.from({ length: teams }, (_, i) => (
                <th key={i} className={i + 1 === user_slot ? "!text-accent" : ""}>
                  {i + 1 === user_slot ? "YOU" : `t${i + 1}`}
                  <span className="num block normal-case tracking-normal text-ink-mute">
                    {fmtPts(totals.get(i + 1) ?? 0)}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.map((row, r) => (
              <tr key={r}>
                <td className="num text-ink-mute">{r + 1}</td>
                {Array.from({ length: teams }, (_, s) => {
                  const p = row[s];
                  return (
                    <td
                      key={s}
                      className={`!whitespace-normal text-[11.5px] leading-tight ${
                        s + 1 === user_slot ? "bg-accent-soft" : ""
                      }`}
                    >
                      {p ? (
                        <>
                          <span className={p.is_user ? "font-medium" : ""}>{p.player_name}</span>
                          <span className="num text-ink-mute"> {p.position}</span>
                        </>
                      ) : (
                        "—"
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
