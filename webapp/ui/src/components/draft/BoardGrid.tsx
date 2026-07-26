import { useEffect, useMemo, useRef } from "react";
import type { DraftState, Pick } from "../../lib/types";
import { POS_COLORS, Segmented } from "../ui";

export type BotSpeed = "slow" | "fast" | "instant";
export const SPEED_MS: Record<BotSpeed, number> = { slow: 2000, fast: 900, instant: 0 };

/** Sleeper-style draft board: teams across, rounds down, snake order,
 * position-colored pick cards, pulsing cell for the pick on the clock. */
export function BoardGrid({
  state,
  speed,
  onSpeed,
}: {
  state: DraftState;
  speed: BotSpeed;
  onSpeed: (s: BotSpeed) => void;
}) {
  const { teams, rounds, user_slot } = state.config;
  const scrollRef = useRef<HTMLDivElement>(null);

  const byOverall = useMemo(() => {
    const m = new Map<number, Pick>();
    state.picks.forEach((p) => m.set(p.overall, p));
    return m;
  }, [state.picks]);

  const currentOverall = state.on_the_clock?.overall ?? null;
  const currentRound = state.on_the_clock?.round ?? rounds;

  // Keep the active round in view as picks come in
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const rowH = 44;
    const target = Math.max(0, (currentRound - 2) * rowH);
    el.scrollTo({ top: target, behavior: "smooth" });
  }, [currentRound, state.picks.length]);

  const overallFor = (round: number, slot: number) =>
    (round - 1) * teams + (round % 2 === 1 ? slot : teams - slot + 1);

  return (
    <div className="glass p-3 mb-4">
      <div className="flex items-center gap-3 mb-2">
        <span className="text-[10.5px] uppercase tracking-[0.1em] text-ink-mute">
          draft board
        </span>
        <span className="num text-[11px] text-ink-soft">
          pick {currentOverall ?? "—"}/{teams * rounds}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-wide text-ink-mute">bot speed</span>
          <Segmented options={["slow", "fast", "instant"]} value={speed} onChange={(v) => onSpeed(v as BotSpeed)} />
        </div>
      </div>
      <div className="overflow-x-auto">
        <div
          className="grid gap-1 min-w-[900px]"
          style={{ gridTemplateColumns: `34px repeat(${teams}, minmax(76px, 1fr))` }}
        >
          <div />
          {Array.from({ length: teams }, (_, i) => i + 1).map((slot) => (
            <div
              key={slot}
              className={`num text-center text-[10.5px] uppercase tracking-wide pb-0.5 ${
                slot === user_slot ? "text-accent font-semibold" : "text-ink-mute"
              }`}
            >
              {slot === user_slot ? "you" : `t${slot}`}
            </div>
          ))}
        </div>
        <div ref={scrollRef} className="max-h-[228px] overflow-y-auto">
          <div
            className="grid gap-1 min-w-[900px]"
            style={{ gridTemplateColumns: `34px repeat(${teams}, minmax(76px, 1fr))` }}
          >
            {Array.from({ length: rounds }, (_, r) => r + 1).map((round) => (
              <BoardRow
                key={round}
                round={round}
                teams={teams}
                userSlot={user_slot}
                byOverall={byOverall}
                currentOverall={currentOverall}
                overallFor={overallFor}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function BoardRow({
  round,
  teams,
  userSlot,
  byOverall,
  currentOverall,
  overallFor,
}: {
  round: number;
  teams: number;
  userSlot: number;
  byOverall: Map<number, Pick>;
  currentOverall: number | null;
  overallFor: (round: number, slot: number) => number;
}) {
  return (
    <>
      <div className="num text-[10.5px] text-ink-mute flex items-center justify-center h-10">
        {round}
      </div>
      {Array.from({ length: teams }, (_, i) => i + 1).map((slot) => {
        const overall = overallFor(round, slot);
        const p = byOverall.get(overall);
        const onClock = overall === currentOverall;
        const c = p ? POS_COLORS[p.position] ?? "var(--color-ink-mute)" : null;
        return (
          <div
            key={slot}
            className={`h-10 rounded-md px-1.5 py-1 text-[10.5px] leading-tight overflow-hidden ${
              onClock ? "clock-pulse" : ""
            } ${slot === userSlot && !p ? "bg-[rgba(45,212,191,0.05)]" : ""}`}
            style={{
              border: `1px solid ${
                p
                  ? `color-mix(in srgb, ${c} 38%, transparent)`
                  : onClock
                    ? "rgba(45,212,191,0.65)"
                    : "rgba(255,255,255,0.07)"
              }`,
              background: p ? `color-mix(in srgb, ${c} 13%, transparent)` : undefined,
            }}
          >
            {p ? (
              <>
                <div className="flex items-baseline gap-1">
                  <span className="num text-[9px] text-ink-mute">
                    {round}.{String(overall - (round - 1) * teams).padStart(2, "0")}
                  </span>
                  <span
                    className="num text-[9px] font-semibold"
                    style={{ color: `color-mix(in srgb, ${c} 75%, white)` }}
                  >
                    {p.position}
                  </span>
                </div>
                <div className={`truncate ${p.is_user ? "text-accent font-semibold" : "text-ink"}`}>
                  {shortName(p.player_name)}
                </div>
              </>
            ) : (
              <span className="num text-[9px] text-ink-mute">
                {onClock ? "on the clock" : `${round}.${String(overall - (round - 1) * teams).padStart(2, "0")}`}
              </span>
            )}
          </div>
        );
      })}
    </>
  );
}

function shortName(name: string): string {
  const parts = name.split(" ");
  if (parts.length < 2) return name;
  return `${parts[0][0]}. ${parts.slice(1).join(" ")}`;
}
