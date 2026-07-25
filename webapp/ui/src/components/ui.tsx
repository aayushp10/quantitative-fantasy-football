import { ReactNode } from "react";
import { ApiError } from "../lib/api";

export function Loading({ label = "loading" }: { label?: string }) {
  return (
    <div className="py-12 text-center text-ink-mute" role="status">
      <span className="num">… {label}</span>
    </div>
  );
}

export function ErrorPanel({ error }: { error: ApiError }) {
  const apiDown = error.status === 0 || error.status === 502 || error.status === 503;
  return (
    <div className="glass my-8 p-6 max-w-xl">
      <div className="font-semibold mb-1">
        {apiDown ? "API not reachable" : `Error ${error.status}`}
      </div>
      <div className="text-ink-soft mb-2">{error.message}</div>
      {apiDown && (
        <div className="text-ink-soft">
          Start the backend first:{" "}
          <code className="num glass-soft px-1.5 py-0.5">make web-api</code>
          {"  "}(or <code className="num glass-soft px-1.5 py-0.5">make web</code> for both).
          {error.status === 503 && " Then build artifacts with make web-data."}
        </div>
      )}
    </div>
  );
}

export const POS_COLORS: Record<string, string> = {
  QB: "var(--color-pos-qb)",
  WR: "var(--color-pos-wr)",
  RB: "var(--color-pos-rb)",
  TE: "var(--color-pos-te)",
  K: "var(--color-pos-k)",
  DST: "var(--color-pos-dst)",
};

/** Sleeper-style position pill. The text label carries identity; color
 * reinforces it (validated palette, see index.css). */
export function PosBadge({ pos, rank }: { pos: string; rank?: number | null }) {
  const c = POS_COLORS[pos] ?? "var(--color-ink-mute)";
  return (
    <span
      className="pos-badge"
      style={{
        color: `color-mix(in srgb, ${c} 70%, white)`,
        background: `color-mix(in srgb, ${c} 17%, transparent)`,
        borderColor: `color-mix(in srgb, ${c} 40%, transparent)`,
      }}
    >
      {pos}
      {rank ?? ""}
    </span>
  );
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <h2 className="text-[10.5px] uppercase tracking-[0.1em] text-ink-mute font-semibold pb-1 mb-2 mt-6 border-b border-rule-strong">
      {children}
    </h2>
  );
}

export function TierBreak({ tier, colSpan }: { tier: number; colSpan: number }) {
  return (
    <tr aria-label={`Tier ${tier} break`}>
      <td colSpan={colSpan} className="!border-b-0 !p-0">
        <div className="flex items-center gap-2 pt-4 pb-1">
          <span className="pill glass-soft px-2.5 py-0.5 text-[10px] uppercase tracking-[0.1em] text-accent whitespace-nowrap">
            tier {tier}
          </span>
          <span className="h-px flex-1 bg-rule-strong" />
        </div>
      </td>
    </tr>
  );
}

export function Segmented({
  options,
  value,
  onChange,
}: {
  options: string[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="seg" role="group">
      {options.map((o) => (
        <button key={o} aria-pressed={value === o} onClick={() => onChange(o)} className="num">
          {o}
        </button>
      ))}
    </div>
  );
}
