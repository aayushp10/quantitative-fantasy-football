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
    <div className="my-8 border border-rule-strong bg-paper-raised p-6 max-w-xl">
      <div className="font-medium mb-1">
        {apiDown ? "API not reachable" : `Error ${error.status}`}
      </div>
      <div className="text-ink-soft mb-2">{error.message}</div>
      {apiDown && (
        <div className="text-ink-soft">
          Start the backend first:{" "}
          <code className="num bg-paper px-1 border border-rule">make web-api</code>
          {"  "}(or <code className="num bg-paper px-1 border border-rule">make web</code> for both).
          {error.status === 503 && " Then build artifacts with make web-data."}
        </div>
      )}
    </div>
  );
}

/** Position rendered as text — identity by label, not color. */
export function Pos({ pos }: { pos: string }) {
  return <span className="num text-ink-soft">{pos}</span>;
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <h2 className="text-[11px] uppercase tracking-[0.08em] text-ink-mute font-medium border-b border-rule-strong pb-1 mb-2 mt-6">
      {children}
    </h2>
  );
}

export function TierBreak({ tier, colSpan }: { tier: number; colSpan: number }) {
  return (
    <tr aria-label={`Tier ${tier} break`}>
      <td colSpan={colSpan} className="!border-b-0 !p-0">
        <div className="flex items-center gap-2 pt-3 pb-0.5">
          <span className="text-[10px] uppercase tracking-[0.1em] text-ink-mute whitespace-nowrap">
            tier {tier}
          </span>
          <span className="h-px flex-1 bg-rule-strong" />
        </div>
      </td>
    </tr>
  );
}
