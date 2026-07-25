import { useState } from "react";
import type { Recommendation, Recommendations, TierCliffAlert } from "../../lib/types";
import { RangeBar } from "../RangeBar";
import { PosBadge } from "../ui";
import { edgeClass, fmtAdp, fmtDec, fmtPct, fmtSigned, fmtVorp } from "../../lib/format";

export function RecPanel({
  recs,
  alerts,
  onClock,
  busy,
  otc,
  userSlot,
  onPick,
  onInspect,
}: {
  recs: Recommendations | null;
  alerts: TierCliffAlert[];
  onClock: boolean;
  busy: boolean;
  otc: { overall: number; round: number; slot: number } | null;
  userSlot: number;
  onPick: (id: string) => void;
  onInspect: (id: string) => void;
}) {
  if (!onClock) {
    return (
      <div className="glass p-4 mb-4 text-ink-soft">
        {busy ? "bots picking…" : otc ? `waiting — pick ${otc.overall}, team ${otc.slot} on the clock` : ""}
      </div>
    );
  }
  if (!recs) {
    return <div className="glass p-4 mb-4 text-ink-mute num">… computing recommendations</div>;
  }
  const [top, ...rest] = recs.recommendations;
  return (
    <div className="mb-5 float-in">
      {alerts.length > 0 && (
        <div
          role="alert"
          className="glass-soft !border-accent/40 px-4 py-2 mb-3 text-[13px]"
          style={{ borderColor: "rgba(45,212,191,0.4)", background: "rgba(45,212,191,0.08)" }}
        >
          <span className="font-semibold uppercase tracking-wide text-[11px] mr-2 text-accent">
            tier cliff
          </span>
          {alerts.map((a) => (
            <span key={a.position} className="mr-4 num">
              {a.position} tier {a.tier}: {a.remaining_in_tier} left, −{a.drop_to_next_tier} VORP below
            </span>
          ))}
        </div>
      )}
      <div className="text-[10.5px] uppercase tracking-[0.1em] text-ink-mute mb-1.5">
        your pick · #{otc?.overall} · round {otc?.round} · slot {userSlot}
      </div>
      {top && <TopCard r={top} onPick={onPick} onInspect={onInspect} busy={busy} />}
      <div className="grid grid-cols-5 gap-2 mt-2">
        {rest.map((r) => (
          <RunnerUp key={r.player_id} r={r} onPick={onPick} onInspect={onInspect} busy={busy} />
        ))}
      </div>
    </div>
  );
}

function Why({ r }: { r: Recommendation }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-1.5">
      <button
        onClick={(e) => {
          e.stopPropagation();
          setOpen(!open);
        }}
        aria-expanded={open}
        className="text-[11px] text-accent hover:underline"
      >
        {open ? "hide" : "why this pick"}
      </button>
      {open && (
        <table className="num text-[11px] mt-1 text-ink-soft">
          <tbody>
            <Row k="vorp" v={fmtVorp(r.vorp)} />
            <Row k="need_weight" v={fmtDec(r.need_weight, 2)} />
            <Row k="need_multiplier" v={`×${fmtDec(r.need_multiplier, 3)}`} />
            <Row k="tier_drop" v={fmtVorp(r.tier_drop)} />
            <Row k="p_survive" v={fmtPct(r.p_survive)} />
            <Row k="urgency = (1−p)·drop" v={fmtDec(r.urgency, 1)} />
            <Row k="rec_score = vorp·mult + urgency" v={fmtDec(r.rec_score, 1)} strong />
          </tbody>
        </table>
      )}
    </div>
  );
}

function Row({ k, v, strong }: { k: string; v: string; strong?: boolean }) {
  return (
    <tr className={strong ? "text-ink font-semibold" : ""}>
      <td className="pr-3">{k}</td>
      <td className="text-right">{v}</td>
    </tr>
  );
}

function TopCard({
  r,
  onPick,
  onInspect,
  busy,
}: {
  r: Recommendation;
  onPick: (id: string) => void;
  onInspect: (id: string) => void;
  busy: boolean;
}) {
  return (
    <div
      className="glass p-4"
      style={{ borderColor: "rgba(45,212,191,0.45)", boxShadow: "0 10px 34px rgba(2,6,18,0.45), 0 0 24px rgba(45,212,191,0.12), inset 0 1px 0 rgba(255,255,255,0.1)" }}
    >
      <div className="flex items-center gap-3">
        <PosBadge pos={r.position} rank={r.pos_rank} />
        <button
          className="text-[17px] font-semibold hover:text-accent"
          onClick={() => onInspect(r.player_id)}
        >
          {r.name}
        </button>
        <span className="num text-ink-soft">
          {r.team ?? "FA"} · tier {r.tier ?? "–"}
        </span>
        <span className="ml-auto num text-[15px] font-semibold">score {fmtDec(r.rec_score, 1)}</span>
        <button
          onClick={() => onPick(r.player_id)}
          disabled={busy}
          className="btn-primary px-5 py-1.5 text-[13px] disabled:opacity-40"
        >
          Draft
        </button>
      </div>
      <div className="flex items-center gap-6 mt-2.5 num text-[13px]">
        <span>
          VORP <span className="font-semibold">{fmtVorp(r.vorp)}</span>
        </span>
        <span>
          ADP {fmtAdp(r.adp)}{" "}
          <span className={edgeClass(r.adp_edge)}>{fmtSigned(r.adp_edge)}</span>
        </span>
        <span>
          survives to your pick{" "}
          <span className={r.p_survive < 0.4 ? "text-edge-neg font-semibold" : "font-semibold"}>
            {fmtPct(r.p_survive)}
          </span>
        </span>
        <RangeBar
          p10={r.season_p10}
          p25={r.season_p25}
          p50={r.season_p50}
          p75={r.season_p75}
          p90={r.season_p90}
          domain={[0, Math.max(1, r.season_p90 ?? 1) * 1.05]}
          width={140}
          accent
        />
      </div>
      <Why r={r} />
    </div>
  );
}

function RunnerUp({
  r,
  onPick,
  onInspect,
  busy,
}: {
  r: Recommendation;
  onPick: (id: string) => void;
  onInspect: (id: string) => void;
  busy: boolean;
}) {
  return (
    <div className="glass-soft p-2.5 flex flex-col">
      <div className="flex items-center gap-1.5 mb-0.5">
        <PosBadge pos={r.position} rank={r.pos_rank} />
        <span className="num text-[10.5px] text-ink-mute">t{r.tier ?? "–"}</span>
      </div>
      <button
        className="text-left font-medium text-[13px] leading-tight hover:text-accent"
        onClick={() => onInspect(r.player_id)}
      >
        {r.name}
      </button>
      <span className="num text-[12px] mt-1">
        v {fmtVorp(r.vorp)} · s {fmtDec(r.rec_score, 1)}
      </span>
      <span className="num text-[12px]">
        surv <span className={r.p_survive < 0.4 ? "text-edge-neg" : ""}>{fmtPct(r.p_survive)}</span>
      </span>
      <Why r={r} />
      <button
        onClick={() => onPick(r.player_id)}
        disabled={busy}
        className="btn-ghost mt-auto px-3 py-0.5 text-[12px] disabled:opacity-40 self-start"
      >
        draft
      </button>
    </div>
  );
}
