import { useState } from "react";
import type { Recommendation, Recommendations, TierCliffAlert } from "../../lib/types";
import { RangeBar } from "../RangeBar";
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
      <div className="border border-rule-strong bg-paper-raised p-4 mb-4 text-ink-soft">
        {busy ? "bots picking…" : otc ? `waiting — pick ${otc.overall}, team ${otc.slot} on the clock` : ""}
      </div>
    );
  }
  if (!recs) {
    return (
      <div className="border border-rule-strong p-4 mb-4 text-ink-mute num">… computing recommendations</div>
    );
  }
  const [top, ...rest] = recs.recommendations;
  return (
    <div className="mb-5">
      {alerts.length > 0 && (
        <div
          role="alert"
          className="border border-accent bg-accent-soft px-3 py-1.5 mb-3 text-[13px]"
        >
          <span className="font-medium uppercase tracking-wide text-[11px] mr-2">tier cliff</span>
          {alerts.map((a) => (
            <span key={a.position} className="mr-4 num">
              {a.position} tier {a.tier}: {a.remaining_in_tier} left, −{a.drop_to_next_tier} VORP below
            </span>
          ))}
        </div>
      )}
      <div className="text-[11px] uppercase tracking-[0.08em] text-ink-mute mb-1.5">
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
    <tr className={strong ? "text-ink font-medium" : ""}>
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
    <div className="border-2 border-accent p-3 bg-paper">
      <div className="flex items-baseline gap-3">
        <button
          className="text-[16px] font-medium hover:text-accent"
          onClick={() => onInspect(r.player_id)}
        >
          {r.name}
        </button>
        <span className="num text-ink-soft">
          {r.position}
          {r.pos_rank} · {r.team ?? "FA"} · tier {r.tier ?? "–"}
        </span>
        <span className="ml-auto num text-[15px] font-medium">score {fmtDec(r.rec_score, 1)}</span>
        <button
          onClick={() => onPick(r.player_id)}
          disabled={busy}
          className="bg-ink text-paper px-4 py-1 text-[13px] hover:bg-accent disabled:opacity-40"
        >
          Draft
        </button>
      </div>
      <div className="flex items-center gap-6 mt-2 num text-[13px]">
        <span>
          VORP <span className="font-medium">{fmtVorp(r.vorp)}</span>
        </span>
        <span>
          ADP {fmtAdp(r.adp)}{" "}
          <span className={edgeClass(r.adp_edge)}>{fmtSigned(r.adp_edge)}</span>
        </span>
        <span>
          survives to your pick{" "}
          <span className={r.p_survive < 0.4 ? "text-edge-neg font-medium" : "font-medium"}>
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
    <div className="border border-rule-strong p-2 flex flex-col">
      <button
        className="text-left font-medium text-[13px] leading-tight hover:text-accent"
        onClick={() => onInspect(r.player_id)}
      >
        {r.name}
      </button>
      <span className="num text-[11px] text-ink-soft">
        {r.position}
        {r.pos_rank} · t{r.tier ?? "–"}
      </span>
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
        className="mt-auto border border-ink px-2 py-0.5 text-[12px] hover:bg-ink hover:text-paper disabled:opacity-40 self-start"
      >
        draft
      </button>
    </div>
  );
}
