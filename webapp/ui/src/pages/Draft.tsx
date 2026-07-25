import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useFormat } from "../App";
import { api, ApiError, useApi } from "../lib/api";
import type {
  AdpBoardRow,
  DraftState,
  Player,
  Recommendations,
  TierCliffAlert,
} from "../lib/types";
import { ErrorPanel, Loading } from "../components/ui";
import { RecPanel } from "../components/draft/RecPanel";
import { AvailableTable } from "../components/draft/AvailableTable";
import { SidePanel } from "../components/draft/SidePanel";
import { EndState } from "../components/draft/EndState";
import { PlayerDrawer } from "../components/draft/PlayerDrawer";

const ACTIVE_KEY = "ff-active-draft";
const DEFAULT_ROSTER: Record<string, number> = {
  QB: 1, RB: 2, WR: 3, TE: 1, FLEX: 1, BN: 7, K: 0, DST: 0,
};

export default function Draft() {
  const [draftId, setDraftId] = useState<string | null>(
    () => localStorage.getItem(ACTIVE_KEY),
  );
  if (!draftId) {
    return (
      <Setup
        onCreated={(id) => {
          localStorage.setItem(ACTIVE_KEY, id);
          setDraftId(id);
        }}
      />
    );
  }
  return (
    <DraftRoom
      draftId={draftId}
      onExit={() => {
        localStorage.removeItem(ACTIVE_KEY);
        setDraftId(null);
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

function Setup({ onCreated }: { onCreated: (id: string) => void }) {
  const { format, meta } = useFormat();
  const [teams, setTeams] = useState(12);
  const [slot, setSlot] = useState(7);
  const [rounds, setRounds] = useState(16);
  const [fmt, setFmt] = useState(format);
  const [roster, setRoster] = useState({ ...DEFAULT_ROSTER });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const create = async () => {
    setBusy(true);
    setErr(null);
    try {
      const res = await api<{ draft_id: string }>("/api/drafts", {
        method: "POST",
        body: JSON.stringify({ teams, user_slot: slot, rounds, format: fmt, roster }),
      });
      onCreated(res.draft_id);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
      setBusy(false);
    }
  };

  const num = (label: string, value: number, set: (n: number) => void, min: number, max: number) => (
    <label className="flex flex-col gap-1 text-[12px] text-ink-soft">
      <span className="uppercase tracking-wide text-[10px] text-ink-mute">{label}</span>
      <input
        type="number"
        className="num border border-rule-strong bg-paper px-2 py-1 w-20"
        value={value}
        min={min}
        max={max}
        onChange={(e) => set(Number(e.target.value))}
      />
    </label>
  );

  return (
    <div className="max-w-xl">
      <h1 className="text-lg font-medium mb-1">Mock draft</h1>
      <p className="text-ink-soft text-[13px] mb-4">
        You draft against {teams - 1} ADP-driven bots with private boards; the model
        recommends picks from VORP, roster need, and survival odds.
      </p>
      <div className="flex gap-4 flex-wrap items-end">
        {num("teams", teams, setTeams, 4, 16)}
        {num("your slot", slot, setSlot, 1, teams)}
        {num("rounds", rounds, setRounds, 4, 25)}
        <label className="flex flex-col gap-1 text-[12px]">
          <span className="uppercase tracking-wide text-[10px] text-ink-mute">format</span>
          <select
            value={fmt}
            onChange={(e) => setFmt(e.target.value)}
            className="num border border-rule-strong bg-paper px-2 py-1"
          >
            {(meta?.formats ?? []).map((f) => (
              <option key={f.key} value={f.key}>
                {f.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="mt-4">
        <span className="uppercase tracking-wide text-[10px] text-ink-mute">roster</span>
        <div className="flex gap-3 mt-1 flex-wrap">
          {Object.keys(roster).map((k) => (
            <label key={k} className="flex items-center gap-1 text-[12px]">
              <span className="num text-ink-soft w-9">{k}</span>
              <input
                type="number"
                min={0}
                max={9}
                className="num border border-rule-strong bg-paper px-1 py-0.5 w-12"
                value={roster[k]}
                onChange={(e) => setRoster({ ...roster, [k]: Number(e.target.value) })}
              />
            </label>
          ))}
        </div>
      </div>
      {err && <div className="text-edge-neg mt-3 text-[13px]">{err}</div>}
      <button
        onClick={create}
        disabled={busy || slot > teams}
        className="mt-5 bg-ink text-paper px-4 py-1.5 text-[13px] disabled:opacity-40 hover:bg-accent"
      >
        {busy ? "creating…" : "Start draft"}
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Room
// ---------------------------------------------------------------------------

function DraftRoom({ draftId, onExit }: { draftId: string; onExit: () => void }) {
  const [state, setState] = useState<DraftState | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [recs, setRecs] = useState<Recommendations | null>(null);
  const [alerts, setAlerts] = useState<TierCliffAlert[]>([]);
  const [busy, setBusy] = useState(false);
  const [drawer, setDrawer] = useState<string | null>(null);
  const newPicksFrom = useRef(0);

  const { data: players, error: pErr } = useApi<Player[]>(
    state ? `/api/players?format=${state.config.format}` : null,
  );
  const { data: adpBoard } = useApi<AdpBoardRow[]>("/api/adp_board");

  const playersById = useMemo(() => {
    const m = new Map<string, Player>();
    (players ?? []).forEach((p) => m.set(p.player_id, p));
    return m;
  }, [players]);
  const adpById = useMemo(() => {
    const m = new Map<string, AdpBoardRow>();
    (adpBoard ?? []).forEach((r) => m.set(r.player_id, r));
    return m;
  }, [adpBoard]);

  const refresh = useCallback(() => {
    api<DraftState>(`/api/drafts/${draftId}`).then(setState).catch(setError);
  }, [draftId]);

  useEffect(refresh, [refresh]);

  const onClock = state?.on_the_clock?.is_user ?? false;

  useEffect(() => {
    if (!state) return;
    if (onClock && !state.complete) {
      api<Recommendations>(`/api/drafts/${draftId}/recommendations?n=6`)
        .then((r) => {
          setRecs(r);
          setAlerts(r.tier_cliff_alerts);
        })
        .catch(() => setRecs(null));
    } else {
      setRecs(null);
    }
  }, [state, onClock, draftId]);

  const pick = async (playerId: string) => {
    if (busy || !onClock) return;
    setBusy(true);
    newPicksFrom.current = state?.picks.length ?? 0;
    try {
      const res = await api<{ state: DraftState; events: { tier_cliff_alerts: TierCliffAlert[] } }>(
        `/api/drafts/${draftId}/pick`,
        { method: "POST", body: JSON.stringify({ player_id: playerId }) },
      );
      setState(res.state);
      setAlerts(res.events.tier_cliff_alerts);
    } catch (e) {
      if (e instanceof ApiError) setError(e);
    } finally {
      setBusy(false);
    }
  };

  const undo = async () => {
    if (busy) return;
    setBusy(true);
    try {
      const res = await api<{ state: DraftState }>(`/api/drafts/${draftId}/undo`, {
        method: "POST",
      });
      setState(res.state);
    } catch {
      /* nothing to undo */
    } finally {
      setBusy(false);
    }
  };

  if (error) return <ErrorPanel error={error} />;
  if (pErr) return <ErrorPanel error={pErr} />;
  if (!state || !players) return <Loading label="draft" />;

  if (state.complete) {
    return (
      <EndState
        state={state}
        playersById={playersById}
        adpById={adpById}
        onExit={onExit}
        onUndo={undo}
      />
    );
  }

  const surviveById = new Map(state.available.map((a) => [a.player_id, a.p_survive]));

  return (
    <div className="flex gap-8 items-start">
      <div className="flex-[2] min-w-0">
        <RecPanel
          recs={recs}
          alerts={alerts}
          onClock={onClock}
          busy={busy}
          otc={state.on_the_clock}
          userSlot={state.config.user_slot}
          onPick={pick}
          onInspect={setDrawer}
        />
        <AvailableTable
          state={state}
          playersById={playersById}
          adpById={adpById}
          surviveById={surviveById}
          canPick={onClock && !busy}
          onPick={pick}
          onInspect={setDrawer}
        />
      </div>
      <SidePanel
        state={state}
        playersById={playersById}
        adpById={adpById}
        needWeights={recs?.need_weights ?? null}
        newPicksFrom={newPicksFrom.current}
        busy={busy}
        onUndo={undo}
        onExit={onExit}
      />
      {drawer && (
        <PlayerDrawer
          playerId={drawer}
          format={state.config.format}
          onClose={() => setDrawer(null)}
        />
      )}
    </div>
  );
}
