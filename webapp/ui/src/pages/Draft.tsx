import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useFormat } from "../App";
import { api, ApiError, useApi } from "../lib/api";
import type {
  AdpBoardRow,
  DraftState,
  Player,
  Recommendations,
  RolloutResult,
  StepResult,
  TierCliffAlert,
} from "../lib/types";
import { ErrorPanel, Loading } from "../components/ui";
import { RecPanel } from "../components/draft/RecPanel";
import { AvailableTable } from "../components/draft/AvailableTable";
import { SidePanel } from "../components/draft/SidePanel";
import { EndState } from "../components/draft/EndState";
import { PlayerDrawer } from "../components/draft/PlayerDrawer";
import { BoardGrid, SPEED_MS, type BotSpeed } from "../components/draft/BoardGrid";

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
  const [clock, setClock] = useState(60);
  const [roster, setRoster] = useState({ ...DEFAULT_ROSTER });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const create = async () => {
    setBusy(true);
    setErr(null);
    try {
      const res = await api<{ draft_id: string }>("/api/drafts", {
        method: "POST",
        body: JSON.stringify({
          teams, user_slot: slot, rounds, format: fmt, roster,
          auto_advance: false,   // the room paces bot picks itself
        }),
      });
      localStorage.setItem(`ff-draft-settings-${res.draft_id}`, JSON.stringify({ clock }));
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
        className="num field px-2 py-1.5 w-20"
        value={value}
        min={min}
        max={max}
        onChange={(e) => set(Number(e.target.value))}
      />
    </label>
  );

  return (
    <div className="max-w-xl glass p-6 float-in">
      <h1 className="text-lg font-semibold mb-1">Mock draft</h1>
      <p className="text-ink-soft text-[13px] mb-4">
        You draft against {teams - 1} ADP-driven bots with private boards; the model
        recommends picks from VORP, roster need, and survival odds — and can
        simulate completed drafts to rank candidates by final roster strength.
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
            className="num field px-2 py-1.5"
          >
            {(meta?.formats ?? []).map((f) => (
              <option key={f.key} value={f.key}>
                {f.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-[12px]">
          <span className="uppercase tracking-wide text-[10px] text-ink-mute">pick clock</span>
          <select
            value={clock}
            onChange={(e) => setClock(Number(e.target.value))}
            className="num field px-2 py-1.5"
          >
            <option value={0}>off</option>
            <option value={30}>0:30</option>
            <option value={60}>1:00</option>
            <option value={90}>1:30</option>
            <option value={120}>2:00</option>
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
                className="num field px-1.5 py-1 w-12"
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
        className="btn-primary mt-5 px-5 py-2 text-[13px] disabled:opacity-40"
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
  const [rollout, setRollout] = useState<RolloutResult | null>(null);
  const [rolloutBusy, setRolloutBusy] = useState(false);
  const [busy, setBusy] = useState(false);
  const [drawer, setDrawer] = useState<string | null>(null);
  const [speed, setSpeed] = useState<BotSpeed>("fast");
  const [queue, setQueue] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(`ff-queue-${draftId}`) ?? "[]");
    } catch {
      return [];
    }
  });
  const [clockLeft, setClockLeft] = useState<number | null>(null);
  const newPicksFrom = useRef(0);
  const steppingRef = useRef(false);
  const deadlineRef = useRef<number | null>(null);

  const clockSecs = useMemo(() => {
    try {
      const s = JSON.parse(localStorage.getItem(`ff-draft-settings-${draftId}`) ?? "{}");
      return typeof s.clock === "number" ? s.clock : 60;
    } catch {
      return 60;
    }
  }, [draftId]);

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
    // Rollouts are on-demand and stale as soon as the board changes
    setRollout(null);
  }, [state, onClock, draftId]);

  // ---- Paced bot picks: one /step call per tick while a bot is on the clock
  useEffect(() => {
    if (!state || state.complete || busy) return;
    const otc = state.on_the_clock;
    if (!otc || otc.is_user) return;
    let cancelled = false;
    const t = setTimeout(async () => {
      if (cancelled || steppingRef.current) return;
      steppingRef.current = true;
      try {
        const res = await api<StepResult>(`/api/drafts/${draftId}/step`, { method: "POST" });
        if (cancelled) return;
        if (res.event && res.event.overall === state.picks.length + 1) {
          const ev = res.event;
          newPicksFrom.current = state.picks.length;
          setState({
            ...state,
            picks: [...state.picks, ev],
            available: state.available.filter((a) => a.player_id !== ev.player_id),
            on_the_clock: res.on_the_clock,
            complete: res.complete,
          });
        } else {
          refresh(); // out of sync (undo race, resumed draft) — full resync
        }
        // Landing on the user or finishing: full state for survival odds etc.
        if (res.complete || res.on_the_clock?.is_user) refresh();
      } catch {
        /* transient server error — the effect re-runs and retries */
      } finally {
        steppingRef.current = false;
      }
    }, SPEED_MS[speed]);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [state, busy, speed, draftId, refresh]);

  // ---- Queue: persist, and drop players as they get drafted
  useEffect(() => {
    localStorage.setItem(`ff-queue-${draftId}`, JSON.stringify(queue));
  }, [queue, draftId]);
  useEffect(() => {
    if (!state) return;
    const drafted = new Set(state.picks.map((p) => p.player_id));
    setQueue((q) => (q.some((id) => drafted.has(id)) ? q.filter((id) => !drafted.has(id)) : q));
  }, [state]);

  const toggleQueue = useCallback((pid: string) => {
    setQueue((q) => (q.includes(pid) ? q.filter((x) => x !== pid) : [...q, pid]));
  }, []);
  const moveQueue = useCallback((pid: string, dir: -1 | 1) => {
    setQueue((q) => {
      const i = q.indexOf(pid);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= q.length) return q;
      const out = [...q];
      [out[i], out[j]] = [out[j], out[i]];
      return out;
    });
  }, []);

  const runRollout = useCallback(async () => {
    if (!onClock) return;
    setRolloutBusy(true);
    try {
      setRollout(await api<RolloutResult>(`/api/drafts/${draftId}/rollout?n=8&sims=24`));
    } catch {
      setRollout(null);
    } finally {
      setRolloutBusy(false);
    }
  }, [draftId, onClock]);

  const pick = useCallback(async (playerId: string) => {
    if (busy || !onClock) return;
    setBusy(true);
    newPicksFrom.current = state?.picks.length ?? 0;
    try {
      const res = await api<{ state: DraftState; events: { tier_cliff_alerts: TierCliffAlert[] } }>(
        `/api/drafts/${draftId}/pick`,
        // advance:false — the stepping loop plays the bots out one by one
        { method: "POST", body: JSON.stringify({ player_id: playerId, advance: false }) },
      );
      setState(res.state);
      setAlerts(res.events.tier_cliff_alerts);
    } catch (e) {
      if (e instanceof ApiError) setError(e);
    } finally {
      setBusy(false);
    }
  }, [busy, onClock, state, draftId]);

  // ---- Pick clock: countdown while the user is on the clock; autodraft on expiry
  const autodraft = useCallback(() => {
    if (busy || !state?.on_the_clock?.is_user) return;
    const availSet = new Set(state.available.map((a) => a.player_id));
    let pid: string | undefined = queue.find((id) => availSet.has(id));
    if (!pid) pid = recs?.recommendations[0]?.player_id;
    if (!pid) {
      let bestV = -Infinity;
      for (const a of state.available) {
        const p = playersById.get(a.player_id);
        if (p?.vorp != null && p.vorp > bestV) {
          bestV = p.vorp;
          pid = a.player_id;
        }
      }
    }
    if (pid) pick(pid);
  }, [busy, state, queue, recs, playersById, pick]);
  const autodraftRef = useRef(autodraft);
  autodraftRef.current = autodraft;

  useEffect(() => {
    if (!onClock || clockSecs <= 0 || !state || state.complete) {
      deadlineRef.current = null;
      setClockLeft(null);
      return;
    }
    if (deadlineRef.current === null) {
      deadlineRef.current = Date.now() + clockSecs * 1000;
      setClockLeft(clockSecs);
    }
    const iv = setInterval(() => {
      const left = Math.max(0, Math.ceil(((deadlineRef.current ?? 0) - Date.now()) / 1000));
      setClockLeft(left);
      if (left <= 0) {
        deadlineRef.current = null;
        clearInterval(iv);
        autodraftRef.current();
      }
    }, 250);
    return () => clearInterval(iv);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onClock, state?.picks.length, clockSecs]);

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
      <div>
        <BoardGrid state={state} speed={speed} onSpeed={setSpeed} />
        <EndState
          state={state}
          playersById={playersById}
          adpById={adpById}
          onExit={onExit}
          onUndo={undo}
        />
      </div>
    );
  }

  const surviveById = new Map(state.available.map((a) => [a.player_id, a.p_survive]));
  const lastPick = state.picks.length > 0 ? state.picks[state.picks.length - 1] : null;

  return (
    <div>
      <BoardGrid state={state} speed={speed} onSpeed={setSpeed} />
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
            rollout={rollout}
            rolloutBusy={rolloutBusy}
            onRollout={runRollout}
            clockLeft={clockLeft}
            lastPick={lastPick}
          />
          <AvailableTable
            state={state}
            playersById={playersById}
            adpById={adpById}
            surviveById={surviveById}
            canPick={onClock && !busy}
            onPick={pick}
            onInspect={setDrawer}
            queued={new Set(queue)}
            onToggleQueue={toggleQueue}
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
          queue={queue}
          onQueueRemove={toggleQueue}
          onQueueMove={moveQueue}
        />
        {drawer && (
          <PlayerDrawer
            playerId={drawer}
            format={state.config.format}
            onClose={() => setDrawer(null)}
          />
        )}
      </div>
    </div>
  );
}
