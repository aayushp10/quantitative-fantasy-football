import { createContext, useContext, useMemo, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { loadFormat, saveFormat, useApi } from "./lib/api";
import type { Meta } from "./lib/types";
import Rankings from "./pages/Rankings";
import PlayerPage from "./pages/PlayerPage";
import Market from "./pages/Market";
import Trust from "./pages/Trust";
import Draft from "./pages/Draft";

interface FormatCtx {
  format: string;
  setFormat: (f: string) => void;
  meta: Meta | null;
}

const Ctx = createContext<FormatCtx>({ format: "12_ppr", setFormat: () => {}, meta: null });
export const useFormat = () => useContext(Ctx);

const NAV = [
  { to: "/", label: "Rankings" },
  { to: "/market", label: "Vs market" },
  { to: "/trust", label: "Trust" },
  { to: "/draft", label: "Mock draft" },
];

export default function App() {
  const [format, setFormatState] = useState(loadFormat());
  const { data: meta } = useApi<Meta>("/api/meta");
  const setFormat = (f: string) => {
    saveFormat(f);
    setFormatState(f);
  };
  const ctx = useMemo(() => ({ format, setFormat, meta }), [format, meta]);

  return (
    <Ctx.Provider value={ctx}>
      <div className="min-w-[1100px] max-w-[1500px] mx-auto px-6 pb-16">
        <header className="flex items-baseline gap-6 border-b border-ink pt-5 pb-2 mb-4">
          <div className="font-medium tracking-tight text-[15px]">
            FF<span className="text-accent">/</span>TERMINAL
            <span className="num text-ink-mute ml-2 text-[12px]">
              {meta ? `${meta.projection_season} · model ${meta.model_version}` : "…"}
            </span>
          </div>
          <nav className="flex gap-4 text-[13px]">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.to === "/"}
                className={({ isActive }) =>
                  isActive
                    ? "text-accent font-medium underline underline-offset-4"
                    : "text-ink-soft hover:text-ink"
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-2 text-[12px]">
            <label htmlFor="fmt" className="text-ink-mute uppercase tracking-wide text-[10px]">
              format
            </label>
            <select
              id="fmt"
              value={format}
              onChange={(e) => setFormat(e.target.value)}
              className="num border border-rule-strong bg-paper px-1.5 py-0.5"
            >
              {(meta?.formats ?? [{ key: format, label: format, league_size: 0, roster: {} }]).map(
                (f) => (
                  <option key={f.key} value={f.key}>
                    {f.label}
                  </option>
                ),
              )}
            </select>
          </div>
        </header>
        <Routes>
          <Route path="/" element={<Rankings />} />
          <Route path="/players/:id" element={<PlayerPage />} />
          <Route path="/market" element={<Market />} />
          <Route path="/trust" element={<Trust />} />
          <Route path="/draft" element={<Draft />} />
        </Routes>
        {meta && (
          <footer className="mt-10 pt-2 border-t border-rule text-[11px] text-ink-mute num">
            trained {meta.seasons_trained[0]}–{meta.seasons_trained[1]} · ADP{" "}
            {meta.adp_source} ({meta.adp_format}, {meta.adp_snapshot_date}) ·{" "}
            {meta.player_count} players · built {meta.build_timestamp}
          </footer>
        )}
      </div>
    </Ctx.Provider>
  );
}
