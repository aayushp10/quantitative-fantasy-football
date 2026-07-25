import { useEffect } from "react";
import { useApi } from "../../lib/api";
import type { PlayerDetail } from "../../lib/types";
import { PlayerCard } from "../PlayerCard";
import { ErrorPanel, Loading } from "../ui";

export function PlayerDrawer({
  playerId,
  format,
  onClose,
}: {
  playerId: string;
  format: string;
  onClose: () => void;
}) {
  const { data, error, loading } = useApi<PlayerDetail>(
    `/api/players/${playerId}?format=${format}`,
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-40" role="dialog" aria-modal="true" aria-label="Player detail">
      <div className="absolute inset-0 bg-[rgba(4,7,16,0.55)] backdrop-blur-[2px]" onClick={onClose} />
      <div className="absolute right-2 top-2 bottom-2 w-[720px] glass !rounded-3xl overflow-y-auto p-6 float-in">
        <button
          onClick={onClose}
          className="text-[12px] text-accent hover:underline mb-3"
          autoFocus
        >
          ← close (esc)
        </button>
        {error && <ErrorPanel error={error} />}
        {loading && <Loading label="player" />}
        {data && <PlayerCard p={data} format={format} />}
      </div>
    </div>
  );
}
