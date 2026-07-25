import { Link, useParams } from "react-router-dom";
import { useFormat } from "../App";
import { useApi } from "../lib/api";
import type { PlayerDetail } from "../lib/types";
import { PlayerCard } from "../components/PlayerCard";
import { ErrorPanel, Loading } from "../components/ui";

export default function PlayerPage() {
  const { id } = useParams();
  const { format } = useFormat();
  const { data, error, loading } = useApi<PlayerDetail>(
    id ? `/api/players/${id}?format=${format}` : null,
  );

  if (error) return <ErrorPanel error={error} />;
  if (loading || !data) return <Loading label="player" />;

  return (
    <div className="max-w-3xl">
      <Link to="/" className="text-[12px] text-accent hover:underline">
        ← rankings
      </Link>
      <div className="mt-2">
        <PlayerCard p={data} />
      </div>
    </div>
  );
}
