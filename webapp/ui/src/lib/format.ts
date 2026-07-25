/** Display rounding: VORP 1dp, percentages 0dp, points 0dp. */

export const fmtVorp = (v: number | null | undefined): string =>
  v == null ? "–" : v.toFixed(1);

export const fmtPts = (v: number | null | undefined): string =>
  v == null ? "–" : Math.round(v).toString();

export const fmtPct = (v: number | null | undefined): string =>
  v == null ? "–" : `${Math.round(v * 100)}%`;

export const fmtSigned = (v: number | null | undefined, dp = 1): string =>
  v == null ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(dp)}`;

export const fmtAdp = (v: number | null | undefined): string =>
  v == null ? "–" : v.toFixed(1);

export const fmtDec = (v: number | null | undefined, dp = 2): string =>
  v == null ? "–" : v.toFixed(dp);

export const fmtGames = (v: number | null | undefined): string =>
  v == null ? "–" : v.toFixed(1);

export const edgeClass = (v: number | null | undefined): string =>
  v == null ? "text-ink-mute" : v > 0 ? "text-edge-pos" : v < 0 ? "text-edge-neg" : "text-ink-soft";
