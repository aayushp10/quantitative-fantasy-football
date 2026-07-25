/**
 * The signature element: season-outcome distribution as a range bar.
 * One visual grammar everywhere — thin p10–p90 track, thick p25–p75 band,
 * ink tick at p50. Table rows share a domain per view so bars are
 * comparable; the large variant labels the quantiles directly.
 */

interface RangeBarProps {
  p10: number | null;
  p25: number | null;
  p50: number | null;
  p75: number | null;
  p90: number | null;
  domain: [number, number];
  width?: number;
  accent?: boolean;
}

export function RangeBar({ p10, p25, p50, p75, p90, domain, width = 96, accent = false }: RangeBarProps) {
  if (p10 == null || p25 == null || p50 == null || p75 == null || p90 == null) {
    return <span className="text-ink-mute">–</span>;
  }
  const [lo, hi] = domain;
  const h = 12;
  const x = (v: number) => ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo || 1)) * width;
  const band = accent ? "var(--color-accent)" : "var(--color-ink-soft)";
  return (
    <svg
      width={width}
      height={h}
      className="block"
      role="img"
      aria-label={`p10 ${Math.round(p10)}, median ${Math.round(p50)}, p90 ${Math.round(p90)}`}
    >
      <line x1={x(p10)} x2={x(p90)} y1={h / 2} y2={h / 2} stroke="var(--color-rule-strong)" strokeWidth={1.5} />
      <rect x={x(p25)} y={h / 2 - 2.5} width={Math.max(1, x(p75) - x(p25))} height={5} fill={band} rx={1} />
      <line x1={x(p50)} x2={x(p50)} y1={0.5} y2={h - 0.5} stroke="var(--color-ink)" strokeWidth={2} />
    </svg>
  );
}

interface RangeBarLargeProps extends Omit<RangeBarProps, "width" | "domain"> {
  domain?: [number, number];
}

/** Player-header variant: own scale, quantiles labeled. */
export function RangeBarLarge({ p10, p25, p50, p75, p90, domain }: RangeBarLargeProps) {
  if (p10 == null || p25 == null || p50 == null || p75 == null || p90 == null) {
    return <div className="text-ink-mute">no distribution available</div>;
  }
  const pad = Math.max(10, (p90 - p10) * 0.12);
  const [lo, hi] = domain ?? [Math.max(0, p10 - pad), p90 + pad];
  const width = 440;
  const h = 56;
  const barY = 30;
  const x = (v: number) => ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo || 1)) * width;
  const label = (v: number, name: string, above: boolean) => (
    <g key={name}>
      <text
        x={x(v)}
        y={above ? 12 : h - 2}
        textAnchor="middle"
        className="num"
        fontSize={11}
        fill={name === "p50" ? "var(--color-ink)" : "var(--color-ink-soft)"}
        fontWeight={name === "p50" ? 600 : 400}
      >
        {Math.round(v)}
      </text>
      <text
        x={x(v)}
        y={above ? 22 : h - 14}
        textAnchor="middle"
        fontSize={9}
        fill="var(--color-ink-mute)"
      >
        {name}
      </text>
    </g>
  );
  return (
    <svg width={width} height={h} className="block max-w-full" role="img"
      aria-label={`Season points: p10 ${Math.round(p10)}, p25 ${Math.round(p25)}, median ${Math.round(p50)}, p75 ${Math.round(p75)}, p90 ${Math.round(p90)}`}>
      <line x1={x(p10)} x2={x(p90)} y1={barY} y2={barY} stroke="var(--color-rule-strong)" strokeWidth={2} />
      <rect x={x(p25)} y={barY - 5} width={Math.max(1, x(p75) - x(p25))} height={10} fill="var(--color-ink-soft)" rx={2} />
      <line x1={x(p50)} x2={x(p50)} y1={barY - 10} y2={barY + 10} stroke="var(--color-ink)" strokeWidth={2.5} />
      {label(p10, "p10", false)}
      {label(p25, "p25", true)}
      {label(p50, "p50", true)}
      {label(p75, "p75", true)}
      {label(p90, "p90", false)}
    </svg>
  );
}
