/**
 * Glass-styled chart kit for player pages. Colors come from the validated
 * dark palette (see index.css): primary series blue #3B76E0, secondary
 * orange #C98110 (blue-orange: CVD-safe pair), boom/bust marks reuse the
 * validated edge pair. Grid and axes are recessive; identity is carried
 * by direct labels and the legend row, never color alone.
 */
import { ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export const SERIES_1 = "#3B76E0";
export const SERIES_2 = "#C98110";
export const BOOM = "#0FA372";
export const BUST = "#C0334D";
const GRID = "rgba(255,255,255,0.07)";
const AXIS = "rgba(255,255,255,0.16)";
const TICK = { fontSize: 10.5, fill: "var(--color-ink-mute)", fontFamily: "var(--font-mono)" };

export function ChartCard({
  title,
  note,
  children,
  height = 190,
}: {
  title: string;
  note?: string;
  children: ReactNode;
  height?: number;
}) {
  return (
    <div className="glass-soft p-4">
      <div className="flex items-baseline gap-2 mb-2">
        <h3 className="text-[10.5px] uppercase tracking-[0.1em] text-ink-soft font-semibold">
          {title}
        </h3>
        {note && <span className="text-[10.5px] text-ink-mute">{note}</span>}
      </div>
      <div style={{ height }}>{children}</div>
    </div>
  );
}

export function GlassTooltip({ children }: { children: ReactNode }) {
  return (
    <div className="glass px-2.5 py-1.5 text-[12px] !rounded-xl">
      {children}
    </div>
  );
}

interface SeasonDatum {
  label: string;
  [k: string]: number | string | null;
}

/** Season bars with an optional distinguished final "projection" bar. */
export function SeasonBars({
  data,
  dataKey,
  projKey,
  format,
  unit,
}: {
  data: SeasonDatum[];
  dataKey: string;
  projKey?: string;
  format: (v: number) => string;
  unit?: string;
}) {
  return (
    <ResponsiveContainer>
      <BarChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: -18 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="label" tick={TICK} stroke={AXIS} tickLine={false} />
        <YAxis tick={TICK} stroke={AXIS} tickLine={false} />
        <Tooltip
          cursor={{ fill: "rgba(255,255,255,0.05)" }}
          content={({ payload, label }) => {
            const row = payload?.[0];
            if (!row) return null;
            const v = row.value as number;
            return (
              <GlassTooltip>
                <span className="text-ink-soft">{label}</span>{" "}
                <span className="num font-semibold">{format(v)}</span>
                {unit && <span className="text-ink-mute"> {unit}</span>}
                {row.dataKey === projKey && <span className="text-accent"> · model projection</span>}
              </GlassTooltip>
            );
          }}
        />
        <Bar dataKey={dataKey} fill={SERIES_1} radius={[5, 5, 0, 0]} maxBarSize={26} />
        {projKey && (
          <Bar
            dataKey={projKey}
            fill="rgba(45,212,191,0.28)"
            stroke="var(--color-accent)"
            strokeDasharray="4 3"
            radius={[5, 5, 0, 0]}
            maxBarSize={26}
          />
        )}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** One or two trend lines over seasons. Two series get a legend row. */
export function TrendLines({
  data,
  series,
  format,
  zeroLine = false,
}: {
  data: SeasonDatum[];
  series: { key: string; label: string; color?: string }[];
  format: (v: number) => string;
  zeroLine?: boolean;
}) {
  const colors = [SERIES_1, SERIES_2];
  return (
    <div className="h-full flex flex-col">
      {series.length > 1 && (
        <div className="flex gap-4 mb-1 text-[10.5px] text-ink-soft">
          {series.map((s, i) => (
            <span key={s.key} className="flex items-center gap-1.5">
              <span
                className="inline-block w-3 h-[3px] rounded-full"
                style={{ background: s.color ?? colors[i] }}
              />
              {s.label}
            </span>
          ))}
        </div>
      )}
      <div className="flex-1 min-h-0">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 10, bottom: 0, left: -18 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="label" tick={TICK} stroke={AXIS} tickLine={false} />
            <YAxis tick={TICK} stroke={AXIS} tickLine={false} tickFormatter={(v) => format(v)} />
            {zeroLine && <ReferenceLine y={0} stroke={AXIS} strokeDasharray="3 3" />}
            <Tooltip
              cursor={{ stroke: AXIS }}
              content={({ payload, label }) => {
                if (!payload?.length) return null;
                return (
                  <GlassTooltip>
                    <div className="text-ink-soft mb-0.5">{label}</div>
                    {payload.map((row) => (
                      <div key={String(row.dataKey)} className="num">
                        <span className="text-ink-mute">
                          {series.find((s) => s.key === row.dataKey)?.label ?? row.dataKey}
                        </span>{" "}
                        <span className="font-semibold">
                          {row.value == null ? "–" : format(row.value as number)}
                        </span>
                      </div>
                    ))}
                  </GlassTooltip>
                );
              }}
            />
            {series.map((s, i) => (
              <Line
                key={s.key}
                dataKey={s.key}
                stroke={s.color ?? colors[i]}
                strokeWidth={2.5}
                dot={{ r: 3.5, fill: s.color ?? colors[i], strokeWidth: 0 }}
                activeDot={{ r: 5 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/** Weekly game-log bars; boom/bust weeks tinted (redundant with height). */
export function WeeklyBars({
  data,
  median,
  boomAt = 20,
  bustAt = 8,
}: {
  data: { week: number; pts: number }[];
  median: number;
  boomAt?: number;
  bustAt?: number;
}) {
  return (
    <ResponsiveContainer>
      <BarChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: -22 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="week" tick={TICK} stroke={AXIS} tickLine={false} />
        <YAxis tick={TICK} stroke={AXIS} tickLine={false} />
        <ReferenceLine
          y={median}
          stroke="rgba(255,255,255,0.35)"
          strokeDasharray="4 3"
          label={{ value: "median", position: "right", fontSize: 9.5, fill: "var(--color-ink-mute)" }}
        />
        <Tooltip
          cursor={{ fill: "rgba(255,255,255,0.05)" }}
          content={({ payload }) => {
            const d = payload?.[0]?.payload as { week: number; pts: number } | undefined;
            if (!d) return null;
            return (
              <GlassTooltip>
                <span className="text-ink-soft">wk {d.week}</span>{" "}
                <span className="num font-semibold">{d.pts.toFixed(1)}</span>
                <span className="text-ink-mute"> pts</span>
              </GlassTooltip>
            );
          }}
        />
        <Bar dataKey="pts" radius={[4, 4, 0, 0]} maxBarSize={16}>
          {data.map((d) => (
            <Cell
              key={d.week}
              fill={d.pts >= boomAt ? BOOM : d.pts < bustAt ? BUST : "rgba(255,255,255,0.38)"}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Horizontal percentile bar vs position cohort. */
export function PercentileBar({
  label,
  value,
  pct,
  color,
}: {
  label: string;
  value: string;
  pct: number | null;
  color: string;
}) {
  return (
    <div className="flex items-center gap-3 py-1">
      <span className="w-36 text-[12px] text-ink-soft shrink-0">{label}</span>
      <div className="h-[7px] flex-1 rounded-full bg-[rgba(255,255,255,0.08)] overflow-hidden">
        {pct != null && (
          <div
            className="h-full rounded-full"
            style={{
              width: `${Math.round(pct * 100)}%`,
              background: `linear-gradient(90deg, color-mix(in srgb, ${color} 55%, transparent), ${color})`,
            }}
          />
        )}
      </div>
      <span className="num text-[12px] w-12 text-right">{value}</span>
      <span className="num text-[10.5px] text-ink-mute w-10 text-right">
        {pct == null ? "–" : `p${Math.round(pct * 100)}`}
      </span>
    </div>
  );
}
