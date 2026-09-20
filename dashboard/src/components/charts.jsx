import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

/*
 * Chart layer. Every chart in the dashboard goes through here so the mark
 * specs cannot drift between panels:
 *
 *   bars     <= 24px, 4px rounded at the DATA end, square at the baseline,
 *            a 2px gap of surface colour between neighbours (never a stroke)
 *   lines    2px, markers >= 8px with a 2px surface ring
 *   areas    ~10% opacity wash, never a saturated block
 *   grid     solid hairline, recessive. The original used strokeDasharray,
 *            which reads as a "threshold" rather than a grid.
 *   labels   selective: the extreme, not a number on every point. Label text
 *            wears an ink token, never the series colour.
 *
 * Text and axes use CSS variables so the theme stays the single source.
 */

const TICK = { fill: 'var(--ink-4)', fontSize: 12 }
const X_AXIS = {
  tick: TICK,
  tickLine: false,
  axisLine: { stroke: 'var(--data-axis)' },
  tickMargin: 8,
}
const Y_AXIS = { tick: TICK, tickLine: false, axisLine: false, width: 44 }
const GRID = { stroke: 'var(--data-grid)', vertical: false }
const CURSOR_BAND = { fill: 'rgba(148, 163, 184, 0.08)' }
const CURSOR_LINE = { stroke: 'var(--data-axis)', strokeWidth: 1 }
const MAX_BAR = 24
const CORNER = 4

/* ------------------------------------------------------------------ */
/* Tooltip: values lead, labels follow, line keys rather than boxes.   */
/* ------------------------------------------------------------------ */

export function DataTooltip({ active, payload, format }) {
  if (!active || !payload?.length) return null
  const { title, rows } = format(payload[0].payload)
  return (
    <div className="chart-tooltip" role="status">
      <p className="chart-tooltip-title">{title}</p>
      {rows.map((row) => (
        <p key={row.label} className="chart-tooltip-row">
          <span className="chart-tooltip-key" style={{ background: row.color }} aria-hidden="true" />
          <strong>{row.value}</strong>
          <span>{row.label}</span>
        </p>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Bar shape: rounded at the data end only.                            */
/* ------------------------------------------------------------------ */

function DataEndBar({ x, y, width, height, payload }) {
  if (![x, y, width, height].every(Number.isFinite)) return null
  const h = Math.abs(height)
  if (!h || width <= 0) return null

  // Recharts reports a negative height for a negative value; normalise so the
  // same code rounds the correct end whichever way the bar points.
  const top = Math.min(y, y + height)
  const gap = width > 4 ? 1 : 0 // 1px each side = the 2px surface gap
  const left = x + gap
  const w = width - gap * 2
  const r = Math.min(CORNER, w / 2, h)
  const right = left + w
  const bottom = top + h

  const d = payload?.negative
    ? `M${left},${top} L${left},${bottom - r} Q${left},${bottom} ${left + r},${bottom} L${right - r},${bottom} Q${right},${bottom} ${right},${bottom - r} L${right},${top} Z`
    : `M${left},${bottom} L${left},${top + r} Q${left},${top} ${left + r},${top} L${right - r},${top} Q${right},${top} ${right},${top + r} L${right},${bottom} Z`

  return <path d={d} fill={payload?.fill ?? 'var(--data-rain)'} fillOpacity={payload?.fillOpacity ?? 1} />
}

function peakIndexOf(data) {
  let best = -1
  let bestValue = 0
  data.forEach((row, i) => {
    if (Number.isFinite(row.value) && row.value > bestValue) {
      best = i
      bestValue = row.value
    }
  })
  return best
}

/**
 * Daily bars. Rows: { label, value, fill?, fillOpacity?, negative? }.
 * `tooltip(row) -> { title, rows: [{label, value, color}] }`.
 */
export function DailyBars({
  data,
  height = 240,
  yDomain,
  yTicks,
  yFormatter,
  referenceLines = [],
  tooltip,
  labelPeak = false,
  peakFormatter = (v) => v.toFixed(1),
  xInterval = 'preserveStartEnd',
  ariaLabel,
}) {
  const peak = labelPeak ? peakIndexOf(data) : -1

  return (
    <div className="chart-box" style={{ height }} role="img" aria-label={ariaLabel}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 22, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid {...GRID} />
          <XAxis dataKey="label" interval={xInterval} {...X_AXIS} />
          <YAxis {...Y_AXIS} domain={yDomain} ticks={yTicks} tickFormatter={yFormatter} />
          {referenceLines.map((line) => (
            <ReferenceLine
              key={line.label ?? line.y}
              y={line.y}
              stroke={line.strong ? 'var(--data-axis)' : 'var(--data-grid)'}
              strokeWidth={1}
              label={
                line.label
                  ? { value: line.label, position: line.labelPosition ?? 'insideTopRight', fill: 'var(--ink-4)', fontSize: 11 }
                  : undefined
              }
            />
          ))}
          <Tooltip
            cursor={CURSOR_BAND}
            content={<DataTooltip format={tooltip} />}
            isAnimationActive={false}
          />
          <Bar dataKey="value" maxBarSize={MAX_BAR} shape={<DataEndBar />} isAnimationActive={false}>
            {peak >= 0 ? (
              <LabelList
                dataKey="value"
                content={(props) =>
                  props.index === peak ? (
                    <text
                      x={props.x + props.width / 2}
                      y={props.y - 8}
                      textAnchor="middle"
                      fontSize={12}
                      fontWeight={600}
                      style={{ fill: 'var(--ink-2)' }}
                    >
                      {peakFormatter(props.value)}
                    </text>
                  ) : null
                }
              />
            ) : null}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/**
 * A single-series line with a ~10% wash beneath it. Rows: { label, value }.
 * Only the peak is labelled; the crosshair tooltip carries the rest.
 */
export function ChanceArea({ data, height = 150, tooltip, yDomain = [0, 100], yTicks = [0, 50, 100], ariaLabel }) {
  const peak = peakIndexOf(data)

  return (
    <div className="chart-box" style={{ height }} role="img" aria-label={ariaLabel}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 22, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid {...GRID} />
          {/* preserveStartEnd + a minimum gap: on a phone, seven labels at the edges of an area chart otherwise touch. */}
          <XAxis dataKey="label" interval="preserveStartEnd" minTickGap={16} {...X_AXIS} />
          <YAxis {...Y_AXIS} domain={yDomain} ticks={yTicks} tickFormatter={(v) => `${v}%`} />
          <Tooltip cursor={CURSOR_LINE} content={<DataTooltip format={tooltip} />} isAnimationActive={false} />
          <Area
            type="monotone"
            dataKey="value"
            stroke="var(--data-rain)"
            strokeWidth={2}
            fill="var(--data-rain)"
            fillOpacity={0.1}
            dot={{ r: 4, fill: 'var(--data-rain)', stroke: 'var(--surface-1-solid)', strokeWidth: 2 }}
            activeDot={{ r: 6, fill: 'var(--data-rain)', stroke: 'var(--surface-1-solid)', strokeWidth: 2 }}
            isAnimationActive={false}
          >
            {peak >= 0 ? (
              <LabelList
                dataKey="value"
                content={(props) =>
                  props.index === peak ? (
                    <text
                      x={props.x}
                      y={props.y - 12}
                      textAnchor="middle"
                      fontSize={12}
                      fontWeight={600}
                      style={{ fill: 'var(--ink-2)' }}
                    >
                      {Math.round(props.value)}%
                    </text>
                  ) : null
                }
              />
            ) : null}
          </Area>
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Legend and table twin                                               */
/* ------------------------------------------------------------------ */

/** Always present for >= 2 encodings; identity is never colour alone. */
export function ChartLegend({ items }) {
  return (
    <ul className="chart-legend" aria-label="Legend">
      {items.map((item) => (
        <li key={item.label}>
          <span
            className="chart-legend-swatch"
            style={{ background: item.color, opacity: item.opacity ?? 1 }}
            aria-hidden="true"
          />
          {item.label}
        </li>
      ))}
    </ul>
  )
}

/**
 * The WCAG-clean equivalent of a chart. Tooltips enhance and never gate, so
 * every value a chart shows must also be reachable without hovering.
 */
export function DataTable({ caption, columns, rows }) {
  return (
    <div className="table-scroll" tabIndex={0} role="region" aria-label={caption}>
      <table className="data-table">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} scope="col" className={column.numeric ? 'numeric' : ''}>
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={row.key ?? i}>
              {columns.map((column) => (
                <td key={column.key} className={column.numeric ? 'numeric' : ''}>
                  {row[column.key] ?? '—'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
