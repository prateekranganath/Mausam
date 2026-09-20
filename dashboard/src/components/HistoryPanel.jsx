import { useMemo, useState } from 'react'
import { ChartLegend, DailyBars, DataTable } from './charts.jsx'
import { Panel, Stat, ViewToggle } from './ui.jsx'
import { fmtDay, fmtMm, fmtNumber } from '../lib/format.js'

const RECENT_DAYS = 30

/**
 * Ninety days of observed-style rainfall with the last thirty emphasised.
 *
 * This replaces a 30/90-day toggle that sat inside the chart card. A filter
 * that lives inside one chart is per-chart state; here the full window is
 * always shown and the recent stretch is simply drawn at full strength while
 * older days recede (same hue at reduced opacity - same entity, less emphasis).
 */
export default function HistoryPanel({ resource }) {
  const [view, setView] = useState('chart')
  const history = resource.data

  const days = useMemo(() => history?.data ?? [], [history])
  const recentStart = Math.max(0, days.length - RECENT_DAYS)

  const bars = days.map((day, index) => ({
    label: fmtDay(day.date),
    value: day.rainfall == null ? null : Number(day.rainfall),
    fill: 'var(--data-rain)',
    fillOpacity: index >= recentStart ? 1 : 0.4,
  }))

  const stats = useMemo(() => {
    const valid = (rows) => rows.map((d) => d.rainfall).filter((v) => Number.isFinite(v))
    const recent = valid(days.slice(recentStart))
    const all = valid(days)
    const sum = (values) => values.reduce((a, b) => a + b, 0)
    return {
      recentTotal: sum(recent),
      allTotal: sum(all),
      peak: all.length ? Math.max(...all) : 0,
      peakDay: days.find((d) => d.rainfall === Math.max(...all))?.date,
      wetDays: recent.filter((v) => v >= 5).length,
    }
  }, [days, recentStart])

  return (
    <Panel
      id="history"
      eyebrow="Reanalysis"
      title="Recent rainfall"
      resource={resource}
      actions={<ViewToggle view={view} onChange={setView} />}
    >
      {history ? (
        <>
          <div className="stat-grid stat-grid-tight">
            <Stat label="Last 30 days" value={fmtNumber(stats.recentTotal)} unit="mm" note={`${stats.wetDays} days above 5 mm`} />
            <Stat label="Last 90 days" value={fmtNumber(stats.allTotal)} unit="mm" />
            <Stat
              label="Wettest day"
              value={fmtNumber(stats.peak, 1)}
              unit="mm"
              note={stats.peakDay ? fmtDay(stats.peakDay) : null}
            />
          </div>

          {view === 'chart' ? (
            <>
              <ChartLegend
                items={[
                  { label: 'Last 30 days', color: 'var(--data-rain)' },
                  { label: 'Days 31 to 90', color: 'var(--data-rain)', opacity: 0.4 },
                ]}
              />
              <DailyBars
                data={bars}
                height={250}
                xInterval={Math.max(1, Math.floor(days.length / 7))}
                ariaLabel={`Daily rainfall over the last ${days.length} days. Total ${fmtMm(stats.allTotal)}.`}
                tooltip={(row) => ({
                  title: row.label,
                  rows: [{ label: 'rainfall', value: row.value == null ? 'no data' : `${fmtNumber(row.value, 1)} mm`, color: 'var(--data-rain)' }],
                })}
              />
            </>
          ) : (
            <DataTable
              caption="Daily rainfall, most recent first"
              columns={[
                { key: 'label', label: 'Day' },
                { key: 'rain', label: 'Rainfall (mm)', numeric: true },
              ]}
              rows={[...days].reverse().map((day) => ({
                key: day.date,
                label: fmtDay(day.date),
                rain: day.rainfall == null ? '—' : fmtNumber(day.rainfall, 1),
              }))}
            />
          )}

          <p className="muted small">
            {history.missing_rainfall_days
              ? `${history.missing_rainfall_days} days have no value. `
              : ''}
            {history.note}
          </p>
        </>
      ) : null}
    </Panel>
  )
}
