import { useMemo, useState } from 'react'
import { ChanceArea, DailyBars, DataTable } from './charts.jsx'
import { EmptyState, Panel, ViewToggle } from './ui.jsx'
import { fmtMm, fmtNumber, fmtWeekday } from '../lib/format.js'

/**
 * The 7-day outlook as two aligned charts rather than one dual-axis chart.
 *
 * Rainfall is in millimetres and chance of rain is a percentage. Plotting both
 * on one plot needs two y-scales, whose relative alignment is arbitrary - it
 * invents a correlation that is not in the data. So each gets its own panel
 * sharing the same x positions (small multiples), and both use the rain hue
 * because they measure the same physical thing.
 */
export default function ForecastPanel({ resource }) {
  const [view, setView] = useState('chart')
  const daily = resource.data?.open_meteo_forecast?.daily

  const rows = useMemo(
    () =>
      (daily ?? []).map((day) => ({
        key: day.time,
        label: fmtWeekday(day.time),
        date: day.time,
        rain: day.precipitation_sum,
        chance: day.precipitation_probability_max,
      })),
    [daily],
  )

  const rainBars = rows.map((row) => ({
    label: row.label,
    value: Number(row.rain ?? 0),
    fill: 'var(--data-rain)',
  }))
  const chance = rows.map((row) => ({ label: row.label, value: Number(row.chance ?? 0) }))
  const total = resource.data?.open_meteo_forecast?.total_precipitation_sum_mm
  const noRain = rainBars.length > 0 && rainBars.every((bar) => !(bar.value > 0))

  return (
    <Panel
      id="forecast"
      eyebrow="Open-Meteo"
      title="Next 7 days"
      resource={resource}
      actions={<ViewToggle view={view} onChange={setView} />}
    >
      {view === 'chart' ? (
        <>
          <div className="chart-caption">
            <span>Rainfall per day</span>
            <span className="muted">7-day total {fmtMm(total, 1)}</span>
          </div>
          {noRain ? (
            // A week of 0 mm draws an empty plot that looks broken. It is a
            // real answer, so state it.
            <EmptyState icon="cloud-rain" title="No rain is forecast for the next 7 days">
              Open-Meteo shows 0 mm on every day. The chance-of-rain chart below shows how confident it is.
            </EmptyState>
          ) : (
          <DailyBars
            data={rainBars}
            height={210}
            labelPeak
            yFormatter={(v) => `${v}`}
            ariaLabel={`Daily rainfall forecast in millimetres. Seven-day total ${fmtMm(total, 1)}.`}
            tooltip={(row) => ({
              title: row.label,
              rows: [{ label: 'rainfall', value: `${fmtNumber(row.value, 1)} mm`, color: 'var(--data-rain)' }],
            })}
          />
          )}

          <div className="chart-caption">
            <span>Chance of rain</span>
            <span className="muted">highest in the day</span>
          </div>
          <ChanceArea
            data={chance}
            height={150}
            ariaLabel="Daily chance of rain as a percentage."
            tooltip={(row) => ({
              title: row.label,
              rows: [{ label: 'chance of rain', value: `${fmtNumber(row.value)}%`, color: 'var(--data-rain)' }],
            })}
          />
        </>
      ) : (
        <DataTable
          caption="Seven-day rainfall forecast"
          columns={[
            { key: 'label', label: 'Day' },
            { key: 'rain', label: 'Rainfall (mm)', numeric: true },
            { key: 'chance', label: 'Chance of rain (%)', numeric: true },
          ]}
          rows={rows.map((row) => ({
            key: row.key,
            label: row.label,
            rain: fmtNumber(row.rain, 1),
            chance: fmtNumber(row.chance),
          }))}
        />
      )}
    </Panel>
  )
}
