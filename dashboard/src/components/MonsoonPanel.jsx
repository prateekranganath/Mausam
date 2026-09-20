import { useMemo, useState } from 'react'
import Icon from './Icon.jsx'
import { ChartLegend, DailyBars, DataTable } from './charts.jsx'
import { Chip, EmptyState, Panel, ViewToggle } from './ui.jsx'
import { fmtDate, fmtDay, fmtNumber, fmtSigned, humanise, symmetricBound } from '../lib/format.js'

/*
 * These two panels replace three numbers the original dashboard invented in
 * the browser:
 *
 *   onsetConfidence = max(20, 100 - rainfallProbability)
 *   breakRisk       = min(98, round(p * 0.7 + (100 - totalRain / 2)))
 *   heavyRainRisk   = min(99, round(meanChance * 0.9))
 *
 * breakRisk saturated at 98 for any forecast under ~40 mm, so "Break spell
 * risk: 98%" is what it printed almost every time, presented with the same
 * authority as model output. The backend measures onset and spell phase
 * directly (/monsoon/onset, /monsoon/phase); these show what it measured.
 */

const ONSET_STEPS = [
  { status: 'pre_onset', label: 'Not yet arrived' },
  { status: 'onset_likely', label: 'Rain started' },
  { status: 'onset_confirmed', label: 'Confirmed' },
  { status: 'post_onset', label: 'Established' },
]

const SEASON_LABEL = { southwest: 'Southwest monsoon', northeast: 'Northeast monsoon' }

export function OnsetPanel({ resource }) {
  const onset = resource.data
  const stepIndex = ONSET_STEPS.findIndex((step) => step.status === onset?.status)
  const clim = onset?.climatology
  const median = clim?.median_onset_date_label

  return (
    <Panel id="onset" eyebrow="Monsoon" title="Has the monsoon arrived?" resource={resource}>
      {onset ? (
        <>
          <p className="muted small">
            {SEASON_LABEL[onset.season] ?? humanise(onset.season)} · {onset.district}
          </p>

          {/* The distinction between "rain started" and "confirmed" is the honest part: persistence can only be checked in hindsight. */}
          <ol className="stepper" aria-label="Monsoon onset progress">
            {ONSET_STEPS.map((step, index) => {
              const reached = stepIndex >= 0 && index <= stepIndex
              const current = index === stepIndex
              return (
                <li
                  key={step.status}
                  className={`step ${reached ? 'is-reached' : ''} ${current ? 'is-current' : ''}`}
                  aria-current={current ? 'step' : undefined}
                >
                  <span className="step-dot">{reached ? <Icon name="check-circle" size={16} /> : null}</span>
                  <span className="step-label">{step.label}</span>
                </li>
              )
            })}
          </ol>

          <p className="status-line">
            <Chip tone={onset.status === 'onset_likely' ? 'moderate' : stepIndex >= 2 ? 'low' : 'neutral'}
              icon={onset.status === 'onset_likely' ? 'alert-triangle' : stepIndex >= 2 ? 'check-circle' : 'clock'}>
              {humanise(onset.status)}
            </Chip>
            <span>{onset.status_description}</span>
          </p>

          <dl className="facts">
            <div>
              <dt>Onset date</dt>
              <dd>{onset.onset_date ? fmtDate(onset.onset_date) : '—'}</dd>
            </div>
            <div>
              <dt>Against normal</dt>
              <dd>
                {onset.anomaly_label ?? '—'}
                {median ? <span className="muted"> · median {median}</span> : null}
              </dd>
            </div>
            <div>
              <dt>Trigger</dt>
              <dd>
                {onset.trigger_7day_rainfall_mm != null
                  ? `${fmtNumber(onset.trigger_7day_rainfall_mm)} mm in 7 days, ${onset.trigger_rainy_days} rainy days`
                  : '—'}
              </dd>
            </div>
            <div>
              <dt>False starts rejected</dt>
              <dd>
                {onset.rejected_false_onsets?.length
                  ? onset.rejected_false_onsets.map((d) => fmtDay(d)).join(', ')
                  : 'None'}
              </dd>
            </div>
          </dl>

          {clim && clim.median_day_of_year == null ? (
            <p className="muted small">
              Not enough seasons on record ({clim.n_seasons} of {clim.min_seasons_required}) for a median onset date.
            </p>
          ) : null}

          <p className="callout callout-quiet">
            <Icon name="info" size={16} />
            <span>
              Local rainfall onset. {onset.not_imd_criterion} It typically runs ahead of IMD’s announcement on the
              southern and eastern coasts.
            </span>
          </p>
        </>
      ) : null}
    </Panel>
  )
}

const PHASE_CHIP = {
  active: { tone: 'info', icon: 'zap', label: 'Active spell' },
  break: { tone: 'moderate', icon: 'pause', label: 'Break spell' },
  normal: { tone: 'neutral', icon: 'minus', label: 'Normal' },
  not_applicable: { tone: 'neutral', icon: 'info', label: 'Not applicable' },
}

const PHASE_FILL = {
  active: 'var(--data-rain)',
  break: 'var(--data-dry)',
}

export function PhasePanel({ resource }) {
  const [view, setView] = useState('chart')
  const phase = resource.data

  const days = useMemo(
    () =>
      (phase?.recent_30_days ?? []).map((day) => ({
        key: day.date,
        label: fmtDay(day.date),
        anomaly: day.anomaly_sd,
        rain: day.rainfall_mm,
        phase: day.phase,
      })),
    [phase],
  )
  const hasAnomaly = days.some((day) => Number.isFinite(day.anomaly))
  const bound = symmetricBound(days.map((day) => day.anomaly), 2)

  // Emphasis, not a rainbow: ordinary days recede to grey and only the spells
  // are coloured, so the eye lands on what the reader needs to see.
  const bars = days.map((day) => ({
    label: day.label,
    value: Number.isFinite(day.anomaly) ? day.anomaly : null,
    negative: Number.isFinite(day.anomaly) && day.anomaly < 0,
    fill: PHASE_FILL[day.phase] ?? 'var(--data-muted)',
    phase: day.phase,
    rain: day.rain,
  }))

  const chip = PHASE_CHIP[phase?.monsoon_phase] ?? PHASE_CHIP.not_applicable

  return (
    <Panel
      id="phase"
      eyebrow="Monsoon"
      title="Active or break spell?"
      resource={resource}
      actions={hasAnomaly ? <ViewToggle view={view} onChange={setView} /> : null}
    >
      {phase ? (
        <>
          <p className="status-line">
            <Chip tone={chip.tone} icon={chip.icon}>{chip.label}</Chip>
            <span>
              {phase.monsoon_phase === 'not_applicable'
                ? phase.phase_description
                : `${phase.days_in_current_phase} ${phase.days_in_current_phase === 1 ? 'day' : 'days'} so far · ${fmtSigned(
                    phase.rainfall_anomaly_sd,
                    2,
                  )} SD against normal for this week`}
            </span>
          </p>

          {!hasAnomaly ? (
            <EmptyState icon="info" title="No spell data in the last 30 days">
              Active and break spells are only defined for June to September.
            </EmptyState>
          ) : view === 'chart' ? (
            <>
              <ChartLegend
                items={[
                  { label: 'Active spell', color: 'var(--data-rain)' },
                  { label: 'Break spell', color: 'var(--data-dry)' },
                  { label: 'Normal', color: 'var(--data-muted)' },
                ]}
              />
              <DailyBars
                data={bars}
                height={230}
                yDomain={[-bound, bound]}
                yFormatter={(v) => `${v > 0 ? '+' : ''}${v}`}
                referenceLines={[
                  { y: 0, strong: true },
                  // Labelled at the LEFT: the recent, most informative days are at the right.
                  { y: 1, label: '+1 SD', labelPosition: 'insideTopLeft' },
                  { y: -1, label: '−1 SD', labelPosition: 'insideBottomLeft' },
                ]}
                xInterval={4}
                ariaLabel="Last 30 days of standardised rainfall anomaly. Days coloured blue are in an active spell, orange in a break spell."
                tooltip={(row) => ({
                  title: row.label,
                  rows: [
                    {
                      label: `standard deviations · ${humanise(row.phase)}`,
                      value: row.value == null ? '—' : fmtSigned(row.value, 2),
                      color: row.fill,
                    },
                    { label: 'rainfall that day', value: `${fmtNumber(row.rain, 1)} mm`, color: 'var(--data-rain)' },
                  ],
                })}
              />
            </>
          ) : (
            <DataTable
              caption="Last 30 days of rainfall anomaly and spell phase"
              columns={[
                { key: 'label', label: 'Day' },
                { key: 'rain', label: 'Rainfall (mm)', numeric: true },
                { key: 'anomaly', label: 'Anomaly (SD)', numeric: true },
                { key: 'phase', label: 'Phase' },
              ]}
              rows={days.map((day) => ({
                key: day.key,
                label: day.label,
                rain: fmtNumber(day.rain, 1),
                anomaly: Number.isFinite(day.anomaly) ? fmtSigned(day.anomaly, 2) : '—',
                phase: humanise(day.phase),
              }))}
            />
          )}

          <p className="callout callout-quiet">
            <Icon name="info" size={16} />
            <span>{phase.caveats}</span>
          </p>
        </>
      ) : null}
    </Panel>
  )
}
