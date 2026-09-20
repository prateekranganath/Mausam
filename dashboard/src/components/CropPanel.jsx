import Icon from './Icon.jsx'
import { Chip, EmptyState, Panel } from './ui.jsx'
import { fmtMm, fmtNumber, humanise } from '../lib/format.js'

/*
 * Real crop advice, replacing a hardcoded three-entry lookup keyed only off
 * the risk level ("low" -> "Rice / maize / pulses", ...). The backend has a
 * rule engine with growth stages, water requirements and attributable
 * recommendations; this surfaces it.
 *
 * Every recommendation shows the rule_id that produced it and the signal values
 * that triggered it, because the point of a deterministic engine is that the
 * advice can be checked rather than merely believed.
 */

const SENSITIVITY = {
  low: { tone: 'low', icon: 'check-circle', label: 'Low drought sensitivity' },
  medium: { tone: 'neutral', icon: 'minus', label: 'Medium drought sensitivity' },
  high: { tone: 'moderate', icon: 'alert-triangle', label: 'High drought sensitivity' },
  critical: { tone: 'high', icon: 'alert-octagon', label: 'Critical stage for water' },
}

const SEVERITY = {
  high: { tone: 'high', icon: 'alert-octagon', label: 'Act now' },
  medium: { tone: 'moderate', icon: 'alert-triangle', label: 'Plan for' },
  info: { tone: 'info', icon: 'info', label: 'Note' },
}

function today() {
  const now = new Date()
  const pad = (n) => String(n).padStart(2, '0')
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`
}

/** Expected rain against what the crop needs this week. */
function WaterMeter({ need, expected, balance }) {
  if (!Number.isFinite(need) || !Number.isFinite(expected)) return null
  const ratio = need > 0 ? expected / need : 1
  const tone = ratio >= 1 ? 'low' : ratio >= 0.6 ? 'moderate' : 'high'
  const icon = tone === 'low' ? 'check-circle' : tone === 'moderate' ? 'alert-triangle' : 'alert-octagon'
  const verdict =
    balance >= 0
      ? `Covered, with about ${fmtNumber(balance)} mm to spare`
      : `Short by about ${fmtNumber(Math.abs(balance))} mm`

  return (
    <div className="water">
      <div className="water-head">
        <span className="stat-label">Water this week</span>
        <Chip tone={tone} icon={icon}>{verdict}</Chip>
      </div>
      <div
        className="meter"
        role="meter"
        aria-label="Forecast rainfall as a share of this stage's weekly water requirement"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(Math.min(ratio, 1) * 100)}
      >
        <div className={`meter-fill meter-${tone}`} style={{ width: `${Math.min(ratio, 1) * 100}%` }} />
      </div>
      <p className="stat-note">
        Forecast {fmtMm(expected)} against roughly {fmtMm(need)} needed
      </p>
    </div>
  )
}

export default function CropPanel({ crops, resource, crop, onCrop, sowingDate, onSowingDate }) {
  const advice = resource.data
  const list = crops.data?.crops ?? []
  const sowing = SENSITIVITY[advice?.stage_drought_sensitivity]

  return (
    <Panel id="crop" eyebrow="Agronomy" title="Crop advice" resource={resource}
      skeleton={<div className="skeleton-stack"><span className="skeleton" style={{ height: 90 }} /><span className="skeleton" style={{ height: 140 }} /></div>}
    >
      {/* Controls sit above the result they scope, in one row. */}
      <div className="controls">
        <label className="field">
          <span className="field-label">Crop</span>
          <select value={crop} onChange={(event) => onCrop(event.target.value)} disabled={!list.length}>
            {list.length === 0 ? <option value={crop}>{humanise(crop)}</option> : null}
            {list.map((item) => (
              <option key={item.key} value={item.key}>{item.display_name}</option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field-label">Sowing date</span>
          <input type="date" value={sowingDate} max={today()} onChange={(event) => onSowingDate(event.target.value)} />
        </label>

        {sowingDate ? (
          <button type="button" className="btn btn-quiet" onClick={() => onSowingDate('')}>
            Not sown yet
          </button>
        ) : null}
      </div>

      {advice ? (
        <div className="crop-result">
          <div className="crop-summary">
            <p className="crop-name">
              <Icon name="sprout" size={20} /> {advice.crop}
            </p>
            {advice.growth_stage ? (
              <p className="status-line">
                <Chip tone="neutral">{humanise(advice.growth_stage)}</Chip>
                {sowing ? <Chip tone={sowing.tone} icon={sowing.icon}>{sowing.label}</Chip> : null}
                <span className="muted small">Day {advice.days_since_sowing} since sowing</span>
              </p>
            ) : advice.sowing_date ? (
              <p className="muted small">
                {advice.past_maturity ? 'This crop is past its expected duration.' : 'Not in a tracked growth stage.'}
              </p>
            ) : (
              <p className="muted small">Not sown yet: showing sowing-window advice. Enter a sowing date for stage-specific guidance.</p>
            )}
          </div>

          <WaterMeter
            need={advice.stage_weekly_water_requirement_mm}
            expected={advice.expected_rainfall_next_7_days_mm}
            balance={advice.water_balance_mm}
          />

          {advice.suppression_note ? (
            <p className="callout callout-info">
              <Icon name="info" size={16} />
              <span>{advice.suppression_note}</span>
            </p>
          ) : null}

          <ul className="reco-list">
            {advice.recommendations.map((rec) => {
              const severity = SEVERITY[rec.severity] ?? SEVERITY.info
              return (
                <li key={rec.rule_id} className={`reco reco-${severity.tone}`}>
                  <span className="reco-icon"><Icon name={severity.icon} size={18} /></span>
                  <div>
                    <p className="reco-label">{severity.label}</p>
                    <p className="reco-action">{rec.action}</p>
                    <details className="reco-why">
                      <summary>Why this advice</summary>
                      <p>{rec.rationale}</p>
                      <p className="mono small">
                        {rec.rule_id}
                        {Object.entries(rec.triggered_by ?? {})
                          .filter(([, value]) => value !== null && value !== undefined)
                          .map(([key, value]) => ` · ${key}=${typeof value === 'number' ? fmtNumber(value, 1) : value}`)
                          .join('')}
                      </p>
                    </details>
                  </div>
                </li>
              )
            })}
          </ul>

          {advice.signals_unavailable?.length ? (
            <details className="reco-why">
              <summary>{advice.signals_unavailable.length} signals were not available</summary>
              <p className="muted small">
                Rules that depend on these did not fire; a missing signal is never treated as a met condition:{' '}
                {advice.signals_unavailable.join(', ')}.
              </p>
            </details>
          ) : null}

          <p className="muted small">{advice.disclaimer}</p>
        </div>
      ) : (
        <EmptyState icon="sprout" title="Choose a crop">Pick a crop to see stage-specific sowing and irrigation advice.</EmptyState>
      )}
    </Panel>
  )
}
