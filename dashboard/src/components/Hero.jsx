import Icon from './Icon.jsx'
import { Chip, ErrorState, Skeleton, Stat } from './ui.jsx'
import { describeRisk, fmtDate, fmtMm, fmtNumber, fmtPercent, monthName } from '../lib/format.js'

/**
 * The headline: one hero figure (the number the page leads with), and the
 * risk read-out beside it.
 *
 * The risk badge always carries an icon and a word, never colour alone. And it
 * never shows a green "Low risk" where the district-month threshold is
 * degenerate - see describeRisk.
 */
export function Hero({ resource }) {
  if (resource.status === 'error') {
    return (
      <section className="hero" aria-label="Headline forecast">
        <ErrorState error={resource.error} onRetry={resource.reload} />
      </section>
    )
  }

  const forecast = resource.data
  if (!forecast) {
    return (
      <section className="hero" aria-label="Headline forecast" aria-busy="true">
        <div className="skeleton-stack">
          <Skeleton height={14} width={180} />
          <Skeleton height={44} width="55%" />
          <Skeleton height={72} width={260} />
        </div>
      </section>
    )
  }

  const risk = describeRisk(forecast)
  const month = monthName(forecast.as_of_date)
  const predicted = forecast.ml_model.predicted_rainfall_mm

  return (
    <section
      className="hero"
      aria-label="Headline forecast"
      aria-busy={resource.status === 'refreshing'}
      data-stale={resource.status === 'refreshing'}
    >
      <div className="hero-copy">
        <p className="eyebrow accent">Field decision support · 7-day outlook</p>
        <h2 className="hero-title">
          {forecast.district}, {forecast.state}
        </h2>
        <p className="hero-meta">
          As of {fmtDate(forecast.as_of_date)} · model {forecast.ml_model.model} v{forecast.ml_model.model_version}
        </p>

        <div className="hero-figure" aria-label={`Expected rain ${fmtNumber(predicted)} millimetres over the next 7 days`}>
          <span className="hero-number">{fmtNumber(predicted)}</span>
          <span className="hero-unit">mm</span>
          <span className="hero-caption">expected rain over the next 7 days</span>
        </div>
      </div>

      <div className={`risk-badge risk-${risk.tone}`}>
        <span className="mini-label">Dry-week risk</span>
        <span className="risk-badge-row">
          <Icon name={risk.icon} size={26} />
          <strong>{risk.label}</strong>
        </span>
        {risk.meaningful ? (
          <span className="risk-badge-sub">
            {fmtPercent(forecast.ml_model.rainfall_probability)} chance the week is unusually dry
          </span>
        ) : null}
      </div>

      {!risk.meaningful && forecast.agreement?.threshold_degenerate ? (
        <p className="callout callout-info hero-callout">
          <Icon name="info" size={18} />
          <span>
            <strong>A dry week is normal here in {month}.</strong> The district’s rainfall threshold is close to 0 mm,
            so an “insufficient rainfall” label can barely fire and a low score would not mean rain is coming. The
            forecast amounts below are still valid.
          </span>
        </p>
      ) : null}
    </section>
  )
}

/** Four supporting tiles. The model-skill tile fails on its own without affecting the rest. */
export function KpiRow({ forecast, metrics }) {
  const data = forecast.data
  if (!data) {
    return (
      <div className="stat-grid" aria-busy="true">
        {[0, 1, 2, 3].map((n) => (
          <article key={n} className="stat">
            <Skeleton height={12} width="60%" />
            <Skeleton height={34} width="45%" />
            <Skeleton height={12} width="80%" />
          </article>
        ))}
      </div>
    )
  }

  const risk = describeRisk(data)
  const agreement = data.agreement ?? {}
  const month = monthName(data.as_of_date)
  const openMeteo = data.open_meteo_forecast?.total_precipitation_sum_mm
  const diff = agreement.difference_mm

  const skill = metrics.data
  const rocModel = skill?.classifier?.roc_auc
  const rocBaseline = skill?.baseline?.roc_auc

  return (
    <div className="stat-grid" data-stale={forecast.status === 'refreshing'}>
      <Stat
        label="Chance of an unusually dry week"
        value={risk.meaningful ? fmtPercent(data.ml_model.rainfall_probability) : 'n/a'}
        tone="accent"
        note={
          risk.meaningful
            ? 'For this district and month, from the trained model'
            : `A dry week is normal for ${month}, so this score is not reported`
        }
      />

      <Stat
        label="Independent forecast (Open-Meteo)"
        value={fmtNumber(openMeteo, 1)}
        unit="mm"
        note={
          Number.isFinite(diff) ? `Model estimate is ${fmtMm(data.ml_model.predicted_rainfall_mm)} (${fmtNumber(Math.abs(diff), 1)} mm apart)` : null
        }
      >
        {agreement.sources_agree === true ? (
          <Chip tone="low" icon="check-circle">Sources agree</Chip>
        ) : agreement.magnitude_diverges ? (
          <Chip tone="moderate" icon="alert-triangle">Sources differ</Chip>
        ) : null}
      </Stat>

      <Stat
        label={`Dry-week threshold · ${month}`}
        value={fmtNumber(agreement.threshold_mm, 1)}
        unit="mm"
        note="7-day rain below this counts as unusually dry for this district"
      />

      <Stat
        label="Model skill (held-out test)"
        value={
          metrics.status === 'error' ? '—' : Number.isFinite(rocModel) ? fmtNumber(rocModel, 2) : '…'
        }
        unit={Number.isFinite(rocModel) ? 'ROC-AUC' : undefined}
        note={
          Number.isFinite(rocBaseline)
            ? `vs ${fmtNumber(rocBaseline, 2)} for a climatology-only baseline. 0.5 is chance.`
            : 'Pooled across all districts'
        }
      />
    </div>
  )
}
