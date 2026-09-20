import { Panel } from './ui.jsx'
import { describeAge, fmtDate, fmtMonthYear, fmtNumber, fmtSigned } from '../lib/format.js'

/*
 * ENSO, IOD and MJO, shown as context.
 *
 * Two things are stated on the surface rather than buried:
 *
 *  1. Each value carries the date it DESCRIBES, and how old that is. These
 *     feeds lag badly - measured, ONI ran ~81 days behind and DMI ~112 - so a
 *     bare "+1.80" would imply a freshness it does not have.
 *  2. They are not inputs to the 7-day model. An ablation showed ONI and DMI
 *     cannot be learned from a 2.5-year training window (only 8% and 16% of
 *     validation values fall inside the training range), and MJO did not
 *     improve validation ROC-AUC either.
 */

const PHASE_LABEL = {
  el_nino: 'El Niño',
  la_nina: 'La Niña',
  neutral: 'Neutral',
  positive_iod: 'Positive IOD',
  negative_iod: 'Negative IOD',
}

function Tile({ kicker, headline, value, unit, asOf, monthly, lag, children }) {
  return (
    <article className="climate-tile">
      <p className="stat-label">{kicker}</p>
      <p className="climate-headline">{headline}</p>
      <p className="climate-value">
        {value}
        <span className="stat-unit">{unit}</span>
      </p>
      <p className="stat-note">
        {monthly ? fmtMonthYear(asOf) : fmtDate(asOf)} · {describeAge(asOf)}
      </p>
      <p className="stat-note muted">Published about {lag} days after the period it describes</p>
      {children}
    </article>
  )
}

export default function ClimateStrip({ resource }) {
  const climate = resource.data

  return (
    <Panel id="climate" eyebrow="Context" title="Large-scale climate drivers" resource={resource}>
      {climate ? (
        <>
          <div className="climate-grid">
            <Tile
              kicker="ENSO · Oceanic Niño Index"
              headline={PHASE_LABEL[climate.enso.phase] ?? climate.enso.phase}
              value={fmtSigned(climate.enso.value, 2)}
              unit="°C"
              asOf={climate.enso.as_of}
              monthly
              lag={climate.enso.publication_lag_days}
            />
            <Tile
              kicker="IOD · Dipole Mode Index"
              headline={PHASE_LABEL[climate.iod.phase] ?? climate.iod.phase}
              value={fmtSigned(climate.iod.value, 2)}
              unit="°C"
              asOf={climate.iod.as_of}
              monthly
              lag={climate.iod.publication_lag_days}
            />
            <Tile
              kicker="MJO · RMM amplitude"
              headline={climate.mjo.phase}
              value={fmtNumber(climate.mjo.value, 2)}
              unit=""
              asOf={climate.mjo.as_of}
              lag={climate.mjo.publication_lag_days}
            />
          </div>
          <p className="callout callout-quiet">
            <span>{climate.note}</span>
          </p>
        </>
      ) : null}
    </Panel>
  )
}
