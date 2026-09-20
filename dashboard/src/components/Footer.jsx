import { fmtDate } from '../lib/format.js'

/**
 * Where the numbers came from. The API reports this on every monsoon response;
 * the original dashboard discarded it, so nothing on screen said that the
 * history was NASA POWER reanalysis spliced to Open-Meteo for the last days.
 */
export default function Footer({ onset, forecast }) {
  const data = onset?.data
  const sources = data?.sources ? Object.entries(data.sources) : []

  return (
    <footer className="footer">
      <div>
        <h3 className="footer-heading">Not an official forecast</h3>
        <p>
          Mausam is a statistical estimate for advisory use. It is not an IMD forecast or declaration; check with your
          local Krishi Vigyan Kendra before acting on it.
        </p>
      </div>

      <div>
        <h3 className="footer-heading">Monsoon record</h3>
        {data ? (
          <p>
            {data.n_days?.toLocaleString('en-GB')} days, {fmtDate(data.data_start)} to {fmtDate(data.data_end)}.
            {sources.length ? ` Sources: ${sources.map(([name, n]) => `${name} (${n.toLocaleString('en-GB')})`).join(', ')}.` : ''}{' '}
            {data.note}
          </p>
        ) : (
          <p>Loading provenance…</p>
        )}
      </div>

      <div>
        <h3 className="footer-heading">Live forecast</h3>
        <p>{forecast?.open_meteo_forecast?.source ?? 'Open-Meteo'}. {forecast?.note}</p>
      </div>
    </footer>
  )
}
