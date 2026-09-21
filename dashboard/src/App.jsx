import { useCallback, useState } from 'react'
import { api } from './api/client.js'
import { useResource } from './hooks/useResource.js'
import AdvisoryPanel from './components/AdvisoryPanel.jsx'
import ClimateStrip from './components/ClimateStrip.jsx'
import CropPanel from './components/CropPanel.jsx'
import DistrictMap from './components/DistrictMap.jsx'
import DistrictPicker from './components/DistrictPicker.jsx'
import Footer from './components/Footer.jsx'
import ForecastPanel from './components/ForecastPanel.jsx'
import { Hero, KpiRow } from './components/Hero.jsx'
import HistoryPanel from './components/HistoryPanel.jsx'
import Icon from './components/Icon.jsx'
import { OnsetPanel, PhasePanel } from './components/MonsoonPanel.jsx'
import TelegramPanel from './components/TelegramPanel.jsx'
import { Chip } from './components/ui.jsx'
import { describeRisk } from './lib/format.js'

const DEFAULT_DISTRICT = 'Thiruvananthapuram'
const DEFAULT_CROP = 'rice_transplanted'

/**
 * The selected district lives in the URL (?district=Nagpur), so a view can be
 * bookmarked or shared. The original kept it in component state only.
 */
function useDistrictParam(fallback) {
  const [district, setDistrict] = useState(
    () => new URLSearchParams(window.location.search).get('district') || fallback,
  )
  const update = useCallback((next) => {
    setDistrict(next)
    const url = new URL(window.location.href)
    url.searchParams.set('district', next)
    window.history.replaceState(null, '', url)
  }, [])
  return [district, update]
}

function ApiStatus({ health }) {
  if (health.status === 'error') {
    return <Chip tone="high" icon="alert-octagon">API offline</Chip>
  }
  if (!health.data) return <Chip tone="neutral" icon="clock">Connecting…</Chip>
  return (
    <Chip tone="low" icon="check-circle" title={`Model ${health.data.classifier} v${health.data.model_version}`}>
      API online · {health.data.n_districts} districts
    </Chip>
  )
}

function OfflineNotice({ message }) {
  return (
    <section className="offline" role="alert">
      <Icon name="alert-octagon" size={28} />
      <div>
        <h2>The forecast service is not reachable</h2>
        <p>{message}</p>
        <p className="muted small">
          From the project root: <code>uvicorn src.api.main:app --reload</code>. If the dashboard is on a different
          origin than <code>localhost:5173</code>, add it to <code>API_ALLOWED_ORIGINS</code> in <code>.env</code>.
        </p>
        <button type="button" className="btn btn-primary" onClick={() => window.location.reload()}>
          <Icon name="refresh" size={16} /> Try again
        </button>
      </div>
    </section>
  )
}

export default function App() {
  const [district, setDistrict] = useDistrictParam(DEFAULT_DISTRICT)
  const [crop, setCrop] = useState(DEFAULT_CROP)
  const [sowingDate, setSowingDate] = useState('')

  // District-independent: fetched once.
  const health = useResource((signal) => api.health(signal), [])
  const districts = useResource((signal) => api.districts(signal), [])
  const metrics = useResource((signal) => api.metrics(signal), [])
  const climate = useResource((signal) => api.climate(signal), [])
  const crops = useResource((signal) => api.crops(signal), [])

  // Per district. Each is its own resource, so one slow or failing endpoint
  // degrades one panel instead of blanking the page.
  const forecast = useResource((signal) => api.forecast(district, signal), [district])
  const historical = useResource((signal) => api.historical(district, 90, signal), [district])
  const onset = useResource((signal) => api.onset(district, signal), [district])
  const phase = useResource((signal) => api.phase(district, signal), [district])
  const cropAdvice = useResource(
    (signal) => api.cropAdvisory(district, { crop, sowingDate }, signal),
    [district, crop, sowingDate],
    { delay: 200 },
  )

  // The forecast analysis, in two steps. The analysis itself is derived by
  // rules on the server and is instant, so it is fetched straight away. The
  // AI paragraph is a separate, slow, rate-limited call to a free-tier model:
  // it waits for the analysis, then debounces, so clicking through districts
  // does not queue a model call for each one, and it can fail without the
  // analysis noticing.
  const analysis = useResource((signal) => api.advisory(district, { polish: false }, signal), [district])
  const aiNote = useResource((signal) => api.advisory(district, { polish: true }, signal), [district], {
    enabled: analysis.status === 'ready',
    delay: 400,
  })

  const offline = health.status === 'error' && health.error?.status === 0

  // Colour the selected district on the map only when the forecast on screen is
  // ITS forecast: while a new district loads, the previous one is still held.
  const selectedRisk = forecast.data?.district === district ? describeRisk(forecast.data) : null

  return (
    <div className="shell">
      <a className="skip-link" href="#main">Skip to content</a>

      <header className="topbar">
        <div className="brand">
          <p className="eyebrow">Monsoon intelligence</p>
          <h1>Mausam Advisory</h1>
        </div>
        <div className="toolbar">
          <ApiStatus health={health} />
          <DistrictPicker
            districts={districts.data?.districts ?? []}
            value={district}
            onChange={setDistrict}
            disabled={!districts.data}
          />
        </div>
      </header>

      <main id="main">
        {offline ? (
          <OfflineNotice message={health.error.message} />
        ) : (
          <>
            <Hero resource={forecast} />
            <KpiRow forecast={forecast} metrics={metrics} />

            <DistrictMap resource={districts} selected={district} risk={selectedRisk} onSelect={setDistrict} />

            <ForecastPanel resource={forecast} />

            <div className="grid-2">
              <OnsetPanel resource={onset} />
              <PhasePanel resource={phase} />
            </div>

            <CropPanel
              crops={crops}
              resource={cropAdvice}
              crop={crop}
              onCrop={setCrop}
              sowingDate={sowingDate}
              onSowingDate={setSowingDate}
            />

            <ClimateStrip resource={climate} />
            <HistoryPanel resource={historical} />
            <AdvisoryPanel analysis={analysis} ai={aiNote} />
            <TelegramPanel district={district} crop={crop} sowingDate={sowingDate} />
          </>
        )}
      </main>

      <Footer onset={onset.data} forecast={forecast.data} />
    </div>
  )
}
