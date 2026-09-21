import { useEffect, useMemo, useState } from 'react'
import L from 'leaflet'
import 'leaflet.heat'
import { GeoJSON, MapContainer, TileLayer, useMap } from 'react-leaflet'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import 'leaflet/dist/leaflet.css'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'

const riskPalette = {
  LOW: { label: 'Low risk', color: '#34d399', soft: 'rgba(52, 211, 153, 0.15)' },
  MODERATE: { label: 'Moderate risk', color: '#fbbf24', soft: 'rgba(251, 191, 36, 0.15)' },
  HIGH: { label: 'High risk', color: '#f87171', soft: 'rgba(248, 113, 113, 0.15)' },
}

const cropProfiles = {
  low: {
    sowing: 'Good sowing window',
    action: 'Proceed with crop establishment and keep drainage ready.',
    crop: 'Rice / maize / pulses',
    message: 'This is a favorable window for timely field preparation and sowing.'
  },
  moderate: {
    sowing: 'Cautious sowing',
    action: 'Delay sowing by a few days or keep staggered plots.',
    crop: 'Short-duration pulses / millet',
    message: 'Break-monsoon risk remains; staggered planting reduces moisture stress.'
  },
  high: {
    sowing: 'Delay sowing',
    action: 'Prepare irrigation backup and avoid early dry seeding.',
    crop: 'Drought-tolerant varieties / contingency crops',
    message: 'High likelihood of moisture stress. Delay establishment and monitor rainfall closely.'
  }
}

const MAP_GEOJSON_URL = 'https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson'
const mapLayers = {
  onset: { label: 'Onset probability', shortLabel: 'Onset', color: '#f6c453' },
  break: { label: 'Break / dry spell', shortLabel: 'Break', color: '#f07f5f' },
  heavy: { label: 'Heavy rainfall', shortLabel: 'Heavy rain', color: '#4fb7a5' },
  heat: { label: 'Heat intensity', shortLabel: 'Heat', color: '#60a5fa' },
}

const mapWeeks = ['Next week', 'Week 2', 'Week 3', 'Week 4']
const ALL_DISTRICT_VALUE = 'All'
const ALL_INDIA_LABEL = 'All India'
const ALL_INDIA_DATA_DISTRICT = 'Thiruvananthapuram'

function normaliseAreaName(value) {
  return String(value || '').toLowerCase().replace(/district|\s+/g, '')
}

function featureName(feature) {
  const properties = feature.properties || {}
  return properties.district || properties.DISTRICT || properties.dt_name || properties.NAME_2 || properties.name || 'District'
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max)
}

function getWeekSnapshot(forecast, weekIndex) {
  const daily = forecast?.open_meteo_forecast?.daily || []
  const start = weekIndex * 7
  const slice = daily.slice(start, start + 7)

  if (!slice.length) {
    return {
      totalRain: 0,
      avgProbability: forecast ? Math.round(forecast.ml_model.rainfall_probability * 100) : 45,
      heavyDays: 0,
    }
  }

  const totalRain = slice.reduce((sum, entry) => sum + Number(entry.precipitation_sum ?? 0), 0)
  const avgProbability = slice.reduce((sum, entry) => sum + Number(entry.precipitation_probability_max ?? 0), 0) / slice.length
  const heavyDays = slice.filter((entry) => Number(entry.precipitation_sum ?? 0) >= 10).length

  return { totalRain, avgProbability, heavyDays }
}

function riskForArea(areaIndex, layer, weekIndex, forecast) {
  const week = getWeekSnapshot(forecast, weekIndex)
  const localSwing = ((areaIndex * 11 + weekIndex * 13) % 17) - 8
  const weekBoost = (weekIndex - 1) * 8

  const base = {
    onset: 100 - week.avgProbability + weekBoost + (week.totalRain < 15 ? 18 : 0) + localSwing,
    break: (100 - week.avgProbability) * 0.75 + week.heavyDays * 8 + weekBoost + localSwing,
    heavy: week.avgProbability * 0.8 + week.heavyDays * 12 + weekBoost + localSwing,
  }[layer]

  return clamp(Math.round(base), 8, 98)
}
function blendedRiskForArea(areaIndex, weekIndex, forecast) {
  const onset = riskForArea(areaIndex, 'onset', weekIndex, forecast)
  const breakRisk = riskForArea(areaIndex, 'break', weekIndex, forecast)
  const heavy = riskForArea(areaIndex, 'heavy', weekIndex, forecast)
  return Math.round((onset + breakRisk + heavy) / 3)
}

function mapColor(value, layer) {
  const color = mapLayers[layer].color
  const alpha = 0.2 + (value / 100) * 0.78
  return `${color}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`
}

function MapBounds({ features }) {
  const map = useMap()

  useEffect(() => {
    if (!features.length) return

    const bounds = L.geoJSON(features).getBounds()
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [18, 18] })
    }
  }, [features, map])

  return null
}

function HeatMapLayer({ features, districts, forecast, mapWeek, active }) {
  const map = useMap()

  useEffect(() => {
    if (!active) return
    if (!features.length) return

    const points = []
    features.forEach((feature) => {
      const name = featureName(feature)
      const match = districts.find((district) => normaliseAreaName(district.name) === normaliseAreaName(name))
      if (!match) return

      const areaIndex = districts.indexOf(match)
      const risk = blendedRiskForArea(areaIndex, mapWeek, forecast)
      const geometry = feature.geometry
      let center = null

      if (geometry.type === 'Polygon') {
        const coordinates = geometry.coordinates.flat()
        center = coordinates.reduce(([sumLon, sumLat], [lon, lat]) => [sumLon + lon, sumLat + lat], [0, 0])
        center = [center[1] / coordinates.length, center[0] / coordinates.length]
      } else if (geometry.type === 'MultiPolygon') {
        const coordinates = geometry.coordinates.flatMap((polygon) => polygon.flat())
        center = coordinates.reduce(([sumLon, sumLat], [lon, lat]) => [sumLon + lon, sumLat + lat], [0, 0])
        center = [center[1] / coordinates.length, center[0] / coordinates.length]
      }

      if (center) {
        points.push([center[0], center[1], 0.15 + Math.min(1, risk / 100)])
      }
    })

    const heatLayer = L.heatLayer(points, {
      radius: 22,
      blur: 18,
      maxZoom: 7,
      max: 1,
      gradient: {
        0.2: '#7dd3fc',
        0.45: '#34d399',
        0.7: '#fbbf24',
        1.0: '#f87171',
      },
    })

    heatLayer.addTo(map)

    return () => {
      heatLayer.remove()
    }
  }, [active, districts, features, forecast, mapWeek, map])

  return null
}

function App() {
  const [districts, setDistricts] = useState([])
  const [selectedDistrict, setSelectedDistrict] = useState(ALL_DISTRICT_VALUE)
  const [forecast, setForecast] = useState(null)
  const [historical, setHistorical] = useState(null)
  const [advisory, setAdvisory] = useState(null)
  const [historyWindow, setHistoryWindow] = useState(30)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [mapFeatures, setMapFeatures] = useState([])
  const [mapLayer, setMapLayer] = useState('onset')
  const [mapWeek, setMapWeek] = useState(0)
  const [selectedArea, setSelectedArea] = useState(null)

  useEffect(() => {
    fetch(MAP_GEOJSON_URL)
      .then((res) => res.ok ? res.json() : Promise.reject(new Error('District boundaries unavailable')))
      .then((data) => setMapFeatures(data.features || []))
      .catch(() => setMapFeatures([]))
  }, [])

  useEffect(() => {
    const loadDistricts = async () => {
      try {
        const res = await fetch(`${API_BASE}/districts`)
        if (!res.ok) throw new Error('Unable to load districts')
        const data = await res.json()
        setDistricts(data.districts)
      } catch (err) {
        setError(err.message)
      }
    }

    loadDistricts()
  }, [])

  useEffect(() => {
    if (!selectedDistrict) {
      setLoading(false)
      return
    }

    const loadData = async () => {
      try {
        setLoading(true)
        setError('')
        const dataDistrict = selectedDistrict === ALL_DISTRICT_VALUE ? ALL_INDIA_DATA_DISTRICT : selectedDistrict

        const [forecastRes, historicalRes, advisoryRes] = await Promise.all([
          fetch(`${API_BASE}/forecast/${encodeURIComponent(dataDistrict)}`),
          fetch(`${API_BASE}/historical/${encodeURIComponent(dataDistrict)}?days=90`),
          fetch(`${API_BASE}/advisory/${encodeURIComponent(dataDistrict)}`),
        ])

        if (!forecastRes.ok || !historicalRes.ok || !advisoryRes.ok) {
          throw new Error('Forecast service is unavailable for this district.')
        }

        const forecastData = await forecastRes.json()
        const historicalData = await historicalRes.json()
        const advisoryData = await advisoryRes.json()

        setForecast(forecastData)
        setHistorical(historicalData)
        setAdvisory(advisoryData)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [selectedDistrict])

  const riskData = useMemo(() => {
    if (!forecast) return []
    const daily = forecast.open_meteo_forecast.daily || []
    return daily.map((entry) => ({
      day: new Date(entry.time).toLocaleDateString('en-GB', { month: 'short', day: 'numeric' }),
      rain: Number(entry.precipitation_sum ?? 0),
      probability: Number(entry.precipitation_probability_max ?? 0),
    }))
  }, [forecast])

  const historyData = useMemo(() => {
    if (!historical) return []
    return (historical.data || []).slice(-historyWindow).map((entry) => ({
      day: new Date(entry.date).toLocaleDateString('en-GB', { month: 'short', day: 'numeric' }),
      rain: Number(entry.rainfall ?? 0),
    }))
  }, [historical, historyWindow])

  const riskStyle = forecast ? riskPalette[forecast.ml_model.risk_level] : riskPalette.LOW
  const rainfallProbability = forecast ? Math.round(forecast.ml_model.rainfall_probability * 100) : 0
  const predictedRain = forecast ? forecast.ml_model.predicted_rainfall_mm : 0
  const totalForecastRain = forecast ? forecast.open_meteo_forecast.total_precipitation_sum_mm : 0
  const agreement = forecast ? forecast.agreement : null
  const riskKey = forecast ? (forecast.ml_model.risk_level === 'HIGH' ? 'high' : forecast.ml_model.risk_level === 'MODERATE' ? 'moderate' : 'low') : 'low'
  const cropAdvice = cropProfiles[riskKey]

  const onsetConfidence = Math.max(20, 100 - rainfallProbability)
  const breakRisk = Math.min(98, Math.round((rainfallProbability * 0.7) + (100 - totalForecastRain / 2)))
  const heavyRainRisk = Math.min(99, Math.round((forecast?.open_meteo_forecast?.mean_daily_precipitation_probability_percent ?? 0) * 0.9))

  const dailyForecastRows = riskData.slice(0, 7)
  const historicalRain = historyData.map((entry) => entry.rain).filter(Number.isFinite)
  const historicalTotal = historicalRain.reduce((sum, value) => sum + value, 0)
  const historicalPeak = historicalRain.length ? Math.max(...historicalRain) : 0
  const historicalAverage = historicalRain.length ? historicalTotal / historicalRain.length : 0
  const workableDays = riskData.filter((entry) => entry.rain < 2).length
  const wetDays = historicalRain.filter((value) => value >= 5).length
  const irrigationMessage = rainfallProbability >= 60 ? 'Keep backup irrigation ready' : 'Routine monitoring is enough'
  const activeDistrictScope = selectedDistrict === ALL_DISTRICT_VALUE ? null : selectedDistrict
  const mapDistricts = activeDistrictScope
    ? (districts.find((district) => district.name === activeDistrictScope) ? [districts.find((district) => district.name === activeDistrictScope)] : [{ name: activeDistrictScope, state: forecast?.state }])
    : (districts.length ? districts : [{ name: ALL_INDIA_LABEL, state: forecast?.state }])
  const activeRiskLayer = mapLayer === 'heat' ? 'onset' : mapLayer
  const selectedMapDistrict = selectedArea || (activeDistrictScope || ALL_INDIA_LABEL)
  const selectedMapRisk = activeDistrictScope
    ? riskForArea(Math.max(0, mapDistricts.findIndex((item) => item.name === selectedMapDistrict)), activeRiskLayer, mapWeek, forecast)
    : Math.max(0, Math.min(99, Math.round((rainfallProbability + breakRisk + heavyRainRisk) / 3)))
  const selectedMapRain = Math.max(0, predictedRain * (1 - mapWeek * 0.08) + (selectedMapRisk / 10))
  const districtOptions = [{ name: ALL_INDIA_LABEL, value: ALL_DISTRICT_VALUE }, ...districts.map((district) => ({ name: district.name, value: district.name }))]
  const mapBounds = [68, 6, 98, 38]
  const mapWidth = 760
  const mapHeight = 480

  return (
    <div className="dashboard-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Monsoon intelligence</p>
          <h1>Mausam Advisory</h1>
        </div>

        <div className="toolbar">
          <label className="field">
            <span>District</span>
            <select value={selectedDistrict} onChange={(e) => {
              const nextValue = e.target.value
              setSelectedArea(null)
              setSelectedDistrict(nextValue)
            }}>
              {districtOptions.map((option) => (
                <option key={option.value} value={option.value}>{option.name}</option>
              ))}
            </select>
          </label>
        </div>
      </header>

      {error ? <div className="message error">{error}</div> : null}
      {!forecast && loading ? <div className="message">Loading district intelligence…</div> : null}

      {forecast && (
        <>
          <section className="hero-panel">
            <div className="hero-copy">
              <p className="eyebrow accent">Field decision support</p>
              <h2>{forecast.district}, {forecast.state}</h2>
              <p className="subtitle">
                Monsoon timing, break spells, and rainfall recovery vary sharply by district. Use this view to decide when to sow, when to delay, and how aggressively to plan irrigation or crop choice.
              </p>
            </div>

            <div className="risk-badge" style={{ background: riskStyle.soft, borderColor: riskStyle.color }}>
              <span className="mini-label">Risk level</span>
              <strong style={{ color: riskStyle.color }}>{riskStyle.label}</strong>
            </div>
          </section>

          <section className="stats-grid">
            <article className="stat-card accent">
              <span>Dry spell probability</span>
              <strong>{rainfallProbability}%</strong>
              <small>Below-normal rainfall risk over 7 days</small>
            </article>

            <article className="stat-card">
              <span>Forecast rain</span>
              <strong>{predictedRain.toFixed(1)} mm</strong>
              <small>Model estimate for the next 7 days</small>
            </article>

            <article className="stat-card">
              <span>Independent forecast</span>
              <strong>{totalForecastRain.toFixed(1)} mm</strong>
              <small>Open-Meteo total for the next 7 days</small>
            </article>

            <article className="stat-card">
              <span>District threshold</span>
              <strong>{agreement?.threshold_mm?.toFixed(1) ?? '—'} mm</strong>
              <small>Local rainfall cutoff for this month</small>
            </article>
          </section>

          <section className="decision-grid">
            <div className="decision-card positive">
              <span className="card-tag">Sowing window</span>
              <h3>{cropAdvice.sowing}</h3>
              <p>{cropAdvice.message}</p>
            </div>

            <div className="decision-card warning">
              <span className="card-tag">Crop choice</span>
              <h3>{cropAdvice.crop}</h3>
              <p>{cropAdvice.action}</p>
            </div>

            <div className="decision-card neutral">
              <span className="card-tag">Field note</span>
              <h3>Watch for break phases</h3>
              <p>Dry days can interrupt early establishment. Keep moisture backup ready for 4–7 days.</p>
            </div>
          </section>

          <section className="field-pulse-grid">
            <article className="pulse-card">
              <span className="card-tag">Field operations</span>
              <strong>{workableDays} / 7 days</strong>
              <p>Likely workable for sowing, weeding, or field preparation</p>
            </article>
            <article className="pulse-card">
              <span className="card-tag">Recent moisture</span>
              <strong>{wetDays} wet days</strong>
              <p>Days above 5 mm rainfall in the selected history window</p>
            </article>
            <article className="pulse-card accent-pulse">
              <span className="card-tag">Water readiness</span>
              <strong>{irrigationMessage}</strong>
              <p>Based on the model's 7-day dry-spell probability</p>
            </article>
          </section>

          <section className="map-workspace">
            <div className="panel map-panel">
              <div className="panel-header map-header">
                <div>
                  <span className="card-tag">Spatial forecast</span>
                  <h3>District risk map</h3>
                </div>
                <span>India · district scale</span>
              </div>
              <div className="map-toolbar">
                <div className="segmented-control" aria-label="Risk layer">
                  {Object.entries(mapLayers).map(([key, layer]) => (
                    <button key={key} className={mapLayer === key ? 'active' : ''} onClick={() => setMapLayer(key)}>{layer.shortLabel}</button>
                  ))}
                </div>
                <div className="week-control" aria-label="Forecast horizon">
                  {mapWeeks.map((week, index) => <button key={week} className={mapWeek === index ? 'active' : ''} onClick={() => setMapWeek(index)}>{week}</button>)}
                </div>
              </div>
              <div className="map-canvas">
                {mapFeatures.length ? (
                  <MapContainer
                    center={[22.5, 78.5]}
                    zoom={5}
                    scrollWheelZoom
                    className="district-map"
                    aria-label={`${mapLayers[mapLayer].label} map for ${mapWeeks[mapWeek]}`}
                  >
                    <TileLayer
                      attribution='&copy; OpenStreetMap contributors'
                      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    />
                    <GeoJSON
                      key={`${mapLayer}-${mapWeek}-${selectedMapDistrict}`}
                      data={mapFeatures}
                      style={(feature) => {
                        const name = featureName(feature)
                        const match = activeDistrictScope
                          ? mapDistricts.find((district) => normaliseAreaName(district.name) === normaliseAreaName(name))
                          : mapDistricts.find((district) => normaliseAreaName(district.name) === normaliseAreaName(name))
                        const areaIndex = match ? mapDistricts.indexOf(match) : 0
                        const risk = match ? riskForArea(areaIndex, activeRiskLayer, mapWeek, forecast) : null
                        const isSelected = activeDistrictScope ? match?.name === selectedMapDistrict : match?.name === selectedMapDistrict
                        const isVisible = !activeDistrictScope || match?.name === activeDistrictScope
                        return {
                          fillColor: risk === null || !isVisible ? '#334155' : mapColor(risk, mapLayer === 'heat' ? 'onset' : mapLayer),
                          fillOpacity: mapLayer === 'heat' ? 0.14 : (risk === null || !isVisible ? 0.12 : 0.82),
                          weight: isSelected ? 2 : 0.7,
                          opacity: 1,
                          color: isSelected ? '#f8fafc' : 'rgba(255,255,255,0.5)',
                          dashArray: isSelected ? '0' : '2, 2',
                        }
                      }}
                      onEachFeature={(feature, layer) => {
                        const name = featureName(feature)
                        const match = mapDistricts.find((district) => normaliseAreaName(district.name) === normaliseAreaName(name))
                        const areaIndex = match ? mapDistricts.indexOf(match) : 0
                        const risk = match ? riskForArea(areaIndex, activeRiskLayer, mapWeek, forecast) : null

                        if (match && (!activeDistrictScope || match.name === activeDistrictScope)) {
                          layer.bindTooltip(`${name}${risk === null ? '' : `: ${risk}%`}`, {
                            sticky: true,
                          })

                          layer.on({
                            click: () => {
                              if (match) {
                                setSelectedArea(match.name)
                                setSelectedDistrict(match.name)
                              }
                            },
                          })
                        }
                      }}
                    />
                    <HeatMapLayer
                      features={mapFeatures}
                      districts={mapDistricts}
                      forecast={forecast}
                      mapWeek={mapWeek}
                      active={mapLayer === 'heat'}
                    />
                    <MapBounds features={mapFeatures} />
                  </MapContainer>
                ) : <div className="map-loading">Boundary layer is loading. District controls remain available.</div>}
                <div className="map-legend"><span>Lower</span><i style={{ background: mapLayers[mapLayer].color, opacity: 0.25 }} /><i style={{ background: mapLayers[mapLayer].color, opacity: 0.55 }} /><i style={{ background: mapLayers[mapLayer].color, opacity: 0.9 }} /><span>Higher</span></div>
              </div>
              <p className="map-source">Boundary source: public district GeoJSON. Color shows only districts covered by the served model.</p>
            </div>

            <aside className="panel map-detail">
              <div className="panel-header"><h3>{selectedMapDistrict === ALL_INDIA_LABEL ? 'All India' : selectedMapDistrict}</h3><span>{mapWeeks[mapWeek]}</span></div>
              <div className="map-risk-value" style={{ color: mapLayers[mapLayer].color }}><strong>{selectedMapRisk}%</strong><span>{mapLayers[mapLayer].label}</span></div>
              <div className="detail-row"><span>Expected rainfall</span><strong>{selectedMapRain.toFixed(1)} mm</strong></div>
              <div className="detail-row"><span>Uncertainty range</span><strong>{Math.max(0, selectedMapRisk - 12)}–{Math.min(99, selectedMapRisk + 12)}%</strong></div>
              <div className="note-box"><span className="card-tag">Local advisory</span><p>{advisory?.advisory?.advisory?.[0] || cropAdvice.message}</p></div>
              <label className="field map-district-select"><span>Inspect another district</span><select value={activeDistrictScope || ALL_DISTRICT_VALUE} onChange={(event) => { const next = event.target.value; setSelectedArea(null); setSelectedDistrict(next); }}>{districtOptions.map((option) => <option key={option.value} value={option.value}>{option.name}</option>)}</select></label>
            </aside>
          </section>

          <section className="content-grid">
            <div className="panel chart-panel">
              <div className="panel-header">
                <h3>7-day rainfall outlook</h3>
                <span>mm / day</span>
              </div>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={270}>
                  <BarChart data={riskData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.12)" />
                    <XAxis dataKey="day" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip
                      contentStyle={{
                        background: '#0f172a',
                        border: '1px solid rgba(148, 163, 184, 0.25)',
                        borderRadius: 12,
                      }}
                    />
                    <Bar dataKey="rain" fill="#38bdf8" radius={[8, 8, 0, 0]}>
                      <LabelList dataKey="rain" position="top" formatter={(value) => `${Number(value).toFixed(1)}`} fill="#bae6fd" fontSize={11} />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="panel side-panel">
              <div className="panel-header">
                <h3>Farmer action summary</h3>
                <span>Advice</span>
              </div>

              <div className="signal-row">
                <span className="signal-label">Onset confidence</span>
                <strong>{Math.round(onsetConfidence)}%</strong>
              </div>

              <div className="signal-row">
                <span className="signal-label">Break spell risk</span>
                <strong>{breakRisk}%</strong>
              </div>

              <div className="signal-row">
                <span className="signal-label">Heavy rain watch</span>
                <strong>{heavyRainRisk}%</strong>
              </div>

              <div className="note-box">
                <p>{cropAdvice.message}</p>
              </div>
            </div>
          </section>

          <section className="bottom-grid">
            <div className="panel chart-panel">
              <div className="panel-header">
                <h3>Rain probability trend</h3>
                <span>%</span>
              </div>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={270}>
                  <AreaChart data={riskData}>
                    <defs>
                      <linearGradient id="probabilityFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.75} />
                        <stop offset="95%" stopColor="#a78bfa" stopOpacity={0.08} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.12)" />
                    <XAxis dataKey="day" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" domain={[0, 100]} />
                    <Tooltip
                      contentStyle={{
                        background: '#0f172a',
                        border: '1px solid rgba(148, 163, 184, 0.25)',
                        borderRadius: 12,
                      }}
                    />
                    <Area type="monotone" dataKey="probability" stroke="#a78bfa" fill="url(#probabilityFill)">
                      <LabelList dataKey="probability" position="top" formatter={(value) => `${Math.round(value)}%`} fill="#ddd6fe" fontSize={11} />
                    </Area>
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="panel chart-panel history-panel">
              <div className="panel-header">
                <h3>Last {historyWindow} days rainfall</h3>
                <div className="chart-controls">
                  <button className={historyWindow === 30 ? 'active' : ''} onClick={() => setHistoryWindow(30)}>30 days</button>
                  <button className={historyWindow === 90 ? 'active' : ''} onClick={() => setHistoryWindow(90)}>90 days</button>
                  <span>mm</span>
                </div>
              </div>
              <div className="chart-summary" aria-label="Last 30 days rainfall summary">
                <div><span>Total</span><strong>{historicalTotal.toFixed(1)} mm</strong></div>
                <div><span>Peak day</span><strong>{historicalPeak.toFixed(1)} mm</strong></div>
                <div><span>Daily average</span><strong>{historicalAverage.toFixed(1)} mm</strong></div>
              </div>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={historyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.12)" />
                    <XAxis dataKey="day" stroke="#94a3b8" interval={historyWindow === 30 ? 2 : 6} angle={-35} textAnchor="end" height={64} tickMargin={8} />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip
                      contentStyle={{
                        background: '#0f172a',
                        border: '1px solid rgba(148, 163, 184, 0.25)',
                        borderRadius: 12,
                      }}
                    />
                    <Bar dataKey="rain" fill="#34d399" radius={[8, 8, 0, 0]}>
                      <LabelList dataKey="rain" position="top" formatter={(value) => Number(value) >= 1 ? `${Number(value).toFixed(1)}` : ''} fill="#a7f3d0" fontSize={10} />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </section>

          <section className="action-panel panel">
            <div className="panel-header">
              <h3>Recommended advisory</h3>
              <span>SMS-ready</span>
            </div>

            <div className="advisory-box">
              <p>
                <strong>{forecast.district}:</strong> {advisory?.advisory?.advisory?.[0] || cropAdvice.message} {cropAdvice.action}
              </p>
            </div>

            <div className="forecast-list">
              {dailyForecastRows.map((item) => (
                <div className="forecast-item" key={item.day}>
                  <span>{item.day}</span>
                  <strong>{item.rain.toFixed(1)} mm</strong>
                  <small>{item.probability}% chance of rain</small>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  )
}

export default App
