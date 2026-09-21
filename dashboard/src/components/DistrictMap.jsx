import { useEffect } from 'react'
import { CircleMarker, MapContainer, TileLayer, Tooltip, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { Panel } from './ui.jsx'

/*
 * A map for choosing a district.
 *
 * This is the map from the teammates' "Restore interactive dashboard map"
 * commit, ported onto the component-based dashboard, with one deliberate
 * difference: it does not colour every district.
 *
 * Their version coloured each polygon from `riskForArea`, which takes ONE
 * district's forecast (the selected one) and adds
 *
 *     localSwing = ((areaIndex * 11 + weekIndex * 13) % 17) - 8
 *
 * a fixed pseudo-random offset by polygon index. Every district on the map was
 * therefore the same forecast plus jitter, and "Week 2/3/4" were invented too,
 * since the forecast only covers 7 days. It was the same class of fabricated
 * figure as the "Break spell risk: 98%" removed from the dashboard earlier, so
 * it is not carried over, and neither is the heat layer built on it.
 *
 * What the map can honestly show is:
 *   - where every servable district is (exact coordinates from GET /districts);
 *   - which one is selected, coloured by ITS real forecast risk;
 *   - a way to choose a district by clicking it.
 * Colouring every district truthfully needs a forecast per district, which is a
 * batch job the API does not have yet (the deferred "risk maps" feature).
 *
 * Markers sit at district centroids rather than drawing polygons. That removes
 * the runtime download of a third-party GeoJSON file, and the name matching it
 * needed to line ~640 polygons up with 313 districts.
 */

const INDIA_BOUNDS = [
  [6.0, 68.0],
  [37.5, 97.5],
]

const TONE_COLOUR = {
  low: 'var(--status-low)',
  moderate: 'var(--status-moderate)',
  high: 'var(--status-high)',
  info: 'var(--status-info)',
}

function FlyToSelected({ position }) {
  const map = useMap()
  useEffect(() => {
    if (!position) return
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    const zoom = Math.max(map.getZoom(), 6)
    if (reduced) map.setView(position, zoom, { animate: false })
    else map.flyTo(position, zoom, { duration: 0.8 })
  }, [position, map])
  return null
}

export default function DistrictMap({ resource, selected, risk, onSelect }) {
  const districts = resource.data?.districts ?? []
  const current = districts.find((district) => district.name === selected)
  const colour = TONE_COLOUR[risk?.tone] ?? 'var(--accent-sky)'

  return (
    <Panel id="map" eyebrow="Explore" title="Choose a district on the map" resource={resource}>
      <div className="map-frame">
        <MapContainer
          bounds={INDIA_BOUNDS}
          minZoom={4}
          maxZoom={10}
          scrollWheelZoom={false}
          className="map-canvas"
          // A dark page should not have a bright basemap, but the dark filter
          // in App.css needs the standard OSM tiles rather than a third-party style.
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />

          {districts
            .filter((district) => district.name !== selected)
            .map((district) => (
              <CircleMarker
                key={district.name}
                center={[district.latitude, district.longitude]}
                radius={5}
                pathOptions={{
                  color: 'var(--surface-1-solid)',
                  weight: 1,
                  fillColor: 'var(--data-muted)',
                  fillOpacity: 0.75,
                }}
                eventHandlers={{ click: () => onSelect(district.name) }}
              >
                <Tooltip direction="top" offset={[0, -4]}>
                  {district.name} · {district.state}
                </Tooltip>
              </CircleMarker>
            ))}

          {/* Drawn last so it sits on top of its neighbours. */}
          {current ? (
            <CircleMarker
              key={`selected-${current.name}`}
              center={[current.latitude, current.longitude]}
              radius={10}
              pathOptions={{ color: 'var(--ink-1)', weight: 2, fillColor: colour, fillOpacity: 0.95 }}
            >
              <Tooltip direction="top" offset={[0, -8]} permanent>
                {current.name}
              </Tooltip>
            </CircleMarker>
          ) : null}

          <FlyToSelected position={current ? [current.latitude, current.longitude] : null} />
        </MapContainer>
      </div>

      <ul className="chart-legend map-legend" aria-label="Map key">
        <li>
          <span className="chart-legend-swatch map-swatch-selected" style={{ background: colour }} aria-hidden="true" />
          Selected district{risk?.meaningful ? `, coloured by its forecast risk (${risk.label.toLowerCase()})` : ''}
        </li>
        <li>
          <span className="chart-legend-swatch" style={{ background: 'var(--data-muted)' }} aria-hidden="true" />
          Other districts: click one to load it
        </li>
      </ul>

      <p className="muted small">
        Only the selected district is coloured, from its own forecast. Colouring every district would need a forecast
        for each one, which is not computed yet. The district search above is the keyboard-accessible way to choose.
      </p>
    </Panel>
  )
}
