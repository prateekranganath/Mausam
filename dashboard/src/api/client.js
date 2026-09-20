/*
 * The single place the dashboard talks to the Mausam API.
 *
 * Every call takes an AbortSignal. The original fetched inline with no way to
 * cancel, so switching districts quickly was a race in which the LAST RESOLVED
 * response won rather than the last requested one - the screen could end up
 * showing a district other than the one selected.
 */

const DEFAULT_BASE = 'http://127.0.0.1:8000'

// Was hardcoded to 127.0.0.1:8000. Set VITE_API_BASE in dashboard/.env.local
// to point at a deployed backend without touching the code.
export const API_BASE = (import.meta.env.VITE_API_BASE || DEFAULT_BASE).replace(/\/+$/, '')

export class ApiError extends Error {
  constructor(message, { status = 0, cause } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.cause = cause
  }
}

/** FastAPI errors arrive as {detail: "text"} or, for 422, {detail: [{msg}, ...]}. */
function detailOf(body, fallback) {
  const detail = body?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((item) => item?.msg).filter(Boolean).join('; ') || fallback
  }
  return fallback
}

async function request(path, { signal, method = 'GET', params } = {}) {
  const query = new URLSearchParams()
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== null && value !== '') query.set(key, String(value))
  }
  const suffix = query.toString() ? `?${query}` : ''

  let response
  try {
    response = await fetch(`${API_BASE}${path}${suffix}`, { method, signal })
  } catch (error) {
    // An abort is a normal part of switching districts, not a failure.
    if (error?.name === 'AbortError') throw error
    throw new ApiError(
      `Cannot reach the Mausam API at ${API_BASE}. Start it with: uvicorn src.api.main:app --reload`,
      { cause: error },
    )
  }

  let body = null
  try {
    body = await response.json()
  } catch {
    // A proxy or crash page is not JSON; fall through to the status message.
  }

  if (!response.ok) {
    throw new ApiError(detailOf(body, `The API returned HTTP ${response.status}.`), {
      status: response.status,
    })
  }
  return body
}

const seg = encodeURIComponent

export const api = {
  health: (signal) => request('/health', { signal }),
  districts: (signal) => request('/districts', { signal }),
  metrics: (signal) => request('/model/metrics', { signal }),
  climate: (signal) => request('/climate/context', { signal }),
  crops: (signal) => request('/crops', { signal }),

  forecast: (district, signal) => request(`/forecast/${seg(district)}`, { signal }),
  historical: (district, days, signal) =>
    request(`/historical/${seg(district)}`, { signal, params: { days } }),
  onset: (district, signal) => request(`/monsoon/onset/${seg(district)}`, { signal }),
  phase: (district, signal) => request(`/monsoon/phase/${seg(district)}`, { signal }),

  // The slow one: a free-tier LLM call, up to OPENROUTER_TIMEOUT_SECONDS.
  advisory: (district, signal) => request(`/advisory/${seg(district)}`, { signal }),

  cropAdvisory: (district, { crop, sowingDate }, signal) =>
    request(`/advisory/crop/${seg(district)}`, {
      signal,
      params: { crop, sowing_date: sowingDate },
    }),

  alertPreview: (district, { crop, sowingDate }, signal) =>
    request('/alerts/telegram/preview', {
      signal,
      params: { district, crop, sowing_date: sowingDate },
    }),
  // The client sends a district and a crop, never message text: the server
  // composes what is delivered, and the recipient is fixed server-side.
  alertSend: (district, { crop, sowingDate }, signal) =>
    request('/alerts/telegram/send', {
      signal,
      method: 'POST',
      params: { district, crop, sowing_date: sowingDate },
    }),
}
