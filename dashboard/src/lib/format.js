/** Formatting and small derivations shared by the panels. */

const num = (value) => (value === null || value === undefined ? NaN : Number(value))

export function fmtNumber(value, digits = 0) {
  const n = num(value)
  return Number.isFinite(n) ? n.toFixed(digits) : '—'
}

export function fmtMm(value, digits = 0) {
  const n = num(value)
  return Number.isFinite(n) ? `${n.toFixed(digits)} mm` : '—'
}

export function fmtPercent(fraction, digits = 0) {
  const n = num(fraction)
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : '—'
}

export function fmtSigned(value, digits = 1) {
  const n = num(value)
  if (!Number.isFinite(n)) return '—'
  return `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n).toFixed(digits)}`
}

/**
 * 'YYYY-MM-DD' -> a local Date. `new Date('2026-09-21')` is parsed as UTC
 * midnight, which renders as the PREVIOUS day west of Greenwich; building the
 * date from parts keeps it on the calendar day the API meant.
 */
export function parseISODate(iso) {
  if (!iso) return null
  const [y, m, d] = String(iso).slice(0, 10).split('-').map(Number)
  if (!y || !m || !d) return null
  return new Date(y, m - 1, d)
}

export function fmtDate(iso, options = { day: 'numeric', month: 'short', year: 'numeric' }) {
  const date = parseISODate(iso)
  return date ? date.toLocaleDateString('en-GB', options) : '—'
}

export const fmtDay = (iso) => fmtDate(iso, { day: 'numeric', month: 'short' })
export const fmtWeekday = (iso) => fmtDate(iso, { weekday: 'short', day: 'numeric' })
export const fmtMonthYear = (iso) => fmtDate(iso, { month: 'short', year: 'numeric' })
export const monthName = (iso) => fmtDate(iso, { month: 'long' })

/** Whole months between an ISO date and now, for "this index is N months old". */
export function monthsSince(iso, now = new Date()) {
  const date = parseISODate(iso)
  if (!date) return null
  return (now.getFullYear() - date.getFullYear()) * 12 + (now.getMonth() - date.getMonth())
}

export function describeAge(iso, now = new Date()) {
  const date = parseISODate(iso)
  if (!date) return ''
  const days = Math.round((now - date) / 86_400_000)
  if (days <= 1) return 'today'
  if (days < 45) return `${days} days old`
  return `${Math.round(days / 30)} months old`
}

/** snake_case / kebab-case API values -> "Sentence case". */
export function humanise(value) {
  if (!value) return ''
  const text = String(value).replace(/[_-]+/g, ' ').trim()
  return text.charAt(0).toUpperCase() + text.slice(1)
}

/**
 * How the risk should be presented.
 *
 * When `agreement.threshold_degenerate` is set the district-month's threshold
 * is ~0 mm, so "insufficient rainfall" can barely fire and a LOW label means
 * nothing - the README's table has 265 of 314 districts in that state in
 * January. The API flags it; the original dashboard ignored the flag and
 * printed a green "Low risk" regardless. It is shown as informational instead,
 * never as reassurance.
 */
export function describeRisk(forecast) {
  const level = forecast?.ml_model?.risk_level
  const degenerate = Boolean(forecast?.agreement?.threshold_degenerate)

  if (degenerate) {
    return {
      level,
      tone: 'info',
      icon: 'info',
      label: 'Dry week is normal here',
      meaningful: false,
    }
  }
  switch (level) {
    case 'HIGH':
      return { level, tone: 'high', icon: 'alert-octagon', label: 'High risk', meaningful: true }
    case 'MODERATE':
      return { level, tone: 'moderate', icon: 'alert-triangle', label: 'Moderate risk', meaningful: true }
    case 'LOW':
      return { level, tone: 'low', icon: 'check-circle', label: 'Low risk', meaningful: true }
    default:
      return { level, tone: 'info', icon: 'info', label: 'Risk unavailable', meaningful: false }
  }
}

/** Largest absolute value, rounded up to a whole number with a floor. */
export function symmetricBound(values, floor = 2) {
  const peak = Math.max(0, ...values.filter(Number.isFinite).map((v) => Math.abs(v)))
  return Math.max(floor, Math.ceil(peak))
}
