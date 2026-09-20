import { useState } from 'react'
import { api } from '../api/client.js'
import { useResource } from '../hooks/useResource.js'
import Icon from './Icon.jsx'
import { Chip, Panel } from './ui.jsx'

const SECTION_LABEL = {
  forecast: 'Forecast',
  onset: 'Monsoon onset',
  monsoon_phase: 'Spell phase',
  crop_advisory: 'Crop advice',
}

/**
 * Preview, then send.
 *
 * The preview is the EXACT text the server would deliver, composed from the
 * same data as every panel above. It works with no Telegram credentials at all,
 * so the whole feature demos on a fresh clone and nobody sends an alert without
 * reading it first. Only the final send needs a bot token and chat id.
 *
 * The browser sends a district and a crop - never message text - and the
 * recipient is fixed server-side. This endpoint has no authentication in front
 * of it, so one that accepted arbitrary text or a chat id would be an open
 * relay for messaging anyone through the project's bot.
 */
export default function TelegramPanel({ district, crop, sowingDate }) {
  const [includeCrop, setIncludeCrop] = useState(true)
  const cropParam = includeCrop ? crop : undefined
  const key = `${district}|${cropParam}|${sowingDate}`

  const preview = useResource(
    (signal) => api.alertPreview(district, { crop: cropParam, sowingDate }, signal),
    [district, cropParam, sowingDate],
    { delay: 500 },
  )

  // A send result belongs to the exact alert it was for; changing district,
  // crop or date invalidates it rather than leaving a stale "Sent".
  const [outcome, setOutcome] = useState({ key: '', status: 'idle' })
  const current = outcome.key === key ? outcome : { status: 'idle' }

  const data = preview.data
  const ready = preview.status === 'ready' && data?.telegram_configured
  const busy = current.status === 'sending'

  async function send() {
    setOutcome({ key, status: 'sending' })
    try {
      const result = await api.alertSend(district, { crop: cropParam, sowingDate })
      setOutcome(
        result.sent
          ? { key, status: 'sent', messageId: result.message_id }
          : { key, status: 'failed', error: result.error },
      )
    } catch (error) {
      setOutcome({ key, status: 'failed', error: error.message })
    }
  }

  return (
    <Panel
      id="telegram"
      eyebrow="Alerts"
      title="Send this as a Telegram alert"
      resource={preview}
      skeleton={
        <div className="skeleton-stack">
          <span className="skeleton" style={{ height: 210 }} />
        </div>
      }
    >
      {data ? (
        <div className="tg">
          <div className="tg-preview">
            <div className="phone" aria-label="Preview of the Telegram message">
              <div className="phone-bar">
                <span className="phone-avatar" aria-hidden="true">M</span>
                <div>
                  <strong>Mausam alerts</strong>
                  <span>bot</span>
                </div>
              </div>
              <div className="bubble">
                <p className="bubble-text">{data.message}</p>
                <span className="bubble-meta">{data.characters} characters</span>
              </div>
            </div>
          </div>

          <div className="tg-side">
            <p className="muted small">
              This is exactly what would be delivered, written from the same data as the panels above. Nothing is sent
              until you press the button.
            </p>

            <p className="chips">
              {data.sections_included.map((section) => (
                <Chip key={section} tone="neutral">{SECTION_LABEL[section] ?? section}</Chip>
              ))}
            </p>

            <label className="check">
              <input type="checkbox" checked={includeCrop} onChange={(event) => setIncludeCrop(event.target.checked)} />
              <span>Include crop advice</span>
            </label>

            <button type="button" className="btn btn-primary" onClick={send} disabled={!ready || busy}>
              <Icon name="send" size={16} />
              {busy ? 'Sending…' : 'Send to Telegram'}
            </button>

            {!data.telegram_configured ? (
              <p className="callout callout-info">
                <Icon name="info" size={16} />
                <span>Sending is disabled. {data.configuration_hint}</span>
              </p>
            ) : null}

            <div aria-live="polite">
              {current.status === 'sent' ? (
                <p className="callout callout-good">
                  <Icon name="check-circle" size={16} />
                  <span>Delivered. Telegram message #{current.messageId}.</span>
                </p>
              ) : null}
              {current.status === 'failed' ? (
                <p className="callout callout-bad" role="alert">
                  <Icon name="alert-octagon" size={16} />
                  <span>Not sent: {current.error}</span>
                </p>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </Panel>
  )
}
