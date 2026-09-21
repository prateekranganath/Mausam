import Icon from './Icon.jsx'
import { Chip, Panel, Skeleton } from './ui.jsx'
import { humanise } from '../lib/format.js'

/*
 * The forecast analysis (GET /advisory/{district}).
 *
 * It has two parts with very different reliability, and the panel treats them
 * that way:
 *
 *   the ANALYSIS   headline, risk, confidence, key factors, where the sources
 *                  differ, actions. Derived by rules from the forecast numbers
 *                  on the server: instant, and it cannot fail because no model
 *                  is involved. This is what the panel is built around.
 *   the AI NOTE    an optional 2-3 sentence plain-language paragraph from a
 *                  free-tier LLM. Slow, rate-limited and sometimes down, so it
 *                  loads separately and its absence is a quiet footnote, not
 *                  an error.
 *
 * The first version of this panel rendered one sentence of an LLM response and
 * showed nothing whenever the model was overloaded. Measured on the live free
 * tier, that was often: a single 503 from one provider meant no advisory at all.
 */

const RISK_TONE = { LOW: 'low', MODERATE: 'moderate', HIGH: 'high' }
const RISK_ICON = { LOW: 'check-circle', MODERATE: 'alert-triangle', HIGH: 'alert-octagon' }

function List({ title, items, ordered = false }) {
  if (!items?.length) return null
  const Tag = ordered ? 'ol' : 'ul'
  return (
    <div className="ai-block">
      <h3 className="ai-heading">{title}</h3>
      <Tag className="ai-list">
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </Tag>
    </div>
  )
}

/** The optional plain-language paragraph, in whatever state it is in. */
function AiNote({ ai, district }) {
  const data = ai.data
  // The AI resource keeps its previous render while it refetches. Its text
  // names a district, so text written for another one must not be shown.
  const current = data && data.forecast?.district === district

  if (ai.status === 'error' || data?.ai_status === 'unavailable') {
    return (
      <p className="callout callout-quiet">
        <Icon name="info" size={16} />
        <span>
          A plain-language note could not be generated right now. The analysis here is complete and does not
          depend on it.{' '}
          <button type="button" className="link-button" onClick={ai.reload}>
            Try again
          </button>
          {/* The real reason (overloaded, rate-limited, bad key...) so a misconfiguration is diagnosable. */}
          {data?.llm_error || ai.error ? (
            <details className="reco-why">
              <summary>Why?</summary>
              <span className="muted small">{data?.llm_error ?? ai.error?.message}</span>
            </details>
          ) : null}
        </span>
      </p>
    )
  }

  if (data?.ai_status === 'unconfigured') {
    return <p className="muted small">The optional plain-language note is switched off (no OpenRouter key is set).</p>
  }

  if (!current || !data?.ai_summary) {
    return (
      <div className="ai-note-loading" role="status">
        <Skeleton height={14} />
        <Skeleton height={14} width="70%" />
        <p className="muted small">Writing a plain-language note. This can take a few seconds; the analysis does not wait for it.</p>
      </div>
    )
  }

  // A note that cites a figure not in the analysis, or puts a number on the
  // wrong unit, never reaches here: the server rejects it and tries the next
  // model, and if none is faithful the analysis stands alone.
  const { text, model } = data.ai_summary
  return (
    <div className="ai-note">
      <p className="ai-summary">{text}</p>
      <p className="muted small">
        Written by {model.replace(/:free$/, '')}. It restates the analysis in plain words and adds nothing to it.
      </p>
    </div>
  )
}

export default function AdvisoryPanel({ analysis, ai }) {
  const data = analysis.data
  const result = data?.analysis
  const district = data?.forecast?.district

  return (
    <Panel id="ai" eyebrow="Analysis" title="Forecast analysis" resource={analysis}>
      {result ? (
        <div className="ai">
          <p className="analysis-headline">{result.headline}</p>

          <p className="status-line">
            {result.risk_meaningful ? (
              <Chip tone={RISK_TONE[result.risk_level] ?? 'neutral'} icon={RISK_ICON[result.risk_level]}>
                {humanise(result.risk_level)} dry-week risk
              </Chip>
            ) : (
              <Chip tone="info" icon="info">Dry week is normal here</Chip>
            )}
            <Chip tone={result.confidence === 'low' ? 'moderate' : 'neutral'} icon={result.confidence === 'low' ? 'alert-triangle' : undefined}>
              {humanise(result.confidence)} confidence
            </Chip>
          </p>

          {result.confidence_reasons.length ? (
            <ul className="reasons">
              {result.confidence_reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          ) : null}

          <div className="ai-note-wrap">
            <h3 className="ai-heading">In plain words</h3>
            <AiNote ai={ai} district={district} />
          </div>

          <div className="ai-columns">
            <List title="Key factors" items={result.key_factors} />
            <List title="Where the sources differ" items={result.model_disagreement} />
            <List title="What to do" items={result.actions} ordered />
          </div>

          <p className="muted small">{data.note}</p>
        </div>
      ) : null}
    </Panel>
  )
}
