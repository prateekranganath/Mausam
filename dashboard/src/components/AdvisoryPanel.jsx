import Icon from './Icon.jsx'
import { Chip, Panel, Skeleton } from './ui.jsx'
import { humanise } from '../lib/format.js'

/*
 * The LLM-written advisory.
 *
 * The original rendered ONE sentence of it, `advisory?.advisory?.advisory?.[0]`,
 * and discarded the summary, key factors, model disagreement, the remaining
 * actions, and the unsupported-numbers warning - from the slowest, most
 * rate-limited call the page makes.
 *
 * It is also the only panel that can be slow (a free-tier model, up to ~90 s),
 * so it owns its own loading state and never blocks the numbers.
 */

const LEVEL_TONE = { low: 'low', moderate: 'moderate', medium: 'moderate', high: 'high' }

const canon = (level) => {
  const value = String(level ?? '').toLowerCase()
  return value === 'medium' ? 'moderate' : value
}

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

function WritingSkeleton() {
  return (
    <div className="skeleton-stack" role="status">
      <p className="muted small">Writing a plain-language summary. On the free model this can take up to a minute; the numbers above do not depend on it.</p>
      <Skeleton height={14} />
      <Skeleton height={14} width="92%" />
      <Skeleton height={14} width="60%" />
    </div>
  )
}

export default function AdvisoryPanel({ resource }) {
  const data = resource.data
  const advisory = data?.advisory
  const unsupported = data?.unsupported_numbers ?? []

  // The LLM restates the forecast, but nothing forces it to agree with the
  // model's own risk level - and it does not always. When they differ, say so
  // rather than leaving two contradictory verdicts side by side on one screen.
  const modelLevel = data?.forecast?.ml_model?.risk_level
  const degenerate = data?.forecast?.agreement?.threshold_degenerate
  const disagrees =
    advisory && modelLevel && !degenerate && canon(advisory.rainfall_risk) !== canon(modelLevel)

  return (
    <Panel
      id="ai"
      eyebrow="AI summary"
      title="Plain-language advisory"
      resource={resource}
      skeleton={<WritingSkeleton />}
    >
      {data && !advisory ? (
        <p className="callout callout-info">
          <Icon name="info" size={16} />
          <span>
            {data.llm_error ?? 'No AI summary is available.'} Every number on this page comes from the forecast models and is
            unaffected.
          </span>
        </p>
      ) : null}

      {advisory ? (
        <div className="ai">
          <p className="ai-summary">{advisory.forecast_summary}</p>

          <p className="status-line">
            <Chip tone={LEVEL_TONE[String(advisory.rainfall_risk).toLowerCase()] ?? 'neutral'}>
              Rainfall risk: {humanise(advisory.rainfall_risk)}
            </Chip>
            <Chip tone="neutral">Confidence: {humanise(advisory.confidence)}</Chip>
          </p>

          {disagrees ? (
            <p className="callout callout-warn">
              <Icon name="alert-triangle" size={16} />
              <span>
                This summary rates the risk <strong>{humanise(advisory.rainfall_risk)}</strong>, but the forecast model
                says <strong>{humanise(modelLevel)}</strong>. The model’s figure, shown at the top of the page, is the
                one to rely on.
              </span>
            </p>
          ) : null}

          <div className="ai-columns">
            <List title="Key factors" items={advisory.key_factors} />
            <List title="Where the sources differ" items={advisory.model_disagreement} />
            <List title="What to do" items={advisory.advisory} ordered />
          </div>

          {unsupported.length ? (
            <p className="callout callout-warn" role="alert">
              <Icon name="alert-triangle" size={16} />
              <span>
                The summary mentions figures that could not be traced back to the forecast data ({unsupported.join(', ')}).
                Check them against the numbers on this page before relying on them.
              </span>
            </p>
          ) : null}

          <p className="muted small">
            Written by {data.llm_model}. It restates the forecast; it does not add to it.
          </p>
        </div>
      ) : null}
    </Panel>
  )
}
