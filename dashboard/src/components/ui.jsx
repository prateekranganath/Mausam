import Icon from './Icon.jsx'

/** A status pill. `tone` is one of low | moderate | high | info | neutral. */
export function Chip({ tone = 'neutral', icon, children, title }) {
  return (
    <span className={`chip chip-${tone}`} title={title}>
      {icon ? <Icon name={icon} size={14} /> : null}
      {children}
    </span>
  )
}

export function Skeleton({ height = 16, width = '100%', className = '' }) {
  return <span className={`skeleton ${className}`} style={{ height, width }} aria-hidden="true" />
}

/**
 * The error a single panel shows for itself. One failing endpoint must not
 * blank the page - the original put every fetch behind one `error` string and
 * one `forecast &&` gate, so a stalled LLM call hid the numeric dashboard that
 * never needed it.
 */
export function ErrorState({ error, onRetry, compact = false }) {
  return (
    <div className={`state state-error ${compact ? 'compact' : ''}`} role="alert">
      <Icon name="alert-triangle" size={20} />
      <div>
        <p className="state-title">{compact ? 'Unavailable' : 'This panel could not load'}</p>
        <p className="state-body">{error?.message ?? 'Something went wrong.'}</p>
        {onRetry ? (
          <button type="button" className="btn btn-quiet" onClick={onRetry}>
            <Icon name="refresh" size={14} /> Try again
          </button>
        ) : null}
      </div>
    </div>
  )
}

export function EmptyState({ icon = 'info', title, children }) {
  return (
    <div className="state">
      <Icon name={icon} size={20} />
      <div>
        <p className="state-title">{title}</p>
        {children ? <p className="state-body">{children}</p> : null}
      </div>
    </div>
  )
}

/**
 * Card chrome shared by every panel.
 *
 * `resource` is the object from useResource. While it is `refreshing` the body
 * holds its previous render at reduced opacity (no skeleton, no layout jump);
 * `aria-busy` tells assistive tech a refresh is in flight.
 */
export function Panel({
  id,
  title,
  eyebrow,
  actions,
  resource,
  skeleton,
  className = '',
  children,
}) {
  const status = resource?.status
  const showSkeleton = status === 'loading' || status === 'idle'
  const showError = status === 'error'

  return (
    <section
      id={id}
      className={`panel ${className}`}
      aria-labelledby={id ? `${id}-title` : undefined}
      aria-busy={status === 'loading' || status === 'refreshing'}
      data-stale={status === 'refreshing'}
    >
      <header className="panel-header">
        <div>
          {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
          <h2 id={id ? `${id}-title` : undefined} className="panel-title">
            {title}
          </h2>
        </div>
        {actions ? <div className="panel-actions">{actions}</div> : null}
      </header>

      <div className="panel-body">
        {showError ? (
          <ErrorState error={resource.error} onRetry={resource.reload} />
        ) : showSkeleton ? (
          (skeleton ?? <DefaultSkeleton />)
        ) : (
          children
        )}
      </div>
    </section>
  )
}

function DefaultSkeleton() {
  return (
    <div className="skeleton-stack" aria-label="Loading">
      <Skeleton height={20} width="40%" />
      <Skeleton height={120} />
      <Skeleton height={16} width="70%" />
    </div>
  )
}

/** A label + value tile. Values use proportional figures, not tabular. */
export function Stat({ label, value, unit, note, tone, children }) {
  return (
    <article className={`stat ${tone ? `stat-${tone}` : ''}`}>
      <p className="stat-label">{label}</p>
      <p className="stat-value">
        {value}
        {unit ? <span className="stat-unit">{unit}</span> : null}
      </p>
      {note ? <p className="stat-note">{note}</p> : null}
      {children}
    </article>
  )
}

export function ViewToggle({ view, onChange }) {
  return (
    <div className="view-toggle" role="group" aria-label="Chart or table view">
      <button
        type="button"
        aria-pressed={view === 'chart'}
        className={view === 'chart' ? 'active' : ''}
        onClick={() => onChange('chart')}
      >
        <Icon name="bar-chart" size={14} /> Chart
      </button>
      <button
        type="button"
        aria-pressed={view === 'table'}
        className={view === 'table' ? 'active' : ''}
        onClick={() => onChange('table')}
      >
        <Icon name="table" size={14} /> Table
      </button>
    </div>
  )
}
