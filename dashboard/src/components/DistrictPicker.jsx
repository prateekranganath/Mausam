import { useEffect, useId, useMemo, useRef, useState } from 'react'
import Icon from './Icon.jsx'

const MAX_RESULTS = 40

/**
 * A searchable district combobox.
 *
 * Replaces a bare <select> holding all 313 districts with no way to search,
 * whose `appearance: none` had also removed the dropdown affordance. Filtering
 * is done locally against the list already loaded, so it is instant; there is
 * no need for a round trip to GET /districts?q=.
 *
 * Follows the ARIA combobox pattern: the input owns focus, arrow keys move an
 * active option announced via aria-activedescendant, Enter selects, Escape
 * closes.
 */
export default function DistrictPicker({ districts, value, onChange, disabled }) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const rootRef = useRef(null)
  const inputRef = useRef(null)
  const inputId = useId()
  const listId = useId()

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return districts.slice(0, MAX_RESULTS)
    const scored = []
    for (const district of districts) {
      const name = district.name.toLowerCase()
      const state = district.state.toLowerCase()
      if (name.startsWith(q)) scored.push([0, district])
      else if (name.includes(q)) scored.push([1, district])
      else if (state === q) scored.push([2, district])
    }
    scored.sort((a, b) => a[0] - b[0] || a[1].name.localeCompare(b[1].name))
    return scored.slice(0, MAX_RESULTS).map(([, district]) => district)
  }, [districts, query])

  useEffect(() => {
    if (!open) return undefined
    const close = (event) => {
      if (!rootRef.current?.contains(event.target)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])

  useEffect(() => {
    if (!open) return
    document.getElementById(`${listId}-${active}`)?.scrollIntoView({ block: 'nearest' })
  }, [active, open, listId])

  const choose = (district) => {
    if (!district) return
    onChange(district.name)
    setOpen(false)
    setQuery('')
    inputRef.current?.blur()
  }

  const onKeyDown = (event) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setOpen(true)
      setActive((i) => Math.min(i + 1, matches.length - 1))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setActive((i) => Math.max(i - 1, 0))
    } else if (event.key === 'Enter' && open) {
      event.preventDefault()
      choose(matches[active])
    } else if (event.key === 'Escape') {
      setOpen(false)
      setQuery('')
    }
  }

  return (
    <div className="combo" ref={rootRef}>
      <label className="field-label" htmlFor={inputId}>
        District
      </label>
      <div className="combo-input">
        <Icon name="search" size={16} />
        <input
          id={inputId}
          ref={inputRef}
          type="text"
          role="combobox"
          autoComplete="off"
          spellCheck="false"
          aria-expanded={open}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={open && matches[active] ? `${listId}-${active}` : undefined}
          disabled={disabled}
          placeholder={disabled ? 'Loading districts…' : 'Search 313 districts'}
          value={open ? query : value}
          onFocus={() => {
            setOpen(true)
            setActive(0)
            setQuery('')
          }}
          onChange={(event) => {
            setQuery(event.target.value)
            setOpen(true)
            setActive(0)
          }}
          onKeyDown={onKeyDown}
        />
        <Icon name="chevron-down" size={16} className="combo-chevron" />
      </div>

      {open ? (
        <ul className="combo-list" id={listId} role="listbox" aria-label="Districts">
          {matches.length === 0 ? (
            <li className="combo-empty" role="presentation">
              No district matches “{query}”
            </li>
          ) : (
            matches.map((district, index) => (
              <li
                key={district.name}
                id={`${listId}-${index}`}
                role="option"
                aria-selected={district.name === value}
                className={`combo-option ${index === active ? 'is-active' : ''}`}
                onMouseEnter={() => setActive(index)}
                // mousedown, not click: the input's blur would otherwise close
                // the list before a click could land.
                onMouseDown={(event) => {
                  event.preventDefault()
                  choose(district)
                }}
              >
                <span>{district.name}</span>
                <span className="combo-state">{district.state}</span>
              </li>
            ))
          )}
        </ul>
      ) : null}
    </div>
  )
}
