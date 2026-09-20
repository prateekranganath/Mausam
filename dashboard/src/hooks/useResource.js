import { useCallback, useEffect, useState } from 'react'

/**
 * Fetch one resource, cancelling it if its inputs change or the component goes
 * away.
 *
 *   status   'idle'       not enabled yet
 *            'loading'    first fetch, nothing to show
 *            'refreshing' refetching; `data` is still the PREVIOUS response
 *            'ready'      `data` is current
 *            'error'      `error` is set, `data` is null
 *
 * `refreshing` exists so a panel can hold its previous render at reduced
 * opacity instead of flashing a skeleton and jumping the layout. Every panel
 * renders the district named INSIDE its own data, so a faded panel is always
 * internally consistent rather than mixing old numbers with a new title.
 *
 * `delay` debounces the request. It matters for the LLM advisory, where a
 * user clicking through ten districts would otherwise queue ten 90-second
 * calls on a server that keeps running them after the browser aborts.
 */
export function useResource(fetcher, deps, { enabled = true, delay = 0 } = {}) {
  const [state, setState] = useState({ data: null, error: null, status: 'idle' })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    if (!enabled) return undefined

    const controller = new AbortController()

    const run = async () => {
      setState((prev) => ({
        data: prev.data,
        error: null,
        status: prev.data ? 'refreshing' : 'loading',
      }))
      try {
        const data = await fetcher(controller.signal)
        if (!controller.signal.aborted) setState({ data, error: null, status: 'ready' })
      } catch (error) {
        if (error?.name === 'AbortError' || controller.signal.aborted) return
        // Drop the data on failure. Showing the previous district's numbers
        // under an error banner for the new one would be actively misleading.
        setState({ data: null, error, status: 'error' })
      }
    }

    const timer = setTimeout(run, delay)
    return () => {
      clearTimeout(timer)
      controller.abort()
    }
    // `fetcher` is rebuilt every render; `deps` says when it actually matters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, attempt])

  const reload = useCallback(() => setAttempt((n) => n + 1), [])

  return { ...state, reload }
}
