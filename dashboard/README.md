# Mausam dashboard

A React + Vite front end for the Mausam API: 7-day rainfall outlook, monsoon
onset, active/break spells, crop advice, climate context, an AI summary and a
Telegram alert preview.

## Run it

The dashboard only displays what the API returns, so start the backend first.

```bash
# 1. from the project root
uvicorn src.api.main:app --reload            # http://127.0.0.1:8000

# 2. in another terminal
cd dashboard
npm install
npm run dev                                   # http://localhost:5173
```

Open **`http://localhost:5173`**, not `http://127.0.0.1:5173`. The browser
sends the page's origin as-is, and the API's CORS allowlist
(`API_ALLOWED_ORIGINS` in the root `.env`) lists `localhost`; `127.0.0.1` is a
different origin and is rejected. Add it there if you need it.

To point at a deployed backend, create `dashboard/.env.local`:

```
VITE_API_BASE=https://your-api.example.com
```

The selected district lives in the URL (`?district=Nagpur`), so a view can be
bookmarked or shared.

```bash
npm run lint      # oxlint
npm run build     # production bundle in dist/
```

If the API is down the page says so once, with the command to start it, rather
than showing nine separate errors.

## What talks to what

| Panel | Endpoint | Notes |
|---|---|---|
| Hero, KPI tiles | `GET /forecast/{district}` | One call feeds the hero, tiles and forecast charts |
| Model skill tile | `GET /model/metrics` | Fails independently of the rest |
| Next 7 days | `GET /forecast/{district}` | Not `/forecast/{district}/series`: that endpoint re-runs the whole forecast to return the same seven rows |
| Has the monsoon arrived? | `GET /monsoon/onset/{district}` | |
| Active or break spell? | `GET /monsoon/phase/{district}` | |
| Crop advice | `GET /crops`, `GET /advisory/crop/{district}` | Rule-based, no LLM |
| Climate drivers | `GET /climate/context` | Context only; not a model input |
| Recent rainfall | `GET /historical/{district}?days=90` | |
| AI summary | `GET /advisory/{district}` | Slow (free-tier LLM); loads last and never blocks anything |
| Telegram alert | `GET /alerts/telegram/preview`, `POST /alerts/telegram/send` | See below |

Every panel is its own resource (`src/hooks/useResource.js`), so a slow or
failing endpoint degrades one panel instead of blanking the page. While a panel
refetches it keeps its previous render at reduced opacity rather than flashing a
skeleton. Switching districts cancels in-flight requests (`AbortController`), so
the screen always ends on the last district you asked for.

## Telegram alerts

The panel shows the **exact message the server would send**, in a
Telegram-styled bubble, then a **Send** button.

- The preview works with **no Telegram credentials at all**, so the feature can
  be demonstrated on a fresh clone. Only Send needs `TELEGRAM_BOT_TOKEN` and
  `TELEGRAM_CHAT_ID` in the root `.env`; without them the button is disabled and
  says why.
- The browser sends a district and a crop, **never message text**. The server
  composes what is delivered, and the recipient is fixed server-side. The
  endpoint has no authentication, so one that accepted arbitrary text or a chat
  id would be an open relay for messaging anyone through the bot. Sends are also
  throttled.

## Design

Dark glassmorphic console, kept from the original. `src/theme.css` holds every
token; nothing else contains a colour literal.

- **Data colours are validated, not picked.** Rainfall is blue `#3987e5` and
  break spells are orange `#d95926`. Both were run through the dataviz palette
  validator against the effective card surface (`#0c1426`): colour-blind
  separation ΔE 26.8 (target ≥ 8), all marks ≥ 3:1 contrast. The original
  sky/violet pair failed that check (deuteranopia ΔE 5.2), so it now only
  colours UI chrome, never data.
- **Colour follows the entity.** Rainfall is the same hue everywhere. Green,
  amber and red are reserved for status and always come with an icon and a word.
- **Chart marks:** bars ≤ 24 px with a 4 px rounded data end, a 2 px surface gap
  between neighbours, solid hairline grid, values labelled selectively (the
  peak, not every point). Mixed units get separate charts, never a dual axis.
- **Every chart has a table twin** (the Chart / Table toggle), and tooltips
  never gate a value.

## Honest by construction

Things the dashboard deliberately does *not* do:

- **It does not invent numbers.** The first version computed "Break spell risk",
  "Onset confidence" and "Heavy rain watch" in the browser from formulas that
  saturated (the break figure read 98% for almost any forecast). Those are gone;
  the panels show what the backend measured.
- **It does not print a green "Low risk" where that label means nothing.** In
  many district-months normal rainfall is already near 0 mm, so the API flags the
  threshold as degenerate. The hero then says "A dry week is normal here" and
  the dry-risk tile shows `n/a`.
- **It flags when the AI summary contradicts the model's risk level**, since the
  LLM restates the forecast but is not forced to agree with it.
- **Onset is labelled as local rainfall onset, not an IMD declaration**, and the
  climate indices show how old each value is.

## Layout

```
src/
  theme.css            design tokens
  index.css, App.css   base + component styles
  api/client.js        the only place that calls the API
  hooks/useResource.js fetch + cancel + keep-previous-render
  lib/format.js        formatting and describeRisk()
  components/          one file per panel, plus charts.jsx and ui.jsx
```
