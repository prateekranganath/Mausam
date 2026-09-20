"""Compose the alert text from real endpoint data.

A pure function over the dicts the API already returns, deliberately kept
free of any I/O so the exact message can be unit tested without a bot
token, a network call, or a configured chat. That matters more than usual
here: this is the only part of the system whose output a farmer reads
verbatim, and it is the part hardest to check by eye once it is inside a
send path.

PLAIN TEXT, NOT MARKDOWN. Telegram's MarkdownV2 requires escaping
`_ * [ ] ( ) ~ \\` > # + - = | { } . !` and four served district names
contain parentheses ("Raipur (CT)", "Raipur (MP)", "Cuddalore (TN)",
"Cuddalore (PY)"). A missed escape is a 400 from Telegram, so the whole
class of bug is removed by sending no markup at all.

THE HONESTY RULES THIS MODULE ENFORCES are the same ones the rest of the
project already applies, carried through to the one surface a user reads
directly:

  - A degenerate risk threshold means the risk label carries no
    information, so the risk line is REPLACED rather than printed. The
    crop engine does exactly this in _suppress_degenerate_dry_risk.
  - `onset_likely` is never worded as a settled onset. The distinction
    between "rain arrived" and "rain arrived and stayed" is the whole
    point of having two states.
  - The not-IMD caveat is unconditional.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

# Telegram rejects a sendMessage body over 4096 characters. Our messages
# run ~400-700, so this is a guard against a pathological advisory rather
# than an expected path -- but a 400 from Telegram at demo time would be a
# silly way to fail.
MAX_TELEGRAM_CHARS = 4096

DISCLAIMER = (
    "Estimate from a statistical model, not an official IMD forecast. "
    "Check with your local KVK before acting."
)

_RISK_WORDING = {
    "HIGH": "HIGH - an unusually dry week is likely",
    "MODERATE": "MODERATE - an unusually dry week is possible",
    "LOW": "LOW - no unusually dry week expected",
}

_PHASE_WORDING = {
    "active": "Active spell - rainfall well above normal",
    "break": "BREAK SPELL - rainfall well below normal",
    "normal": "Normal - rainfall within its usual range",
}

_SEVERITY_MARK = {"high": "!", "medium": "-", "info": "-"}


def _format_date(value: str | None) -> str:
    """ISO date to something readable in a chat. Falls back to the raw
    string rather than raising: a malformed date must not lose the alert."""
    if not value:
        return ""
    try:
        return dt.date.fromisoformat(str(value)[:10]).strftime("%d %b %Y")
    except ValueError:
        return str(value)


def _risk_block(forecast: dict[str, Any]) -> list[str]:
    model = forecast.get("ml_model") or {}
    agreement = forecast.get("agreement") or {}
    open_meteo = forecast.get("open_meteo_forecast") or {}

    lines: list[str] = []

    if agreement.get("threshold_degenerate"):
        # The label cannot fire here, so printing "LOW risk" would be
        # technically true and completely uninformative.
        lines.append(
            "RISK: not meaningful here - a dry week is normal for this "
            "district at this time of year, so the dry-risk score is not reported."
        )
    else:
        level = str(model.get("risk_level", "")).upper()
        probability = model.get("rainfall_probability")
        wording = _RISK_WORDING.get(level, level or "unknown")
        if probability is not None:
            wording += f" ({round(float(probability) * 100)}% chance)"
        lines.append(f"RISK: {wording}")

    predicted = model.get("predicted_rainfall_mm")
    if predicted is not None:
        lines.append(f"Expected rain, next 7 days: {round(float(predicted))} mm")

    om_total = open_meteo.get("total_precipitation_sum_mm")
    if om_total is not None:
        note = ""
        if agreement.get("sources_agree") is True:
            note = " (agrees with our estimate)"
        elif agreement.get("magnitude_diverges") is True:
            note = " (differs from our estimate - treat both as uncertain)"
        lines.append(f"Open-Meteo forecast: {round(float(om_total))} mm{note}")

    return lines


def _onset_line(onset: dict[str, Any]) -> str | None:
    status = onset.get("status")
    date = _format_date(onset.get("onset_date"))
    anomaly = onset.get("anomaly_label")

    if status in {"onset_confirmed", "post_onset"} and date:
        line = f"Monsoon: arrived {date}"
        if anomaly:
            line += f", {anomaly}"
        return line
    if status == "onset_likely":
        # Never let this read as settled. The persistence window has not
        # closed, so this could still turn out to be a false onset.
        return (
            f"Monsoon: rain has started{f' around {date}' if date else ''}, but it is "
            "NOT yet confirmed to have settled in - it could still break."
        )
    if status == "pre_onset":
        return "Monsoon: has not arrived here yet."
    if status == "no_onset_detected":
        return "Monsoon: no onset detected in this season's record."
    return None


def _phase_line(phase: dict[str, Any]) -> str | None:
    name = phase.get("monsoon_phase")
    if name in (None, "not_applicable"):
        return None
    wording = _PHASE_WORDING.get(name, str(name))
    days = phase.get("days_in_current_phase")
    return f"Current phase: {wording}" + (f", {days} days so far" if days else "")


def _crop_block(crop: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    name = crop.get("crop")
    stage = crop.get("growth_stage")
    sensitivity = crop.get("stage_drought_sensitivity")

    heading = name or "Crop"
    if stage:
        heading += f", {str(stage).replace('_', ' ')}"
        if sensitivity in {"critical", "high"}:
            heading += f" ({sensitivity} stage for water)"
    lines.append(f"{heading}:")

    balance = crop.get("water_balance_mm")
    need = crop.get("stage_weekly_water_requirement_mm")
    has_balance_line = balance is not None and need is not None
    if has_balance_line:
        if balance < 0:
            lines.append(f"- Needs ~{round(float(need))} mm this week; short by {abs(round(float(balance)))} mm")
        else:
            lines.append(f"- Needs ~{round(float(need))} mm this week; forecast covers it")

    # A "status" recommendation only restates that nothing needs doing,
    # which the water-balance line above has already said. In a chat
    # message that reads as padding, so it is dropped when the balance
    # line is present and kept when it is the only thing to report.
    recommendations = [
        rec for rec in (crop.get("recommendations") or [])
        if not (has_balance_line and rec.get("category") == "status")
    ]
    for rec in recommendations[:3]:
        mark = _SEVERITY_MARK.get(rec.get("severity", "info"), "-")
        lines.append(f"{mark} {rec.get('action', '')}")

    if crop.get("suppression_note"):
        lines.append("- Some rainfall-risk advice was withheld as not meaningful here.")

    return lines


def compose_alert(
    forecast: dict[str, Any],
    onset: dict[str, Any] | None = None,
    phase: dict[str, Any] | None = None,
    crop: dict[str, Any] | None = None,
) -> str:
    """The alert text for one district.

    Only `forecast` is required. Every other section is included when the
    data is available and silently omitted when it is not, so a partial
    outage degrades the message rather than failing the alert.
    """
    district = forecast.get("district", "this district")
    state = forecast.get("state")
    header = f"Mausam alert - {district}" + (f", {state}" if state else "")

    blocks: list[list[str]] = [[header, _format_date(forecast.get("as_of_date"))]]
    blocks.append(_risk_block(forecast))

    monsoon = [line for line in (
        _onset_line(onset) if onset else None,
        _phase_line(phase) if phase else None,
    ) if line]
    if monsoon:
        blocks.append(monsoon)

    if crop:
        blocks.append(_crop_block(crop))

    blocks.append([DISCLAIMER])

    text = "\n\n".join("\n".join(line for line in block if line) for block in blocks)
    if len(text) > MAX_TELEGRAM_CHARS:
        # Keep the disclaimer: it is the part that must never be dropped.
        keep = MAX_TELEGRAM_CHARS - len(DISCLAIMER) - 6
        text = text[:keep].rstrip() + "\n...\n\n" + DISCLAIMER
    return text
