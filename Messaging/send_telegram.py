"""CLI entry point for sending a Mausam alert over Telegram.

The sending logic now lives in src/alerts/telegram.py and the message
wording in src/alerts/compose.py, so the API can reuse both and both can
be unit tested without a bot token. This file stays at its original path
and keeps its original `send_monsoon_alert(...)` signature so anything
already calling it continues to work.

Usage:
    # a real alert for a district, composed from live forecast data
    python Messaging/send_telegram.py --district Nagpur
    python Messaging/send_telegram.py --district Nagpur --crop cotton --sowing-date 2026-06-20

    # show the message without sending it (needs no bot token)
    python Messaging/send_telegram.py --district Nagpur --preview
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alerts import telegram
from src.alerts.compose import compose_alert


def send_monsoon_alert(risk_level: str, district: str, rainfall_mm, chat_id: str = None) -> dict:
    """Send a simple monsoon risk alert.

    Kept for backward compatibility with the original script. Prefer
    `python Messaging/send_telegram.py --district <name>`, which composes
    the message from real forecast, onset, phase and crop data instead of
    values passed in by hand.
    """
    message = (
        f"Mausam alert: {risk_level} risk of insufficient rainfall in {district} "
        f"over the next 7 days. Expected rainfall: {rainfall_mm}mm. "
        f"Please plan sowing/irrigation accordingly."
    )
    result = telegram.send_message(message, chat_id=chat_id)
    print(f"Message sent. Message ID: {result.message_id}" if result.sent
          else f"Failed to send message: {result.error}")
    return result.to_dict()


def _live_alert(district: str, crop: str | None, sowing_date: str | None) -> str:
    """Compose from real data, using the same code path the API uses."""
    from src.api.state import ServiceState, servable_districts
    from src.advisory import engine as crop_engine
    from src.data.history import district_history, recent_soil_and_dryness
    from src.forecasting.district_registry import get_district_config
    from src.ml.predict import RainfallRiskPredictor
    from src.monsoon import active_break, onset as onset_module

    import pandas as pd

    from src.forecasting.agreement import compute_agreement, lookup_threshold

    cfg = get_district_config(district)
    predictor = RainfallRiskPredictor.load_local_all_india()
    raw = predictor.predict_live(cfg.district)

    # Compute agreement here too, exactly as the API does. Passing an empty
    # dict would silently disable the composer's degenerate-threshold rule,
    # so this path would print a meaningless "LOW risk" in districts where
    # the API correctly refuses to.
    month = pd.Timestamp(raw["as_of_date"]).month
    threshold = lookup_threshold(predictor.preprocessor.risk_threshold_table_, cfg.district, month)
    agreement = compute_agreement(
        raw["local_model"]["predicted_rainfall_mm"],
        raw["open_meteo_forecast"]["total_precipitation_sum_mm"],
        threshold,
    )

    forecast = {
        "district": cfg.district,
        "state": cfg.state,
        "as_of_date": raw["as_of_date"],
        "ml_model": raw["local_model"],
        "open_meteo_forecast": raw["open_meteo_forecast"],
        "agreement": agreement,
    }

    onset_result = phase_result = crop_result = None
    try:
        history = district_history(cfg)
        if not history.empty:
            onset_result = onset_module.onset_status(history, cfg.district)
            phase_result = active_break.current_phase(history, cfg.district)
            if crop:
                crop_result = crop_engine.advise(
                    crop, sowing_date=sowing_date, forecast=forecast,
                    onset=onset_result, phase=phase_result,
                    soil=recent_soil_and_dryness(history),
                )
    except Exception as exc:  # noqa: BLE001 - context is optional
        print(f"(monsoon context unavailable: {exc})", file=sys.stderr)

    return compose_alert(forecast, onset=onset_result, phase=phase_result, crop=crop_result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--district", default=None, help="District to compose a real alert for")
    parser.add_argument("--crop", default=None, help="Optional crop key (see GET /crops)")
    parser.add_argument("--sowing-date", default=None, help="Optional YYYY-MM-DD")
    parser.add_argument("--preview", action="store_true", help="Print the message without sending (needs no token)")
    args = parser.parse_args()

    if not args.district:
        # Original demo behaviour, so the bare script still does something.
        send_monsoon_alert(risk_level="HIGH", district="Thiruvananthapuram", rainfall_mm=12)
        return

    message = _live_alert(args.district, args.crop, args.sowing_date)
    print("-" * 60)
    print(message)
    print("-" * 60)

    if args.preview:
        return
    if not telegram.is_configured():
        raise SystemExit(telegram.configuration_hint())

    result = telegram.send_message(message)
    print(f"Sent. Telegram message id: {result.message_id}" if result.sent
          else f"Not sent: {result.error}")


if __name__ == "__main__":
    main()
