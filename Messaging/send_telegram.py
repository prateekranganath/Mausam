import os
import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_monsoon_alert(risk_level: str, district: str, rainfall_mm, chat_id: str = None):
    """Send a monsoon risk alert via Telegram bot.

    Args:
        risk_level: e.g. 'HIGH', 'MODERATE', 'LOW'
        district: district/block name
        rainfall_mm: predicted rainfall in mm (number or string)
        chat_id: Telegram chat ID to send to. Defaults to TELEGRAM_CHAT_ID from .env.
    """
    message = (
        f"Monsoon Alert: {risk_level} risk of insufficient rainfall in {district} "
        f"over the next 7 days. Expected rainfall: {rainfall_mm}mm. "
        f"Please plan sowing/irrigation accordingly."
    )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id or CHAT_ID,
        "text": message,
    }

    response = requests.post(url, data=payload)
    result = response.json()

    if result.get("ok"):
        print(f"Message sent successfully. Message ID: {result['result']['message_id']}")
    else:
        print(f"Failed to send message: {result}")

    return result


if __name__ == "__main__":
    send_monsoon_alert(risk_level="HIGH", district="Thiruvananthapuram", rainfall_mm=12)