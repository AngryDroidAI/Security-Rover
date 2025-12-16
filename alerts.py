# core/alerts.py
import requests
import time

TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_alert(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("[ALERT]", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=5)
    except Exception as e:
        print("Alert error:", e)

