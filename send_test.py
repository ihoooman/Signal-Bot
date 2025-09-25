import os, requests
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/xrpbot/.env"))
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

msg = "سلام هومن! ✅ تست ارسال پیام از ربات انجام شد."
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
r = requests.post(url, json={"chat_id": CHAT_ID, "text": msg}, timeout=20)
print("Status:", r.status_code, "| Response:", r.text)
