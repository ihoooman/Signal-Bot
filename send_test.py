import os, requests
from pathlib import Path
from dotenv import load_dotenv

repo_root = Path(__file__).resolve().parent
for candidate in (os.getenv("ENV_FILE"), repo_root / ".env", Path("~/xrpbot/.env").expanduser()):
    if not candidate:
        continue
    candidate_path = Path(candidate).expanduser()
    if candidate_path.exists():
        load_dotenv(candidate_path)
        break
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise RuntimeError("Both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set for send_test.py")

msg = "سلام هومن! ✅ تست ارسال پیام از ربات انجام شد."
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
r = requests.post(url, json={"chat_id": CHAT_ID, "text": msg}, timeout=20)
print("Status:", r.status_code, "| Response:", r.text)
