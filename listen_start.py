#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, requests
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
for candidate in (os.getenv("ENV_FILE"), REPO_ROOT / ".env", Path("~/xrpbot/.env").expanduser()):
    if not candidate:
        continue
    candidate_path = Path(candidate).expanduser()
    if candidate_path.exists():
        load_dotenv(candidate_path)
        break

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is required to read Telegram updates")
API   = f"https://api.telegram.org/bot{TOKEN}"

ROOT = os.path.dirname(__file__)
SUBS_FILE   = os.path.join(ROOT, "subscribers.json")
OFFSET_FILE = os.path.join(ROOT, "offset.json")

def load_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False)

def main():
    subs = load_json(SUBS_FILE, [])
    offd = load_json(OFFSET_FILE, {"offset": 0})
    offset = int(offd.get("offset", 0))

    # یک درخواست کوتاه (بدون long-poll) کافیست؛ چون اکشن‌ها هر چند دقیقه اجرا می‌شوند
    r = requests.get(f"{API}/getUpdates", params={"offset": offset, "timeout": 0}, timeout=20)
    r.raise_for_status()
    updates = r.json().get("result", [])

    changed = False
    max_update_id = offset
    for upd in updates:
        max_update_id = max(max_update_id, upd["update_id"])
        msg = upd.get("message") or upd.get("channel_post")
        if not msg: 
            continue
        text = (msg.get("text") or "").strip().lower()
        chat = msg["chat"]
        chat_id = chat["id"]

        if text in ("/start", "/subscribe", "start"):
            if chat_id not in subs:
                subs.append(chat_id)
                changed = True

    # به‌روزرسانی offset برای جلوگیری از پردازش تکراری
    if max_update_id != offset:
        offd["offset"] = max_update_id + 1
        changed = True

    if changed:
        save_json(SUBS_FILE, subs)
        save_json(OFFSET_FILE, offd)
        # چاپ برای لاگ اکشن‌ها
        print(f"Updated subscribers: {len(subs)}, offset: {offd['offset']}")

if __name__ == "__main__":
    main()
