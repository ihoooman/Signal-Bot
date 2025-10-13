#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import inspect
import logging
import os, json, math, re, requests, sys, time, uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import TYPE_CHECKING, Callable
from dotenv import load_dotenv
try:
    from telegram import Bot as TelegramBot, KeyboardButton, ReplyKeyboardMarkup
    from telegram.error import RetryAfter, TimedOut
    from telegram.ext import CommandHandler
except ModuleNotFoundError:  # pragma: no cover - fallback when dependency missing
    TelegramBot = None
    KeyboardButton = None
    ReplyKeyboardMarkup = None

    class CommandHandler:  # type: ignore[empty-body]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("python-telegram-bot is required for CommandHandler")

    class RetryAfter(Exception):
        def __init__(self, retry_after: float = 1.0):
            super().__init__("Rate limited")
            self.retry_after = retry_after

    class TimedOut(Exception):
        pass

SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

if TYPE_CHECKING:  # pragma: no cover - typing only
    from telegram import Update
    from telegram.ext import ContextTypes

import migrate_db

from signal_bot.services.symbol_resolver import ResolutionCandidate, ResolutionResult, resolve_instrument

from subscriptions import (
    add_to_watchlist,
    count_subscribers,
    describe_backend,
    donation_totals,
    get_subscriber,
    get_user_watchlist,
    list_recent_donations,
    load_latest_summary,
    remove_from_watchlist,
    save_donation,
    save_summary,
    upsert_subscriber,
)
from trigger_xrp_bot import render_compact_summary, run_summary_once, send_telegram

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
LOGGER = logging.getLogger("signal_bot.listen_start")


REPO_ROOT = Path(__file__).resolve().parent
for candidate in (os.getenv("ENV_FILE"), REPO_ROOT / ".env", Path("~/xrpbot/.env").expanduser()):
    if not candidate:
        continue
    candidate_path = Path(candidate).expanduser()
    if candidate_path.exists():
        load_dotenv(candidate_path)
        break

BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is required to read Telegram updates")
API   = f"https://api.telegram.org/bot{BOT_TOKEN}"

BROADCAST_SLEEP_MS = int(os.getenv("BROADCAST_SLEEP_MS", "0"))


class _RequestsUpdate:
    __slots__ = ("_data",)

    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return self._data


class _RequestsBot:
    def __init__(self, token: str):
        self._token = token

    def get_updates(self, offset: int, timeout: int, allowed_updates=None):
        params = {"offset": offset, "timeout": timeout}
        if allowed_updates:
            params["allowed_updates"] = list(allowed_updates)
        try:
            response = requests.get(f"https://api.telegram.org/bot{self._token}/getUpdates", params=params, timeout=20 + timeout)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network fallback
            print(f"Failed to fetch updates via requests: {exc}")
            return []
        payload = response.json().get("result", [])
        return [_RequestsUpdate(item) for item in payload]


if TelegramBot is not None:
    BOT = TelegramBot(BOT_TOKEN)
else:  # pragma: no cover - fallback when dependency missing
    BOT = _RequestsBot(BOT_TOKEN)

ROOT = Path(__file__).resolve().parent
SUBS_FILE = Path(os.getenv("SUBSCRIBERS_DB_PATH") or os.getenv("SUBSCRIBERS_PATH", str(ROOT / "subscribers.sqlite3"))
    ).expanduser()
OFFSET_FILE = Path(os.getenv("OFFSET_FILE", str(ROOT / "data" / "offset.txt"))).expanduser()
LEGACY_OFFSET_FILE = Path(os.getenv("LEGACY_OFFSET_FILE", str(ROOT / "offset.json"))).expanduser()
STATE_FILE = Path(os.getenv("CONVERSATION_STATE_PATH", str(ROOT / "conversation_state.json"))).expanduser()
SNAPSHOT_PATH = Path(os.getenv("SNAPSHOT_PATH", str(ROOT / "data" / "last_summary.json"))).expanduser()
EXCHANGE_PAIRS_PATH = Path(
    os.getenv("EXCHANGE_PAIRS_PATH", str(ROOT / "data" / "exchange_pairs.json"))
).expanduser()
SNAPSHOT_MAX_AGE = timedelta(hours=2)

SCHEMA_LOGGER = logging.getLogger("signal_bot.schema")
migrate_db.verify_or_create(path=SUBS_FILE, logger=SCHEMA_LOGGER)

ALLOWED_UPDATES = ("message", "callback_query", "pre_checkout_query")


def _run_bot_get_updates(offset: int, timeout_value: int):
    async def _invoke():
        result = BOT.get_updates(
            offset=offset,
            timeout=timeout_value,
            allowed_updates=ALLOWED_UPDATES,
        )
        if inspect.isawaitable(result):
            return await result
        awaitable = getattr(result, "__await__", None)
        if callable(awaitable):
            return await result  # type: ignore[func-returns-value]
        return result

    try:
        return asyncio.run(_invoke())
    except RuntimeError as exc:
        if "asyncio.run()" in str(exc):
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(_invoke())
            finally:
                asyncio.set_event_loop(None)
                loop.close()
        raise


def _parse_donation_tiers(raw: str | None) -> list[int]:
    tiers: list[int] = []
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError:
            continue
        if value > 0:
            tiers.append(value)
    return tiers


def _parse_admin_ids(raw: str | None) -> set[int]:
    ids: set[int] = set()
    for chunk in (raw or "").replace(",", " ").split():
        try:
            ids.add(int(chunk))
        except (TypeError, ValueError):
            continue
    return ids


def _normalise_pair_token(text: str | None) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", (text or "")).upper()


@lru_cache(maxsize=1)
def _exchange_pairs_data() -> list[dict[str, object]]:
    try:
        payload = json.loads(EXCHANGE_PAIRS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception as exc:  # pragma: no cover - malformed data
        LOGGER.warning("Failed to load exchange pairs from %s: %s", EXCHANGE_PAIRS_PATH, exc)
        return []

    if isinstance(payload, dict):
        payload = payload.get("pairs", [])

    if not isinstance(payload, list):
        return []

    cleaned: list[dict[str, object]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        pair = _normalise_pair_token(str(entry.get("pair") or ""))
        if not pair or not pair.endswith("USDT"):
            continue
        symbol = _normalise_pair_token(str(entry.get("symbol") or ""))
        name = str(entry.get("name") or "")
        aliases_raw = entry.get("aliases") or []
        if isinstance(aliases_raw, str):
            aliases_raw = [aliases_raw]
        aliases = [str(alias) for alias in aliases_raw if isinstance(alias, str)]
        cleaned.append({"pair": pair, "symbol": symbol, "name": name, "aliases": aliases})
    return cleaned


@lru_cache(maxsize=1)
def _exchange_pair_lookup() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in _exchange_pairs_data():
        pair = str(entry.get("pair"))
        symbol = str(entry.get("symbol") or "")
        name = str(entry.get("name") or "")
        aliases = list(entry.get("aliases") or [])

        tokens = {pair, symbol}
        if name:
            tokens.add(name)
            tokens.update(name.split())
        tokens.update(aliases)
        for alias in list(tokens):
            normalised = _normalise_pair_token(alias)
            if not normalised:
                continue
            mapping.setdefault(normalised, pair)
            if not normalised.endswith("USDT"):
                composed = f"{normalised}USDT"
                mapping.setdefault(composed, pair)
    return mapping


def _match_exchange_pair(raw_text: str) -> str | None:
    token = _normalise_pair_token(raw_text)
    if not token:
        return None
    lookup = _exchange_pair_lookup()
    pair = lookup.get(token)
    if pair:
        return pair
    if not token.endswith("USDT"):
        pair = lookup.get(f"{token}USDT")
        if pair:
            return pair
    result = resolve_instrument(raw_text)
    if result.status == "resolved" and result.pair:
        return result.pair
    return None


DONATION_TIERS = _parse_donation_tiers(os.getenv("DONATION_TIERS", "100,500,1000"))
DONATION_TITLE = os.getenv("DONATION_TITLE", "حمایت با استارز")
DONATION_DESCRIPTION = os.getenv(
    "DONATION_DESCRIPTION",
    "با پرداخت استارز از ادامه فعالیت ربات حمایت کنید.",
)
DONATION_START_PARAMETER = os.getenv("DONATION_START_PARAMETER", "donate_stars")
TERMS_MESSAGE = os.getenv(
    "TERMS_MESSAGE",
    "شرایط استفاده: پرداخت با استارز به منزله حمایت داوطلبانه است.",
)
HELP_MESSAGE = os.getenv(
    "HELP_MESSAGE",
    "دستورهای در دسترس:\n"
    "• /menu — نمایش منوی دکمه‌ای\n"
    "• /get — دریافت آخرین اسنپ‌شات\n"
    "• /add — افزودن ارز به واچ‌لیست\n"
    "• /remove — حذف ارز از واچ‌لیست\n"
    "• /donate — دونیت با استارز\n"
    "• /terms — شرایط استفاده\n"
    "• /paysupport — پشتیبانی پرداخت",
)
PAY_SUPPORT_MESSAGE = os.getenv(
    "PAY_SUPPORT_MESSAGE",
    "پشتیبانی پرداخت: برای پیگیری مشکلات پرداخت با پشتیبانی ربات در تماس باشید.",
)
ADMIN_CHAT_IDS = _parse_admin_ids(os.getenv("ADMIN_CHAT_IDS"))

DEBUG_NOT_ALLOWED_MESSAGE = "⛔ فقط برای ادمین"
UNKNOWN_COMMAND_MESSAGE = "دستور ناشناخته است. برای مشاهده گزینه‌ها /menu را ارسال کنید."

MENU_MARKUP = {
    "inline_keyboard": [
        [
            {
                "text": "📬 دریافت فوری / Get updates now",
                "callback_data": "get_updates_now",
            },
            {
                "text": "➕ افزودن ارز / Add asset",
                "callback_data": "add_asset_start",
            },
        ],
        [
            {
                "text": "🗑️ حذف ارز / Remove asset",
                "callback_data": "remove_asset_start",
            }
        ],
        [
            {
                "text": "💖 دونیت با استارز / Donate with Stars",
                "callback_data": "donate_stars_start",
            }
        ],
    ]
}

REMOVE_PAGE_SIZE = 10
CONTACT_PROMPT_INTERVAL = timedelta(minutes=10)
CONTACT_PENDING_MESSAGE = (
    "شماره شما هنوز تأیید نشده است. لطفاً دکمه «📱 ارسال شماره من» را بزنید یا دوباره /start را ارسال کنید."
)


def _parse_contact_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def ensure_contact_prompt(
    existing: dict | None,
    chat_id: int | str,
    *,
    command: str | None = None,
    prompt_callback: Callable[[bool], None] | None = None,
) -> tuple[bool, bool]:
    normalized_command = (command or "").strip().lstrip("/").lower()
    if is_admin(chat_id) or normalized_command == "debug":
        return False, False

    if existing and existing.get("phone_number"):
        return False, False

    now = datetime.now(timezone.utc)
    awaiting = bool(existing.get("awaiting_contact")) if isinstance(existing, dict) else False
    prompted_at = (
        _parse_contact_timestamp(existing.get("contact_prompted_at"))
        if isinstance(existing, dict)
        else None
    )
    should_send_keyboard = (
        not awaiting
        or prompted_at is None
        or now - prompted_at >= CONTACT_PROMPT_INTERVAL
    )
    record_changed = False

    if should_send_keyboard:
        LOGGER.info("Prompting %s to share contact details", chat_id)
        already_registered = bool(
            isinstance(existing, dict) and existing.get("phone_number")
        )
        if prompt_callback is not None:
            try:
                prompt_callback(already_registered)
            except Exception as exc:  # pragma: no cover - callback errors
                LOGGER.warning(
                    "Contact prompt callback failed for %s: %s", chat_id, exc
                )
        else:
            try:
                send_start_prompt(chat_id, already_registered=already_registered)
            except Exception as exc:  # pragma: no cover - network failure
                print(f"Failed to send contact prompt to {chat_id}: {exc}")
        try:
            record_changed, _ = upsert_subscriber(
                chat_id,
                phone_number=existing.get("phone_number") if isinstance(existing, dict) else None,
                first_name=existing.get("first_name") if isinstance(existing, dict) else None,
                last_name=existing.get("last_name") if isinstance(existing, dict) else None,
                username=existing.get("username") if isinstance(existing, dict) else None,
                is_subscribed=bool(existing.get("is_subscribed")) if isinstance(existing, dict) else False,
                awaiting_contact=True,
                contact_prompted_at=now,
                path=SUBS_FILE,
            )
        except Exception as exc:  # pragma: no cover - db failure
            LOGGER.exception("Failed to record contact prompt state for %s", chat_id)
        else:
            if record_changed:
                LOGGER.info("Marked subscriber %s as awaiting contact", chat_id)
    else:
        try:
            send_telegram(str(chat_id), CONTACT_PENDING_MESSAGE)
        except Exception as exc:  # pragma: no cover - network failure
            print(f"Failed to send awaiting-contact notice to {chat_id}: {exc}")

    return True, record_changed

def load_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False)


def load_offset(path: Path, *, legacy_path: Path | None = None) -> int:
    candidate = Path(path)
    try:
        with open(candidate, "r", encoding="utf-8") as fh:
            raw = fh.read().strip()
    except FileNotFoundError:
        if legacy_path:
            legacy_data = load_json(legacy_path, {"offset": 0})
            try:
                return int(legacy_data.get("offset", 0))
            except Exception:
                return 0
        return 0
    except Exception:
        return 0
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def save_offset(path: Path, value: int) -> None:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    with open(candidate, "w", encoding="utf-8") as fh:
        fh.write(f"{max(0, int(value))}")


def load_state() -> dict:
    state = load_json(STATE_FILE, {"conversations": {}})
    if not isinstance(state, dict):
        state = {"conversations": {}}
    conv = state.get("conversations")
    if not isinstance(conv, dict):
        state["conversations"] = {}
    pending = state.get("pending_invoices")
    if not isinstance(pending, dict):
        state["pending_invoices"] = {}
    return state


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_json(STATE_FILE, state)


@lru_cache(maxsize=1)
def _tehran_zone() -> ZoneInfo:
    tz_name = os.getenv("TIMEZONE", "Asia/Tehran")
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("Asia/Tehran")


def _parse_summary_timestamp(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None

    tz = _tehran_zone()
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S",):
            try:
                parsed = datetime.strptime(raw, fmt)
            except ValueError:
                continue
            else:
                return parsed.replace(tzinfo=tz)
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


def _load_snapshot(path: Path | None = None) -> dict | None:
    summary = load_latest_summary(path=SUBS_FILE)
    if summary:
        return summary

    candidate = Path(path) if path else SNAPSHOT_PATH
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _save_snapshot(payload: dict, path: Path | None = None) -> None:
    save_summary(payload, path=SUBS_FILE)

    candidate = Path(path) if path else SNAPSHOT_PATH
    candidate.parent.mkdir(parents=True, exist_ok=True)
    with open(candidate, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _snapshot_is_fresh(snapshot: dict | None, *, max_age: timedelta = SNAPSHOT_MAX_AGE) -> bool:
    if not snapshot or not isinstance(snapshot, dict):
        return False
    generated = snapshot.get("generated_at")
    if not isinstance(generated, str):
        return False
    parsed = _parse_summary_timestamp(generated)
    if not parsed:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    now = datetime.now(parsed.tzinfo)
    try:
        return (now - parsed) <= max_age
    except TypeError:
        return False


def load_snapshot() -> dict | None:
    return _load_snapshot()


def _conversation_bucket(state: dict) -> dict:
    bucket = state.setdefault("conversations", {})
    if not isinstance(bucket, dict):
        bucket = {}
        state["conversations"] = bucket
    return bucket


def get_conversation(state: dict, chat_id: int | str) -> dict | None:
    return _conversation_bucket(state).get(str(chat_id))


def set_conversation(state: dict, chat_id: int | str, data: dict) -> None:
    _conversation_bucket(state)[str(chat_id)] = data


def clear_conversation(state: dict, chat_id: int | str) -> None:
    _conversation_bucket(state).pop(str(chat_id), None)


def _invoice_bucket(state: dict) -> dict:
    bucket = state.setdefault("pending_invoices", {})
    if not isinstance(bucket, dict):
        bucket = {}
        state["pending_invoices"] = bucket
    return bucket


def _register_pending_invoice(state: dict, payload: str, chat_id: int | str, amount: int) -> None:
    _invoice_bucket(state)[payload] = {
        "chat_id": str(chat_id),
        "stars_amount": int(amount),
        "started_at": time.time(),
    }


def _pop_pending_invoice(state: dict, payload: str) -> dict | None:
    return _invoice_bucket(state).pop(payload, None)


def is_admin(chat_id: int | str) -> bool:
    try:
        return int(str(chat_id)) in ADMIN_CHAT_IDS
    except (TypeError, ValueError):
        return False


def _format_stars(amount: int) -> str:
    return f"{int(amount)}⭐"


def _build_debug_message(chat_id: int | str) -> str:
    current = get_subscriber(chat_id, path=SUBS_FILE)
    if current:
        subscribed = "✅" if current.get("is_subscribed") else "❌"
        return (
            "اطلاعات شما در دیتابیس:\n"
            f"• chat_id: {current.get('chat_id')}\n"
            f"• phone_number: {current.get('phone_number', '')}\n"
            f"• subscribed: {subscribed}"
        )
    return "هیچ اطلاعاتی برای شما ثبت نشده است."


async def debug_info(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    message = getattr(update, "effective_message", None)
    if message is None:
        return

    user = getattr(update, "effective_user", None)
    user_id = getattr(user, "id", None)
    if not is_admin(user_id):
        LOGGER.info("Rejected /debug command from user=%s", user_id)
        await message.reply_text(DEBUG_NOT_ALLOWED_MESSAGE)
        return

    chat = getattr(update, "effective_chat", None)
    chat_id = getattr(chat, "id", getattr(message, "chat_id", user_id))
    backend = describe_backend(path=SUBS_FILE)
    LOGGER.info("Handling /debug for user=%s backend=%s", user_id, backend)
    await message.reply_text(_build_debug_message(chat_id))


def _send_summary_message(chat_id: int | str, payload: dict | None) -> None:
    message = render_compact_summary(payload)
    try:
        send_telegram(str(chat_id), message)
    except Exception as exc:  # pragma: no cover - network failure
        print(f"Failed to send summary to {chat_id}: {exc}")


def _run_live_summary(chat_id: int | str) -> dict:
    watchlist = get_user_watchlist(chat_id, path=SUBS_FILE)
    return run_summary_once(watchlist)


def send_summary_for_chat(chat_id: int | str, *, force_live: bool = False) -> dict | None:
    snapshot: dict | None = None
    if not force_live:
        snapshot = _load_snapshot()
        if snapshot and _snapshot_is_fresh(snapshot):
            _send_summary_message(chat_id, snapshot)
            return snapshot

    try:
        summary = _run_live_summary(chat_id)
    except Exception as exc:
        print(f"Failed to compute live summary for {chat_id}: {exc}")
        if snapshot and not force_live:
            try:
                _send_summary_message(chat_id, snapshot)
            except Exception as send_exc:  # pragma: no cover - network failure
                print(f"Failed to deliver cached summary to {chat_id}: {send_exc}")
            return snapshot
        try:
            send_telegram(
                str(chat_id),
                "محاسبه لحظه‌ای با خطا مواجه شد. لطفاً بعداً دوباره تلاش کنید.",
            )
        except Exception as send_exc:  # pragma: no cover - network failure
            print(f"Failed to notify live summary error for {chat_id}: {send_exc}")
        return None

    try:
        _save_snapshot(summary)
    except Exception as exc:  # pragma: no cover - disk error
        print(f"Failed to persist summary snapshot: {exc}")

    _send_summary_message(chat_id, summary)
    return summary


def _build_donation_keyboard() -> dict:
    rows: list[list[dict[str, str]]] = []
    if DONATION_TIERS:
        tier_buttons: list[dict[str, str]] = []
        for tier in DONATION_TIERS:
            tier_buttons.append(
                {
                    "text": _format_stars(tier),
                    "callback_data": f"donate:tier:{tier}",
                }
            )
        rows.append(tier_buttons)
    rows.append(
        [
            {
                "text": "✏️ مبلغ دلخواه / Custom amount",
                "callback_data": "donate:custom",
            }
        ]
    )
    return {"inline_keyboard": rows}


def _make_invoice_payload(chat_id: int | str, amount: int) -> str:
    nonce = uuid.uuid4().hex[:6]
    return f"don|{chat_id}|{amount}|{int(time.time()*1000)}|{nonce}"


def _handle_invoice_error(chat_id: int | str, error: Exception) -> None:
    description = ""
    if isinstance(error, requests.HTTPError):
        response = error.response  # type: ignore[attr-defined]
        if response is not None:
            try:
                data = response.json()
                description = data.get("description", "")
            except ValueError:
                description = getattr(response, "text", "") or ""
    else:
        description = str(error)

    friendly = "ارسال فاکتور با خطا مواجه شد. لطفاً نسخه تلگرام خود را به‌روز کنید و دوباره تلاش کنید."
    if "STAR" in description.upper() or "XTR" in description.upper():
        friendly = (
            "به نظر می‌رسد این دستگاه از پرداخت استارز پشتیبانی نمی‌کند."
            " لطفاً تلگرام خود را به آخرین نسخه به‌روزرسانی کنید."
        )

    try:
        send_telegram(str(chat_id), friendly)
    except Exception as exc:  # pragma: no cover - best effort notification
        print(f"Failed to notify invoice error to {chat_id}: {exc}")

    print(f"Invoice error for {chat_id}: {description or error}")


def _send_stars_invoice(state: dict, chat_id: int | str, amount: int) -> bool:
    amount_int = int(amount)
    if amount_int <= 0:
        try:
            send_telegram(str(chat_id), "مبلغ باید بزرگ‌تر از صفر باشد.")
        except Exception as exc:  # pragma: no cover - notification failure
            print(f"Failed to notify invalid amount to {chat_id}: {exc}")
        return False

    payload = _make_invoice_payload(chat_id, amount_int)
    invoice = {
        "chat_id": chat_id,
        "title": DONATION_TITLE,
        "description": DONATION_DESCRIPTION,
        "payload": payload,
        "currency": "XTR",
        "provider_token": "",
        "prices": [{"label": "Donation", "amount": amount_int}],
        "start_parameter": DONATION_START_PARAMETER,
    }

    try:
        response = requests.post(f"{API}/sendInvoice", json=invoice, timeout=20)
        response.raise_for_status()
    except requests.HTTPError as exc:
        _handle_invoice_error(chat_id, exc)
        return False
    except requests.RequestException as exc:  # pragma: no cover - network issue
        print(f"Failed to reach Telegram for invoice: {exc}")
        try:
            send_telegram(str(chat_id), "امکان برقراری ارتباط با تلگرام وجود ندارد. بعداً تلاش کنید.")
        except Exception as inner:  # pragma: no cover - notification failure
            print(f"Failed to send invoice fallback message to {chat_id}: {inner}")
        return False

    _register_pending_invoice(state, payload, chat_id, amount_int)
    try:
        send_telegram(
            str(chat_id),
            f"فاکتور { _format_stars(amount_int) } ارسال شد. لطفاً در تلگرام پرداخت را تکمیل کنید.",
        )
    except Exception as exc:  # pragma: no cover
        print(f"Failed to send invoice confirmation to {chat_id}: {exc}")
    return True


def handle_donate_stars_start(state: dict, chat_id: int | str) -> bool:
    if not DONATION_TIERS:
        try:
            send_telegram(
                str(chat_id),
                "سطح‌های دونیت تنظیم نشده‌اند. لطفاً بعداً دوباره امتحان کنید.",
            )
        except Exception as exc:  # pragma: no cover
            print(f"Failed to notify missing donation tiers to {chat_id}: {exc}")
        return False

    keyboard = _build_donation_keyboard()
    payload = {
        "chat_id": chat_id,
        "text": "یکی از مبالغ زیر را برای دونیت انتخاب کنید:\nChoose a Stars amount.",
        "reply_markup": keyboard,
    }

    try:
        response = requests.post(f"{API}/sendMessage", json=payload, timeout=20)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure
        print(f"Failed to send donation options to {chat_id}: {exc}")
    return False


def handle_donate_tier(state: dict, chat_id: int | str, amount: int) -> bool:
    success = _send_stars_invoice(state, chat_id, amount)
    return success


def handle_donate_custom_prompt(state: dict, chat_id: int | str) -> bool:
    set_conversation(state, chat_id, {"state": "await_donation_custom"})
    try:
        send_telegram(
            str(chat_id),
            "لطفاً مقدار استارز را به صورت عدد ارسال کنید (مثلاً 250).",
        )
    except Exception as exc:  # pragma: no cover
        print(f"Failed to prompt custom donation amount for {chat_id}: {exc}")
    return True


def handle_donate_custom_text(state: dict, chat_id: int | str, raw_text: str) -> bool:
    convo = get_conversation(state, chat_id) or {}
    if convo.get("state") != "await_donation_custom":
        return False

    digits = re.sub(r"\D", "", raw_text or "")
    if not digits:
        try:
            send_telegram(
                str(chat_id),
                "مقدار معتبر نیست. فقط عدد ارسال کنید (مثلاً 1000).",
            )
        except Exception as exc:  # pragma: no cover
            print(f"Failed to warn about invalid custom amount for {chat_id}: {exc}")
        return False

    amount = int(digits)
    success = _send_stars_invoice(state, chat_id, amount)
    if success:
        clear_conversation(state, chat_id)
    return success


def handle_pre_checkout_query(query: dict) -> None:
    payload = {"pre_checkout_query_id": query.get("id"), "ok": True}
    try:
        response = requests.post(f"{API}/answerPreCheckoutQuery", json=payload, timeout=20)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - best effort acknowledgement
        print(f"Failed to answer pre-checkout query {query.get('id')}: {exc}")


def handle_successful_payment(state: dict, chat_id: int | str, payment: dict) -> bool:
    payload = payment.get("invoice_payload") or ""
    pending = _pop_pending_invoice(state, payload) if payload else None

    stars_amount = int(payment.get("total_amount", 0))
    if stars_amount <= 0 and pending:
        stars_amount = int(pending.get("stars_amount", 0))

    charge_id = payment.get("telegram_payment_charge_id", "")
    provider_charge_id = payment.get("provider_payment_charge_id", "")
    duration_ms = 0
    if pending and pending.get("started_at"):
        duration_ms = int(max(0, time.time() - pending["started_at"]) * 1000)

    reference = {
        "invoice_payload": payload,
        "telegram_payment_charge_id": charge_id,
        "provider_payment_charge_id": provider_charge_id,
    }

    if stars_amount > 0:
        try:
            save_donation(
                chat_id,
                stars_amount,
                json.dumps(reference, ensure_ascii=False),
                path=SUBS_FILE,
            )
        except Exception as exc:  # pragma: no cover - persistence failure
            print(f"Failed to persist donation for {chat_id}: {exc}")

    thanks = (
        f"🙏 سپاس از حمایت شما! {_format_stars(stars_amount)} دریافت شد."
        if stars_amount > 0
        else "پرداخت با موفقیت دریافت شد."
    )
    try:
        send_telegram(str(chat_id), thanks)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to send thank-you message to {chat_id}: {exc}")

    print(
        "Donation recorded:",
        json.dumps(
            {
                "user_id": str(chat_id),
                "stars_amount": stars_amount,
                "telegram_payment_charge_id": charge_id,
                "duration_ms": duration_ms,
            },
            ensure_ascii=False,
        ),
    )

    return bool(pending)


def handle_admin_donations(chat_id: int | str) -> None:
    rows = list_recent_donations(path=SUBS_FILE)
    totals = donation_totals(path=SUBS_FILE)
    lines = ["آخرین دونیت‌ها:"]
    if not rows:
        lines.append("هنوز دونیتی ثبت نشده است.")
    else:
        for row in rows:
            lines.append(
                f"{row['created_at']} – {row['user_id']} – {_format_stars(row['stars_amount'])}"
            )
    lines.append(
        f"جمع کل: {totals['count']} پرداخت | {_format_stars(totals['total'])}"
    )
    try:
        send_telegram(str(chat_id), "\n".join(lines))
    except Exception as exc:  # pragma: no cover
        print(f"Failed to send donation summary to {chat_id}: {exc}")


def handle_refund_request(chat_id: int | str, raw_text: str) -> None:
    parts = (raw_text or "").strip().split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        try:
            send_telegram(str(chat_id), "استفاده: /refund <telegram_payment_charge_id>")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to send refund usage to {chat_id}: {exc}")
        return

    charge_id = parts[1].strip()
    payload = {"user_id": chat_id, "telegram_payment_charge_id": charge_id}
    try:
        response = requests.post(f"{API}/refundStarPayment", json=payload, timeout=20)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure
        print(f"Refund request failed for {chat_id}: {exc}")
        try:
            send_telegram(str(chat_id), f"درخواست بازپرداخت ناموفق بود: {exc}")
        except Exception as inner:  # pragma: no cover
            print(f"Failed to send refund failure message to {chat_id}: {inner}")
        return

    try:
        send_telegram(
            str(chat_id),
            f"درخواست بازپرداخت برای شناسه {charge_id} ارسال شد.",
        )
    except Exception as exc:  # pragma: no cover
        print(f"Failed to confirm refund to {chat_id}: {exc}")


def _compose_start_prompt_text(already_registered: bool = False) -> str:
    if already_registered:
        return (
            "سلام! شماره شما قبلاً ثبت شده است؛ در صورت نیاز به بروزرسانی یا تأیید مجدد"
            " لطفاً روی دکمه «📱 ارسال شماره من» بزنید."
        )
    return (
        "سلام! برای فعال‌سازی سیگنال‌ها لطفاً روی دکمه «📱 ارسال شماره من» بزنید تا شماره تلگرام شما ثبت شود.\n"
        "در صورت ثبت قبلی، می‌توانید مجدداً شماره را بفرستید تا اطلاعات به‌روزرسانی شود."
    )


def send_start_prompt(chat_id: int | str, already_registered: bool = False) -> None:
    text = _compose_start_prompt_text(already_registered)

    keyboard = {
        "keyboard": [[{"text": "📱 ارسال شماره من", "request_contact": True}]],
        "resize_keyboard": True,
        "one_time_keyboard": True,
    }
    payload = {"chat_id": chat_id, "text": text, "reply_markup": keyboard}
    resp = requests.post(f"{API}/sendMessage", json=payload, timeout=20)
    resp.raise_for_status()


def send_contact_confirmation(chat_id: int | str) -> None:
    text = (
        "شماره شما با موفقیت ثبت و اشتراک فعال شد. ✅\n"
        "از این پس سیگنال‌ها و خلاصه‌ها برای شما ارسال می‌شود."
    )
    payload = {
        "chat_id": chat_id,
        "text": text,
        "reply_markup": {"remove_keyboard": True},
    }
    resp = requests.post(f"{API}/sendMessage", json=payload, timeout=20)
    resp.raise_for_status()


def send_menu(chat_id: int | str, *, prepend_text: str | None = None) -> None:
    watchlist = get_user_watchlist(chat_id, path=SUBS_FILE)
    watchlist_line = (
        "Watchlist: " + ", ".join(sorted(watchlist))
        if watchlist
        else "Watchlist: defaults (USDT quote)."
    )
    lines = [
        prepend_text or "از دکمه‌های زیر استفاده کنید:",
        "📬 دریافت فوری / Get updates now — خلاصه لحظه‌ای",
        "➕ افزودن ارز / Add asset — پیش‌فرض USDT",
        "🗑️ حذف ارز / Remove asset — مدیریت واچ‌لیست",
        "💖 دونیت با استارز / Donate with Stars — پرداخت با XTR",
        "",
        watchlist_line,
    ]
    payload = {
        "chat_id": chat_id,
        "text": "\n".join(line for line in lines if line is not None),
        "reply_markup": MENU_MARKUP,
    }
    resp = requests.post(f"{API}/sendMessage", json=payload, timeout=20)
    resp.raise_for_status()


def send_unsubscribe_confirmation(chat_id: int | str) -> None:
    text = (
        "اشتراک شما غیرفعال شد. برای فعال‌سازی دوباره، هر زمان /start را بفرستید و شماره خود را ارسال کنید."
    )
    payload = {"chat_id": chat_id, "text": text, "reply_markup": {"remove_keyboard": True}}
    resp = requests.post(f"{API}/sendMessage", json=payload, timeout=20)
    resp.raise_for_status()


def answer_callback(callback_id: str, *, text: str | None = None) -> None:
    payload = {"callback_query_id": callback_id}
    if text:
        payload["text"] = text
    resp = requests.post(f"{API}/answerCallbackQuery", json=payload, timeout=20)
    resp.raise_for_status()


def handle_add_asset_start(state: dict, chat_id: int | str) -> bool:
    set_conversation(state, chat_id, {"state": "await_symbol"})
    try:
        send_telegram(
            str(chat_id),
            "لطفاً نماد یا نام دارایی را ارسال کنید (مثال: sol یا solana). پیش‌فرض USDT است.",
        )
    except Exception as exc:  # pragma: no cover - network failure
        print(f"Failed to prompt for asset from {chat_id}: {exc}")
    return True


def _confirm_watchlist_add(chat_id: int | str, pair: str, added: bool) -> None:
    if added:
        message = f"ارز {pair} با موفقیت اضافه شد ✅"
    else:
        message = f"ارز {pair} پیش‌تر به واچ‌لیست شما اضافه شده بود ⚠️"
    try:
        send_telegram(str(chat_id), message)
    except Exception as exc:  # pragma: no cover - network failure
        print(f"Failed to confirm watchlist update for {chat_id}: {exc}")


def handle_add_asset_pick(state: dict, chat_id: int | str, pair: str) -> bool:
    convo = get_conversation(state, chat_id) or {}
    if convo.get("state") != "await_pick":
        try:
            send_telegram(str(chat_id), "درخواست انتخاب دارایی معتبر نیست. لطفاً دوباره تلاش کنید.")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to notify invalid pick for {chat_id}: {exc}")
        return False

    valid_pairs = {candidate.upper() for candidate in convo.get("candidates", [])}
    pair_norm = (pair or "").upper()
    if pair_norm not in valid_pairs:
        try:
            send_telegram(str(chat_id), "انتخاب ارسال‌شده در گزینه‌ها نبود. لطفاً دوباره تلاش کنید.")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to warn about invalid candidate for {chat_id}: {exc}")
        return False

    added = add_to_watchlist(chat_id, pair_norm, path=SUBS_FILE)
    _confirm_watchlist_add(chat_id, pair_norm, added)
    clear_conversation(state, chat_id)
    try:
        send_menu(chat_id)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to send menu after pick for {chat_id}: {exc}")
    return True


def _send_ambiguous_options(chat_id: int | str, candidates: list[ResolutionCandidate]) -> None:
    keyboard = {
        "inline_keyboard": [
            [{"text": candidate.label, "callback_data": f"addasset:pick:{candidate.pair}"}]
            for candidate in candidates
        ]
    }
    payload = {
        "chat_id": chat_id,
        "text": "چند گزینه مشابه یافت شد، لطفاً انتخاب کنید:",
        "reply_markup": keyboard,
    }
    resp = requests.post(f"{API}/sendMessage", json=payload, timeout=20)
    resp.raise_for_status()


def handle_add_asset_text(state: dict, chat_id: int | str, raw_text: str) -> bool:
    pair = _match_exchange_pair(raw_text)
    if pair:
        added = add_to_watchlist(chat_id, pair, path=SUBS_FILE)
        _confirm_watchlist_add(chat_id, pair, added)
        clear_conversation(state, chat_id)
        try:
            send_menu(chat_id)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to send menu after add for {chat_id}: {exc}")
        return True

    result: ResolutionResult = resolve_instrument(raw_text)
    if result.status == "error":
        if result.reason == "EMPTY_INPUT":
            msg = "ورودی خالی بود. لطفاً نماد یا نام دارایی را بفرستید (مانند SOL)."
        else:
            msg = "نماد پشتیبانی نمی‌شود. مثال: SOL یا SOLUSDT (پیش‌فرض USDT)."
        try:
            send_telegram(str(chat_id), msg)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to notify invalid input for {chat_id}: {exc}")
        return False

    if result.status == "resolved" and result.pair:
        added = add_to_watchlist(chat_id, result.pair, path=SUBS_FILE)
        _confirm_watchlist_add(chat_id, result.pair, added)
        clear_conversation(state, chat_id)
        try:
            send_menu(chat_id)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to send menu after add for {chat_id}: {exc}")
        return True

    candidates = (result.candidates or [])[:3]
    if candidates:
        set_conversation(
            state,
            chat_id,
            {
                "state": "await_pick",
                "candidates": [candidate.pair for candidate in candidates],
            },
        )
        _send_ambiguous_options(chat_id, candidates)
        return True

    try:
        send_telegram(str(chat_id), "هیچ نتیجه‌ای یافت نشد. لطفاً دوباره تلاش کنید.")
    except Exception as exc:  # pragma: no cover
        print(f"Failed to notify no result for {chat_id}: {exc}")
    return False


def _send_remove_page(chat_id: int | str, pairs: list[str], page: int) -> None:
    total_pages = max(1, math.ceil(len(pairs) / REMOVE_PAGE_SIZE))
    page_index = max(0, min(page, total_pages - 1))
    start = page_index * REMOVE_PAGE_SIZE
    slice_pairs = pairs[start : start + REMOVE_PAGE_SIZE]
    keyboard = [
        [{"text": pair, "callback_data": f"remove_asset:pick:{pair}"}] for pair in slice_pairs
    ]
    if len(pairs) > REMOVE_PAGE_SIZE:
        nav_row: list[dict[str, str]] = []
        if page_index > 0:
            nav_row.append({"text": "« Prev", "callback_data": f"remove_asset:page:{page_index - 1}"})
        if page_index < total_pages - 1:
            nav_row.append({"text": "Next »", "callback_data": f"remove_asset:page:{page_index + 1}"})
        if nav_row:
            keyboard.append(nav_row)

    payload = {
        "chat_id": chat_id,
        "text": "دارایی مورد نظر برای حذف را انتخاب کنید.\nSelect asset to remove."
        + (f"\nصفحه {page_index + 1} از {total_pages}" if total_pages > 1 else ""),
        "reply_markup": {"inline_keyboard": keyboard},
    }
    resp = requests.post(f"{API}/sendMessage", json=payload, timeout=20)
    resp.raise_for_status()


def handle_remove_asset_start(state: dict, chat_id: int | str) -> bool:
    watchlist = get_user_watchlist(chat_id, path=SUBS_FILE)
    if not watchlist:
        try:
            send_telegram(str(chat_id), "لیست خالی است / Your watchlist is empty.")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to notify empty watchlist for {chat_id}: {exc}")
        return False

    payload = {"state": "await_remove_pick", "pairs": watchlist, "page": 0}
    set_conversation(state, chat_id, payload)
    try:
        _send_remove_page(chat_id, watchlist, 0)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to send remove menu to {chat_id}: {exc}")
    return True


def handle_remove_asset_page(state: dict, chat_id: int | str, page: int) -> bool:
    convo = get_conversation(state, chat_id) or {}
    if convo.get("state") != "await_remove_pick":
        return False
    pairs = list(convo.get("pairs") or [])
    if not pairs:
        return False
    convo["page"] = page
    set_conversation(state, chat_id, convo)
    try:
        _send_remove_page(chat_id, pairs, page)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to paginate remove menu for {chat_id}: {exc}")
    return True


def _send_remove_confirm(chat_id: int | str, pair: str) -> None:
    keyboard = {
        "inline_keyboard": [
            [
                {
                    "text": "✅ بله / Yes",
                    "callback_data": f"remove_asset:confirm:{pair}:yes",
                },
                {
                    "text": "↩️ خیر / No",
                    "callback_data": f"remove_asset:confirm:{pair}:no",
                },
            ]
        ]
    }
    payload = {
        "chat_id": chat_id,
        "text": f"{pair}\nحذف شود؟ / Remove?",
        "reply_markup": keyboard,
    }
    resp = requests.post(f"{API}/sendMessage", json=payload, timeout=20)
    resp.raise_for_status()


def handle_remove_asset_pick(state: dict, chat_id: int | str, pair: str) -> bool:
    convo = get_conversation(state, chat_id) or {}
    if convo.get("state") != "await_remove_pick":
        try:
            send_telegram(str(chat_id), "درخواست حذف معتبر نیست. / Invalid remove request.")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to warn invalid remove pick for {chat_id}: {exc}")
        return False

    pairs = [item.upper() for item in convo.get("pairs", [])]
    pair_norm = (pair or "").upper()
    if pair_norm not in pairs:
        try:
            send_telegram(str(chat_id), "گزینه انتخاب‌شده یافت نشد. / Option not available.")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to warn missing remove option for {chat_id}: {exc}")
        return False

    set_conversation(
        state,
        chat_id,
        {
            "state": "await_remove_confirm",
            "pair": pair_norm,
            "pairs": pairs,
            "page": convo.get("page", 0),
        },
    )
    try:
        _send_remove_confirm(chat_id, pair_norm)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to send remove confirmation to {chat_id}: {exc}")
    return True


def handle_remove_asset_confirm(
    state: dict, chat_id: int | str, pair: str, decision: str
) -> bool:
    convo = get_conversation(state, chat_id) or {}
    if convo.get("state") != "await_remove_confirm":
        return False

    expected_pair = (convo.get("pair") or "").upper()
    pair_norm = (pair or "").upper()
    if expected_pair != pair_norm:
        return False

    if decision == "yes":
        removed = remove_from_watchlist(chat_id, pair_norm, path=SUBS_FILE)
        message = (
            f"🗑️ {pair_norm} حذف شد / Removed."
            if removed
            else f"{pair_norm} در واچ‌لیست نبود / Not found."
        )
        try:
            send_telegram(str(chat_id), message)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to confirm removal for {chat_id}: {exc}")
        clear_conversation(state, chat_id)
        try:
            send_menu(chat_id)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to send menu after removal for {chat_id}: {exc}")
        return True

    if decision == "no":
        try:
            send_telegram(str(chat_id), "عملیات لغو شد. / Cancelled.")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to send cancel notice for {chat_id}: {exc}")
        pairs = [item.upper() for item in convo.get("pairs", [])]
        set_conversation(
            state,
            chat_id,
            {"state": "await_remove_pick", "pairs": pairs, "page": convo.get("page", 0)},
        )
        try:
            _send_remove_page(chat_id, pairs, convo.get("page", 0))
        except Exception as exc:  # pragma: no cover
            print(f"Failed to resend remove menu for {chat_id}: {exc}")
        return True

    return False


def handle_get_updates_now(chat_id: int | str) -> None:
    send_summary_for_chat(chat_id, force_live=True)


def _fetch_updates(offset: int, timeout: float) -> list[dict]:
    timeout_value = int(max(0, math.ceil(timeout)))
    try:
        updates = _run_bot_get_updates(offset, timeout_value)
        if updates is None:
            return []
        if not isinstance(updates, list):
            try:
                updates = list(updates)
            except TypeError:
                updates = [updates]
    except RetryAfter as exc:  # pragma: no cover - network timing
        sleep_for = getattr(exc, "retry_after", 1)
        time.sleep(float(sleep_for))
        return []
    except TimedOut:  # pragma: no cover - network timing
        return []
    except Exception as exc:  # pragma: no cover - network failure
        print(f"Failed to fetch updates: {exc}")
        return []
    return [upd.to_dict() for upd in updates]


def process_updates(duration_seconds: float | None = None, poll_timeout: float = 10.0) -> None:
    legacy_candidate = (
        LEGACY_OFFSET_FILE
        if LEGACY_OFFSET_FILE != OFFSET_FILE and LEGACY_OFFSET_FILE.exists()
        else None
    )
    offset = load_offset(OFFSET_FILE, legacy_path=legacy_candidate)
    last_processed_update_id = offset - 1 if offset > 0 else None

    state = load_state()
    state_changed = False
    subs_changed = False
    offset_changed = False

    def handle_batch(updates: list[dict]) -> None:
        nonlocal offset, state_changed, subs_changed, offset_changed, last_processed_update_id
        if not updates:
            return

        for upd in updates:
            if not isinstance(upd, dict):
                continue
            update_id = upd.get("update_id")
            if (
                update_id is not None
                and last_processed_update_id is not None
                and update_id <= last_processed_update_id
            ):
                continue
            try:
                pre_checkout = upd.get("pre_checkout_query")
                if pre_checkout:
                    handle_pre_checkout_query(pre_checkout)
                    continue

                callback = upd.get("callback_query")
                if callback:
                    callback_id = callback.get("id")
                    try:
                        answer_callback(callback_id or "")
                    except Exception as exc:  # pragma: no cover
                        print(f"Failed to answer callback {callback_id}: {exc}")
                    message = callback.get("message") or {}
                    chat = message.get("chat") or {}
                    chat_id = chat.get("id")
                    if not chat_id:
                        continue
                    data = (callback.get("data") or "").strip()
                    existing = get_subscriber(chat_id, path=SUBS_FILE)
                    requires_contact = data != "get_updates_now"
                    if requires_contact:
                        handled, changed = ensure_contact_prompt(
                            existing, chat_id, command=data
                        )
                        subs_changed = subs_changed or changed
                        if handled:
                            continue
                    if data == "get_updates_now":
                        handle_get_updates_now(chat_id)
                    elif data == "add_asset_start":
                        state_changed = handle_add_asset_start(state, chat_id) or state_changed
                    elif data == "remove_asset_start":
                        state_changed = handle_remove_asset_start(state, chat_id) or state_changed
                    elif data == "donate_stars_start":
                        state_changed = handle_donate_stars_start(state, chat_id) or state_changed
                    elif data.startswith("donate:tier:"):
                        try:
                            amount = int(data.split(":", 2)[2])
                        except ValueError:
                            amount = 0
                        state_changed = handle_donate_tier(state, chat_id, amount) or state_changed
                    elif data == "donate:custom":
                        state_changed = handle_donate_custom_prompt(state, chat_id) or state_changed
                    elif data.startswith("remove_asset:page:"):
                        try:
                            page = int(data.rsplit(":", 1)[1])
                        except ValueError:
                            page = 0
                        state_changed = handle_remove_asset_page(state, chat_id, page) or state_changed
                    elif data.startswith("remove_asset:pick:"):
                        pair = data.split(":", 2)[2]
                        state_changed = handle_remove_asset_pick(state, chat_id, pair) or state_changed
                    elif data.startswith("remove_asset:confirm:"):
                        parts = data.split(":")
                        if len(parts) == 4:
                            _, _, pair, decision = parts
                            state_changed = (
                                handle_remove_asset_confirm(state, chat_id, pair, decision)
                                or state_changed
                            )
                    elif data.startswith("addasset:pick:"):
                        pair = data.split(":", 2)[2]
                        state_changed = handle_add_asset_pick(state, chat_id, pair) or state_changed
                    else:
                        try:
                            send_telegram(str(chat_id), "دستور ناشناخته است.")
                        except Exception as exc:  # pragma: no cover
                            print(f"Failed to notify unknown callback for {chat_id}: {exc}")
                    continue

                msg = upd.get("message") or upd.get("channel_post")
                if not msg:
                    continue

                chat = msg.get("chat", {}) or {}
                chat_id = chat.get("id")
                if not chat_id:
                    continue

                existing = get_subscriber(chat_id, path=SUBS_FILE)
                conversation = get_conversation(state, chat_id)

                payment = msg.get("successful_payment")
                if payment:
                    state_changed = handle_successful_payment(state, chat_id, payment) or state_changed
                    continue

                contact = msg.get("contact")
                if contact:
                    contact_user_id = contact.get("user_id")
                    if contact_user_id and str(contact_user_id) != str(chat_id):
                        continue
                    phone = contact.get("phone_number")
                    if not phone:
                        continue
                    sender = msg.get("from", {}) or {}
                    first_name = contact.get("first_name") or sender.get("first_name") or ""
                    last_name = contact.get("last_name") or sender.get("last_name") or ""
                    username = sender.get("username") or ""

                    LOGGER.info("Received contact payload from chat_id=%s", chat_id)
                    try:
                        changed, saved = upsert_subscriber(
                            chat_id,
                            phone_number=phone,
                            first_name=first_name,
                            last_name=last_name,
                            username=username,
                            is_subscribed=True,
                            awaiting_contact=False,
                            contact_prompted_at=None,
                            path=SUBS_FILE,
                        )
                        subs_changed = subs_changed or changed
                        if saved:
                            phone_for_log = saved.get("phone_number", "")
                            if isinstance(phone_for_log, str) and len(phone_for_log) > 6:
                                phone_for_log = f"{phone_for_log[:4]}…{phone_for_log[-2:]}"
                            LOGGER.info(
                                "Saved contact for chat_id=%s phone=%s",
                                chat_id,
                                phone_for_log,
                            )
                    except Exception as exc:
                        LOGGER.exception("Failed to save contact for %s", chat_id)
                        saved = None
                    else:
                        if saved:
                            clear_conversation(state, chat_id)
                            try:
                                send_contact_confirmation(chat_id)
                                send_menu(
                                    chat_id,
                                    prepend_text="اشتراک شما فعال است؛ گزینه‌های زیر در دسترس هستند.",
                                )
                            except Exception as exc:
                                print(f"Failed to send confirmation/menu to {chat_id}: {exc}")
                            state_changed = True
                        else:
                            try:
                                send_telegram(
                                    str(chat_id),
                                    "ثبت شماره با خطا مواجه شد. لطفاً چند لحظه بعد دوباره تلاش کنید.",
                                )
                            except Exception as exc:
                                print(f"Failed to notify contact failure for {chat_id}: {exc}")
                    continue

                raw_text = msg.get("text") or ""
                text_lower = raw_text.strip().lower()
                command_name = (
                    text_lower.lstrip("/") if text_lower.startswith("/") else text_lower
                )

                if text_lower == "/debug":
                    if not is_admin(chat_id):
                        try:
                            send_telegram(str(chat_id), DEBUG_NOT_ALLOWED_MESSAGE)
                        except Exception as exc:
                            print(f"Failed to send non-admin debug notice to {chat_id}: {exc}")
                        continue

                    try:
                        send_telegram(str(chat_id), _build_debug_message(chat_id))
                    except Exception as exc:
                        print(f"Failed to send debug info to {chat_id}: {exc}")
                    continue

                if text_lower in ("/start", "/subscribe", "start"):
                    if existing and existing.get("phone_number"):
                        try:
                            send_start_prompt(chat_id, already_registered=True)
                        except Exception as exc:
                            print(f"Failed to send start prompt to {chat_id}: {exc}")
                    else:
                        handled, changed = ensure_contact_prompt(
                            existing, chat_id, command="start"
                        )
                        subs_changed = subs_changed or changed
                    continue

                if text_lower in ("/stop", "/unsubscribe", "stop"):
                    try:
                        changed, _ = (
                            upsert_subscriber(
                                chat_id,
                                phone_number=existing.get("phone_number") if existing else None,
                                first_name=existing.get("first_name") if existing else None,
                                last_name=existing.get("last_name") if existing else None,
                                username=existing.get("username") if existing else None,
                                is_subscribed=False,
                                awaiting_contact=False,
                                contact_prompted_at=None,
                                path=SUBS_FILE,
                            )
                            if existing
                            else upsert_subscriber(
                                chat_id,
                                is_subscribed=False,
                                awaiting_contact=False,
                                contact_prompted_at=None,
                                path=SUBS_FILE,
                            )
                        )
                        subs_changed = subs_changed or changed
                    except Exception as exc:
                        print(f"Failed to update subscription for {chat_id}: {exc}")
                    clear_conversation(state, chat_id)
                    try:
                        send_unsubscribe_confirmation(chat_id)
                    except Exception as exc:
                        print(f"Failed to send unsubscribe confirmation to {chat_id}: {exc}")
                    state_changed = True
                    continue

                contact_required_commands = {
                    "/menu",
                    "menu",
                    "/get",
                    "get",
                    "/add",
                    "add",
                    "/remove",
                    "remove",
                    "/donate",
                    "donate",
                }
                if text_lower in contact_required_commands:
                    handled, changed = ensure_contact_prompt(
                        existing, chat_id, command=command_name
                    )
                    subs_changed = subs_changed or changed
                    if handled:
                        continue

                if text_lower in ("/menu", "menu"):
                    try:
                        send_menu(chat_id)
                    except Exception as exc:
                        print(f"Failed to send menu to {chat_id}: {exc}")
                    continue

                if text_lower in ("/get", "get"):
                    send_summary_for_chat(chat_id)
                    continue

                if text_lower in ("/donate", "donate"):
                    state_changed = handle_donate_stars_start(state, chat_id) or state_changed
                    continue

                if text_lower in ("/add", "add"):
                    state_changed = handle_add_asset_start(state, chat_id) or state_changed
                    continue

                if text_lower in ("/remove", "remove"):
                    state_changed = handle_remove_asset_start(state, chat_id) or state_changed
                    continue

                if text_lower.startswith("/terms"):
                    try:
                        send_telegram(str(chat_id), TERMS_MESSAGE)
                    except Exception as exc:  # pragma: no cover
                        print(f"Failed to send terms message to {chat_id}: {exc}")
                    continue

                if text_lower.startswith("/paysupport"):
                    try:
                        send_telegram(str(chat_id), PAY_SUPPORT_MESSAGE)
                    except Exception as exc:  # pragma: no cover
                        print(f"Failed to send pay support message to {chat_id}: {exc}")
                    continue

                if text_lower in ("/help", "help"):
                    try:
                        send_telegram(str(chat_id), HELP_MESSAGE)
                    except Exception as exc:  # pragma: no cover
                        print(f"Failed to send help message to {chat_id}: {exc}")
                    continue

                if text_lower.startswith("/donations"):
                    if not is_admin(chat_id):
                        try:
                            send_telegram(str(chat_id), "دسترسی مجاز نیست.")
                        except Exception as exc:  # pragma: no cover
                            print(f"Failed to send unauthorized notice to {chat_id}: {exc}")
                    else:
                        handle_admin_donations(chat_id)
                    continue

                if text_lower.startswith("/refund"):
                    if not is_admin(chat_id):
                        try:
                            send_telegram(str(chat_id), "دسترسی مجاز نیست.")
                        except Exception as exc:  # pragma: no cover
                            print(f"Failed to send unauthorized refund notice to {chat_id}: {exc}")
                    else:
                        handle_refund_request(chat_id, raw_text)
                    continue

                if text_lower in ("/cancel", "cancel"):
                    if conversation:
                        clear_conversation(state, chat_id)
                        state_changed = True
                        try:
                            send_telegram(str(chat_id), "فرآیند جاری لغو شد.")
                        except Exception as exc:  # pragma: no cover
                            print(f"Failed to confirm cancel to {chat_id}: {exc}")
                    else:
                        try:
                            send_telegram(str(chat_id), "هیچ فرآیند فعالی برای لغو وجود ندارد.")
                        except Exception as exc:  # pragma: no cover
                            print(f"Failed to notify no-op cancel for {chat_id}: {exc}")
                    continue

                if conversation and conversation.get("state") == "await_donation_custom" and raw_text.strip():
                    state_changed = handle_donate_custom_text(state, chat_id, raw_text) or state_changed
                    continue

                if conversation and conversation.get("state") == "await_symbol" and raw_text.strip():
                    state_changed = handle_add_asset_text(state, chat_id, raw_text) or state_changed
                    continue

                if raw_text.strip():
                    try:
                        send_telegram(str(chat_id), UNKNOWN_COMMAND_MESSAGE)
                    except Exception as exc:  # pragma: no cover
                        print(f"Failed to send fallback reply to {chat_id}: {exc}")
            finally:
                if update_id is not None and (
                    last_processed_update_id is None or update_id > last_processed_update_id
                ):
                    last_processed_update_id = update_id

        if last_processed_update_id is not None:
            next_offset = last_processed_update_id + 1
            if next_offset != offset:
                offset = next_offset
                offset_changed = True

    def flush_changes() -> None:
        nonlocal state_changed, subs_changed, offset_changed
        if not (state_changed or subs_changed or offset_changed):
            return

        offset_was_updated = offset_changed
        if offset_changed:
            save_offset(OFFSET_FILE, offset)
            offset_changed = False
        if state_changed:
            save_state(state)
            state_changed = False
        if subs_changed or offset_was_updated:
            total = count_subscribers(path=SUBS_FILE)
            print(f"Updated subscribers: {total}, offset: {offset}")
            subs_changed = False

    try:
        if duration_seconds is None:
            poll_timeout_value = max(0.0, poll_timeout)
            while True:
                handle_batch(_fetch_updates(offset, poll_timeout_value))
                flush_changes()
        elif duration_seconds > 0:
            deadline = time.monotonic() + duration_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                timeout = min(poll_timeout, remaining)
                if timeout <= 0:
                    timeout = 0
                handle_batch(_fetch_updates(offset, timeout))
                flush_changes()
                if remaining <= 0:
                    break
        else:
            handle_batch(_fetch_updates(offset, 0))
    finally:
        flush_changes()



async def on_start(update, context):
    chat = getattr(update, "effective_chat", None)
    chat_id = getattr(chat, "id", None)
    if chat_id is None:
        return

    existing = get_subscriber(chat_id, path=SUBS_FILE)

    prompt_requests: list[bool] = []

    def _prompt(already_registered: bool) -> None:
        prompt_requests.append(bool(already_registered))

    handled, _ = ensure_contact_prompt(
        existing,
        chat_id,
        command="start",
        prompt_callback=_prompt,
    )

    if prompt_requests:
        already_registered = prompt_requests[-1]
        message = getattr(update, "message", None)
        if (
            message is not None
            and KeyboardButton is not None
            and ReplyKeyboardMarkup is not None
        ):
            keyboard = [[KeyboardButton("📱 ارسال شماره من", request_contact=True)]]
            reply_markup = ReplyKeyboardMarkup(
                keyboard,
                resize_keyboard=True,
                one_time_keyboard=True,
            )
            text = _compose_start_prompt_text(already_registered)
            try:
                await message.reply_text(text, reply_markup=reply_markup)
            except Exception:
                await asyncio.to_thread(
                    send_start_prompt,
                    chat_id,
                    already_registered=already_registered,
                )
        else:
            await asyncio.to_thread(
                send_start_prompt,
                chat_id,
                already_registered=already_registered,
            )
        return

    if handled:
        return

    await asyncio.to_thread(send_menu, chat_id)


def register_start_handler(app):
    app.add_handler(CommandHandler("start", on_start))


def main():  # pragma: no cover - retained for manual execution
    process_updates()


if __name__ == "__main__":
    main()
