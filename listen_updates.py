#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Long-polling runner that reuses ``listen_start`` helpers with persistence."""

import asyncio
import logging
import threading
import weakref
from typing import Optional

import listen_start

__all__ = ["load_offset", "save_offset", "main", "run_polling_main"]

LOGGER = logging.getLogger("signal_bot.listen_updates")
REGISTER_LOGGER = logging.getLogger("register")

KNOWN_COMMANDS = frozenset({"/start", "/menu", "/get", "/add", "/remove", "/debug"})
_REGISTERED_APPS = weakref.WeakSet()


async def on_unknown(update, context) -> None:
    message = getattr(update, "effective_message", None)
    if message is None:
        return
    text = getattr(message, "text", None)
    if isinstance(text, str):
        command = text.strip().split()[0] if text.strip() else ""
        if command:
            normalized = command.split("@", 1)[0].lower()
            if normalized in KNOWN_COMMANDS:
                LOGGER.debug(
                    "Ignoring known command in unknown handler: command=%s", command
                )
                return
    chat = getattr(update, "effective_chat", None)
    LOGGER.info("Routing unknown command for chat_id=%s", getattr(chat, "id", None))
    await message.reply_text(listen_start.UNKNOWN_COMMAND_MESSAGE)


def load_offset() -> Optional[int]:
    """Read the persisted update offset if available."""
    legacy_candidate = (
        listen_start.LEGACY_OFFSET_FILE
        if listen_start.LEGACY_OFFSET_FILE != listen_start.OFFSET_FILE
        and listen_start.LEGACY_OFFSET_FILE.exists()
        else None
    )
    value = listen_start.load_offset(listen_start.OFFSET_FILE, legacy_path=legacy_candidate)
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def save_offset(value: int) -> None:
    """Persist the provided update offset to disk."""
    listen_start.save_offset(listen_start.OFFSET_FILE, int(value))


def main(stop_event: Optional[threading.Event] = None, poll_timeout: float = 10.0) -> None:
    """Start the Telegram long-polling loop.

    When ``stop_event`` is provided, the loop checks for graceful shutdown
    between polling windows so Render can terminate the process cleanly.
    """

    poll_timeout_value = max(0.0, float(poll_timeout))

    try:
        if stop_event is None:
            listen_start.process_updates(poll_timeout=poll_timeout_value)
            return

        while not stop_event.is_set():
            listen_start.process_updates(
                duration_seconds=poll_timeout_value,
                poll_timeout=poll_timeout_value,
            )
    except SystemExit:
        raise
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        pass


if __name__ == "__main__":  # pragma: no cover - retained for scripts
    main()


class _LegacyPollingWorker:
    """Sequential dispatcher that mirrors ``listen_start.process_updates`` behaviour."""

    def __init__(self) -> None:
        legacy_candidate = (
            listen_start.LEGACY_OFFSET_FILE
            if listen_start.LEGACY_OFFSET_FILE != listen_start.OFFSET_FILE
            and listen_start.LEGACY_OFFSET_FILE.exists()
            else None
        )
        self._offset = listen_start.load_offset(
            listen_start.OFFSET_FILE, legacy_path=legacy_candidate
        )
        self._last_processed_update_id = self._offset - 1 if self._offset > 0 else None
        self._state = listen_start.load_state()
        self._state_changed = False
        self._subs_changed = False
        self._offset_changed = False

    async def process_update(self, update) -> None:
        if update is None:
            return
        await asyncio.to_thread(self._process_update_dict, update.to_dict())

    def _process_update_dict(self, upd: dict) -> None:
        if not isinstance(upd, dict):
            return

        update_id = upd.get("update_id")
        if (
            update_id is not None
            and self._last_processed_update_id is not None
            and update_id <= self._last_processed_update_id
        ):
            return

        try:
            pre_checkout = upd.get("pre_checkout_query")
            if pre_checkout:
                listen_start.handle_pre_checkout_query(pre_checkout)
                return

            callback = upd.get("callback_query")
            if callback:
                self._handle_callback(callback)
                return

            message = upd.get("message") or upd.get("channel_post")
            if message:
                self._handle_message(message)
        finally:
            if update_id is not None and (
                self._last_processed_update_id is None
                or update_id > self._last_processed_update_id
            ):
                self._last_processed_update_id = update_id
                next_offset = update_id + 1
                if next_offset != self._offset:
                    self._offset = next_offset
                    self._offset_changed = True
            self._flush_changes()

    def _handle_callback(self, callback: dict) -> None:
        callback_id = callback.get("id")
        try:
            listen_start.answer_callback(callback_id or "")
        except Exception as exc:  # pragma: no cover - network reliability
            print(f"Failed to answer callback {callback_id}: {exc}")

        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if not chat_id:
            return

        data = (callback.get("data") or "").strip()
        existing = listen_start.get_subscriber(chat_id, path=listen_start.SUBS_FILE)
        requires_contact = data != "get_updates_now"
        if requires_contact:
            handled, changed = listen_start.ensure_contact_prompt(existing, chat_id)
            if changed:
                self._subs_changed = True
            if handled:
                return

        if data == "get_updates_now":
            listen_start.handle_get_updates_now(chat_id)
            return
        if data == "add_asset_start":
            self._state_changed = (
                listen_start.handle_add_asset_start(self._state, chat_id)
                or self._state_changed
            )
            return
        if data == "remove_asset_start":
            self._state_changed = (
                listen_start.handle_remove_asset_start(self._state, chat_id)
                or self._state_changed
            )
            return
        if data == "donate_stars_start":
            self._state_changed = (
                listen_start.handle_donate_stars_start(self._state, chat_id)
                or self._state_changed
            )
            return
        if data.startswith("donate:tier:"):
            try:
                amount = int(data.split(":", 2)[2])
            except ValueError:
                amount = 0
            self._state_changed = (
                listen_start.handle_donate_tier(self._state, chat_id, amount)
                or self._state_changed
            )
            return
        if data == "donate:custom":
            self._state_changed = (
                listen_start.handle_donate_custom_prompt(self._state, chat_id)
                or self._state_changed
            )
            return
        if data.startswith("remove_asset:page:"):
            try:
                page = int(data.rsplit(":", 1)[1])
            except ValueError:
                page = 0
            self._state_changed = (
                listen_start.handle_remove_asset_page(self._state, chat_id, page)
                or self._state_changed
            )
            return
        if data.startswith("remove_asset:pick:"):
            pair = data.split(":", 2)[2]
            self._state_changed = (
                listen_start.handle_remove_asset_pick(self._state, chat_id, pair)
                or self._state_changed
            )
            return
        if data.startswith("remove_asset:confirm:"):
            parts = data.split(":")
            if len(parts) == 4:
                _, _, pair, decision = parts
                self._state_changed = (
                    listen_start.handle_remove_asset_confirm(
                        self._state, chat_id, pair, decision
                    )
                    or self._state_changed
                )
            return
        if data.startswith("addasset:pick:"):
            pair = data.split(":", 2)[2]
            self._state_changed = (
                listen_start.handle_add_asset_pick(self._state, chat_id, pair)
                or self._state_changed
            )
            return
        try:
            listen_start.send_telegram(str(chat_id), "دستور ناشناخته است.")
        except Exception as exc:  # pragma: no cover - network reliability
            print(f"Failed to notify unknown callback for {chat_id}: {exc}")

    def _handle_message(self, message: dict) -> None:
        chat = message.get("chat", {}) or {}
        chat_id = chat.get("id")
        if not chat_id:
            return

        existing = listen_start.get_subscriber(chat_id, path=listen_start.SUBS_FILE)
        conversation = listen_start.get_conversation(self._state, chat_id)

        payment = message.get("successful_payment")
        if payment:
            self._state_changed = (
                listen_start.handle_successful_payment(self._state, chat_id, payment)
                or self._state_changed
            )
            return

        contact = message.get("contact")
        if contact:
            self._handle_contact(chat_id, message, contact, existing)
            return

        raw_text = message.get("text") or ""
        text_lower = raw_text.strip().lower()

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
            "/start",
            "start"
        }
        if text_lower in contact_required_commands:
            handled, changed = listen_start.ensure_contact_prompt(existing, chat_id)
            if changed:
                self._subs_changed = True
            if handled:
                return

        if text_lower in ("/menu", "menu"):
            self._send_menu(chat_id)
            return
        if text_lower in ("/get", "get"):
            listen_start.send_summary_for_chat(chat_id)
            return
        if text_lower in ("/donate", "donate"):
            self._state_changed = (
                listen_start.handle_donate_stars_start(self._state, chat_id)
                or self._state_changed
            )
            return
        if text_lower in ("/add", "add"):
            self._state_changed = (
                listen_start.handle_add_asset_start(self._state, chat_id)
                or self._state_changed
            )
            return
        if text_lower in ("/remove", "remove"):
            self._state_changed = (
                listen_start.handle_remove_asset_start(self._state, chat_id)
                or self._state_changed
            )
            return
        if text_lower.startswith("/terms"):
            self._send_text(chat_id, listen_start.TERMS_MESSAGE)
            return
        if text_lower.startswith("/paysupport"):
            self._send_text(chat_id, listen_start.PAY_SUPPORT_MESSAGE)
            return
        if text_lower in ("/help", "help"):
            self._send_text(chat_id, listen_start.HELP_MESSAGE)
            return
        if text_lower.startswith("/donations"):
            if not listen_start.is_admin(chat_id):
                self._send_text(chat_id, "دسترسی مجاز نیست.")
            else:
                listen_start.handle_admin_donations(chat_id)
            return
        if text_lower.startswith("/refund"):
            if not listen_start.is_admin(chat_id):
                self._send_text(chat_id, "دسترسی مجاز نیست.")
            else:
                listen_start.handle_refund_request(chat_id, raw_text)
            return
        if text_lower in ("/cancel", "cancel"):
            if conversation:
                listen_start.clear_conversation(self._state, chat_id)
                self._state_changed = True
                self._send_text(chat_id, "فرآیند جاری لغو شد.")
            else:
                self._send_text(chat_id, "هیچ فرآیند فعالی برای لغو وجود ندارد.")
            return
        if (
            conversation
            and conversation.get("state") == "await_donation_custom"
            and raw_text.strip()
        ):
            self._state_changed = (
                listen_start.handle_donate_custom_text(
                    self._state, chat_id, raw_text
                )
                or self._state_changed
            )
            return
        if conversation and conversation.get("state") == "await_symbol" and raw_text.strip():
            self._state_changed = (
                listen_start.handle_add_asset_text(self._state, chat_id, raw_text)
                or self._state_changed
            )
            return
        if raw_text.strip():
            self._send_text(
                chat_id,
                "دستور ناشناخته است. برای مشاهده گزینه‌ها /menu را ارسال کنید.",
            )

    def _handle_contact(
        self, chat_id, message: dict, contact: dict, existing: Optional[dict]
    ) -> None:
        contact_user_id = contact.get("user_id")
        if contact_user_id and str(contact_user_id) != str(chat_id):
            return
        phone = contact.get("phone_number")
        if not phone:
            return
        sender = message.get("from", {}) or {}
        first_name = (
            contact.get("first_name")
            or sender.get("first_name")
            or ""
        )
        last_name = contact.get("last_name") or sender.get("last_name") or ""
        username = sender.get("username") or ""
        try:
            changed, _ = listen_start.upsert_subscriber(
                chat_id,
                phone_number=phone,
                first_name=first_name,
                last_name=last_name,
                username=username,
                is_subscribed=True,
                awaiting_contact=False,
                contact_prompted_at=None,
                path=listen_start.SUBS_FILE,
            )
            if changed:
                self._subs_changed = True
        except Exception as exc:  # pragma: no cover - persistence errors
            print(f"Failed to save contact for {chat_id}: {exc}")
        listen_start.clear_conversation(self._state, chat_id)
        try:
            listen_start.send_contact_confirmation(chat_id)
            listen_start.send_menu(
                chat_id,
                prepend_text="اشتراک شما فعال است؛ گزینه‌های زیر در دسترس هستند.",
            )
        except Exception as exc:  # pragma: no cover - network reliability
            print(f"Failed to send confirmation/menu to {chat_id}: {exc}")
        self._state_changed = True

    def _send_menu(self, chat_id) -> None:
        try:
            listen_start.send_menu(chat_id)
        except Exception as exc:  # pragma: no cover - network reliability
            print(f"Failed to send menu to {chat_id}: {exc}")

    def _send_text(self, chat_id, text: str) -> None:
        try:
            listen_start.send_telegram(str(chat_id), text)
        except Exception as exc:  # pragma: no cover - network reliability
            print(f"Failed to send message to {chat_id}: {exc}")

    def _flush_changes(self) -> None:
        if not (self._state_changed or self._subs_changed or self._offset_changed):
            return

        offset_was_updated = self._offset_changed
        if self._offset_changed:
            listen_start.save_offset(listen_start.OFFSET_FILE, self._offset)
            self._offset_changed = False
        if self._state_changed:
            listen_start.save_state(self._state)
            self._state_changed = False
        if self._subs_changed or offset_was_updated:
            total = listen_start.count_subscribers(path=listen_start.SUBS_FILE)
            print(f"Updated subscribers: {total}, offset: {self._offset}")
            self._subs_changed = False


def _register_handlers(
    app,
    dispatch_callback,
    *,
    command_handler_cls=None,
    message_handler_cls=None,
    type_handler_cls=None,
    command_filter=None,
    update_cls=None,
):
    if getattr(app, "_handlers_registered", False) or app in _REGISTERED_APPS:
        LOGGER.debug("Skipping handler registration on %r; already configured", app)
        return

    if (
        command_handler_cls is None
        or message_handler_cls is None
        or type_handler_cls is None
        or command_filter is None
        or update_cls is None
    ):
        from telegram import Update
        from telegram.ext import CommandHandler, MessageHandler, TypeHandler, filters

        command_handler_cls = CommandHandler
        message_handler_cls = MessageHandler
        type_handler_cls = TypeHandler
        command_filter = filters.COMMAND
        update_cls = Update

    app.add_handler(command_handler_cls("debug", listen_start.debug_info))
    REGISTER_LOGGER.info("/debug handler registered")
    app.add_handler(type_handler_cls(update_cls, dispatch_callback), group=1)
    REGISTER_LOGGER.info("/start handler registered")
    REGISTER_LOGGER.info("/menu handler registered")
    REGISTER_LOGGER.info("/get handler registered")
    app.add_handler(message_handler_cls(command_filter, on_unknown), group=2)
    REGISTER_LOGGER.info("unknown handler registered LAST")

    try:
        setattr(app, "_handlers_registered", True)
    except AttributeError:
        try:
            _REGISTERED_APPS.add(app)
        except TypeError:
            LOGGER.debug(
                "Unable to track handler registration for %r; unsupported weakref",
                app,
            )


def run_polling_main() -> None:
    """Start the python-telegram-bot polling loop."""

    try:
        from telegram import Update
        from telegram.ext import ApplicationBuilder, ContextTypes
    except ModuleNotFoundError as exc:  # pragma: no cover - missing dependency
        raise RuntimeError("python-telegram-bot is required for polling") from exc

    from trigger_xrp_bot import get_bot_token

    worker = _LegacyPollingWorker()

    async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = getattr(update, "effective_message", None)
        text = getattr(message, "text", None) if message else None
        if isinstance(text, str):
            stripped = text.strip()
            if stripped.lower().startswith("/debug"):
                return
            command = stripped.split()[0].split("@", 1)[0].lower()
            if command == "/start":
                return
        await worker.process_update(update)

    app = (
        ApplicationBuilder()
        .token(get_bot_token())
        .concurrent_updates(False)
        .build()
    )
    _register_handlers(app, _dispatch)
    import listen_start

    listen_start.register_start_handler(app)
    app.run_polling(drop_pending_updates=True, stop_signals=None)
