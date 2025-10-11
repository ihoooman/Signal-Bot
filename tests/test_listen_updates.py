import asyncio
import unittest

import listen_start
import listen_updates


class DummyCommandHandler:
    def __init__(self, name, callback):
        self.name = name
        self.callback = callback


class DummyTypeHandler:
    def __init__(self, update_cls, callback):
        self.update_cls = update_cls
        self.callback = callback


class DummyMessageHandler:
    def __init__(self, filter_value, callback):
        self.filter_value = filter_value
        self.callback = callback


class DummyApp:
    def __init__(self):
        self.calls = []

    def add_handler(self, handler, group=None):
        self.calls.append((handler, group))


class ListenUpdatesTests(unittest.TestCase):
    def test_register_handlers_order(self):
        app = DummyApp()
        dispatch = object()

        listen_updates._register_handlers(
            app,
            dispatch,
            command_handler_cls=DummyCommandHandler,
            message_handler_cls=DummyMessageHandler,
            type_handler_cls=DummyTypeHandler,
            command_filter="COMMAND",
            update_cls="Update",
        )

        self.assertEqual(len(app.calls), 3)
        first, second, third = app.calls
        self.assertIsInstance(first[0], DummyCommandHandler)
        self.assertEqual(first[0].name, "debug")
        self.assertIs(second[0].callback, dispatch)
        self.assertEqual(second[1], 1)
        self.assertIsInstance(third[0], DummyMessageHandler)
        self.assertEqual(third[0].filter_value, "COMMAND")
        self.assertEqual(third[1], 2)

    def test_on_unknown_replies_with_default_message(self):
        class DummyMessage:
            def __init__(self):
                self.replies = []

            async def reply_text(self, text):
                self.replies.append(text)

        class DummyUpdate:
            def __init__(self):
                self.effective_message = DummyMessage()
                self.effective_chat = type("Chat", (), {"id": 42})()

        update = DummyUpdate()
        asyncio.run(listen_updates.on_unknown(update, object()))

        self.assertEqual(update.effective_message.replies, [listen_start.UNKNOWN_COMMAND_MESSAGE])


if __name__ == "__main__":
    unittest.main()
