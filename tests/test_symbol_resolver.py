import sys
from pathlib import Path
import unittest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from signal_bot.services.symbol_resolver import resolve_instrument


class SymbolResolverTests(unittest.TestCase):
    def test_alias_resolves_to_default_quote(self):
        result = resolve_instrument(" sol ")
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.pair, "SOLUSDT")

    def test_explicit_quote_is_preserved(self):
        result = resolve_instrument("solusdc")
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.pair, "SOLUSDC")

    def test_unknown_symbol_returns_error(self):
        result = resolve_instrument("unknowncoin")
        self.assertEqual(result.status, "error")
        self.assertEqual(result.reason, "NOT_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
