import json
import sys
import tempfile
import time
from pathlib import Path
import unittest
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import signal_bot.services.symbol_resolver as symbol_resolver
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

    def test_resolves_from_cached_binance_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "supported_pairs.json"
            cache_payload = {
                "fetched_at": time.time(),
                "binance": [
                    {
                        "symbol": "FARTUSDT",
                        "baseAsset": "FART",
                        "quoteAsset": "USDT",
                        "status": "TRADING",
                        "isSpotTradingAllowed": True,
                    }
                ],
                "coingecko": [],
            }
            cache_path.write_text(json.dumps(cache_payload))

            with mock.patch.object(symbol_resolver, "_SUPPORTED_PAIRS_PATH", cache_path):
                symbol_resolver._SUPPORTED_MARKET_CACHE = None
                symbol_resolver._SUPPORTED_MARKET_CACHE_TS = 0.0
                result = resolve_instrument("fart")

            symbol_resolver._SUPPORTED_MARKET_CACHE = None
            symbol_resolver._SUPPORTED_MARKET_CACHE_TS = 0.0

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.pair, "FARTUSDT")


if __name__ == "__main__":
    unittest.main()
