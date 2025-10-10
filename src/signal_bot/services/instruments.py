"""Instrument definitions and helpers for the Signal Bot."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence

SUPPORTED_QUOTES: Sequence[str] = ("USDT", "USDC", "USD")


@dataclass(frozen=True)
class Instrument:
    """Represents a tradable instrument supported by the analysis pipeline."""

    pair: str
    symbol: str
    display_name: str
    quote: str = "USDT"
    aliases: tuple[str, ...] = ()

    def with_quote(self, quote: str) -> "Instrument":
        return Instrument(
            pair=f"{self.symbol}{quote}",
            symbol=self.symbol,
            display_name=self.display_name,
            quote=quote,
            aliases=self.aliases,
        )


_DEFAULT_INSTRUMENTS: Sequence[Instrument] = (
    Instrument("XRPUSDT", "XRP", "XRP", aliases=("RIPPLE",)),
    Instrument("BTCUSDT", "BTC", "BTC", aliases=("BITCOIN",)),
    Instrument("ETHUSDT", "ETH", "ETH", aliases=("ETHEREUM", "ETHER")),
    Instrument("SOLUSDT", "SOL", "SOL", aliases=("SOLANA",)),
)


@lru_cache(maxsize=1)
def get_supported_instruments() -> List[Instrument]:
    """Return the list of base instruments supported by the analysis pipeline."""

    return [Instrument(inst.pair, inst.symbol, inst.display_name, inst.quote, inst.aliases) for inst in _DEFAULT_INSTRUMENTS]


@lru_cache(maxsize=1)
def get_default_pairs() -> List[str]:
    """Return the canonical default instrument pairs for broadcasts."""

    return [inst.pair for inst in _DEFAULT_INSTRUMENTS]


def _build_alias_index() -> dict[str, Instrument]:
    mapping: dict[str, Instrument] = {}
    for inst in get_supported_instruments():
        mapping[inst.symbol.upper()] = inst
        mapping[inst.display_name.upper()] = inst
        for alias in inst.aliases:
            mapping[alias.upper()] = inst
    return mapping


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, Instrument]:
    return _build_alias_index()


def list_alias_tokens() -> List[str]:
    return list(_alias_index().keys())


def get_instrument_by_symbol(symbol: str) -> Instrument | None:
    return _alias_index().get(symbol.upper())


def get_instrument_by_pair(pair: str) -> Instrument | None:
    pair_up = pair.upper()
    base_index = {inst.symbol.upper(): inst for inst in get_supported_instruments()}
    direct = next((inst for inst in get_supported_instruments() if inst.pair.upper() == pair_up), None)
    if direct:
        return direct

    for symbol, inst in base_index.items():
        if not pair_up.startswith(symbol):
            continue
        quote = pair_up[len(symbol) :]
        if not quote:
            return inst
        if quote not in SUPPORTED_QUOTES:
            continue
        return inst.with_quote(quote)
    return None


def iter_supported_pairs() -> Iterable[str]:
    for inst in get_supported_instruments():
        yield inst.pair
