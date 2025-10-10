"""Utilities for resolving user-provided asset names into canonical trading pairs."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import List

from .instruments import (
    SUPPORTED_QUOTES,
    get_default_pairs,
    get_instrument_by_pair,
    get_instrument_by_symbol,
    list_alias_tokens,
)

_ALIAS_MAP = {
    "XRP": "XRP",
    "RIPPLE": "XRP",
    "BTC": "BTC",
    "BITCOIN": "BTC",
    "ETH": "ETH",
    "ETHER": "ETH",
    "ETHEREUM": "ETH",
    "SOL": "SOL",
    "SOLANA": "SOL",
}


@dataclass(slots=True)
class ResolutionCandidate:
    pair: str
    label: str


@dataclass(slots=True)
class ResolutionResult:
    status: str
    pair: str | None = None
    candidates: List[ResolutionCandidate] | None = None
    reason: str | None = None


def _normalise(text: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "", text or "")
    return cleaned.upper()


def _alias_for(token: str) -> str | None:
    if not token:
        return None
    if token in _ALIAS_MAP:
        return _ALIAS_MAP[token]
    inst = get_instrument_by_symbol(token)
    return inst.symbol if inst else None


def _resolve_pair_from_components(base: str | None, quote: str | None) -> str | None:
    if not base:
        return None
    quote = quote or "USDT"
    candidate = f"{base}{quote}".upper()
    inst = get_instrument_by_pair(candidate)
    return inst.pair if inst else None


def _extract_quote(token: str) -> tuple[str | None, str | None]:
    for quote in SUPPORTED_QUOTES:
        if token.endswith(quote):
            base_token = token[: -len(quote)]
            if base_token:
                return base_token, quote
    return token, None


def _format_label(pair: str) -> str:
    inst = get_instrument_by_pair(pair)
    if not inst:
        return pair
    return f"{inst.symbol}/{inst.quote}"


def resolve_instrument(user_input: str) -> ResolutionResult:
    if not user_input or not user_input.strip():
        return ResolutionResult(status="error", reason="EMPTY_INPUT")

    token = _normalise(user_input)
    if not token:
        return ResolutionResult(status="error", reason="EMPTY_INPUT")

    inst = get_instrument_by_pair(token)
    if inst:
        return ResolutionResult(status="resolved", pair=inst.pair)

    base_token, explicit_quote = _extract_quote(token)
    base_symbol = _alias_for(base_token)
    if base_symbol:
        pair = _resolve_pair_from_components(base_symbol, explicit_quote)
        if pair:
            return ResolutionResult(status="resolved", pair=pair)

    alias_symbol = _alias_for(token)
    if alias_symbol:
        pair = _resolve_pair_from_components(alias_symbol, explicit_quote)
        if pair:
            return ResolutionResult(status="resolved", pair=pair)

    # Fuzzy search using aliases and known pairs
    all_tokens: List[str] = list(dict.fromkeys(list_alias_tokens() + list(get_default_pairs())))
    matches = difflib.get_close_matches(token, all_tokens, n=3, cutoff=0.5)

    candidates: List[ResolutionCandidate] = []
    seen: set[str] = set()
    for match in matches:
        base_sym = _alias_for(match)
        if base_sym:
            pair = _resolve_pair_from_components(base_sym, explicit_quote)
        else:
            inst = get_instrument_by_pair(match)
            pair = inst.pair if inst else None
        if not pair or pair in seen:
            continue
        seen.add(pair)
        candidates.append(ResolutionCandidate(pair=pair, label=_format_label(pair)))

    if len(candidates) == 1:
        return ResolutionResult(status="resolved", pair=candidates[0].pair)

    if candidates:
        return ResolutionResult(status="ambiguous", candidates=candidates[:3])

    return ResolutionResult(status="error", reason="NOT_SUPPORTED")


__all__ = [
    "ResolutionCandidate",
    "ResolutionResult",
    "resolve_instrument",
]
