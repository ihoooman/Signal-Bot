"""Utilities for resolving user-provided asset names into canonical trading pairs."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import requests

from .instruments import (
    SUPPORTED_QUOTES,
    get_default_pairs,
    get_instrument_by_pair,
    get_instrument_by_symbol,
    list_alias_tokens,
)

logger = logging.getLogger(__name__)

_SUPPORTED_PAIRS_PATH = Path(__file__).resolve().parents[2] / "data" / "supported_pairs.json"
_CACHE_TTL_SECONDS = 24 * 60 * 60


@dataclass(slots=True)
class _SupportedMarketData:
    binance_pairs: Dict[str, str]
    coingecko_tokens: Dict[str, str]
    coingecko_symbol_to_name: Dict[str, str]
    fuzzy_tokens: tuple[str, ...]


_SUPPORTED_MARKET_CACHE: _SupportedMarketData | None = None
_SUPPORTED_MARKET_CACHE_TS: float = 0.0

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


def _read_cached_pairs() -> dict[str, object] | None:
    try:
        with _SUPPORTED_PAIRS_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        logger.debug("resolve_instrument: failed to read supported pairs cache", exc_info=True)
        return None
    return payload if isinstance(payload, dict) else None


def _write_cached_pairs(data: dict[str, object]) -> None:
    try:
        _SUPPORTED_PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _SUPPORTED_PAIRS_PATH.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle)
        tmp_path.replace(_SUPPORTED_PAIRS_PATH)
    except OSError:
        logger.debug("resolve_instrument: failed to write supported pairs cache", exc_info=True)


def _fetch_binance_pairs() -> list[dict[str, object]]:
    try:
        response = requests.get(
            "https://api.binance.com/api/v3/exchangeInfo",
            params={"permissions": "SPOT"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        symbols = payload.get("symbols") if isinstance(payload, dict) else None
        if not isinstance(symbols, list):
            return []
        cleaned: list[dict[str, object]] = []
        for entry in symbols:
            if not isinstance(entry, dict):
                continue
            cleaned.append(
                {
                    "symbol": entry.get("symbol"),
                    "baseAsset": entry.get("baseAsset"),
                    "quoteAsset": entry.get("quoteAsset"),
                    "status": entry.get("status"),
                    "isSpotTradingAllowed": entry.get("isSpotTradingAllowed"),
                }
            )
        return cleaned
    except requests.RequestException:
        logger.debug("resolve_instrument: failed to fetch Binance exchangeInfo", exc_info=True)
    except ValueError:
        logger.debug("resolve_instrument: invalid JSON from Binance exchangeInfo", exc_info=True)
    return []


def _fetch_coingecko_list() -> list[dict[str, object]]:
    try:
        response = requests.get("https://api.coingecko.com/api/v3/coins/list", timeout=10)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return []
        cleaned: list[dict[str, object]] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            cleaned.append(
                {
                    "id": entry.get("id"),
                    "symbol": entry.get("symbol"),
                    "name": entry.get("name"),
                }
            )
        return cleaned
    except requests.RequestException:
        logger.debug("resolve_instrument: failed to fetch CoinGecko list", exc_info=True)
    except ValueError:
        logger.debug("resolve_instrument: invalid JSON from CoinGecko list", exc_info=True)
    return []


def _download_supported_pairs() -> dict[str, object] | None:
    binance = _fetch_binance_pairs()
    coingecko = _fetch_coingecko_list()
    if not binance and not coingecko:
        return None
    return {
        "fetched_at": time.time(),
        "binance": binance,
        "coingecko": coingecko,
    }


def _get_supported_pairs_data() -> dict[str, object]:
    cached = _read_cached_pairs()
    now = time.time()
    fetched_at: float = 0.0
    if cached:
        try:
            fetched_at = float(cached.get("fetched_at", 0.0))
        except (TypeError, ValueError):
            fetched_at = 0.0
    if cached and fetched_at and (now - fetched_at) <= _CACHE_TTL_SECONDS:
        return cached

    refreshed = _download_supported_pairs()
    if refreshed:
        _write_cached_pairs(refreshed)
        return refreshed

    if cached:
        return cached

    return {"fetched_at": now, "binance": [], "coingecko": []}


def _build_supported_market_data(raw: dict[str, object]) -> _SupportedMarketData:
    binance_pairs: Dict[str, str] = {}
    coingecko_tokens: Dict[str, str] = {}
    coingecko_symbol_to_name: Dict[str, str] = {}
    fuzzy_tokens: list[str] = []

    for entry in raw.get("binance", []) if isinstance(raw, dict) else []:
        if not isinstance(entry, dict):
            continue
        symbol = _normalise(str(entry.get("symbol") or ""))
        base_asset = _normalise(str(entry.get("baseAsset") or ""))
        quote_asset = _normalise(str(entry.get("quoteAsset") or ""))
        status = str(entry.get("status") or "").upper()
        spot_allowed = entry.get("isSpotTradingAllowed", True)
        if not symbol or quote_asset != "USDT":
            continue
        if status and status != "TRADING":
            continue
        if spot_allowed is False:
            continue
        binance_pairs[symbol] = symbol
        if symbol not in fuzzy_tokens:
            fuzzy_tokens.append(symbol)
        if base_asset:
            binance_pairs.setdefault(base_asset, symbol)
            if base_asset not in fuzzy_tokens:
                fuzzy_tokens.append(base_asset)

    coingecko_entries = raw.get("coingecko") if isinstance(raw, dict) else None
    if isinstance(coingecko_entries, list):
        for entry in coingecko_entries:
            if not isinstance(entry, dict):
                continue
            symbol = _normalise(str(entry.get("symbol") or ""))
            name = _normalise(str(entry.get("name") or ""))
            coin_id = _normalise(str(entry.get("id") or ""))
            if not symbol:
                continue
            coingecko_symbol_to_name[symbol] = str(entry.get("name") or symbol)
            for token in (symbol, name, coin_id):
                if not token:
                    continue
                coingecko_tokens.setdefault(token, symbol)
                if token not in fuzzy_tokens:
                    fuzzy_tokens.append(token)

    return _SupportedMarketData(
        binance_pairs=binance_pairs,
        coingecko_tokens=coingecko_tokens,
        coingecko_symbol_to_name=coingecko_symbol_to_name,
        fuzzy_tokens=tuple(fuzzy_tokens),
    )


def _load_supported_market_data() -> _SupportedMarketData:
    global _SUPPORTED_MARKET_CACHE, _SUPPORTED_MARKET_CACHE_TS
    raw = _get_supported_pairs_data()
    fetched_at = 0.0
    try:
        fetched_at = float(raw.get("fetched_at", 0.0))
    except (TypeError, ValueError):
        fetched_at = 0.0
    if _SUPPORTED_MARKET_CACHE and _SUPPORTED_MARKET_CACHE_TS == fetched_at:
        return _SUPPORTED_MARKET_CACHE
    support = _build_supported_market_data(raw)
    _SUPPORTED_MARKET_CACHE = support
    _SUPPORTED_MARKET_CACHE_TS = fetched_at
    return support


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


def _format_label(pair: str, support: _SupportedMarketData | None = None) -> str:
    inst = get_instrument_by_pair(pair)
    if inst:
        return f"{inst.symbol}/{inst.quote}"
    base, quote = _extract_quote(pair)
    if support and base:
        name = support.coingecko_symbol_to_name.get(base)
        if name and quote:
            return f"{name} ({base}/{quote})"
        if name:
            return name
    if base and quote:
        return f"{base}/{quote}"
    if len(pair) > 4:
        return f"{pair[:-4]}/{pair[-4:]}"
    return pair


def _resolve_with_market_support(
    token: str,
    base_token: str | None,
    explicit_quote: str | None,
    support: _SupportedMarketData,
) -> str | None:
    allow_usdt_default = explicit_quote in (None, "USDT")
    lookup_tokens: list[str] = []
    lookup_candidates = [token]
    if allow_usdt_default and base_token and base_token not in lookup_candidates:
        lookup_candidates.append(base_token)
    for candidate in lookup_candidates:
        if candidate and candidate not in lookup_tokens:
            lookup_tokens.append(candidate)

    for candidate in lookup_tokens:
        pair = support.binance_pairs.get(candidate)
        if pair:
            logger.debug("resolve_instrument: token=%s source=binance pair=%s", token, pair)
            return pair

    gecko_tokens: list[str] = []
    if token.endswith("USDT") and len(token) > 4:
        base_candidate = token[:-4]
        if base_candidate and base_candidate not in gecko_tokens:
            gecko_tokens.append(base_candidate)
    if allow_usdt_default and base_token and base_token not in gecko_tokens:
        gecko_tokens.append(base_token)
    if token not in gecko_tokens:
        gecko_tokens.append(token)

    for candidate in gecko_tokens:
        symbol = support.coingecko_tokens.get(candidate)
        if not symbol:
            continue
        pair = f"{symbol}USDT"
        logger.debug("resolve_instrument: token=%s source=coingecko pair=%s", token, pair)
        return pair

    return None


def _levenshtein_distance(left: str, right: str, max_distance: int = 2) -> int | None:
    if left == right:
        return 0
    if not left:
        return len(right) if len(right) <= max_distance else None
    if not right:
        return len(left) if len(left) <= max_distance else None
    if abs(len(left) - len(right)) > max_distance:
        return None
    if len(left) < len(right):
        left, right = right, left

    previous_row = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current_row = [i]
        row_min = current_row[0]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (0 if left_char == right_char else 1)
            best = min(insert_cost, delete_cost, replace_cost)
            current_row.append(best)
            if best < row_min:
                row_min = best
        if row_min > max_distance:
            return None
        previous_row = current_row
    distance = previous_row[-1]
    return distance if distance <= max_distance else None


def _suggest_pairs(
    token: str,
    base_token: str | None,
    explicit_quote: str | None,
    support: _SupportedMarketData,
) -> List[ResolutionCandidate]:
    tokens: set[str] = set(support.fuzzy_tokens)
    tokens.update(_normalise(alias) for alias in list_alias_tokens())
    tokens.update(_normalise(pair) for pair in get_default_pairs())
    results: list[tuple[int, str, str]] = []
    seen_pairs: set[str] = set()

    for candidate in tokens:
        if not candidate or candidate == token:
            continue
        distance = _levenshtein_distance(token, candidate)
        if distance is None:
            continue
        pair = support.binance_pairs.get(candidate)
        if not pair:
            symbol = support.coingecko_tokens.get(candidate)
            if symbol:
                pair = f"{symbol}USDT"
            else:
                alias_symbol = _alias_for(candidate)
                if alias_symbol:
                    pair = _resolve_pair_from_components(alias_symbol, explicit_quote)
                else:
                    inst = get_instrument_by_pair(candidate)
                    pair = inst.pair if inst else None
        if not pair or pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        results.append((distance, candidate, pair))

    results.sort(key=lambda entry: (entry[0], entry[1]))
    return [
        ResolutionCandidate(pair=pair, label=_format_label(pair, support))
        for _, _, pair in results
    ]


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

    support = _load_supported_market_data()
    pair = _resolve_with_market_support(token, base_token, explicit_quote, support)
    if pair:
        return ResolutionResult(status="resolved", pair=pair)

    candidates = _suggest_pairs(token, base_token, explicit_quote, support)
    if candidates:
        logger.debug(
            "resolve_instrument: token=%s fallback=fuzzy suggestions=%s",
            token,
            [candidate.pair for candidate in candidates],
        )
        if len(candidates) == 1:
            return ResolutionResult(status="resolved", pair=candidates[0].pair)
        return ResolutionResult(status="ambiguous", candidates=candidates[:3])

    return ResolutionResult(status="error", reason="NOT_SUPPORTED")


__all__ = [
    "ResolutionCandidate",
    "ResolutionResult",
    "resolve_instrument",
]
