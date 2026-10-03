import json
from datetime import datetime, timezone

import requests

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_URL = "https://clob.polymarket.com/book"

MIN_VOLUME_24H = 1000.0
MARKET_LIMIT = 100
MAX_PAGES = 20
MAX_HOURS_TO_END = 24

CRYPTO_KEYWORDS = (
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "solana",
    "sol",
)


def to_float(value):
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_json_list(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []

    return value if isinstance(value, list) else []


def parse_datetime(value):
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    except (TypeError, ValueError):
        return None


def get_volume_24h(market):
    return to_float(
        market.get(
            "volume24hrClob",
            market.get("volume24hr", 0),
        )
    )


def is_crypto_market(market):
    question = str(market.get("question", "")).lower()
    slug = str(market.get("slug", "")).lower()

    searchable = f"{question} {slug}"

    has_btc = (
        "bitcoin" in searchable
        or "btc" in searchable
    )

    has_up_down = (
        "up or down" in searchable
        or "up-down" in searchable
        or "up_down" in searchable
    )

    return has_btc and has_up_down


def is_short_crypto_market(market, now):
    if not is_crypto_market(market):
        return False, "not crypto"

    end_time = parse_datetime(market.get("endDate"))

    if end_time is None:
        return False, "missing/invalid endDate"

    hours_left = (
        end_time - now
    ).total_seconds() / 3600

    if hours_left <= 0:
        return False, "already ended"

    if hours_left > MAX_HOURS_TO_END:
        return False, "ends after 24h"

    return True, hours_left


def fetch_markets(session):
    markets = []

    for page in range(MAX_PAGES):
        response = session.get(
            GAMMA_URL,
            params={
                "active": "true",
                "closed": "false",
                "order": "volume24hr",
                "ascending": "false",
                "limit": MARKET_LIMIT,
                "offset": page * MARKET_LIMIT,
            },
            timeout=20,
        )

        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        markets.extend(batch)

        print(
            f"Page {page + 1}: fetched {len(batch)} markets"
        )

        if len(batch) < MARKET_LIMIT:
            break

    return markets


def get_book(token_id, session):
    response = session.get(
        CLOB_URL,
        params={"token_id": str(token_id)},
        timeout=15,
    )

    response.raise_for_status()

    return response.json()


def inspect_book(book):
    bids = [
        (
            to_float(level.get("price")),
            to_float(level.get("size")),
        )
        for level in book.get("bids", [])
        if to_float(level.get("size")) > 0
    ]

    asks = [
        (
            to_float(level.get("price")),
            to_float(level.get("size")),
        )
        for level in book.get("asks", [])
        if to_float(level.get("size")) > 0
    ]

    if not bids or not asks:
        return None

    best_bid = max(price for price, _ in bids)
    best_ask = min(price for price, _ in asks)

    if best_bid >= best_ask:
        return None

    bid_depth_5 = sum(
        size
        for _, size in sorted(
            bids,
            key=lambda x: x[0],
            reverse=True,
        )[:5]
    )

    ask_depth_5 = sum(
        size
        for _, size in sorted(
            asks,
            key=lambda x: x[0],
        )[:5]
    )

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": best_ask - best_bid,
        "mid": (best_bid + best_ask) / 2,
        "bid_depth_5": bid_depth_5,
        "ask_depth_5": ask_depth_5,
    }


def main():
    session = requests.Session()
    now = datetime.now(timezone.utc)

    print("Scanning Polymarket markets...")
    print(f"Minimum 24h volume: ${MIN_VOLUME_24H:,.0f}")
    print(f"Maximum time to end: {MAX_HOURS_TO_END} hours")
    print(f"Maximum pages: {MAX_PAGES}")

    try:
        markets = fetch_markets(session)
    except requests.RequestException as exc:
        print("Failed to fetch markets:", exc)
        return

    print("\nTotal markets fetched:", len(markets))

    candidates = []
    rejected_crypto = 0
    rejected_time = 0
    rejected_volume = 0

    for market in markets:
        if not is_crypto_market(market):
            rejected_crypto += 1
            continue

        matched, result = is_short_crypto_market(market, now)

        if not matched:
            rejected_time += 1
            continue

        volume_24h = get_volume_24h(market)

        if volume_24h < MIN_VOLUME_24H:
            rejected_volume += 1
            continue

        candidates.append({
            "market": market,
            "hours_left": result,
            "volume_24h": volume_24h,
        })

    print("\nFILTER SUMMARY")
    print("Crypto markets rejected:", rejected_crypto)
    print("Crypto markets rejected by time:", rejected_time)
    print("Rejected by volume:", rejected_volume)
    print("Short crypto candidates:", len(candidates))

    results = []

    for candidate in candidates:
        market = candidate["market"]

        question = market.get("question", "Unknown")
        tokens = parse_json_list(
            market.get("clobTokenIds")
        )
        outcomes = parse_json_list(
            market.get("outcomes")
        )

        for index, token_id in enumerate(tokens):
            try:
                book = get_book(token_id, session)
                stats = inspect_book(book)

                if stats is None:
                    print(
                        f"No valid two-sided book: {question}"
                    )
                    continue

                outcome = (
                    outcomes[index]
                    if index < len(outcomes)
                    else f"Outcome {index}"
                )

                results.append({
                    "question": question,
                    "slug": market.get("slug", ""),
                    "outcome": outcome,
                    "token_id": str(token_id),
                    "end_date": market.get("endDate"),
                    "hours_left": candidate["hours_left"],
                    "volume_24h": candidate["volume_24h"],
                    "liquidity": to_float(
                        market.get("liquidity")
                    ),
                    **stats,
                })

            except (
                requests.RequestException,
                ValueError,
                TypeError,
            ) as exc:
                print(f"Skipped token {token_id}: {exc}")

    results.sort(
        key=lambda item: (
            -item["volume_24h"],
            item["spread"],
            -(
                item["bid_depth_5"]
                + item["ask_depth_5"]
            ),
        )
    )

    print("\n" + "=" * 70)
    print("VALID ORDER BOOKS:", len(results))
    print("=" * 70)

    for index, item in enumerate(results[:10], start=1):
        print("\n" + "-" * 70)
        print(f"#{index} | Outcome: {item['outcome']}")
        print("Market:", item["question"])
        print("Slug:", item["slug"])
        print("Token ID:", item["token_id"])
        print("End date:", item["end_date"])
        print(f"Ends in: {item['hours_left']:.2f} hours")
        print(f"24h Volume: ${item['volume_24h']:,.2f}")
        print(f"Liquidity: ${item['liquidity']:,.2f}")

        print(
            f"Bid: {item['best_bid']:.4f} | "
            f"Ask: {item['best_ask']:.4f} | "
            f"Spread: {item['spread']:.4f} | "
            f"Mid: {item['mid']:.4f}"
        )

        print(
            f"Top-5 depth — "
            f"Bid: {item['bid_depth_5']:.2f} | "
            f"Ask: {item['ask_depth_5']:.2f}"
        )


if __name__ == "__main__":
    main()