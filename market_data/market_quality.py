import asyncio
import json
import requests
import websockets

from datetime import datetime, timezone

from market_data.scan_active_markets import (
    fetch_markets,
    is_crypto_market,
    is_short_crypto_market,
    get_volume_24h,
    parse_json_list,
    MIN_VOLUME_24H,
)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
SCAN_LIMIT = 10
INSPECTION_SECONDS = 30


async def inspect_market(token_id, duration=INSPECTION_SECONDS):
    """Collect valid two-sided BBO observations from the market WebSocket."""
    from market_data.live_orderbook import LiveOrderBook

    orderbook = LiveOrderBook(token_id)

    bids = []
    asks = []
    mids = []
    spreads = []
    event_count = 0
    error = None

    try:
        async with websockets.connect(
            WS_URL,
            ping_interval=20,
            open_timeout=10,
        ) as ws:
            await ws.send(json.dumps({
                "assets_ids": [token_id],
                "type": "market",
            }))

            end_time = asyncio.get_running_loop().time() + duration

            while asyncio.get_running_loop().time() < end_time:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=2)
                except asyncio.TimeoutError:
                    continue

                data = json.loads(message)
                events = data if isinstance(data, list) else [data]

                for event in events:
                    if not isinstance(event, dict):
                        continue

                    snapshot = orderbook.process_event(event)
                    event_count += 1

                    if not isinstance(snapshot, dict):
                        continue

                    bid = snapshot.get("best_bid")
                    ask = snapshot.get("best_ask")

                    if bid is None or ask is None:
                        continue

                    bid = float(bid)
                    ask = float(ask)

                    # Ignore crossed or invalid BBO snapshots.
                    if bid <= 0 or ask <= 0 or bid > ask:
                        continue

                    bids.append(bid)
                    asks.append(ask)
                    mids.append((bid + ask) / 2)
                    spreads.append(ask - bid)

    except Exception as exc:
        error = str(exc)

    return {
        "events": event_count,
        "bbo_samples": len(mids),
        "unique_bbo": len(set(zip(bids, asks))),
        "unique_mid_prices": len(set(mids)),
        "price_changes": sum(
            mids[i] != mids[i - 1] for i in range(1, len(mids))
        ),
        "spread_changes": sum(
            spreads[i] != spreads[i - 1]
            for i in range(1, len(spreads))
        ),
        "avg_spread": round(sum(spreads) / len(spreads), 6) if spreads else None,
        "best_bid": bids[-1] if bids else None,
        "best_ask": asks[-1] if asks else None,
        "error": error,
    }


def quality_score(stats):
    """Heuristic data-quality score; not a trading signal."""
    if stats.get("error"):
        return 0.0

    samples = stats.get("bbo_samples", 0)
    unique_bbo = stats.get("unique_bbo", 0)
    price_changes = stats.get("price_changes", 0)
    spread_changes = stats.get("spread_changes", 0)

    if samples == 0:
        return 0.0

    score = 0.0
    score += min(samples / 30, 1) * 40
    score += min(unique_bbo / 5, 1) * 30
    score += min(price_changes / 3, 1) * 20
    score += min(spread_changes / 3, 1) * 10

    return round(score, 2)


def get_candidates():
    session = requests.Session()
    now = datetime.now(timezone.utc)

    print("Scanning markets using active scanner filters...")

    markets = fetch_markets(session)
    print("Total markets fetched:", len(markets))

    candidates = []

    for market in markets:
        if not is_crypto_market(market):
            continue

        matched, _ = is_short_crypto_market(market, now)
        if not matched:
            continue

        volume = get_volume_24h(market)
        if volume < MIN_VOLUME_24H:
            continue

        tokens = parse_json_list(market.get("clobTokenIds"))

        for token_id in tokens:
            candidates.append({
                "token_id": str(token_id),
                "question": market.get("question", "Unknown"),
                "volume_24h": volume,
            })

    # Avoid inspecting the same token more than once.
    unique_candidates = {}
    for candidate in candidates:
        unique_candidates[candidate["token_id"]] = candidate

    candidates = list(unique_candidates.values())

    print("Crypto candidate tokens:", len(candidates))
    return candidates


async def main():
    candidates = get_candidates()
    results = []

    for i, candidate in enumerate(candidates[:SCAN_LIMIT], 1):
        print(f"\n[{i}/{min(len(candidates), SCAN_LIMIT)}] {candidate['question']}")

        stats = await inspect_market(
            candidate["token_id"],
            duration=INSPECTION_SECONDS,
        )

        stats["question"] = candidate["question"]
        stats["token_id"] = candidate["token_id"]
        stats["volume_24h"] = candidate["volume_24h"]
        stats["score"] = quality_score(stats)

        results.append(stats)

        if stats.get("error"):
            print("WebSocket error:", stats["error"])

    results.sort(key=lambda item: item["score"], reverse=True)

    print("\n" + "=" * 100)
    print("MARKET QUALITY")
    print("=" * 100)

    for result in results:
        print(
            f"Score={result['score']:5.1f} | "
            f"Events={result['events']:5d} | "
            f"BBO={result['bbo_samples']:4d} | "
            f"UniqueBBO={result['unique_bbo']:3d} | "
            f"UniqueMid={result['unique_mid_prices']:3d} | "
            f"Changes={result['price_changes']:3d} | "
            f"AvgSpread={result['avg_spread']} | "
            f"Token={result['token_id']} | "
            f"{result['question']}"
        )


if __name__ == "__main__":
    asyncio.run(main())