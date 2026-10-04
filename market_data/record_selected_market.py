import asyncio
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import websockets

from market_data.recorder import MarketRecorder


GAMMA_URL = "https://gamma-api.polymarket.com"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

MARKET_PREFIX = "btc-updown-5m-"
SLOT_SECONDS = 300
DISCOVERY_RETRY_SECONDS = 5
RECONNECT_SECONDS = 2


def get_current_slot():
    now = int(datetime.now(timezone.utc).timestamp())
    return (now // SLOT_SECONDS) * SLOT_SECONDS


def fetch_market_tokens(slug):
    url = f"{GAMMA_URL}/events/slug/{slug}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PredAlpha/1.0"},
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        event = json.loads(response.read().decode("utf-8"))

    for market in event.get("markets", []):
        outcomes = market.get("outcomes", [])
        token_ids = market.get("clobTokenIds", [])

        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)

        if len(outcomes) != len(token_ids):
            continue

        outcome_map = {
            str(outcome).strip().lower(): str(token_id)
            for outcome, token_id in zip(outcomes, token_ids)
        }

        if "up" in outcome_map and "down" in outcome_map:
            return {
                "UP": outcome_map["up"],
                "DOWN": outcome_map["down"],
            }

    raise ValueError(f"Could not find Up/Down tokens for {slug}")


async def discover_market():
    while True:
        slot = get_current_slot()
        slug = f"{MARKET_PREFIX}{slot}"

        try:
            tokens = await asyncio.to_thread(fetch_market_tokens, slug)
            return slug, slot, tokens
        except Exception as exc:
            print(f"Market not ready: {slug} | {exc}")
            await asyncio.sleep(DISCOVERY_RETRY_SECONDS)


async def record_market(slug, slot, tokens):
    token_ids = list(tokens.values())
    end_time = slot + SLOT_SECONDS

    filename = f"{slug}.jsonl"
    recorder = MarketRecorder(
        output_dir="data/raw",
        filename=filename,
    )

    event_count = 0

    print(f"\nMarket: {slug}")
    print(f"UP token:   {tokens['UP']}")
    print(f"DOWN token: {tokens['DOWN']}")
    print(f"Output: data/raw/{filename}")

    # Save market/token mapping separately for later labeling.
    metadata = {
        "market": slug,
        "slot_start_utc": datetime.fromtimestamp(
            slot, timezone.utc
        ).isoformat(),
        "slot_end_utc": datetime.fromtimestamp(
            end_time, timezone.utc
        ).isoformat(),
        "tokens": tokens,
    }
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/raw", f"{slug}.meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    while datetime.now(timezone.utc).timestamp() < end_time:
        try:
            print("Connecting to market WebSocket...")

            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                open_timeout=10,
                close_timeout=5,
            ) as ws:
                await ws.send(json.dumps({
                    "assets_ids": token_ids,
                    "type": "market",
                }))

                print("Connected. Recording...")

                while datetime.now(timezone.utc).timestamp() < end_time:
                    remaining = end_time - datetime.now(
                        timezone.utc
                    ).timestamp()

                    try:
                        message = await asyncio.wait_for(
                            ws.recv(),
                            timeout=min(2, max(0.1, remaining)),
                        )
                    except asyncio.TimeoutError:
                        continue

                    data = json.loads(message)
                    events = data if isinstance(data, list) else [data]

                    for event in events:
                        if not isinstance(event, dict):
                            continue

                        changes = event.get("price_changes", [])

                        if isinstance(changes, list) and changes:
                            for change in changes:
                                if not isinstance(change, dict):
                                    continue

                                asset_id = str(change.get("asset_id", ""))
                                if asset_id not in token_ids:
                                    continue

                                record = {
                                    **event,
                                    **change,
                                    "event_type": event.get("event_type"),
                                    "timestamp": event.get("timestamp"),
                                }
                                recorder.record(asset_id, record)
                                event_count += 1
                        else:
                            asset_id = str(event.get("asset_id", ""))
                            if asset_id in token_ids:
                                recorder.record(asset_id, event)
                                event_count += 1

                        if event_count > 0 and event_count % 1000 == 0:
                            print(f"Recorded events: {event_count}")

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if datetime.now(timezone.utc).timestamp() < end_time:
                print(
                    f"WebSocket error: {type(exc).__name__}: {exc}"
                )
                print(f"Reconnecting in {RECONNECT_SECONDS}s...")
                await asyncio.sleep(RECONNECT_SECONDS)

    print(f"Finished {slug} | Events: {event_count}")


async def main():
    print("PredAlpha automatic Polymarket recorder started.")
    print("Press CTRL+C to stop.\n")

    while True:
        slug, slot, tokens = await discover_market()
        await record_market(slug, slot, tokens)

        # Wait until the next 5-minute slot.
        while get_current_slot() <= slot:
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nRecorder stopped by user.")