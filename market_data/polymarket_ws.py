import asyncio
import json
import websockets

from market_data.recorder import MarketRecorder

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

ASSET_IDS = [
    # Mark Kelly
    "32338220190071351435772801779725302244575775216413325951443816017994629993401",
    "25659310674993675562345759665114759892400026242514633218387667107987341231962",

    # Rahm Emanuel
    "70071592420137476676935286377781779672157004436137616627487590484756055232944",
    "107829893465244243112531790315844862173967342141753896250561816305880797209350",

    # Wes Moore
    "11343337042526652606304508556838778144915122118685013460142819817079620240439",
    "19117858613830240204442128472263675815111662346578092643025820800380629933012",
]


async def connect_market(asset_ids: list[str]):
    recorder = MarketRecorder(filename="polymarket_bbo_active.jsonl")

    async with websockets.connect(
        WS_URL,
        ping_interval=20,
        ping_timeout=20,
    ) as websocket:

        subscription = {
            "assets_ids": asset_ids,
            "type": "market",
            "custom_feature_enabled": True,
        }

        await websocket.send(json.dumps(subscription))

        print("Connected to Polymarket")
        print(f"Subscribed to {len(asset_ids)} assets")
        print("Recording to data/raw/polymarket_bbo_active.jsonl")
        print("Press Ctrl+C to stop.")

        async for message in websocket:
            try:
                data = json.loads(message)
                recorder.record("multi", data)

                events = data if isinstance(data, list) else [data]
                for event in events:
                    if isinstance(event, dict):
                        print("Event:", event.get("event_type", "unknown"))

            except json.JSONDecodeError:
                print("Skipped invalid JSON message")


if __name__ == "__main__":
    try:
        asyncio.run(connect_market(ASSET_IDS))
    except KeyboardInterrupt:
        print("\nCapture stopped.")