import asyncio
import json
import websockets
from market_data.recorder import MarketRecorder

TOKEN_IDS = [
    "32664731353322490016663940157951165038326534411865610104571929420891932752044",  # Up
    "83368144530639073518031297526937851295808514385655078316894643930693648829402",   # Down
]

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
DURATION_SECONDS = 1800  # 10 minutes
OUTPUT_FILENAME = "btc_oct3_up_down_session_20261002_04.jsonl"


async def main():
    recorder = MarketRecorder(
        output_dir="data/raw",
        filename=OUTPUT_FILENAME
    )
    event_count = 0

    async with websockets.connect(
        WS_URL,
        ping_interval=20,
        open_timeout=10
    ) as ws:
        await ws.send(json.dumps({
            "assets_ids": TOKEN_IDS,
            "type": "market"
        }))

        print("Connected. Recording both BTC outcomes for 30 minutes...")
        end_time = asyncio.get_running_loop().time() + DURATION_SECONDS

        while asyncio.get_running_loop().time() < end_time:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=2)
            except asyncio.TimeoutError:
                continue

            data = json.loads(message)
            events = data if isinstance(data, list) else [data]

            for event in events:
                if isinstance(event, dict):
                    asset_id = str(event.get("asset_id", ""))
                    
                    if asset_id in TOKEN_IDS:
                        print(
                            "EVENT:",
                            event.get("event_type"),
                            "asset:",
                            asset_id[:12],
                            "keys:",
                            list(event.keys())
                        )
                        recorder.record(asset_id, event)
                        event_count += 1

                        if event_count % 100 == 0:
                            print(f"Recorded {event_count} events")

    print(f"Recording complete. Total events: {event_count}")
    print(f"Saved: data/raw/{OUTPUT_FILENAME}")


if __name__ == "__main__":
    asyncio.run(main())