import asyncio
import json
import websockets

from market_data.recorder import MarketRecorder


WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


async def connect_market(asset_id: str):

    recorder = MarketRecorder()

    async with websockets.connect(WS_URL) as websocket:

        subscription = {
            "assets_ids": [asset_id],
            "type": "market"
        }

        await websocket.send(
            json.dumps(subscription)
        )

        print("Connected to Polymarket")
        print(f"Subscribed: {asset_id}")

        async for message in websocket:

            data = json.loads(message)

            recorder.record(data)

            print(data)


if __name__ == "__main__":

    asset_id = input(
        "Enter Polymarket asset ID: "
    ).strip()

    asyncio.run(
        connect_market(asset_id)
    )