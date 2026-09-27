import asyncio
import json
import websockets

from market_data.orderbook import OrderBook


WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class LiveOrderBook:

    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self.book = OrderBook()

    async def run(self):

        async with websockets.connect(WS_URL) as websocket:

            subscription = {
                "assets_ids": [self.asset_id],
                "type": "market",
            }

            await websocket.send(
                json.dumps(subscription)
            )

            print("Connected.")
            print(f"Asset: {self.asset_id}")

            async for message in websocket:

                data = json.loads(message)

                self.process_event(data)

    def process_event(self, data):

        print("\nEVENT:")
        print(data)

        # Event normalization will be implemented
        # after inspecting the actual feed format.


if __name__ == "__main__":

    asset_id = input(
        "Enter Polymarket token ID: "
    ).strip()

    client = LiveOrderBook(asset_id)

    asyncio.run(client.run())