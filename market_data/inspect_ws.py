import asyncio
import json
import websockets

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


async def main():

    asset_id = input(
        "Enter Polymarket token ID: "
    ).strip()

    async with websockets.connect(
        WS_URL
    ) as websocket:

        subscription = {
            "assets_ids": [asset_id],
            "type": "market",
        }

        await websocket.send(
            json.dumps(subscription)
        )

        print("\nConnected.")
        print("Waiting for events...\n")

        count = 0

        async for message in websocket:

            data = json.loads(message)

            print("=" * 80)

            print(
                json.dumps(
                    data,
                    indent=2
                )
            )

            count += 1

            if count >= 5:
                break


if __name__ == "__main__":
    asyncio.run(main())