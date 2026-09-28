import asyncio
import json

import websockets
from market_data.recorder import MarketRecorder
from market_data.orderbook import OrderBook


WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class LiveOrderBook:

    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self.book = OrderBook()
        self.recorder = MarketRecorder()

    def apply_level(self, side, price, size):
        price = float(price)
        size = float(size)

        if side == "bid":
            self.book.update_bid(price, size)

        elif side == "ask":
            self.book.update_ask(price, size)

    def process_event(self, data):

        # Handle batched events
        if isinstance(data, list):

            snapshot = None

            for event in data:
                result = self.process_event(event)

                if result is not None:
                    snapshot = result

            return snapshot

        if not isinstance(data, dict):
            return None

        event_type = data.get(
            "event_type",
            data.get("type")
        )

        # ==================================================
        # INITIAL ORDER BOOK
        # ==================================================

        if event_type == "book":

            # A book event represents the subscribed asset.
            # We load its bids and asks.

            bids = data.get("bids", [])
            asks = data.get("asks", [])

            for level in bids:

                price = level.get("price")
                size = level.get("size")

                if price is not None and size is not None:

                    self.apply_level(
                        "bid",
                        price,
                        size
                    )

            for level in asks:

                price = level.get("price")
                size = level.get("size")

                if price is not None and size is not None:

                    self.apply_level(
                        "ask",
                        price,
                        size
                    )

        # ==================================================
        # PRICE CHANGE
        # ==================================================

        elif event_type == "price_change":

            changes = data.get(
                "price_changes",
                []
            )

            for change in changes:

                # IMPORTANT:
                # A price_change event can contain updates
                # for both outcome tokens.
                #
                # Only process our subscribed asset.

                change_asset_id = change.get(
                    "asset_id"
                )

                if change_asset_id != self.asset_id:
                    continue

                price = change.get("price")
                size = change.get("size")
                side = change.get("side")

                if price is None or size is None:
                    continue

                if side == "BUY":
                    mapped_side = "bid"

                elif side == "SELL":
                    mapped_side = "ask"

                else:
                    continue

                self.apply_level(
                    mapped_side,
                    price,
                    size
                )

        # ==================================================
        # SNAPSHOT + VALIDATION
        # ==================================================

        snapshot = self.book.snapshot()

        bid = snapshot["best_bid"]
        ask = snapshot["best_ask"]

        # Order book must satisfy:
        #
        # best_bid <= best_ask

        if (
            bid is not None
            and ask is not None
            and bid > ask
        ):

            print(
                "WARNING: INVALID ORDER BOOK"
            )

            print(
                f"bid={bid} "
                f"ask={ask}"
            )

        return snapshot

    async def run(self):

        async with websockets.connect(
            WS_URL
        ) as websocket:

            subscription = {
                "assets_ids": [self.asset_id],
                "type": "market",
            }

            await websocket.send(
                json.dumps(subscription)
            )

            print("Connected.")
            print(
                f"Asset: {self.asset_id}"
            )

            async for message in websocket:

                data = json.loads(message)

                snapshot = self.process_event(
                    data
                )

                if snapshot is None:
                    continue
                self.recorder.record(
                    asset_id=self.asset_id,
                    snapshot=snapshot,
                    event=data
                )
                bid = snapshot["best_bid"]
                ask = snapshot["best_ask"]

                # ==================================================
                # VALID BOOK
                # ==================================================

                if (
                    bid is not None
                    and ask is not None
                    and bid <= ask
                ):

                    print(
                        f"BID={bid:.4f} "
                        f"ASK={ask:.4f} "
                        f"MID={snapshot['mid_price']:.4f} "
                        f"SPREAD={snapshot['spread']:.4f} "
                        f"BID_DEPTH={snapshot['bid_depth']:.2f} "
                        f"ASK_DEPTH={snapshot['ask_depth']:.2f}"
                    )

                # ==================================================
                # INVALID BOOK
                # ==================================================

                else:

                    print(
                        "INVALID BOOK -> "
                        f"BID={bid} "
                        f"ASK={ask}"
                    )


if __name__ == "__main__":

    asset_id = input(
        "Enter Polymarket token ID: "
    ).strip()

    client = LiveOrderBook(
        asset_id
    )

    try:

        asyncio.run(
            client.run()
        )

    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )