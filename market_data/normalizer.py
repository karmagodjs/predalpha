from datetime import datetime, timezone

from market_data.orderbook import OrderBook
from market_data.snapshot import MarketSnapshot


class MarketNormalizer:

    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self.book = OrderBook()
        self.market_id = None

    @staticmethod
    def _convert_timestamp(timestamp):
        if timestamp is None:
            return datetime.now(timezone.utc).isoformat()

        try:
            # Polymarket timestamp = milliseconds
            ts = int(timestamp)

            return datetime.fromtimestamp(
                ts / 1000,
                tz=timezone.utc
            ).isoformat()

        except (ValueError, TypeError):
            return str(timestamp)

    def process(self, data):

        event_type = data.get("event_type")

        if event_type == "book":
            return self._process_book(data)

        if event_type == "price_change":
            return self._process_price_change(data)

        return None

    def _process_book(self, data):

        # Ignore other assets
        if data.get("asset_id") != self.asset_id:
            return None

        self.market_id = data.get(
            "market",
            self.market_id
        )

        # Reset order book
        self.book = OrderBook()

        # Load bids
        for level in data.get("bids", []):

            price = float(level["price"])
            size = float(level["size"])

            self.book.update_bid(
                price,
                size
            )

        # Load asks
        for level in data.get("asks", []):

            price = float(level["price"])
            size = float(level["size"])

            self.book.update_ask(
                price,
                size
            )

        return self._create_snapshot(
            data.get("timestamp")
        )

    def _process_price_change(self, data):

        self.market_id = data.get(
            "market",
            self.market_id
        )

        updated = False

        for change in data.get(
            "price_changes",
            []
        ):

            # IMPORTANT:
            # Ignore complementary token
            if change.get("asset_id") != self.asset_id:
                continue

            price = float(
                change["price"]
            )

            size = float(
                change["size"]
            )

            side = change.get("side")

            if side == "BUY":

                self.book.update_bid(
                    price,
                    size
                )

                updated = True

            elif side == "SELL":

                self.book.update_ask(
                    price,
                    size
                )

                updated = True

        if not updated:
            return None

        return self._create_snapshot(
            data.get("timestamp")
        )

    def _create_snapshot(self, timestamp):

        best_bid = self.book.best_bid
        best_ask = self.book.best_ask

        mid_price = self.book.mid_price
        spread = self.book.spread

        # Total depth
        bid_depth = sum(
            self.book.bids.values()
        )

        ask_depth = sum(
            self.book.asks.values()
        )

        total_depth = (
            bid_depth + ask_depth
        )

        # Order-book imbalance
        if total_depth > 0:

            imbalance = (
                bid_depth - ask_depth
            ) / total_depth

        else:

            imbalance = None

        # Best-level quantities
        best_bid_size = (
            self.book.bids.get(
                best_bid,
                0.0
            )
            if best_bid is not None
            else 0.0
        )

        best_ask_size = (
            self.book.asks.get(
                best_ask,
                0.0
            )
            if best_ask is not None
            else 0.0
        )

        # Standard microprice
        denominator = (
            best_bid_size
            + best_ask_size
        )

        if (
            best_bid is not None
            and best_ask is not None
            and denominator > 0
        ):

            microprice = (
                best_ask * best_bid_size
                + best_bid * best_ask_size
            ) / denominator

        else:

            microprice = None

        snapshot = MarketSnapshot(

            timestamp=self._convert_timestamp(
                timestamp
            ),

            market_id=str(
                self.market_id
            ),

            best_bid=best_bid,

            best_ask=best_ask,

            mid_price=mid_price,

            spread=spread,

            bid_depth=bid_depth,

            ask_depth=ask_depth,

            imbalance=imbalance,

            microprice=microprice,
        )

        return snapshot.to_dict()