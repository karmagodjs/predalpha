from bisect import bisect_left, insort


class OptimizedOrderBook:
    def __init__(self):
        self.bids = {}
        self.asks = {}
        self._bid_prices = []  # ascending
        self._ask_prices = []  # ascending

    def _update(self, side, prices, price, size):
        price = float(price)
        size = float(size)

        if size < 0:
            raise ValueError("Order size cannot be negative.")

        if size == 0:
            if price in side:
                del side[price]
                idx = bisect_left(prices, price)
                if idx < len(prices) and prices[idx] == price:
                    prices.pop(idx)
            return

        if price not in side:
            insort(prices, price)

        side[price] = size

    def update_bid(self, price, size):
        self._update(self.bids, self._bid_prices, price, size)

    def update_ask(self, price, size):
        self._update(self.asks, self._ask_prices, price, size)

    @property
    def best_bid(self):
        return self._bid_prices[-1] if self._bid_prices else None

    @property
    def best_ask(self):
        return self._ask_prices[0] if self._ask_prices else None

    @property
    def mid_price(self):
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self):
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def snapshot(self):
        return {
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "bid_depth": sum(self.bids.values()),
            "ask_depth": sum(self.asks.values()),
        }