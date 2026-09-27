from dataclasses import dataclass, field
from typing import Dict

@dataclass
class OrderBook:
    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)
    
    def update_bid(self, price: float, size: float):
        if size <= 0:
            self.bids.pop(price, None)
        else:
            self.bids[price] = size

    def update_ask(self, price: float, size: float):
        if size <= 0:
            self.asks.pop(price, None)
        else:
            self.asks[price] = size

    @property
    def best_bid(self):
        return max(self.bids.keys()) if self.bids else None

    @property
    def best_ask(self):
        return min(self.asks.keys()) if self.asks else None

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