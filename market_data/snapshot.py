from dataclasses import dataclass


@dataclass
class MarketSnapshot:

    timestamp: str
    market_id: str

    best_bid: float | None
    best_ask: float | None

    mid_price: float | None
    spread: float | None

    bid_depth: float
    ask_depth: float

    imbalance: float | None
    microprice: float | None

    def to_dict(self):

        return {
            "timestamp": self.timestamp,
            "market_id": self.market_id,

            "best_bid": self.best_bid,
            "best_ask": self.best_ask,

            "mid_price": self.mid_price,
            "spread": self.spread,

            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,

            "imbalance": self.imbalance,
            "microprice": self.microprice,
        }