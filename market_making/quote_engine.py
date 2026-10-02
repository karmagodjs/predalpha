class QuoteEngine:

    def __init__(
        self,
        base_spread_bps=5.0,
        inventory_skew=0.10,
    ):

        self.base_spread_bps = (
            base_spread_bps
        )

        self.inventory_skew = (
            inventory_skew
        )

    def generate_quotes(
        self,
        fair_value,
        inventory,
    ):

        spread = (
            fair_value
            * self.base_spread_bps
            / 10_000
        )

        skew = (
            inventory
            * self.inventory_skew
        )

        bid = (
            fair_value
            - spread
            - skew
        )

        ask = (
            fair_value
            + spread
            - skew
        )

        return {
            "bid": bid,
            "ask": ask,
        }