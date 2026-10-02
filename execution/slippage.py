class SlippageModel:

    def __init__(self, slippage_bps=1.0):

        self.slippage_bps = slippage_bps

    def apply(self, price, side):

        slippage = (
            price
            * self.slippage_bps
            / 10_000
        )

        if side == "BUY":

            execution_price = (
                price + slippage
            )

        elif side == "SELL":

            execution_price = (
                price - slippage
            )

        else:

            raise ValueError(
                f"Invalid side: {side}"
            )

        return execution_price, abs(
            execution_price - price
        )