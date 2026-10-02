class TransactionCostModel:

    def __init__(
        self,
        fee_rate=0.0,
    ):

        self.fee_rate = fee_rate

    def calculate(
        self,
        price,
        quantity,
    ):

        notional = (
            price * quantity
        )

        return (
            notional
            * self.fee_rate
        )