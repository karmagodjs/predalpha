class PositionManager:

    def __init__(self):

        self.position = 0.0
        self.avg_price = 0.0
        self.realized_pnl = 0.0

    def apply_fill(
        self,
        side,
        price,
        quantity,
        fee,
    ):

        if side == "BUY":

            total_cost = (
                self.avg_price
                * self.position
                + price * quantity
            )

            new_position = (
                self.position
                + quantity
            )

            if new_position > 0:

                self.avg_price = (
                    total_cost
                    / new_position
                )

            self.position = (
                new_position
            )

            self.realized_pnl -= fee

        elif side == "SELL":

            if quantity > self.position:

                raise ValueError(
                    "Cannot sell more "
                    "than current position."
                )

            pnl = (
                price
                - self.avg_price
            ) * quantity

            self.realized_pnl += (
                pnl - fee
            )

            self.position -= quantity

            if self.position == 0:

                self.avg_price = 0.0

        else:

            raise ValueError(
                f"Invalid side: {side}"
            )