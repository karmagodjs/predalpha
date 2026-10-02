class RiskEngine:

    def __init__(self, limits):

        self.limits = limits

    def check_order(
        self,
        side,
        quantity,
        current_position,
        current_equity,
        peak_equity,
        daily_pnl,
        trade_count,
    ):

        # Order-size limit

        if (
            quantity
            > self.limits.max_order_size
        ):

            return False, (
                "MAX_ORDER_SIZE"
            )

        # Position limit

        if side == "BUY":

            projected_position = (
                current_position
                + quantity
            )

        else:

            projected_position = (
                current_position
                - quantity
            )

        if abs(
            projected_position
        ) > self.limits.max_position:

            return False, (
                "MAX_POSITION"
            )

        # Daily loss

        if (
            daily_pnl
            <= -self.limits.max_daily_loss
        ):

            return False, (
                "MAX_DAILY_LOSS"
            )

        # Drawdown

        if peak_equity > 0:

            drawdown = (
                peak_equity
                - current_equity
            ) / peak_equity

            if (
                drawdown
                >= self.limits.max_drawdown
            ):

                return False, (
                    "MAX_DRAWDOWN"
                )

        # Trade count

        if (
            trade_count
            >= self.limits.max_trades
        ):

            return False, (
                "MAX_TRADES"
            )

        return True, "APPROVED"