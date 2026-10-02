class CircuitBreaker:

    def __init__(
        self,
        max_loss,
        max_drawdown,
    ):

        self.max_loss = max_loss
        self.max_drawdown = (
            max_drawdown
        )

        self.triggered = False

    def check(
        self,
        daily_pnl,
        drawdown,
    ):

        if (
            daily_pnl
            <= -self.max_loss
        ):

            self.triggered = True

        if (
            drawdown
            >= self.max_drawdown
        ):

            self.triggered = True

        return self.triggered