from dataclasses import dataclass


@dataclass
class Fill:
    timestamp: str
    side: str
    price: float
    quantity: float
    fee: float
    slippage: float


class ExecutionSimulator:
    def __init__(
        self,
        fee_rate=0.0,
        slippage_bps=1.0,
    ):
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps

    def execute(self, timestamp, side, price, quantity):
        slippage = price * self.slippage_bps / 10_000

        if side == "BUY":
            execution_price = price + slippage
        elif side == "SELL":
            execution_price = price - slippage
        else:
            raise ValueError(f"Invalid side: {side}")

        notional = execution_price * quantity
        fee = notional * self.fee_rate

        return Fill(
            timestamp=timestamp,
            side=side,
            price=execution_price,
            quantity=quantity,
            fee=fee,
            slippage=abs(execution_price - price),
        )