import uuid

from execution.orders import (
    Order,
    OrderSide,
    OrderType,
)


class SignalExecutor:

    def __init__(
        self,
        trade_size=100.0,
    ):

        self.trade_size = (
            trade_size
        )

    def create_order(
        self,
        signal,
        timestamp,
        price,
    ):

        if signal == "UP":

            side = OrderSide.BUY

        elif signal == "DOWN":

            side = OrderSide.SELL

        else:

            return None

        quantity = (
            self.trade_size
            / price
        )

        return Order(
            order_id=str(
                uuid.uuid4()
            ),
            timestamp=timestamp,
            side=side,
            order_type=OrderType.MARKET,
            price=price,
            quantity=quantity,
        )