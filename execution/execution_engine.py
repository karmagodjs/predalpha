from execution.slippage import (
    SlippageModel
)

from execution.costs import (
    TransactionCostModel
)

from execution.fills import Fill


class ExecutionEngine:

    def __init__(
        self,
        slippage_bps=1.0,
        fee_rate=0.0,
    ):

        self.slippage = SlippageModel(
            slippage_bps
        )

        self.costs = (
            TransactionCostModel(
                fee_rate
            )
        )

    def execute(
        self,
        order,
    ):

        execution_price, slippage = (
            self.slippage.apply(
                order.price,
                order.side.value,
            )
        )

        fee = self.costs.calculate(
            execution_price,
            order.quantity,
        )

        return Fill(
            order_id=order.order_id,
            timestamp=order.timestamp,
            side=order.side.value,
            price=execution_price,
            quantity=order.quantity,
            fee=fee,
            slippage=slippage,
        )