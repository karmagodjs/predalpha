import pandas as pd

from backtesting.execution import ExecutionSimulator
from backtesting.portfolio import Portfolio


class BacktestEngine:

    def __init__(
        self,
        initial_cash=10_000.0,
        fee_rate=0.0,
        slippage_bps=1.0,
        trade_size=100.0,
    ):

        self.portfolio = Portfolio(
            initial_cash
        )

        self.execution = ExecutionSimulator(
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
        )

        self.trade_size = trade_size

        self.equity_curve = []
        self.trades = []

    def run(self, df):

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        for _, row in df.iterrows():

            signal = row["signal"]

            # Always use RAW market price
            price = row["mid_price_raw"]

            timestamp = row["timestamp"]

            # ==========================================
            # BUY
            # ==========================================

            if (
                signal == "UP"
                and self.portfolio.position == 0
            ):

                quantity = (
                    self.trade_size / price
                )

                fill = self.execution.execute(
                    timestamp,
                    "BUY",
                    price,
                    quantity,
                )

                success = self.portfolio.buy(
                    fill.price,
                    fill.quantity,
                    fill.fee,
                )

                if success:

                    self.trades.append(
                        {
                            "timestamp": timestamp,
                            "side": "BUY",
                            "price": fill.price,
                            "quantity": fill.quantity,
                            "fee": fill.fee,
                            "slippage": fill.slippage,
                        }
                    )

            # ==========================================
            # SELL
            # ==========================================

            elif (
                signal == "DOWN"
                and self.portfolio.position > 0
            ):

                quantity = (
                    self.portfolio.position
                )

                fill = self.execution.execute(
                    timestamp,
                    "SELL",
                    price,
                    quantity,
                )

                success = self.portfolio.sell(
                    fill.price,
                    fill.quantity,
                    fill.fee,
                )

                if success:

                    self.trades.append(
                        {
                            "timestamp": timestamp,
                            "side": "SELL",
                            "price": fill.price,
                            "quantity": fill.quantity,
                            "fee": fill.fee,
                            "slippage": fill.slippage,
                        }
                    )

            # ==========================================
            # EQUITY
            # ==========================================

            equity = self.portfolio.equity(
                price
            )

            self.equity_curve.append(
                {
                    "timestamp": timestamp,
                    "equity": equity,
                    "cash": self.portfolio.cash,
                    "position": self.portfolio.position,
                    "price": price,
                }
            )

        return pd.DataFrame(
            self.equity_curve
        )