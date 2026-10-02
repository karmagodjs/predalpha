from dataclasses import dataclass


@dataclass
class RiskLimits:

    max_position: float = 1_000.0

    max_order_size: float = 100.0

    max_daily_loss: float = 100.0

    max_drawdown: float = 0.05

    max_trades: int = 1_000