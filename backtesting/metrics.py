import numpy as np


def calculate_metrics(equity_curve):

    if equity_curve.empty:

        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "final_equity": 0.0,
            "num_observations": 0,
        }

    equity = (
        equity_curve["equity"]
        .astype(float)
    )

    returns = (
        equity
        .pct_change()
        .dropna()
    )

    total_return = (
        equity.iloc[-1]
        / equity.iloc[0]
        - 1
    )

    if (
        len(returns) > 1
        and returns.std() > 0
    ):

        sharpe = (
            returns.mean()
            / returns.std()
        ) * np.sqrt(
            len(returns)
        )

    else:

        sharpe = 0.0

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity - running_max
    ) / running_max

    max_drawdown = (
        drawdown.min()
    )

    return {
        "total_return": float(
            total_return
        ),
        "sharpe": float(
            sharpe
        ),
        "max_drawdown": float(
            max_drawdown
        ),
        "final_equity": float(
            equity.iloc[-1]
        ),
        "num_observations": len(
            equity
        ),
    }