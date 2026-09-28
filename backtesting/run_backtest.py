from pathlib import Path

import pandas as pd

from backtesting.engine import (
    BacktestEngine
)

from backtesting.metrics import (
    calculate_metrics
)


INPUT = Path(
    "data/processed/test_predictions.parquet"
)


def main():

    print("=" * 60)
    print("PREDALPHA EVENT-DRIVEN BACKTEST")
    print("=" * 60)

    if not INPUT.exists():

        raise FileNotFoundError(
            f"Missing: {INPUT}"
        )

    df = pd.read_parquet(
        INPUT
    )

    # --------------------------------------------------
    # Signal mapping
    # --------------------------------------------------

    df["signal"] = "NO_TRADE"

    df.loc[
        df["prediction"] == 0,
        "signal"
    ] = "DOWN"

    df.loc[
        df["prediction"] == 1,
        "signal"
    ] = "FLAT"

    df.loc[
        df["prediction"] == 2,
        "signal"
    ] = "UP"

    # --------------------------------------------------
    # Raw price validation
    # --------------------------------------------------

    if "mid_price_raw" not in df.columns:

        raise ValueError(
            "mid_price_raw is required "
            "for backtesting."
        )

    print(
        "\nRows:",
        len(df)
    )

    print(
        "\nSignals:"
    )

    print(
        df["signal"]
        .value_counts()
    )

    # --------------------------------------------------
    # Engine
    # --------------------------------------------------

    engine = BacktestEngine(
        initial_cash=10_000.0,
        fee_rate=0.0,
        slippage_bps=1.0,
        trade_size=100.0,
    )

    equity_curve = engine.run(
        df
    )

    metrics = calculate_metrics(
        equity_curve
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print("BACKTEST RESULTS")

    print("=" * 60)

    print(
        f"Total return: "
        f"{metrics['total_return']:.4%}"
    )

    print(
        f"Sharpe: "
        f"{metrics['sharpe']:.4f}"
    )

    print(
        f"Max drawdown: "
        f"{metrics['max_drawdown']:.4%}"
    )

    print(
        f"Final equity: "
        f"${metrics['final_equity']:.2f}"
    )

    print(
        f"Trades: "
        f"{len(engine.trades)}"
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    equity_curve.to_parquet(
        "data/processed/"
        "backtest_equity.parquet",
        index=False,
    )

    pd.DataFrame(
        engine.trades
    ).to_parquet(
        "data/processed/"
        "backtest_trades.parquet",
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        "data/processed/backtest_equity.parquet"
    )

    print(
        "data/processed/backtest_trades.parquet"
    )


if __name__ == "__main__":
    main()