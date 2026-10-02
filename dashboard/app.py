import streamlit as st
import pandas as pd
from dashboard.logging_config import get_logger

logger = get_logger()
logger.info("Dashboard session started")

from dashboard.data_service import (
    load_market_data,
    load_optional_parquet,
    load_optional_csv,
    latest_row,
    numeric_value,
)
from dashboard.components import show_metric, show_table, show_line_chart

st.set_page_config(
    page_title="PredAlpha | Quant Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("PredAlpha")
st.caption("Market Microstructure & Quant Research Dashboard")

with st.sidebar:
    st.header("Controls")
    st.caption("Read-only research dashboard")
    auto_refresh = st.checkbox("Refresh manually", value=False)
    if st.button("Reload data"):
        st.rerun()

market_df, market_source = load_market_data()
latest = latest_row(market_df)

st.caption(f"Market data source: `{market_source}`")

# 8.2 — Market data
st.header("8.2 · Market Data")
if latest is None:
    st.warning("Market data not found. Add a supported parquet file under data/processed.")
else:
    c1, c2, c3, c4 = st.columns(4)
    show_metric("Best bid", numeric_value(latest, ["best_bid"]))
    show_metric("Best ask", numeric_value(latest, ["best_ask"]))
    show_metric("Mid price", numeric_value(latest, ["mid_price"]))
    show_metric("Spread", numeric_value(latest, ["spread"]))

    show_line_chart(
        market_df,
        ["mid_price", "best_bid", "best_ask"],
        "Price history",
    )

# 8.3 — Order book / microstructure
st.divider()
st.header("8.3 · Order Book & Microstructure")

if latest is not None:
    c1, c2, c3, c4 = st.columns(4)
    show_metric("Bid depth", numeric_value(latest, ["bid_depth"]))
    show_metric("Ask depth", numeric_value(latest, ["ask_depth"]))
    show_metric("Imbalance", numeric_value(latest, ["imbalance", "depth_imbalance"]))
    show_metric("Microprice", numeric_value(latest, ["microprice"]))

    depth_cols = [c for c in ["bid_depth", "ask_depth"] if c in market_df.columns]
    if depth_cols:
        show_line_chart(market_df, depth_cols, "Depth history")

    book_cols = [
        c for c in ["best_bid", "best_ask", "bid_depth", "ask_depth"]
        if c in market_df.columns
    ]
    if book_cols:
        show_table(market_df[book_cols], "Latest order book snapshots")
else:
    st.info("Order book data is unavailable until market data is loaded.")

# 8.4 — Model signal
st.divider()
st.header("8.4 · Model Signal")

pred_df, pred_source = load_optional_parquet([
    "predictions.parquet",
    "fold_predictions.parquet",
    "test_predictions.parquet",
])

st.caption(f"Prediction source: `{pred_source}`")
pred = latest_row(pred_df)

if pred is None:
    st.info("No prediction file found. Model signal is not available.")
else:
    p1, p2, p3 = st.columns(3)
    show_metric("Prediction", pred.get("prediction", pred.get("predicted_label", "N/A")))
    show_metric("Confidence", numeric_value(pred, ["confidence", "probability", "max_probability"]))
    show_metric("Actual label", pred.get("label", "N/A"))
    show_table(pred_df, "Recent predictions")

# 8.5 — Backtest / performance
st.divider()
st.header("8.5 · Backtest Performance")

equity_df, equity_source = load_optional_parquet([
    "equity_curve.parquet",
    "backtest_equity.parquet",
])

trades_df, trades_source = load_optional_parquet([
    "trades.parquet",
    "backtest_trades.parquet",
])

summary_df, summary_source = load_optional_csv([
    "backtest_summary.csv",
    "summary.csv",
])

st.caption(
    f"Equity: `{equity_source}` · Trades: `{trades_source}` · "
    f"Summary: `{summary_source}`"
)

if not equity_df.empty:
    equity_col = next(
        (c for c in ["equity", "portfolio_value", "total_equity"] if c in equity_df.columns),
        None,
    )
    if equity_col:
        show_line_chart(equity_df, [equity_col], "Equity curve")
    else:
        st.info("Equity file found, but no recognized equity column exists.")
else:
    st.info("No backtest equity curve found.")

if not summary_df.empty:
    show_table(summary_df, "Backtest summary", max_rows=10)

if not trades_df.empty:
    show_metric("Recorded trades", len(trades_df))
    show_table(trades_df, "Recent trades")

st.divider()
st.warning(
    "Research only. Missing panels indicate missing source files, not zero performance. "
    "This dashboard does not place orders."
)

# 8.6 — Risk and execution monitoring
st.divider()
st.header("8.6 · Risk & Execution Monitor")

from dashboard.monitoring import load_risk_status, load_execution_events

risk_status, risk_source = load_risk_status()
events_df, events_source = load_execution_events()

st.caption(f"Risk source: `{risk_source}` · Execution source: `{events_source}`")

if risk_status:
    risk_cols = st.columns(min(4, max(1, len(risk_status))))
    for col, (key, value) in zip(risk_cols, risk_status.items()):
        col.metric(str(key).replace("_", " ").title(), str(value))
else:
    st.info("Risk status unavailable. No risk report was found.")

if not events_df.empty:
    st.subheader("Recent execution events")
    st.dataframe(events_df.tail(20), use_container_width=True)
else:
    st.info("No execution events found. Execution monitoring is read-only.")
