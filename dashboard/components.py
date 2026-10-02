import streamlit as st

def show_metric(label, value, help_text=None):
    if value is None:
        st.metric(label, "N/A", help=help_text)
    elif isinstance(value, (int, float)):
        st.metric(label, f"{value:,.6g}", help=help_text)
    else:
        st.metric(label, str(value), help=help_text)

def show_table(df, title, max_rows=20):
    st.subheader(title)
    if df is None or df.empty:
        st.info("No data available yet.")
    else:
        st.dataframe(df.tail(max_rows), use_container_width=True)

def show_line_chart(df, columns, title):
    st.subheader(title)
    available = [col for col in columns if col in df.columns]
    if df.empty or not available:
        st.info("Required time-series columns are not available.")
        return

    chart_df = df[available].copy()
    if "timestamp" in df.columns:
        chart_df.index = df["timestamp"]
    st.line_chart(chart_df, use_container_width=True)
