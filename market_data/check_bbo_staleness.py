from pathlib import Path
import pandas as pd

INPUT = Path(
    "data/processed/btc_88000_session_20261002_01_bbo_1s.parquet"
)


def main():
    df = pd.read_parquet(INPUT).sort_index()

    bbo = df[["bid", "ask"]].copy()

    unchanged = (
        bbo["bid"].eq(bbo["bid"].shift())
        & bbo["ask"].eq(bbo["ask"].shift())
    )

    # Consecutive unchanged seconds
    groups = (~unchanged).cumsum()
    stale_runs = unchanged.groupby(groups).sum()
    stale_runs = stale_runs[stale_runs > 0]

    print("Total rows:", len(df))
    print("Unchanged BBO seconds:", int(unchanged.sum()))
    print("Changed BBO seconds:", int((~unchanged).sum()))
    print("Longest unchanged run:", int(stale_runs.max()) if len(stale_runs) else 0)
    print("Runs >= 10 seconds:", int((stale_runs >= 10).sum()))
    print("Runs >= 30 seconds:", int((stale_runs >= 30).sum()))

    print("\nTop 10 longest unchanged runs:")
    print(stale_runs.sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()