from pathlib import Path

import pandas as pd


class ParquetWriter:

    def __init__(
        self,
        path="data/processed/market_data.parquet"
    ):
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def write(self, snapshots):

        df = pd.DataFrame(snapshots)

        if df.empty:
            return

        df.to_parquet(
            self.path,
            index=False
        )

        print(
            f"Saved {len(df)} rows → {self.path}"
        )