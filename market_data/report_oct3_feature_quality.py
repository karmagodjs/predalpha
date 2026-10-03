from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data/processed")
INPUT = BASE / "btc_oct3_up_down_session_20261002_04_combined.parquet"
OUTPUT = BASE / "btc_oct3_up_down_session_20261002_04_quality_report.txt"

def main():
    df = pd.read_parquet(INPUT)

    numeric = df.select_dtypes(include=[np.number])
    report = []

    report.append("OCT 3 FEATURE QUALITY REPORT")
    report.append("=" * 40)
    report.append(f"Rows: {len(df)}")
    report.append(f"Unique paired timestamps: {df['timestamp'].nunique()}")
    report.append(f"Columns: {len(df.columns)}")
    report.append(f"Duplicate rows: {df.duplicated().sum()}")
    report.append(f"Missing values: {df.isna().sum().sum()}")
    report.append(f"Infinite numeric values: {np.isinf(numeric).sum().sum()}")

    report.append("\nMissing values by column:")
    report.append(df.isna().sum().to_string())

    report.append("\nOutcome distribution:")
    report.append(df["outcome"].value_counts().to_string())

    report.append("\nNumeric feature summary:")
    report.append(numeric.describe().round(6).to_string())

    report.append("\nTimestamp range:")
    report.append(str(df["timestamp"].min()))
    report.append(str(df["timestamp"].max()))

    text = "\n".join(report)
    OUTPUT.write_text(text, encoding="utf-8")

    print(text)
    print("\nSaved:", OUTPUT)

if __name__ == "__main__":
    main()