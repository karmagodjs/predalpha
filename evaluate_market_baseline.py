import pandas as pd
from pathlib import Path

path = Path("data/processed/btc_oct3_new_markets_labeled.parquet")
df = pd.read_parquet(path)

# One observation per independent market
markets = (
    df.groupby(["market", "label"], as_index=False)
      .agg(
          quote_rows=("timestamp", "size"),
          mean_up_mid=("up_mid_price", "mean"),
          mean_down_mid=("down_mid_price", "mean"),
          mean_combined_spread=("combined_spread", "mean"),
          mean_pair_age_ms=("pair_age_ms", "mean"),
      )
)

if markets["market"].nunique() != 2:
    raise ValueError("Expected exactly 2 independent markets for this baseline.")

# Balanced labels: deterministic majority baseline, tie-break = UP
counts = markets["label"].value_counts()
baseline_class = "UP" if counts.get("UP", 0) >= counts.get("DOWN", 0) else "DOWN"
markets["baseline_prediction"] = baseline_class
markets["correct"] = markets["label"] == markets["baseline_prediction"]

accuracy = markets["correct"].mean()
report = [
    "MARKET-LEVEL BASELINE REPORT",
    f"Independent markets: {len(markets)}",
    f"Label counts: {markets['label'].value_counts().to_dict()}",
    f"Baseline: always predict {baseline_class}",
    f"Correct: {int(markets['correct'].sum())}/{len(markets)}",
    f"Accuracy: {accuracy:.3f}",
    "",
    "Market-level rows:",
    markets.to_string(index=False),
    "",
    "LIMITATION: only 2 independent outcomes; this is a sanity-check baseline,",
    "not a statistically meaningful ML evaluation. Do not use quote rows",
    "as independent train/test samples."
]

text = "\n".join(report)
print(text)

Path("reports").mkdir(exist_ok=True)
Path("reports/baseline_report.txt").write_text(text, encoding="utf-8")
markets.to_csv("data/processed/btc_oct3_market_level_baseline.csv", index=False)

print("\nSaved: reports/baseline_report.txt")
print("Saved: data/processed/btc_oct3_market_level_baseline.csv")
