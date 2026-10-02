from pathlib import Path
import pandas as pd

METRICS_PATH = Path("data/processed/walk_forward/summary.csv")
REGIME_PATH = Path("data/processed/market_regimes.parquet")
OUTPUT_PATH = Path("data/processed/phase6_research_report.md")

def main():
    sections = ["# PredAlpha — Phase 6 Research Report\n"]

    sections.append("## Walk-forward metrics")
    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        sections.append(metrics.to_markdown(index=False))
    else:
        sections.append(
            "Metrics file not found. Generate out-of-sample fold predictions "
            "and run the walk-forward metrics evaluation first."
        )

    sections.append("\n## Market regime distribution")
    if REGIME_PATH.exists():
        regimes = pd.read_parquet(REGIME_PATH)
        if "regime" in regimes.columns:
            counts = regimes["regime"].value_counts(dropna=False).rename_axis(
                "regime"
            ).reset_index(name="rows")
            sections.append(counts.to_markdown(index=False))
        else:
            sections.append("Regime file has no 'regime' column.")
    else:
        sections.append("Regime file not found.")

    sections.append(
        "\n## Limitations\n"
        "- Results depend on the quality and coverage of recorded market data.\n"
        "- A small dataset cannot establish robust out-of-sample performance.\n"
        "- Classification metrics do not establish profitability.\n"
        "- Transaction costs, slippage, latency and execution assumptions "
        "must be evaluated separately.\n"
        "- Constant or near-constant prices make return-based validation "
        "uninformative."
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Saved report: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()