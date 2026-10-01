import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from benchmarks.latency_benchmark import run_once
from market_data.orderbook import OrderBook


def main(iterations=10_000):
    book = OrderBook()

    for i in range(1, 21):
        book.update_bid(round(0.05 - i * 0.001, 3), i * 10)
        book.update_ask(round(0.06 + i * 0.001, 3), i * 10)

    values = []

    for i in range(iterations):
        elapsed, _, _, _ = run_once(book, i)
        values.append(elapsed)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "iterations": iterations,
        "unit": "nanoseconds",
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(max(values)),
        "note": "Local Python benchmark; excludes network and exchange latency."
    }

    output = Path("data/processed/latency_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()