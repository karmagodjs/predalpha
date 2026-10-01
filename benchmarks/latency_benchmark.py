import time
import statistics
import pandas as pd

from market_data.orderbook import OrderBook
from features.microstructure import (
    order_book_imbalance,
    microprice,
    spread_bps,
)


def percentile(values, p):
    return statistics.quantiles(values, n=100)[p - 1]


def run_once(book, i):
    start = time.perf_counter_ns()

    price = round(0.03 + (i % 20) * 0.001, 3)
    size = float((i % 100) + 1)

    book.update_bid(price, size)
    book.update_ask(price + 0.002, size)

    imbalance = order_book_imbalance(book)
    micro = microprice(book)
    spread = spread_bps(book)

    elapsed = time.perf_counter_ns() - start

    return elapsed, imbalance, micro, spread


def main(iterations=10_000):
    book = OrderBook()

    for i in range(1, 21):
        book.update_bid(round(0.05 - i * 0.001, 3), i * 10)
        book.update_ask(round(0.06 + i * 0.001, 3), i * 10)

    latencies = []

    for i in range(iterations):
        elapsed, _, _, _ = run_once(book, i)
        latencies.append(elapsed)

    latencies.sort()

    print("PredAlpha local pipeline latency")
    print(f"Samples: {len(latencies)}")
    print(f"Mean:   {statistics.mean(latencies):,.0f} ns")
    print(f"Median: {statistics.median(latencies):,.0f} ns")
    print(f"P95:    {latencies[int(0.95 * len(latencies))]:,.0f} ns")
    print(f"P99:    {latencies[int(0.99 * len(latencies))]:,.0f} ns")
    print(f"Max:    {max(latencies):,.0f} ns")


if __name__ == "__main__":
    main()