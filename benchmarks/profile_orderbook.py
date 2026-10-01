import statistics
import time

from market_data.orderbook import OrderBook


def make_book(book):
    for i in range(1, 21):
        book.update_bid(round(0.50 - i * 0.001, 3), i * 10)
        book.update_ask(round(0.51 + i * 0.001, 3), i * 10)
    return book


def workload(book, iterations=50_000):
    start = time.perf_counter_ns()

    for i in range(iterations):
        price = round(0.45 + (i % 20) * 0.001, 3)
        size = float((i % 100) + 1)

        book.update_bid(price, size)
        book.update_ask(price + 0.02, size)

        _ = book.best_bid
        _ = book.best_ask
        _ = book.mid_price
        _ = book.spread

    return (time.perf_counter_ns() - start) / iterations


def main():
    results = []

    for _ in range(5):
        book = make_book(OrderBook())
        results.append(workload(book))

    print(f"Python baseline median: {statistics.median(results):,.1f} ns/op")
    print(f"Python baseline min:    {min(results):,.1f} ns/op")
    print(f"Python baseline max:    {max(results):,.1f} ns/op")


if __name__ == "__main__":
    main()