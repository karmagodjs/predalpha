from market_data.optimized_orderbook import OptimizedOrderBook


def test_best_prices_and_spread():
    book = OptimizedOrderBook()
    book.update_bid(0.40, 10)
    book.update_bid(0.42, 5)
    book.update_ask(0.45, 8)
    book.update_ask(0.44, 6)

    assert book.best_bid == 0.42
    assert book.best_ask == 0.44
    assert abs(book.mid_price - 0.43) < 1e-12
    assert abs(book.spread - 0.02) < 1e-12


def test_delete_price_level():
    book = OptimizedOrderBook()
    book.update_bid(0.40, 10)
    book.update_bid(0.42, 5)

    book.update_bid(0.42, 0)

    assert book.best_bid == 0.40
    assert 0.42 not in book.bids


def test_update_existing_level():
    book = OptimizedOrderBook()
    book.update_ask(0.50, 10)
    book.update_ask(0.50, 25)

    assert book.best_ask == 0.50
    assert book.asks[0.50] == 25
    assert len(book._ask_prices) == 1