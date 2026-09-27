import pytest

from market_data.orderbook import OrderBook
from features.microstructure import (
    order_book_imbalance,
    microprice
)


def test_orderbook_basic():

    book = OrderBook()

    book.update_bid(0.48, 100)
    book.update_bid(0.47, 200)

    book.update_ask(0.52, 150)
    book.update_ask(0.53, 250)

    assert book.best_bid == pytest.approx(0.48)
    assert book.best_ask == pytest.approx(0.52)

    assert book.mid_price == pytest.approx(0.50)
    assert book.spread == pytest.approx(0.04)


def test_order_book_imbalance():

    book = OrderBook()

    book.update_bid(0.48, 100)
    book.update_ask(0.52, 100)

    imbalance = order_book_imbalance(book)

    assert imbalance == pytest.approx(0.0)


def test_microprice():

    book = OrderBook()

    book.update_bid(0.48, 100)
    book.update_ask(0.52, 100)

    price = microprice(book)

    assert price == pytest.approx(0.50)