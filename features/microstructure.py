def order_book_imbalance(book, levels=5):

    bids = sorted(
        book.bids.items(),
        reverse=True
    )[:levels]

    asks = sorted(
        book.asks.items()
    )[:levels]
    bid_volume = sum(size for _, size in bids)
    ask_volume = sum(size for _, size in asks)
    total = bid_volume + ask_volume

    if total == 0:
        return 0.0

    return (bid_volume - ask_volume) / total


def microprice(book):

    if book.best_bid is None or book.best_ask is None:
        return None
    bid_size = book.bids[book.best_bid]
    ask_size = book.asks[book.best_ask]
    total = bid_size + ask_size

    if total == 0:
        return None

    return (
        book.best_ask * bid_size +
        book.best_bid * ask_size
    ) / total