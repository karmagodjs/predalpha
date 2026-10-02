#include <map>
#include <stdexcept>

class OrderBook {
public:
    void update_bid(double price, double size) {
        update(bids, price, size);
    }

    void update_ask(double price, double size) {
        update(asks, price, size);
    }

    double best_bid() const {
        if (bids.empty()) return -1.0;
        return bids.rbegin()->first;
    }

    double best_ask() const {
        if (asks.empty()) return -1.0;
        return asks.begin()->first;
    }

    double mid_price() const {
        double bid = best_bid();
        double ask = best_ask();
        if (bid < 0.0 || ask < 0.0) return -1.0;
        return (bid + ask) / 2.0;
    }

    double spread() const {
        double bid = best_bid();
        double ask = best_ask();
        if (bid < 0.0 || ask < 0.0) return -1.0;
        return ask - bid;
    }

private:
    std::map<double, double> bids;
    std::map<double, double> asks;

    static void update(
        std::map<double, double>& side,
        double price,
        double size
    ) {
        if (size < 0.0) {
            throw std::invalid_argument("Order size cannot be negative");
        }

        if (size == 0.0) {
            side.erase(price);
        } else {
            side[price] = size;
        }
    }
};