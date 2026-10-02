#include <pybind11/pybind11.h>
#include "orderbook.cpp"

namespace py = pybind11;

PYBIND11_MODULE(orderbook_cpp, m) {
    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("update_bid", &OrderBook::update_bid)
        .def("update_ask", &OrderBook::update_ask)
        .def_property_readonly("best_bid", &OrderBook::best_bid)
        .def_property_readonly("best_ask", &OrderBook::best_ask)
        .def_property_readonly("mid_price", &OrderBook::mid_price)
        .def_property_readonly("spread", &OrderBook::spread);
}