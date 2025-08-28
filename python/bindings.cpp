#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "surprise_metrics.h"

namespace py = pybind11;
using namespace surprise_metrics;

PYBIND11_MODULE(pysurprise_metrics, m) {
    m.doc() = "High-performance surprise metrics calculator";
    
    py::class_<MetricsCalculator>(m, "MetricsCalculator")
        .def(py::init<int, size_t>(), 
             py::arg("num_gpus") = 1,
             py::arg("buffer_size") = 1000000)
        .def("set_garch_params", &MetricsCalculator::set_garch_params)
        .def("set_jump_threshold", &MetricsCalculator::set_jump_threshold)
        .def("set_window_size", &MetricsCalculator::set_window_size)
        .def("process_trades_batch",
             [](MetricsCalculator& self, 
                py::array_t<int64_t> timestamps,
                py::array_t<float> prices,
                py::array_t<int64_t> sizes) {
                 
                 auto ts = timestamps.unchecked<1>();
                 auto p = prices.unchecked<1>();
                 auto s = sizes.unchecked<1>();
                 
                 std::vector<Trade> trades;
                 trades.reserve(ts.shape(0));
                 
                 for (py::ssize_t i = 0; i < ts.shape(0); i++) {
                     Trade trade;
                     trade.timestamp = std::chrono::nanoseconds(ts(i));
                     trade.price = p(i);
                     trade.size = s(i);
                     trades.push_back(trade);
                 }
                 
                 self.process_trades(trades);
                 auto metrics = self.get_metrics();
                 
                 // Convert to numpy array
                 py::array_t<float> result({(py::ssize_t)metrics.size(), 6});
                 auto r = result.mutable_unchecked<2>();
                 
                 for (py::ssize_t i = 0; i < metrics.size(); i++) {
                     r(i, 0) = metrics[i].timestamp.count();
                     r(i, 1) = metrics[i].standardized_return;
                     r(i, 2) = metrics[i].lee_mykland_stat;
                     r(i, 3) = metrics[i].bns_stat;
                     r(i, 4) = metrics[i].trade_intensity_zscore;
                     r(i, 5) = metrics[i].jump_detected ? 1.0f : 0.0f;
                 }
                 
                 return result;
             });
}
