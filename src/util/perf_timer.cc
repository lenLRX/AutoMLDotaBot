//
// Created by len on 5/4/20.
//

#include "util/perf_timer.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <sstream>

using namespace std::chrono;

PerfTimer::PerfTimer(const char *file, int line, const char *func) {
    file_ = file;
    line_ = line;
    func_ = func;
    start = steady_clock::now();
}

PerfTimer::~PerfTimer() {
    auto end = steady_clock::now();
    auto duration = end - start;
    microseconds duration_us = duration_cast<microseconds>(duration);

    static auto loss_logger = spdlog::basic_logger_mt("perf_logger", "perf.log", true);
    std::stringstream ss;
    ss << file_ << ":" << line_ << " func:" << func_
      << " duration:" << duration_us.count() << "us";
    loss_logger->info(ss.str());
}
