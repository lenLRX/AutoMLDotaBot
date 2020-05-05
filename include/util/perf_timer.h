//
// Created by len on 5/4/20.
//

#ifndef AUTOMLDOTABOT_PERF_TIMER_H
#define AUTOMLDOTABOT_PERF_TIMER_H

#include <chrono>

#include "spdlog/spdlog.h"

class PerfTimer
{
public:
    PerfTimer(const char* file, int line, const char* func);
    ~PerfTimer();

private:
    std::chrono::steady_clock::time_point start;
    const char* file_;
    int line_;
    const char* func_;
};

#define _CONCAT_(x, y) x##y
#define __CONCAT__(x, y) _CONCAT_(x, y)

#define PERF_TIMER() \
    auto __CONCAT__(temp_perf_obj_, __LINE__) = PerfTimer(__FILE__, __LINE__, __FUNCTION__)

#endif //AUTOMLDOTABOT_PERF_TIMER_H
