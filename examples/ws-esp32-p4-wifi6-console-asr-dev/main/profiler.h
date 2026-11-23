/**
 * @file profiler.h
 * @brief Performance profiling utilities
 */

#ifndef PROFILER_H
#define PROFILER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "esp_timer.h"

typedef struct {
    const char* name;
    uint64_t start_time;
    uint64_t total_time;
    uint32_t count;
} profile_timer_t;

#define PROFILE_START(timer) \
    do { (timer).start_time = esp_timer_get_time(); } while(0)

#define PROFILE_END(timer) \
    do { \
        uint64_t end = esp_timer_get_time(); \
        (timer).total_time += (end - (timer).start_time); \
        (timer).count++; \
    } while(0)

#define PROFILE_AVERAGE_MS(timer) \
    ((timer).count > 0 ? (float)((timer).total_time) / (timer).count / 1000.0f : 0.0f)

#define PROFILE_TOTAL_MS(timer) \
    ((float)((timer).total_time) / 1000.0f)

void profile_print_stats(profile_timer_t* timers, int num_timers);
void profile_reset(profile_timer_t* timer);

#ifdef __cplusplus
}
#endif

#endif // PROFILER_H
