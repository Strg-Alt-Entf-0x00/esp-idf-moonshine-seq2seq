/**
 * @file profiler.c
 * @brief Performance profiling implementation
 */

#include "profiler.h"
#include "esp_log.h"

static const char* TAG = "PROFILER";

void profile_print_stats(profile_timer_t* timers, int num_timers)
{
    ESP_LOGI(TAG, "==== Performance Statistics ====");
    for (int i = 0; i < num_timers; i++) {
        float avg_ms = PROFILE_AVERAGE_MS(timers[i]);
        float total_ms = PROFILE_TOTAL_MS(timers[i]);
        ESP_LOGI(TAG, "%s: %.2f ms (avg), %.2f ms (total), %d calls",
                 timers[i].name, avg_ms, total_ms, timers[i].count);
    }
    ESP_LOGI(TAG, "================================");
}

void profile_reset(profile_timer_t* timer)
{
    timer->start_time = 0;
    timer->total_time = 0;
    timer->count = 0;
}
