/**
 * @file debug_log.h
 * @brief Professional Debug Logging System with Runtime Control
 * 
 * Features:
 * - Compile-time log level filtering (zero overhead for disabled logs)
 * - Runtime log level control per module
 * - Color-coded output
 * - Performance counters
 * - Memory tracking
 */

#ifndef DEBUG_LOG_H
#define DEBUG_LOG_H

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include <stdint.h>
#include <stdbool.h>

// Debug Log Levels (compile-time maximum level)
#ifndef ASR_LOG_LEVEL_NONE
#define ASR_LOG_LEVEL_NONE      0
#define ASR_LOG_LEVEL_ERROR     1
#define ASR_LOG_LEVEL_WARN      2
#define ASR_LOG_LEVEL_INFO      3
#define ASR_LOG_LEVEL_DEBUG     4
#define ASR_LOG_LEVEL_VERBOSE   5
#endif

// Compile-time maximum log level (can be overridden by app_config.h)
#ifndef ASR_LOG_COMPILE_LEVEL
#define ASR_LOG_COMPILE_LEVEL   ASR_LOG_LEVEL_DEBUG
#endif

// Runtime default log level
#ifndef ASR_LOG_DEFAULT_LEVEL
#define ASR_LOG_DEFAULT_LEVEL   ASR_LOG_LEVEL_INFO
#endif

// Debug flags defaults (can be overridden by app_config.h)
#ifndef ASR_DEBUG_AUDIO_STATS
#define ASR_DEBUG_AUDIO_STATS   1
#endif
#ifndef ASR_DEBUG_MEMORY_STATS
#define ASR_DEBUG_MEMORY_STATS  1
#endif
#ifndef ASR_DEBUG_TIMING
#define ASR_DEBUG_TIMING        1
#endif
#ifndef ASR_DEBUG_I2S_DETAILS
#define ASR_DEBUG_I2S_DETAILS   0
#endif
#ifndef ASR_DEBUG_INFERENCE
#define ASR_DEBUG_INFERENCE     0
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Debug module IDs
typedef enum {
    DBG_MODULE_MAIN = 0,
    DBG_MODULE_AUDIO,
    DBG_MODULE_INFERENCE,
    DBG_MODULE_DECODER,
    DBG_MODULE_AFE,
    DBG_MODULE_FEATURES,
    DBG_MODULE_I2S,
    DBG_MODULE_MAX
} debug_module_t;

// Runtime debug flags
extern uint32_t g_debug_flags;

#define DBG_FLAG_AUDIO_STATS    (1 << 0)
#define DBG_FLAG_MEMORY_STATS   (1 << 1)
#define DBG_FLAG_TIMING         (1 << 2)
#define DBG_FLAG_I2S_DETAILS    (1 << 3)
#define DBG_FLAG_INFERENCE      (1 << 4)
#define DBG_FLAG_RING_BUFFER    (1 << 5)

// Macros for compile-time log level filtering
#if ASR_LOG_COMPILE_LEVEL >= ASR_LOG_LEVEL_ERROR
    #define LOG_ERROR(tag, format, ...) ESP_LOGE(tag, format, ##__VA_ARGS__)
#else
    #define LOG_ERROR(tag, format, ...) ((void)0)
#endif

#if ASR_LOG_COMPILE_LEVEL >= ASR_LOG_LEVEL_WARN
    #define LOG_WARN(tag, format, ...) ESP_LOGW(tag, format, ##__VA_ARGS__)
#else
    #define LOG_WARN(tag, format, ...) ((void)0)
#endif

#if ASR_LOG_COMPILE_LEVEL >= ASR_LOG_LEVEL_INFO
    #define LOG_INFO(tag, format, ...) ESP_LOGI(tag, format, ##__VA_ARGS__)
#else
    #define LOG_INFO(tag, format, ...) ((void)0)
#endif

#if ASR_LOG_COMPILE_LEVEL >= ASR_LOG_LEVEL_DEBUG
    #define LOG_DEBUG(tag, format, ...) ESP_LOGD(tag, format, ##__VA_ARGS__)
#else
    #define LOG_DEBUG(tag, format, ...) ((void)0)
#endif

#if ASR_LOG_COMPILE_LEVEL >= ASR_LOG_LEVEL_VERBOSE
    #define LOG_VERBOSE(tag, format, ...) ESP_LOGV(tag, format, ##__VA_ARGS__)
#else
    #define LOG_VERBOSE(tag, format, ...) ((void)0)
#endif

// Conditional debug logs (only if flag is set)
#define LOG_DEBUG_IF(flag, tag, format, ...) \
    do { if (g_debug_flags & flag) ESP_LOGI(tag, "[DBG] " format, ##__VA_ARGS__); } while(0)

// Performance measurement macros
#define PERF_START(name) int64_t _perf_start_##name = esp_timer_get_time()

#define PERF_END(name, tag) \
    do { \
        if (g_debug_flags & DBG_FLAG_TIMING) { \
            int64_t _perf_duration = esp_timer_get_time() - _perf_start_##name; \
            ESP_LOGI(tag, "[PERF] %s: %lld us", #name, _perf_duration); \
        } \
    } while(0)

// Memory tracking macros
#define MEM_TRACK(tag, label) \
    do { \
        if (g_debug_flags & DBG_FLAG_MEMORY_STATS) { \
            size_t free_heap = esp_get_free_heap_size(); \
            size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM); \
            ESP_LOGI(tag, "[MEM] %s - Heap: %d, PSRAM: %d", label, free_heap, free_psram); \
        } \
    } while(0)

// Debug initialization
void debug_init(void);

// Runtime log level control
void debug_set_level(debug_module_t module, esp_log_level_t level);
esp_log_level_t debug_get_level(debug_module_t module);

// Debug flag control
void debug_enable_flag(uint32_t flag);
void debug_disable_flag(uint32_t flag);
bool debug_is_flag_enabled(uint32_t flag);
void debug_set_flags(uint32_t flags);
uint32_t debug_get_flags(void);

// Print debug status
void debug_print_status(void);

#ifdef __cplusplus
}
#endif

#endif // DEBUG_LOG_H
