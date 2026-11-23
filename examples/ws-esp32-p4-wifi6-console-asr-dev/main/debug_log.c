/**
 * @file debug_log.c
 * @brief Professional Debug Logging Implementation
 */

#include "debug_log.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_system.h"
#include <string.h>

static const char* TAG = "DEBUG";

// Runtime debug flags
uint32_t g_debug_flags = 0;

// Per-module log levels
static esp_log_level_t g_module_log_levels[DBG_MODULE_MAX];

// Module names for display
static const char* g_module_names[DBG_MODULE_MAX] = {
    "MAIN",
    "AUDIO",
    "INFERENCE",
    "DECODER",
    "AFE",
    "FEATURES",
    "I2S"
};

void debug_init(void)
{
    // Initialize with default log levels
    for (int i = 0; i < DBG_MODULE_MAX; i++) {
        g_module_log_levels[i] = ASR_LOG_DEFAULT_LEVEL;
    }

    // Initialize debug flags from config
    g_debug_flags = 0;
#if ASR_DEBUG_AUDIO_STATS
    g_debug_flags |= DBG_FLAG_AUDIO_STATS;
#endif
#if ASR_DEBUG_MEMORY_STATS
    g_debug_flags |= DBG_FLAG_MEMORY_STATS;
#endif
#if ASR_DEBUG_TIMING
    g_debug_flags |= DBG_FLAG_TIMING;
#endif
#if ASR_DEBUG_I2S_DETAILS
    g_debug_flags |= DBG_FLAG_I2S_DETAILS;
#endif
#if ASR_DEBUG_INFERENCE
    g_debug_flags |= DBG_FLAG_INFERENCE;
#endif

    ESP_LOGI(TAG, "Debug system initialized");
    ESP_LOGI(TAG, "Compile-time max level: %d", ASR_LOG_COMPILE_LEVEL);
    ESP_LOGI(TAG, "Runtime default level: %d", ASR_LOG_DEFAULT_LEVEL);
    ESP_LOGI(TAG, "Debug flags: 0x%08X", g_debug_flags);
}

void debug_set_level(debug_module_t module, esp_log_level_t level)
{
    if (module >= DBG_MODULE_MAX) return;
    g_module_log_levels[module] = level;
    ESP_LOGI(TAG, "Module %s log level set to %d", g_module_names[module], level);
}

esp_log_level_t debug_get_level(debug_module_t module)
{
    if (module >= DBG_MODULE_MAX) return ESP_LOG_NONE;
    return g_module_log_levels[module];
}

void debug_enable_flag(uint32_t flag)
{
    g_debug_flags |= flag;
    ESP_LOGI(TAG, "Debug flag 0x%08X enabled (flags=0x%08X)", flag, g_debug_flags);
}

void debug_disable_flag(uint32_t flag)
{
    g_debug_flags &= ~flag;
    ESP_LOGI(TAG, "Debug flag 0x%08X disabled (flags=0x%08X)", flag, g_debug_flags);
}

bool debug_is_flag_enabled(uint32_t flag)
{
    return (g_debug_flags & flag) != 0;
}

void debug_set_flags(uint32_t flags)
{
    g_debug_flags = flags;
    ESP_LOGI(TAG, "Debug flags set to 0x%08X", g_debug_flags);
}

uint32_t debug_get_flags(void)
{
    return g_debug_flags;
}

void debug_print_status(void)
{
    ESP_LOGI(TAG, "=== Debug System Status ===");
    ESP_LOGI(TAG, "Compile-time max level: %d", ASR_LOG_COMPILE_LEVEL);
    
    ESP_LOGI(TAG, "\nModule Log Levels:");
    for (int i = 0; i < DBG_MODULE_MAX; i++) {
        const char* level_str;
        switch (g_module_log_levels[i]) {
            case ESP_LOG_NONE:    level_str = "NONE"; break;
            case ESP_LOG_ERROR:   level_str = "ERROR"; break;
            case ESP_LOG_WARN:    level_str = "WARN"; break;
            case ESP_LOG_INFO:    level_str = "INFO"; break;
            case ESP_LOG_DEBUG:   level_str = "DEBUG"; break;
            case ESP_LOG_VERBOSE: level_str = "VERBOSE"; break;
            default:              level_str = "UNKNOWN"; break;
        }
        ESP_LOGI(TAG, "  %-12s: %s", g_module_names[i], level_str);
    }
    
    ESP_LOGI(TAG, "\nDebug Flags (0x%08X):", g_debug_flags);
    ESP_LOGI(TAG, "  Audio Stats   : %s", (g_debug_flags & DBG_FLAG_AUDIO_STATS) ? "ON" : "OFF");
    ESP_LOGI(TAG, "  Memory Stats  : %s", (g_debug_flags & DBG_FLAG_MEMORY_STATS) ? "ON" : "OFF");
    ESP_LOGI(TAG, "  Timing        : %s", (g_debug_flags & DBG_FLAG_TIMING) ? "ON" : "OFF");
    ESP_LOGI(TAG, "  I2S Details   : %s", (g_debug_flags & DBG_FLAG_I2S_DETAILS) ? "ON" : "OFF");
    ESP_LOGI(TAG, "  Inference     : %s", (g_debug_flags & DBG_FLAG_INFERENCE) ? "ON" : "OFF");
    ESP_LOGI(TAG, "  Ring Buffer   : %s", (g_debug_flags & DBG_FLAG_RING_BUFFER) ? "ON" : "OFF");
    
    ESP_LOGI(TAG, "\nMemory Status:");
    size_t free_heap = esp_get_free_heap_size();
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t total_psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "  Free Heap : %d bytes", free_heap);
    ESP_LOGI(TAG, "  Free PSRAM: %d / %d bytes (%.1f%%)", 
             free_psram, total_psram, (float)free_psram * 100.0f / total_psram);
    
    ESP_LOGI(TAG, "===========================");
}
