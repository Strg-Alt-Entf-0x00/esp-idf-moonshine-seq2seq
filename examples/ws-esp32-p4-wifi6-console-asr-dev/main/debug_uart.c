/**
 * @file debug_uart.c
 * @brief Debug UART Implementation (UART1)
 */

#include "debug_uart.h"
#include "app_config.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <stdio.h>
#include <string.h>

static const char* TAG = "debug_uart";

// State
static bool g_initialized = false;
static bool g_enabled = true;
static bool g_timestamps = true;

// Configuration
#define DEBUG_UART_TX_BUF_SIZE  (2048)
#define DEBUG_UART_RX_BUF_SIZE  (256)    // Minimal RX buffer (required by driver)
#define DEBUG_UART_QUEUE_SIZE   (0)      // No event queue needed

/* ==================================================================
 * INITIALIZATION
 * ================================================================== */

esp_err_t debug_uart_init(void) {
    if (g_initialized) {
        ESP_LOGW(TAG, "Debug UART already initialized");
        return ESP_OK;
    }
    
#if ASR_DEBUG_UART_ENABLED
    ESP_LOGI(TAG, "Initializing Debug UART (UART%d @ %d baud)", 
             ASR_DEBUG_UART_NUM, ASR_DEBUG_UART_BAUD_RATE);
    
    // UART configuration
    uart_config_t uart_config = {
        .baud_rate = ASR_DEBUG_UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
#if ASR_DEBUG_UART_FLOW_CTRL
        .flow_ctrl = UART_HW_FLOWCTRL_CTS_RTS,  // Hardware flow control enabled
        .rx_flow_ctrl_thresh = 122,
#else
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,  // No flow control (sufficient for 115200)
#endif
        .source_clk = UART_SCLK_DEFAULT,
    };
    
    // Configure UART parameters
    esp_err_t ret = uart_param_config(ASR_DEBUG_UART_NUM, &uart_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure UART: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Set UART pins
    ret = uart_set_pin(
        ASR_DEBUG_UART_NUM,
        ASR_DEBUG_UART_TX,   // TX: GPIO2
        ASR_DEBUG_UART_RX,   // RX: GPIO3 (not used, but must be set)
#if ASR_DEBUG_UART_FLOW_CTRL
        ASR_DEBUG_UART_RTS,  // RTS: GPIO4 (only if flow control enabled)
        ASR_DEBUG_UART_CTS   // CTS: GPIO5 (only if flow control enabled)
#else
        UART_PIN_NO_CHANGE,  // RTS not used
        UART_PIN_NO_CHANGE   // CTS not used
#endif
    );
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set UART pins: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Install UART driver
    ret = uart_driver_install(
        ASR_DEBUG_UART_NUM,
        DEBUG_UART_RX_BUF_SIZE,  // RX buffer (0 for output-only)
        DEBUG_UART_TX_BUF_SIZE,  // TX buffer
        DEBUG_UART_QUEUE_SIZE,   // Event queue size
        NULL,                    // Event queue handle
        0                        // Interrupt flags
    );
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to install UART driver: %s", esp_err_to_name(ret));
        return ret;
    }
    
    g_initialized = true;
    
    // Send initialization message
    debug_uart_printf("\r\n");
    debug_uart_printf("================================================================================\r\n");
    debug_uart_printf("ESP32-P4 ASR - Debug UART Initialized\r\n");
    debug_uart_printf("================================================================================\r\n");
    debug_uart_printf("UART%d @ %d baud on GPIO%d (TX) / GPIO%d (RX)\r\n",
                     ASR_DEBUG_UART_NUM, ASR_DEBUG_UART_BAUD_RATE,
                     ASR_DEBUG_UART_TX, ASR_DEBUG_UART_RX);
#if ASR_DEBUG_UART_FLOW_CTRL
    debug_uart_printf("Hardware Flow Control: ENABLED (RTS=GPIO%d, CTS=GPIO%d)\r\n",
                     ASR_DEBUG_UART_RTS, ASR_DEBUG_UART_CTS);
#else
    debug_uart_printf("Hardware Flow Control: DISABLED\r\n");
#endif
    debug_uart_printf("================================================================================\r\n");
    debug_uart_printf("\r\n");
    
    ESP_LOGI(TAG, "✓ Debug UART initialized successfully");
    return ESP_OK;
    
#else
    ESP_LOGW(TAG, "Debug UART disabled in configuration");
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

esp_err_t debug_uart_deinit(void) {
    if (!g_initialized) {
        return ESP_OK;
    }
    
    uart_driver_delete(ASR_DEBUG_UART_NUM);
    g_initialized = false;
    
    return ESP_OK;
}

bool debug_uart_is_initialized(void) {
    return g_initialized;
}

/* ==================================================================
 * BASIC OUTPUT
 * ================================================================== */

int debug_uart_write(const uint8_t* data, size_t length) {
    if (!g_initialized || !g_enabled || !data || length == 0) {
        return 0;
    }
    
    return uart_write_bytes(ASR_DEBUG_UART_NUM, (const char*)data, length);
}

int debug_uart_puts(const char* str) {
    if (!str) {
        return 0;
    }
    return debug_uart_write((const uint8_t*)str, strlen(str));
}

int debug_uart_printf(const char* format, ...) {
    if (!g_initialized || !g_enabled || !format) {
        return 0;
    }
    
    char buffer[256];
    va_list args;
    va_start(args, format);
    int len = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    if (len > 0 && len < (int)sizeof(buffer)) {
        return debug_uart_write((const uint8_t*)buffer, len);
    }
    
    return len;
}

/* ==================================================================
 * TELEMETRY - STRUCTURED LOGGING
 * ================================================================== */

void debug_uart_log_audio(uint8_t buffer_fill_percent, bool vad_active, uint32_t frames_processed) {
    if (!g_initialized || !g_enabled) return;
    
    if (g_timestamps) {
        int64_t time_us = esp_timer_get_time();
        debug_uart_printf("[%lld] ", time_us / 1000);  // milliseconds
    }
    
    debug_uart_printf("AUDIO,FILL=%d,VAD=%d,FRAMES=%lu\r\n",
                     buffer_fill_percent, vad_active ? 1 : 0, frames_processed);
}

void debug_uart_log_inference(uint32_t time_ms, float rtf, const char* engine) {
    if (!g_initialized || !g_enabled) return;
    
    if (g_timestamps) {
        int64_t time_us = esp_timer_get_time();
        debug_uart_printf("[%lld] ", time_us / 1000);
    }
    
    debug_uart_printf("INFERENCE,TIME_MS=%lu,RTF=%.3f,ENGINE=%s\r\n",
                     time_ms, rtf, engine ? engine : "unknown");
}

void debug_uart_log_memory(size_t heap_free, size_t psram_free, size_t largest_block) {
    if (!g_initialized || !g_enabled) return;
    
    if (g_timestamps) {
        int64_t time_us = esp_timer_get_time();
        debug_uart_printf("[%lld] ", time_us / 1000);
    }
    
    debug_uart_printf("MEMORY,HEAP=%zu,PSRAM=%zu,LARGEST=%zu\r\n",
                     heap_free, psram_free, largest_block);
}

void debug_uart_log_result(const char* text, float confidence, uint32_t time_ms) {
    if (!g_initialized || !g_enabled) return;
    
    if (g_timestamps) {
        int64_t time_us = esp_timer_get_time();
        debug_uart_printf("[%lld] ", time_us / 1000);
    }
    
    debug_uart_printf("RESULT,TEXT=\"%s\",CONF=%.2f,TIME_MS=%lu\r\n",
                     text ? text : "", confidence, time_ms);
}

/* ==================================================================
 * PERFORMANCE PROFILING
 * ================================================================== */

void debug_uart_log_perf_marker(const char* marker, int64_t time_us) {
    if (!g_initialized || !g_enabled) return;
    
    debug_uart_printf("PERF,MARKER=%s,TIME_US=%lld\r\n",
                     marker ? marker : "unknown", time_us);
}

void debug_uart_log_perf_interval(const char* stage, int64_t duration_us) {
    if (!g_initialized || !g_enabled) return;
    
    debug_uart_printf("PERF,STAGE=%s,DURATION_US=%lld\r\n",
                     stage ? stage : "unknown", duration_us);
}

/* ==================================================================
 * ADVANCED - BINARY DATA STREAMING
 * ================================================================== */

void debug_uart_stream_audio(const int16_t* samples, uint32_t num_samples) {
    if (!g_initialized || !g_enabled || !samples) return;
    
    // Simple binary format: header + data
    debug_uart_printf("AUDIO_STREAM,%lu\r\n", num_samples);
    debug_uart_write((const uint8_t*)samples, num_samples * sizeof(int16_t));
    debug_uart_printf("\r\n");
}

void debug_uart_stream_features(const float* features, uint32_t num_frames, uint32_t num_mels) {
    if (!g_initialized || !g_enabled || !features) return;
    
    // Simple binary format: header + data
    debug_uart_printf("FEATURE_STREAM,%lu,%lu\r\n", num_frames, num_mels);
    debug_uart_write((const uint8_t*)features, num_frames * num_mels * sizeof(float));
    debug_uart_printf("\r\n");
}

/* ==================================================================
 * CONFIGURATION
 * ================================================================== */

void debug_uart_set_enabled(bool enable) {
    g_enabled = enable;
}

void debug_uart_set_timestamps(bool enable) {
    g_timestamps = enable;
}
