/**
 * @file debug_uart.h
 * @brief Secondary UART for Debug Logging and Telemetry
 * 
 * Provides a dedicated UART channel (UART1) for debug output and telemetry.
 * 
 * Features:
 * - Real-time debug logging
 * - Performance metrics and telemetry
 * - Audio statistics and monitoring
 * - Optional hardware flow control (RTS/CTS)
 * 
 * Hardware:
 * - UART1 on GPIO2 (TX) / GPIO3 (RX)
 * - Optional flow control: GPIO4 (RTS) / GPIO5 (CTS)
 * - FT232 USB-Serial adapter (COM13)
 * - Baudrate: 115200 (human-readable)
 */

#ifndef DEBUG_UART_H
#define DEBUG_UART_H

#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ==================================================================
 * INITIALIZATION
 * ================================================================== */

/**
 * @brief Initialize debug UART (UART1)
 * 
 * Sets up UART1 on GPIO2/3 for debug output at 115200 baud.
 * Optional hardware flow control on GPIO4/5 if enabled in config.
 * 
 * @return ESP_OK on success
 */
esp_err_t debug_uart_init(void);

/**
 * @brief Deinitialize debug UART
 * 
 * @return ESP_OK on success
 */
esp_err_t debug_uart_deinit(void);

/**
 * @brief Check if debug UART is initialized
 * 
 * @return true if initialized
 */
bool debug_uart_is_initialized(void);

/* ==================================================================
 * BASIC OUTPUT
 * ================================================================== */

/**
 * @brief Print formatted string to debug UART
 * 
 * Works like printf() but outputs to UART1.
 * Non-blocking, buffered output.
 * 
 * @param format Printf-style format string
 * @param ... Variable arguments
 * @return Number of bytes written, or -1 on error
 */
int debug_uart_printf(const char* format, ...) __attribute__((format(printf, 1, 2)));

/**
 * @brief Write raw data to debug UART
 * 
 * @param data Data buffer
 * @param length Number of bytes to write
 * @return Number of bytes written
 */
int debug_uart_write(const uint8_t* data, size_t length);

/**
 * @brief Write string to debug UART
 * 
 * @param str Null-terminated string
 * @return Number of bytes written
 */
int debug_uart_puts(const char* str);

/* ==================================================================
 * TELEMETRY - STRUCTURED LOGGING
 * ================================================================== */

/**
 * @brief Log audio processing statistics
 * 
 * Format: "AUDIO,FILL=%d,VAD=%d,FRAMES=%d"
 * 
 * @param buffer_fill_percent Ring buffer fill percentage (0-100)
 * @param vad_active Voice activity detected (0/1)
 * @param frames_processed Number of frames processed
 */
void debug_uart_log_audio(uint8_t buffer_fill_percent, bool vad_active, uint32_t frames_processed);

/**
 * @brief Log inference timing and performance
 * 
 * Format: "INFERENCE,TIME_MS=%d,RTF=%.3f,ENGINE=%s"
 * 
 * @param time_ms Inference time in milliseconds
 * @param rtf Real-time factor (inference_time / audio_duration)
 * @param engine Engine name ("encoder", "decoder", "end2end")
 */
void debug_uart_log_inference(uint32_t time_ms, float rtf, const char* engine);

/**
 * @brief Log memory usage statistics
 * 
 * Format: "MEMORY,HEAP=%d,PSRAM=%d,LARGEST=%d"
 * 
 * @param heap_free Free heap memory in bytes
 * @param psram_free Free PSRAM in bytes
 * @param largest_block Largest contiguous block in bytes
 */
void debug_uart_log_memory(size_t heap_free, size_t psram_free, size_t largest_block);

/**
 * @brief Log recognition result
 * 
 * Format: "RESULT,TEXT=%s,CONF=%.2f,TIME_MS=%d"
 * 
 * @param text Recognized text
 * @param confidence Confidence score (0.0-1.0)
 * @param time_ms Recognition time in milliseconds
 */
void debug_uart_log_result(const char* text, float confidence, uint32_t time_ms);

/* ==================================================================
 * PERFORMANCE PROFILING
 * ================================================================== */

/**
 * @brief Log performance marker with timestamp
 * 
 * Format: "PERF,MARKER=%s,TIME_US=%lld"
 * 
 * @param marker Marker name
 * @param time_us Timestamp in microseconds
 */
void debug_uart_log_perf_marker(const char* marker, int64_t time_us);

/**
 * @brief Log performance interval
 * 
 * Format: "PERF,STAGE=%s,DURATION_US=%lld"
 * 
 * @param stage Processing stage name
 * @param duration_us Duration in microseconds
 */
void debug_uart_log_perf_interval(const char* stage, int64_t duration_us);

/* ==================================================================
 * ADVANCED - BINARY DATA STREAMING (OPTIONAL)
 * ================================================================== */

/**
 * @brief Stream raw audio samples
 * 
 * Useful for visualizing audio input in real-time.
 * Format: Binary int16 samples
 * 
 * @param samples Audio sample buffer
 * @param num_samples Number of samples
 */
void debug_uart_stream_audio(const int16_t* samples, uint32_t num_samples);

/**
 * @brief Stream mel-spectrogram features
 * 
 * Useful for visualizing features in real-time.
 * Format: Binary float32 features
 * 
 * @param features Feature buffer (num_frames × num_mels)
 * @param num_frames Number of frames
 * @param num_mels Number of mel bins per frame
 */
void debug_uart_stream_features(const float* features, uint32_t num_frames, uint32_t num_mels);

/* ==================================================================
 * CONFIGURATION
 * ================================================================== */

/**
 * @brief Enable/disable debug UART output
 * 
 * @param enable true to enable, false to disable
 */
void debug_uart_set_enabled(bool enable);

/**
 * @brief Set timestamp mode for log messages
 * 
 * @param enable true to prefix messages with timestamps
 */
void debug_uart_set_timestamps(bool enable);

#ifdef __cplusplus
}
#endif

#endif // DEBUG_UART_H
