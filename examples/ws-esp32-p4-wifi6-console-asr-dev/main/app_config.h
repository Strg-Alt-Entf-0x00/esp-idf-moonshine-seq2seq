/**
 * @file app_config.h
 * @brief ESP32-P4 ASR Application Configuration
 *
 * This file contains all configuration parameters for the ASR system.
 */

#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include "sdkconfig.h"
#include "soc/gpio_num.h"
#include "driver/uart.h"

#ifdef __cplusplus
extern "C" {
#endif

// Debug/Telemetry UART (UART0) - Console stays at 115200, no baudrate change!
// UART0 (GPIO37/38) is used by console, no additional configuration needed
#define ASR_DEBUG_UART_ENABLED      0  // UART0 console is always available
#define ASR_DEBUG_UART_NUM          UART_NUM_0  // Not used, console handles this
#define ASR_UART_TX_PIN             GPIO_NUM_37  // Console TX
#define ASR_UART_RX_PIN             GPIO_NUM_38  // Console RX

// Audio Channels
#define ASR_SAMPLE_RATE         16000
#define ASR_BITS_PER_SAMPLE     16
#define ASR_CHANNELS            1  // Mono

// ==================== OUTPUT CONFIGURATION ====================

// Maximum transcript length
#define ASR_MAX_TRANSCRIPT_LEN  256

// Output via UART
#define ASR_UART_NUM            UART_NUM_0
#define ASR_UART_BAUD_RATE      115200

// ==================== DEBUG / TESTING ====================

// Debug Log Levels (compile-time maximum level)
#define ASR_LOG_LEVEL_NONE      0
#define ASR_LOG_LEVEL_ERROR     1
#define ASR_LOG_LEVEL_WARN      2
#define ASR_LOG_LEVEL_INFO      3
#define ASR_LOG_LEVEL_DEBUG     4
#define ASR_LOG_LEVEL_VERBOSE   5

// Compile-time maximum log level (logs above this are removed at compile time)
#define ASR_LOG_COMPILE_LEVEL   ASR_LOG_LEVEL_DEBUG

// Runtime default log level (can be changed at runtime)
#define ASR_LOG_DEFAULT_LEVEL   ASR_LOG_LEVEL_INFO

// Debug flags (can be changed at runtime via console commands)
#define ASR_DEBUG_AUDIO_STATS   1  // Show audio buffer statistics
#define ASR_DEBUG_MEMORY_STATS  1  // Show memory usage
#define ASR_DEBUG_TIMING        1  // Show timing/performance stats
#define ASR_DEBUG_I2S_DETAILS   0  // Detailed I2S buffer info
#define ASR_DEBUG_INFERENCE     0  // Detailed inference debug

// Save audio to SD card (for debugging)
#define ASR_SAVE_AUDIO          0

#ifdef __cplusplus
}
#endif

#endif // APP_CONFIG_H
