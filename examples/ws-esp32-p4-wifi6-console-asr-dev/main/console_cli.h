/**
 * @file console_cli.h
 * @brief Console Command Line Interface for ESP32-P4 ASR
 * 
 * Provides interactive console commands for system control and debugging.
 * 
 * Supported Commands:
 * - list [.ext]     List SD card files (all or filtered by extension)
 * - help            Show available commands
 * - info            Show system information
 * - reboot          Reboot the system
 * 
 * @author ESP32-P4 ASR Project
 * @date 2026-08-17
 */

#ifndef CONSOLE_CLI_H
#define CONSOLE_CLI_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize console CLI system
 * 
 * Creates a background task that monitors UART0 for console commands.
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t console_cli_init(void);

/**
 * @brief Process a single console command
 * 
 * @param command_line Command string to process
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t console_cli_process_command(const char* command_line);

/**
 * @brief List files on SD card
 * 
 * @param extension File extension filter (e.g., ".wav", ".espm") or NULL for all files
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t console_cli_list_files(const char* extension);

/**
 * @brief List an SD-card directory recursively
 *
 * @param path Absolute path rooted at /sd, or NULL/empty for /sd
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t console_cli_list_directory(const char* path);

/**
 * @brief Print help information
 */
void console_cli_print_help(void);

/**
 * @brief Print system information
 */
void console_cli_print_info(void);

#ifdef __cplusplus
}
#endif

#endif // CONSOLE_CLI_H
