/**
 * @file console_cli.c
 * @brief Console Command Line Interface Implementation
 */

#include "console_cli.h"

#include "esp_log.h"
#include "esp_system.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

static const char *TAG = "CONSOLE_CLI";

#define CONSOLE_UART_NUM UART_NUM_0
#define CONSOLE_RX_BUF_SIZE 256
#define MAX_COMMAND_LENGTH 128
#define MAX_LIST_PATH 512

// Forward declaration - g_fs_manager is a C++ object, we'll check it via pointer
extern void* g_fs_manager;  // FilesystemManager* in main.cpp


// Helper function to check if SD is mounted (calls C++ method via main.cpp)


/**
 * @brief List files recursively in a directory
 */
static esp_err_t list_directory_recursive(const char* path, const char* extension,
                                          int level, size_t* file_count,
                                          size_t* dir_count)
{
    DIR* dir = opendir(path);
    if (!dir) {
        ESP_LOGE(TAG, "Failed to open directory: %s", path);
        return ESP_FAIL;
    }
    
    struct dirent* entry;
    struct stat st;
    char full_path[MAX_LIST_PATH];
    esp_err_t result = ESP_OK;
    
    while ((entry = readdir(dir)) != NULL) {
        // Skip . and ..
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        const int path_length = snprintf(full_path, sizeof(full_path), "%s/%s",
                                         path, entry->d_name);
        if (path_length < 0 || (size_t)path_length >= sizeof(full_path)) {
            ESP_LOGW(TAG, "Skipping path that exceeds the listing limit");
            result = ESP_ERR_INVALID_SIZE;
            continue;
        }
        
        if (stat(full_path, &st) != 0) {
            ESP_LOGW(TAG, "Could not inspect SD entry: %s", full_path);
            result = ESP_FAIL;
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            for (int i = 0; i < level; i++) {
                printf("  ");
            }
            printf("📁 %s/\n", entry->d_name);
            ++(*dir_count);

            if (list_directory_recursive(full_path, extension, level + 1,
                                         file_count, dir_count) != ESP_OK) {
                result = ESP_FAIL;
            }
        } else {
            bool show = extension == NULL || strlen(extension) == 0;
            if (!show) {
                const char* file_ext = strrchr(entry->d_name, '.');
                show = file_ext && strcmp(file_ext, extension) == 0;
            }

            if (show) {
                for (int i = 0; i < level; i++) {
                    printf("  ");
                }

                char size_str[32];
                if (st.st_size < 1024) {
                    snprintf(size_str, sizeof(size_str), "%llu B",
                             (unsigned long long)st.st_size);
                } else if (st.st_size < 1024 * 1024) {
                    snprintf(size_str, sizeof(size_str), "%.1f KB",
                             st.st_size / 1024.0);
                } else {
                    snprintf(size_str, sizeof(size_str), "%.2f MB",
                             st.st_size / (1024.0 * 1024.0));
                }

                printf("📄 %-40s %10s\n", entry->d_name, size_str);
                ++(*file_count);
            }
        }
    }
    
    closedir(dir);
    return result;
}

static esp_err_t list_directory_contents(const char* path, const char* extension)
{
    size_t file_count = 0;
    size_t dir_count = 0;

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    if (extension && strlen(extension) > 0) {
        printf("SD Card Contents (%s, filter: %s)\n", path, extension);
    } else {
        printf("SD Card Contents (%s)\n", path);
    }
    printf("═══════════════════════════════════════════════════════════════\n");

    const esp_err_t result = list_directory_recursive(path, extension, 0,
                                                     &file_count, &dir_count);
    if (result == ESP_OK && extension && strlen(extension) > 0) {
        printf("\nFound %zu file(s) matching '%s'\n", file_count, extension);
    } else if (result == ESP_OK) {
        printf("\nTotal: %zu file(s), %zu director%s\n", file_count, dir_count,
               dir_count == 1 ? "y" : "ies");
    }

    printf("═══════════════════════════════════════════════════════════════\n\n");
    return result;
}

esp_err_t console_cli_list_files(const char* extension)
{
    return list_directory_contents("/sd", extension);
}

static bool path_is_within_sd_root(const char* path)
{
    if (!path || strncmp(path, "/sd", 3) != 0 ||
        (path[3] != '\0' && path[3] != '/')) {
        return false;
    }

    const char* component = path + 3;
    while (*component) {
        while (*component == '/') {
            ++component;
        }
        if (!*component) {
            break;
        }

        const char* separator = strchr(component, '/');
        const size_t component_length = separator
            ? (size_t)(separator - component) : strlen(component);
        if (component_length == 2 && component[0] == '.' &&
            component[1] == '.') {
            return false;
        }
        component = separator ? separator : component + component_length;
    }
    return true;
}

esp_err_t console_cli_list_directory(const char* path)
{
    const char* directory = (path && path[0]) ? path : "/sd";
    if (!path_is_within_sd_root(directory)) {
        ESP_LOGE(TAG, "Invalid SD directory path: %s", directory);
        return ESP_ERR_INVALID_ARG;
    }
    if (strlen(directory) >= MAX_LIST_PATH) {
        ESP_LOGE(TAG, "SD directory path exceeds the listing limit");
        return ESP_ERR_INVALID_SIZE;
    }
    return list_directory_contents(directory, NULL);
}

void console_cli_print_help(void)
{
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("ESP32-P4 ASR Console Commands\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("\nAvailable Commands:\n");
    printf("  ls [<path>]    - Recursively list an SD-card directory\n");
    printf("  list           - List all files on the SD card\n");
    printf("  list .espm     - List model files (.espm)\n");
    printf("  list .wav      - List audio files (.wav)\n");
    printf("  list .txt      - List text files (.txt)\n");
    printf("  list .log      - List log files (.log)\n");
    printf("  list <.ext>    - List files with any extension\n");
    printf("\n");
    printf("  info           - Show system information\n");
    printf("  help           - Show this help message\n");
    printf("  reboot         - Reboot the system\n");
    printf("\n");
    printf("Note: Type commands and press Enter\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
}

void console_cli_print_info(void)
{
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("ESP32-P4 ASR System Information\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    // Chip info
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    printf("\nChip: ESP32-P4\n");
    printf("  Cores: %d\n", chip_info.cores);
    printf("  Silicon Revision: %d\n", chip_info.revision);
    
    // Memory info
    printf("\nMemory:\n");
    printf("  Free Heap: %lu KB\n", esp_get_free_heap_size() / 1024);
    printf("  Min Free Heap: %lu KB\n", esp_get_minimum_free_heap_size() / 1024);
    
    size_t psram_size = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    if (psram_size > 0) {
        printf("  PSRAM Total: %u MB\n", (unsigned int)(psram_size / (1024 * 1024)));
        printf("  PSRAM Free: %u MB\n", (unsigned int)(psram_free / (1024 * 1024)));
    }
    

    
    // Uptime
    uint32_t uptime_sec = esp_log_timestamp() / 1000;
    printf("\nUptime: %lu seconds (%lu min)\n", uptime_sec, uptime_sec / 60);
    
    printf("═══════════════════════════════════════════════════════════════\n\n");
}

esp_err_t console_cli_process_command(const char* command_line)
{
    if (!command_line || strlen(command_line) == 0) {
        return ESP_OK;
    }
    
    // Trim whitespace
    while (*command_line == ' ' || *command_line == '\t') {
        command_line++;
    }
    
    // Parse command and arguments
    char cmd[32] = {0};
    char arg[96] = {0};
    
    sscanf(command_line, "%31s %95s", cmd, arg);
    
    // Process commands
    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "?") == 0) {
        console_cli_print_help();
    }
    else if (strcmp(cmd, "info") == 0) {
        console_cli_print_info();
    }
    else if (strcmp(cmd, "list") == 0) {
        const char* filter = (strlen(arg) > 0) ? arg : NULL;
        console_cli_list_files(filter);
    }
    else if (strcmp(cmd, "ls") == 0) {
        const char* path = (strlen(arg) > 0) ? arg : "/sd";
        const esp_err_t error = console_cli_list_directory(path);
        if (error != ESP_OK) {
            ESP_LOGE(TAG, "Could not list directory %s: %s",
                     path, esp_err_to_name(error));
        }
    }
    else if (strcmp(cmd, "reboot") == 0 || strcmp(cmd, "reset") == 0) {
        printf("\nRebooting system...\n\n");
        vTaskDelay(pdMS_TO_TICKS(100));
        esp_restart();
    }
    else if (strcmp(cmd, "clear") == 0 || strcmp(cmd, "cls") == 0) {
        printf("\033[2J\033[H"); // ANSI escape codes to clear screen
    }
    else {
        printf("Unknown command: %s\n", cmd);
        printf("Type 'help' for available commands\n\n");
        return ESP_ERR_INVALID_ARG;
    }
    
    return ESP_OK;
}

/**
 * @brief Console CLI task - monitors UART for commands
 */
static void console_cli_task(void* pvParameters)
{
    char command_buffer[MAX_COMMAND_LENGTH];
    int buffer_pos = 0;
    uint8_t data;
    
    ESP_LOGI(TAG, "Console CLI ready - type 'help' for commands");
    
    while (1) {
        // Read one byte from UART0 (console)
        int len = uart_read_bytes(CONSOLE_UART_NUM, &data, 1, pdMS_TO_TICKS(100));
        
        if (len > 0) {
            // Echo character
            uart_write_bytes(CONSOLE_UART_NUM, (const char*)&data, 1);
            
            if (data == '\r' || data == '\n') {
                // Command complete
                uart_write_bytes(CONSOLE_UART_NUM, "\n", 1);
                
                if (buffer_pos > 0) {
                    command_buffer[buffer_pos] = '\0';
                    console_cli_process_command(command_buffer);
                    buffer_pos = 0;
                }
                
                // Print prompt
                uart_write_bytes(CONSOLE_UART_NUM, "> ", 2);
            }
            else if (data == '\b' || data == 127) {
                // Backspace
                if (buffer_pos > 0) {
                    buffer_pos--;
                    uart_write_bytes(CONSOLE_UART_NUM, " \b", 2);
                }
            }
            else if (buffer_pos < MAX_COMMAND_LENGTH - 1) {
                // Add character to buffer
                command_buffer[buffer_pos++] = data;
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

esp_err_t console_cli_init(void)
{
    ESP_LOGI(TAG, "Initializing console CLI...");
    
    // UART0 is already initialized by ESP-IDF for console
    // Just create the CLI task
    
    BaseType_t ret = xTaskCreate(
        console_cli_task,
        "console_cli",
        4096,
        NULL,
        5,  // Priority
        NULL
    );
    
    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create console CLI task");
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "Console CLI initialized successfully");
    
    return ESP_OK;
}
