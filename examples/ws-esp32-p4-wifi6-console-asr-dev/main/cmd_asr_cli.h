#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Register ASR console commands
 * 
 * Registers commands like LOAD, UNLOAD, INFER, MEM, LS, PING
 * with the esp_console subsystem.
 */
void cmd_asr_cli_register(void);

#ifdef __cplusplus
}
#endif
