#ifndef MOONSHINE_TOKENIZER_H
#define MOONSHINE_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MOONSHINE_MAX_TOKEN_LENGTH 32

/**
 * @brief Initialize the tokenizer from an in-memory ESPM vocabulary section.
 * 
 * @param buffer Pointer to vocab binary data
 * @param size Size in bytes
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t moonshine_tokenizer_init_from_buffer(const uint8_t* buffer, size_t size);

uint32_t moonshine_tokenizer_get_vocab_size(void);

/**
 * @brief Free tokenizer resources
 */
void moonshine_tokenizer_free(void);

/**
 * @brief Get the string representation of a token ID
 * 
 * @param token_id The token ID to decode
 * @return Pointer to a null-terminated string, or NULL if invalid token
 */
const char* moonshine_tokenizer_get_string(uint32_t token_id);

/**
 * @brief Decode an array of token IDs into a single string
 * 
 * @param tokens Array of token IDs
 * @param count Number of tokens in the array
 * @param output_str Output buffer for the string
 * @param max_len Maximum length of the output buffer
 * @return Number of bytes written, or -1 on error
 */
int moonshine_tokenizer_decode(const uint32_t* tokens, size_t count, char* output_str, size_t max_len);

#ifdef __cplusplus
}
#endif

#endif // MOONSHINE_TOKENIZER_H
