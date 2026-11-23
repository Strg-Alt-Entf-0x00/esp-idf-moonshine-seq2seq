#include "moonshine_tokenizer.h"
#include "moonshine.h"
#include <stdbool.h>
#include <string.h>
#include "esp_log.h"
#include "esp_heap_caps.h"

static const char* TAG = "moonshine_tokenizer";

typedef struct {
    char* string;
    uint16_t length;
} TokenEntry;

static TokenEntry* g_vocab = NULL;
static uint32_t g_vocab_size = 0;
static char* g_string_pool = NULL;

static bool read_token_length(const uint8_t *buffer, size_t size, size_t *offset,
                              uint32_t *out_length)
{
    if (*offset >= size) {
        return false;
    }
    uint32_t length = buffer[(*offset)++];
    if (length >= 128) {
        if (*offset >= size) {
            return false;
        }
        const uint32_t high = buffer[(*offset)++];
        if (high > 127) {
            return false;
        }
        length = (high * 128U) + length - 128U;
        if (length < 128) {
            return false;
        }
    }
    if (length > size - *offset) {
        return false;
    }
    *out_length = length;
    return true;
}

static bool valid_utf8(const uint8_t *data, size_t size)
{
    for (size_t i = 0; i < size;) {
        const uint8_t first = data[i++];
        if (first <= 0x7FU) {
            continue;
        }
        if (first >= 0xC2U && first <= 0xDFU) {
            if (i >= size || (data[i++] & 0xC0U) != 0x80U) {
                return false;
            }
            continue;
        }
        if (first >= 0xE0U && first <= 0xEFU) {
            if (size - i < 2) {
                return false;
            }
            const uint8_t second = data[i++];
            const uint8_t third = data[i++];
            if ((third & 0xC0U) != 0x80U ||
                (first == 0xE0U && (second < 0xA0U || second > 0xBFU)) ||
                (first == 0xEDU && (second < 0x80U || second > 0x9FU)) ||
                ((first != 0xE0U && first != 0xEDU) &&
                 (second & 0xC0U) != 0x80U)) {
                return false;
            }
            continue;
        }
        if (first >= 0xF0U && first <= 0xF4U) {
            if (size - i < 3) {
                return false;
            }
            const uint8_t second = data[i++];
            const uint8_t third = data[i++];
            const uint8_t fourth = data[i++];
            if ((third & 0xC0U) != 0x80U || (fourth & 0xC0U) != 0x80U ||
                (first == 0xF0U && (second < 0x90U || second > 0xBFU)) ||
                (first == 0xF4U && (second < 0x80U || second > 0x8FU)) ||
                (first != 0xF0U && first != 0xF4U &&
                 (second & 0xC0U) != 0x80U)) {
                return false;
            }
            continue;
        }
        return false;
    }
    return true;
}

esp_err_t moonshine_tokenizer_init_from_buffer(const uint8_t* buffer, size_t size) {
    if (!buffer || size == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    if (g_vocab != NULL) {
        ESP_LOGW(TAG, "Tokenizer already initialized");
        return ESP_OK;
    }

    // First pass: count tokens
    size_t offset = 0;
    uint32_t token_count = 0;
    while (offset < size) {
        uint32_t length;
        if (!read_token_length(buffer, size, &offset, &length) ||
            token_count >= MOONSHINE_MAX_VOCAB_SIZE ||
            memchr(buffer + offset, '\0', length) ||
            !valid_utf8(buffer + offset, length)) {
            ESP_LOGE(TAG, "Invalid or oversized vocabulary at byte %u", (unsigned)offset);
            return ESP_ERR_INVALID_SIZE;
        }
        offset += length;
        token_count++;
    }
    if (token_count == 0) {
        return ESP_ERR_INVALID_SIZE;
    }
    if (size > UINT32_MAX - token_count) {
        return ESP_ERR_INVALID_SIZE;
    }

    ESP_LOGI(TAG, "Found %lu tokens in vocab buffer", (unsigned long)token_count);

    g_vocab = (TokenEntry*)heap_caps_calloc(token_count, sizeof(TokenEntry), MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (!g_vocab) {
        return ESP_ERR_NO_MEM;
    }

    // Allocate string pool
    g_string_pool = (char*)heap_caps_malloc(size + token_count, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (!g_string_pool) {
        ESP_LOGE(TAG, "Failed to allocate string pool (size=%lu)", (unsigned long)(size + token_count));
        heap_caps_free(g_vocab);
        g_vocab = NULL;
        return ESP_ERR_NO_MEM;
    }

    // Second pass: extract strings
    offset = 0;
    uint32_t pool_offset = 0;
    for (uint32_t i = 0; i < token_count; i++) {
        uint32_t length;
        if (!read_token_length(buffer, size, &offset, &length)) {
            heap_caps_free(g_vocab);
            heap_caps_free(g_string_pool);
            g_vocab = NULL;
            g_string_pool = NULL;
            return ESP_ERR_INVALID_SIZE;
        }

        g_vocab[i].length = length;
        g_vocab[i].string = &g_string_pool[pool_offset];
        
        if (length > 0) {
            memcpy(g_vocab[i].string, &buffer[offset], length);
        }
        g_vocab[i].string[length] = '\0';
        
        pool_offset += length + 1;
        offset += length;
    }

    g_vocab_size = token_count;
    ESP_LOGI(TAG, "Tokenizer initialization complete (%lu tokens)", (unsigned long)token_count);
    return ESP_OK;
}

uint32_t moonshine_tokenizer_get_vocab_size(void)
{
    return g_vocab_size;
}

void moonshine_tokenizer_free(void) {
    if (g_vocab) {
        heap_caps_free(g_vocab);
        g_vocab = NULL;
        g_vocab_size = 0;
    }
    if (g_string_pool) {
        heap_caps_free(g_string_pool);
        g_string_pool = NULL;
    }
}

const char* moonshine_tokenizer_get_string(uint32_t token_id) {
    if (token_id >= g_vocab_size || !g_vocab) {
        return NULL;
    }
    return g_vocab[token_id].string;
}

int moonshine_tokenizer_decode(const uint32_t* tokens, size_t count, char* output_str, size_t max_len) {
    if (!g_vocab || !output_str || max_len == 0) return -1;
    
    output_str[0] = '\0';
    size_t current_len = 0;
    
    for (size_t i = 0; i < count; i++) {
        uint32_t token_id = tokens[i];
        
        // Skip special tokens
        if (token_id == 0 || token_id == 1 || token_id == 2 || token_id == 3) {
            continue; 
        }
        
        const char* str = moonshine_tokenizer_get_string(token_id);
        if (str) {
            // Replace U+2581 (SPIECE_UNDERLINE) with space
            // U+2581 is E2 96 81 in UTF-8
            const char* p = str;
            while (*p && current_len < max_len - 1) {
                if ((uint8_t)p[0] == 0xE2 && p[1] && p[2] &&
                    (uint8_t)p[1] == 0x96 && (uint8_t)p[2] == 0x81) {
                    output_str[current_len++] = ' ';
                    p += 3;
                } else {
                    output_str[current_len++] = *p++;
                }
            }
        }
    }
    
    output_str[current_len] = '\0';
    return current_len;
}

esp_err_t moonshine_tokens_to_text(
    const int32_t* tokens,
    uint32_t num_tokens,
    char* text,
    size_t text_len
) {
    if (moonshine_tokenizer_decode((const uint32_t*)tokens, num_tokens, text, text_len) >= 0) {
        return ESP_OK;
    }
    return ESP_FAIL;
}
