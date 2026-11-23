/**
 * @file moonshine_context.c
 * @brief Moonshine Inference Context Management
 * 
 * Manages runtime buffers and state for inference operations.
 */

#include "moonshine.h"
#include "moonshine_parallel.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <string.h>

static const char* TAG = "moonshine_context";

esp_err_t moonshine_create_context(
    moonshine_model_t* model,
    uint32_t max_seq_len,
    moonshine_context_t* context
) {
    if (!model || !context || !model->loaded) {
        ESP_LOGE(TAG, "Invalid parameters");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Ensure dual-core worker is initialized
    moonshine_parallel_init();
    
    ESP_LOGI(TAG, "Creating inference context (max_seq_len=%lu)", max_seq_len);
    
    // Zero out context
    memset(context, 0, sizeof(moonshine_context_t));
    
    context->model = model;
    context->max_seq_len = max_seq_len;
    
    const uint32_t hidden_size = model->header.hidden_size;
    
    // Allocate encoder output buffer
    // Size: max_seq_len × hidden_size × sizeof(float)
    size_t encoder_size = max_seq_len * hidden_size * sizeof(float);
    context->encoder_output = heap_caps_malloc(encoder_size, MALLOC_CAP_SPIRAM);
    if (!context->encoder_output) {
        ESP_LOGE(TAG, "Failed to allocate encoder output (%zu bytes)", encoder_size);
        moonshine_destroy_context(context);
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "  Encoder output: %zu bytes (%.2f MB)", 
             encoder_size, encoder_size / 1024.0 / 1024.0);
    
    // Allocate decoder output buffer (optional - allocated on-demand for decoder)
    // For encoder-only inference, this is NULL to save memory (~8MB for vocab_size=32768)
    // Will be allocated when decoder inference is needed
    context->decoder_output = NULL;
    
    ESP_LOGI(TAG, "  Decoder output: NULL (allocate on-demand for decoder inference)");
    
    // Allocate token buffer
    size_t tokens_size = max_seq_len * sizeof(int32_t);
    context->tokens = heap_caps_malloc(tokens_size, MALLOC_CAP_DEFAULT);
    if (!context->tokens) {
        ESP_LOGE(TAG, "Failed to allocate tokens (%zu bytes)", tokens_size);
        moonshine_destroy_context(context);
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "  Tokens: %zu bytes", tokens_size);
    
    context->initialized = true;
    
    // Print memory summary
    size_t total_alloc = encoder_size + tokens_size;  // decoder_size excluded for now
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t free_heap = heap_caps_get_free_size(MALLOC_CAP_DEFAULT);
    
    ESP_LOGI(TAG, "Context created:");
    ESP_LOGI(TAG, "  Total allocated: %zu bytes (%.2f MB)", 
             total_alloc, total_alloc / 1024.0 / 1024.0);
    ESP_LOGI(TAG, "  PSRAM free: %zu bytes (%.2f MB)", 
             free_psram, free_psram / 1024.0 / 1024.0);
    ESP_LOGI(TAG, "  Heap free: %zu bytes (%.2f KB)", 
             free_heap, free_heap / 1024.0);
    
    return ESP_OK;
}

esp_err_t moonshine_destroy_context(moonshine_context_t* context) {
    if (!context) {
        return ESP_ERR_INVALID_ARG;
    }
    
    ESP_LOGI(TAG, "Destroying inference context");
    
    if (context->encoder_output) {
        free(context->encoder_output);
    }
    
    if (context->decoder_output) {
        free(context->decoder_output);
    }
    
    if (context->tokens) {
        free(context->tokens);
    }
    
    memset(context, 0, sizeof(moonshine_context_t));
    
    ESP_LOGI(TAG, "Context destroyed");
    
    return ESP_OK;
}
