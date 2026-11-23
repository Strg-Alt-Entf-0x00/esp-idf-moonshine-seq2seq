/**
 * @file moonshine_loader.c
 * @brief Shared model utilities and CRC32 implementation.
 */

#include "moonshine.h"

#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"

static const char *TAG = "moonshine_loader";

static uint32_t s_crc32_table[256];
static bool s_crc32_table_ready;
static portMUX_TYPE s_crc32_lock = portMUX_INITIALIZER_UNLOCKED;

static void ensure_crc32_table(void)
{
    portENTER_CRITICAL(&s_crc32_lock);
    if (!s_crc32_table_ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t value = i;
            for (uint32_t bit = 0; bit < 8; ++bit) {
                value = (value & 1U) ? (value >> 1) ^ 0xEDB88320U : value >> 1;
            }
            s_crc32_table[i] = value;
        }
        s_crc32_table_ready = true;
    }
    portEXIT_CRITICAL(&s_crc32_lock);
}

uint32_t moonshine_crc32_update(uint32_t crc, const uint8_t *data, size_t len)
{
    ensure_crc32_table();
    for (size_t i = 0; i < len; ++i) {
        crc = s_crc32_table[(crc ^ data[i]) & 0xFFU] ^ (crc >> 8);
    }
    return crc;
}

uint32_t moonshine_crc32(const uint8_t *data, size_t len)
{
    return moonshine_crc32_update(0xFFFFFFFFU, data, len) ^ 0xFFFFFFFFU;
}

esp_err_t moonshine_init(void)
{
    const size_t psram_size = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    if (psram_size == 0) {
        ESP_LOGE(TAG, "PSRAM is not available");
        return ESP_ERR_NOT_SUPPORTED;
    }
    ESP_LOGI(TAG, "PSRAM available: %zu bytes", psram_size);
    return ESP_OK;
}

esp_err_t moonshine_unload_model(moonshine_model_t *model)
{
    if (!model) {
        return ESP_ERR_INVALID_ARG;
    }

    if (model->tensors) {
        for (uint32_t i = 0; i < model->num_tensors; ++i) {
            if (model->tensors[i].scales && !model->scales_in_buffer) {
                heap_caps_free(model->tensors[i].scales);
            }
            if (model->tensors[i].data && !model->owns_tensor_data) {
                heap_caps_free(model->tensors[i].data);
            }
        }
        heap_caps_free(model->tensors);
    }
    if (model->psram_buffer && !model->is_static_arena) {
        heap_caps_free(model->psram_buffer);
    }

    memset(model, 0, sizeof(*model));
    return ESP_OK;
}

esp_err_t moonshine_verify_model(const moonshine_model_t *model)
{
    if (!model || !model->loaded || !model->tensors || model->num_tensors == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    if (model->header.hidden_size == 0 ||
        model->header.hidden_size > MOONSHINE_MAX_HIDDEN_SIZE ||
        model->header.encoder_layers == 0 ||
        model->header.encoder_layers > MOONSHINE_MAX_LAYERS ||
        model->header.decoder_layers == 0 ||
        model->header.decoder_layers > MOONSHINE_MAX_LAYERS ||
        model->header.vocab_size == 0 ||
        model->header.vocab_size > MOONSHINE_MAX_VOCAB_SIZE ||
        (model->header.quant_type != MOONSHINE_QUANT_INT8 &&
         model->header.quant_type != MOONSHINE_QUANT_INT8_PER_CH &&
         model->header.quant_type != MOONSHINE_QUANT_MIXED) ||
        (model->header.part != MOONSHINE_PART_ENCODER &&
         model->header.part != MOONSHINE_PART_DECODER)) {
        ESP_LOGE(TAG, "Model metadata is outside runtime limits");
        return ESP_ERR_INVALID_ARG;
    }

    for (uint32_t i = 0; i < model->num_tensors; ++i) {
        const moonshine_tensor_t *tensor = &model->tensors[i];
        if (!tensor->data || tensor->data_size != tensor->num_elements ||
            tensor->num_channels == 0 || !tensor->scales) {
            ESP_LOGE(TAG, "Tensor %u has invalid data or scale metadata", (unsigned)i);
            return ESP_ERR_INVALID_STATE;
        }
    }
    return ESP_OK;
}

esp_err_t moonshine_get_model_info(const moonshine_model_t *model,
                                   char *buffer, size_t buffer_len)
{
    if (!model || !buffer || buffer_len == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    const char *version = model->header.quant_type == MOONSHINE_QUANT_INT8
        ? "INT8" : model->header.quant_type == MOONSHINE_QUANT_INT8_PER_CH
        ? "INT8-per-channel" : model->header.quant_type == MOONSHINE_QUANT_MIXED
        ? "Mixed INT8" : "Unknown";
    const char *type = model->header.part == MOONSHINE_PART_ENCODER
        ? "Encoder" : model->header.part == MOONSHINE_PART_DECODER
        ? "Decoder" : "Unknown";
    const int written = snprintf(
        buffer, buffer_len,
        "Moonshine %s model\n"
        "  Quantization: %s\n"
        "  Parameters: %u\n"
        "  Hidden size: %u\n"
        "  Encoder layers: %u\n"
        "  Decoder layers: %u\n"
        "  Tensors: %u\n"
        "  Resident tensor data: %.2f MiB",
        type, version, (unsigned)model->header.total_params,
        (unsigned)model->header.hidden_size,
        (unsigned)model->header.encoder_layers,
        (unsigned)model->header.decoder_layers,
        (unsigned)model->num_tensors,
        model->total_size / (1024.0 * 1024.0));
    if (written < 0 || (size_t)written >= buffer_len) {
        return ESP_ERR_INVALID_SIZE;
    }
    return ESP_OK;
}
