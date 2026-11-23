/**
 * VAD Filter - PulseVAD Integration
 * 
 * Lightweight pre-filter for Moonshine ASR to avoid expensive inference on silence/noise.
 * Uses PulseVAD (2.1k params, < 0.15ms latency) for voice activity detection.
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "vad_filter.h"
#include "esp_pulsevad.h"

static const char* TAG = "VAD_FILTER";

#define PULSEVAD_CHUNK_SAMPLES 3200  // 200ms @ 16kHz (PulseVAD requirement)

struct vad_filter_t {
    vad_filter_config_t config;
    pulsevad_handle_t pulsevad;
    
    // Internal buffering (for handling non-3200 sample chunks)
    int16_t* sample_buffer;
    uint32_t buffer_pos;
    bool buffer_active;
    bool last_speech_state;
};

// ============================================================================
// Helper: RMS Energy Calculation
// ============================================================================

static float calculate_rms(const int16_t* audio, uint32_t len) {
    if (len == 0) return 0.0f;
    
    double sum_sq = 0.0;
    for (uint32_t i = 0; i < len; i++) {
        float sample = (float)audio[i] / 32768.0f;
        sum_sq += sample * sample;
    }
    return (float)sqrt(sum_sq / (double)len);
}

// ============================================================================
// Initialization
// ============================================================================

esp_err_t vad_filter_init(const vad_filter_config_t* config, vad_filter_handle_t* handle) {
    if (!config || !handle) return ESP_ERR_INVALID_ARG;
    
    vad_filter_handle_t filter = (vad_filter_handle_t)calloc(1, sizeof(struct vad_filter_t));
    if (!filter) return ESP_ERR_NO_MEM;
    
    filter->config = *config;
    filter->last_speech_state = false;
    
    // Initialize PulseVAD
    pulsevad_config_t pulsevad_cfg = {
        .speech_threshold = config->vad_threshold,
        .min_speech_ms = config->min_speech_ms,
        .min_silence_ms = config->min_silence_ms
    };
    
    esp_err_t ret = pulsevad_init(&pulsevad_cfg, &filter->pulsevad);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize PulseVAD: %d", ret);
        free(filter);
        return ret;
    }
    
    // Allocate internal buffer for chunk alignment
    filter->sample_buffer = (int16_t*)heap_caps_malloc(
        PULSEVAD_CHUNK_SAMPLES * sizeof(int16_t),
        MALLOC_CAP_INTERNAL  // Use SRAM for buffer (fast access)
    );
    
    if (!filter->sample_buffer) {
        ESP_LOGE(TAG, "Failed to allocate sample buffer");
        pulsevad_free(filter->pulsevad);
        free(filter);
        return ESP_ERR_NO_MEM;
    }
    
    filter->buffer_pos = 0;
    filter->buffer_active = false;
    
    ESP_LOGI(TAG, "VAD filter initialized (PulseVAD 2.1k)");
    ESP_LOGI(TAG, "  RMS Threshold: %.5f", config->rms_threshold);
    ESP_LOGI(TAG, "  VAD Threshold: %.2f", config->vad_threshold);
    ESP_LOGI(TAG, "  Hysteresis: %lu ms speech / %lu ms silence", 
             config->min_speech_ms, config->min_silence_ms);
    
    *handle = filter;
    return ESP_OK;
}

// ============================================================================
// Processing
// ============================================================================

esp_err_t vad_filter_process(
    vad_filter_handle_t handle,
    const int16_t* audio,
    uint32_t audio_len,
    bool* has_speech
) {
    if (!handle || !audio || !has_speech) return ESP_ERR_INVALID_ARG;

    // Optional fast energy check: only bypass if truly near-zero digital silence (< 0.0005)
    // and no samples are currently buffered, otherwise buffer must remain continuous
    float rms = calculate_rms(audio, audio_len);
    if (handle->config.rms_threshold > 0.0f && rms < handle->config.rms_threshold && handle->buffer_pos == 0) {
        ESP_LOGD(TAG, "VAD digital silence bypass (RMS: %.5f)", rms);
        handle->last_speech_state = false;
        *has_speech = false;
        return ESP_OK;
    }

    // Process audio in strict 3,200 sample chunks (200 ms @ 16 kHz)
    uint32_t offset = 0;
    bool speech_detected_any = false;
    bool ran_inference = false;

    // 1. If buffer has residual samples from previous chunk, fill it first
    if (handle->buffer_pos > 0) {
        uint32_t needed = PULSEVAD_CHUNK_SAMPLES - handle->buffer_pos;
        uint32_t to_copy = (audio_len < needed) ? audio_len : needed;
        memcpy(handle->sample_buffer + handle->buffer_pos, audio, to_copy * sizeof(int16_t));
        handle->buffer_pos += to_copy;
        offset += to_copy;

        if (handle->buffer_pos == PULSEVAD_CHUNK_SAMPLES) {
            float prob = 0.0f;
            bool is_speech = false;
            esp_err_t ret = pulsevad_process_i16_chunk(handle->pulsevad, handle->sample_buffer, &prob, &is_speech);
            ran_inference = true;
            if (ret == ESP_OK && is_speech) {
                speech_detected_any = true;
                ESP_LOGI(TAG, "VAD Accepted (buffered chunk): prob=%.3f, rms=%.4f", prob, rms);
            }
            handle->buffer_pos = 0;
        }
    }

    // 2. Process complete 3,200 sample chunks directly from input
    while (offset + PULSEVAD_CHUNK_SAMPLES <= audio_len) {
        float prob = 0.0f;
        bool is_speech = false;
        esp_err_t ret = pulsevad_process_i16_chunk(handle->pulsevad, audio + offset, &prob, &is_speech);
        ran_inference = true;
        if (ret == ESP_OK && is_speech) {
            speech_detected_any = true;
            ESP_LOGI(TAG, "VAD Accepted (chunk %lu): prob=%.3f, rms=%.4f",
                     (unsigned long)(offset / PULSEVAD_CHUNK_SAMPLES), prob, rms);
        }
        offset += PULSEVAD_CHUNK_SAMPLES;
    }

    // 3. Stash remaining samples in internal accumulator buffer
    uint32_t remaining = audio_len - offset;
    if (remaining > 0) {
        memcpy(handle->sample_buffer + handle->buffer_pos, audio + offset, remaining * sizeof(int16_t));
        handle->buffer_pos += remaining;
        ESP_LOGD(TAG, "VAD buffer accumulating: %lu / %d samples",
                 (unsigned long)handle->buffer_pos, PULSEVAD_CHUNK_SAMPLES);
    }

    if (ran_inference) {
        handle->last_speech_state = speech_detected_any;
        *has_speech = speech_detected_any;
    } else {
        *has_speech = handle->last_speech_state;
    }
    return ESP_OK;
}

// ============================================================================
// Reset
// ============================================================================

esp_err_t vad_filter_reset(vad_filter_handle_t handle) {
    if (!handle) return ESP_ERR_INVALID_ARG;
    handle->buffer_pos = 0;
    handle->buffer_active = false;
    handle->last_speech_state = false;
    if (handle->pulsevad) {
        pulsevad_reset(handle->pulsevad);
    }
    ESP_LOGD(TAG, "VAD filter buffer and state reset");
    return ESP_OK;
}

// ============================================================================
// Cleanup
// ============================================================================

esp_err_t vad_filter_deinit(vad_filter_handle_t handle) {
    if (!handle) return ESP_ERR_INVALID_ARG;
    
    if (handle->pulsevad) {
        pulsevad_free(handle->pulsevad);
    }
    
    if (handle->sample_buffer) {
        heap_caps_free(handle->sample_buffer);
    }
    
    free(handle);
    
    ESP_LOGI(TAG, "VAD filter deinitialized");
    return ESP_OK;
}

