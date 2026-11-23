/**
 * @file moonshine_audio_utils.c
 * @brief Audio preprocessing utilities implementation
 * 
 * @copyright 2026
 */

#include "moonshine_audio_utils.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

// ESP-DSP includes
#include "esp_dsp.h"
#include "dsps_fir.h"
#include "dsps_resampler.h"
#include "dsps_mulc.h"

static const char* TAG = "MOONSHINE_AUDIO";

/* ========================================================================
 * RESAMPLER IMPLEMENTATION
 * ======================================================================== */

#define DOWNSAMPLE_TAPS 32
#define DOWNSAMPLE_PHASES 64

static int32_t resample_downsample_sinc(
    const moonshine_resampler_t* resampler,
    const int16_t* input,
    int32_t input_length,
    int16_t* output,
    int32_t output_max_length)
{
    if (resampler->input_rate == 0 || resampler->output_rate == 0) {
        return -ESP_ERR_INVALID_STATE;
    }

    const uint64_t output_count =
        ((uint64_t)input_length * resampler->output_rate) / resampler->input_rate;
    if (output_count > INT32_MAX) {
        return -ESP_ERR_INVALID_SIZE;
    }

    const int32_t output_length =
        output_count < (uint64_t)output_max_length
            ? (int32_t)output_count
            : output_max_length;
    const size_t coefficient_count = DOWNSAMPLE_TAPS * DOWNSAMPLE_PHASES;
    float* coefficients = heap_caps_malloc(
        coefficient_count * sizeof(float), MALLOC_CAP_DEFAULT);
    if (!coefficients) {
        return -ESP_ERR_NO_MEM;
    }

    const float pi = 3.14159265358979323846f;
    const float cutoff = resampler->resample_factor * 0.94f;
    for (int32_t phase = 0; phase < DOWNSAMPLE_PHASES; ++phase) {
        const float fraction = (float)phase / DOWNSAMPLE_PHASES;
        float coefficient_sum = 0.0f;
        for (int32_t tap = 0; tap < DOWNSAMPLE_TAPS; ++tap) {
            const float distance =
                (float)(tap - (DOWNSAMPLE_TAPS / 2 - 1)) - fraction;
            float coefficient = 0.0f;
            if (fabsf(distance) <= DOWNSAMPLE_TAPS / 2) {
                const float window = 0.5f + 0.5f *
                    cosf(pi * distance / (DOWNSAMPLE_TAPS / 2));
                const float argument = cutoff * distance;
                const float sinc = fabsf(argument) < 1e-7f
                    ? 1.0f
                    : sinf(pi * argument) / (pi * argument);
                coefficient = cutoff * sinc * window;
            }
            coefficients[phase * DOWNSAMPLE_TAPS + tap] = coefficient;
            coefficient_sum += coefficient;
        }
        if (fabsf(coefficient_sum) < 1e-7f) {
            heap_caps_free(coefficients);
            return -ESP_ERR_INVALID_RESPONSE;
        }
        for (int32_t tap = 0; tap < DOWNSAMPLE_TAPS; ++tap) {
            coefficients[phase * DOWNSAMPLE_TAPS + tap] /= coefficient_sum;
        }
    }

    for (int32_t sample = 0; sample < output_length; ++sample) {
        const double input_position =
            ((double)sample * resampler->input_rate) / resampler->output_rate;
        int32_t center = (int32_t)input_position;
        int32_t phase = (int32_t)(
            (input_position - center) * DOWNSAMPLE_PHASES + 0.5);
        if (phase == DOWNSAMPLE_PHASES) {
            phase = 0;
            ++center;
        }

        float value = 0.0f;
        float weight = 0.0f;
        for (int32_t tap = 0; tap < DOWNSAMPLE_TAPS; ++tap) {
            const int32_t input_index =
                center + tap - (DOWNSAMPLE_TAPS / 2 - 1);
            if (input_index < 0 || input_index >= input_length) {
                continue;
            }
            const float coefficient =
                coefficients[phase * DOWNSAMPLE_TAPS + tap];
            value += input[input_index] * coefficient;
            weight += coefficient;
        }
        if (fabsf(weight) < 1e-7f) {
            heap_caps_free(coefficients);
            return -ESP_ERR_INVALID_RESPONSE;
        }

        value /= weight;
        if (value > INT16_MAX) value = INT16_MAX;
        if (value < INT16_MIN) value = INT16_MIN;
        output[sample] = (int16_t)value;
    }

    heap_caps_free(coefficients);
    return output_length;
}

esp_err_t moonshine_resampler_init(
    moonshine_resampler_t* resampler,
    uint32_t input_rate,
    uint32_t output_rate
)
{
    if (!resampler || input_rate == 0 || output_rate == 0) {
        ESP_LOGE(TAG, "Invalid resampler parameters");
        return ESP_ERR_INVALID_ARG;
    }

    // Calculate resample factor: output / input
    float resample_factor = (float)output_rate / (float)input_rate;
    
    ESP_LOGI(TAG, "Initializing resampler: %lu Hz → %lu Hz (factor: %.4f)",
             input_rate, output_rate, resample_factor);

    dsps_resample_ph_t* dsp_resampler = NULL;
    if (resample_factor >= 1.0f) {
        dsp_resampler = heap_caps_malloc(sizeof(dsps_resample_ph_t), MALLOC_CAP_DEFAULT);
        if (!dsp_resampler) {
            ESP_LOGE(TAG, "Failed to allocate resampler structure");
            return ESP_ERR_NO_MEM;
        }

        const esp_err_t ret = dsps_resampler_ph_init(dsp_resampler, resample_factor);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "dsps_resampler_ph_init failed: %d", ret);
            heap_caps_free(dsp_resampler);
            return ret;
        }
    }
    
    // Store in context
    resampler->input_rate = input_rate;
    resampler->output_rate = output_rate;
    resampler->resample_factor = resample_factor;
    resampler->dsp_resampler = dsp_resampler;
    
    return ESP_OK;
}

int32_t moonshine_resample(
    moonshine_resampler_t* resampler,
    const int16_t* input,
    int32_t input_length,
    int16_t* output,
    int32_t output_max_length
)
{
    if (!resampler || !input || !output || input_length <= 0 || output_max_length <= 0) {
        ESP_LOGE(TAG, "Invalid resample parameters");
        return -ESP_ERR_INVALID_ARG;
    }

    if (!resampler->dsp_resampler) {
        if (resampler->resample_factor >= 1.0f) {
            ESP_LOGE(TAG, "Resampler not initialized");
            return -ESP_ERR_INVALID_STATE;
        }
        return resample_downsample_sinc(
            resampler, input, input_length, output, output_max_length);
    }
    
    dsps_resample_ph_t* dsp_resampler = (dsps_resample_ph_t*)resampler->dsp_resampler;
    
    // Allocate temporary float buffers for DSP operations
    // ESP-DSP works with float32, we need to convert int16 ↔ float32
    float* input_f32 = (float*)heap_caps_malloc(input_length * sizeof(float), MALLOC_CAP_DEFAULT);
    
    // Calculate expected output size
    int32_t expected_output = (int32_t)((float)input_length * resampler->resample_factor) + 16;
    if (expected_output > output_max_length) {
        expected_output = output_max_length;
    }
    
    float* output_f32 = (float*)heap_caps_malloc(expected_output * sizeof(float), MALLOC_CAP_DEFAULT);
    
    if (!input_f32 || !output_f32) {
        ESP_LOGE(TAG, "Failed to allocate float buffers for resampling");
        if (input_f32) free(input_f32);
        if (output_f32) free(output_f32);
        return -ESP_ERR_NO_MEM;
    }
    
    // Convert int16 → float32: scale from [-32768, 32767] to approximately [-1.0, 1.0]
    for (int32_t i = 0; i < input_length; i++) {
        input_f32[i] = (float)input[i] / 32768.0f;
    }
    
    // Execute poly-phase resampling (cubic interpolation)
    int32_t output_samples = dsps_resampler_ph_exec(
        dsp_resampler,
        input_f32,
        output_f32,
        input_length
    );
    
    // Check output size
    if (output_samples < 0) {
        ESP_LOGE(TAG, "dsps_resampler_ph_exec failed: %ld", output_samples);
        free(input_f32);
        free(output_f32);
        return output_samples;
    }
    
    if (output_samples > output_max_length) {
        ESP_LOGW(TAG, "Resampler output (%ld) exceeds buffer size (%ld), truncating",
                 output_samples, output_max_length);
        output_samples = output_max_length;
    }
    
    // Convert float32 → int16: scale from [-1.0, 1.0] to [-32768, 32767]
    for (int32_t i = 0; i < output_samples; i++) {
        float val = output_f32[i] * 32768.0f;
        // Clamp to int16 range
        if (val > 32767.0f) val = 32767.0f;
        if (val < -32768.0f) val = -32768.0f;
        output[i] = (int16_t)val;
    }
    
    // Cleanup
    free(input_f32);
    free(output_f32);
    
    return output_samples;
}

void moonshine_resampler_free(moonshine_resampler_t* resampler)
{
    if (!resampler) {
        return;
    }

    // Free ESP-DSP resampler structure
    if (resampler->dsp_resampler) {
        free(resampler->dsp_resampler);
        resampler->dsp_resampler = NULL;
    }
    
    memset(resampler, 0, sizeof(moonshine_resampler_t));
}

/* ========================================================================
 * FORMAT CONVERSION IMPLEMENTATION
 * ======================================================================== */

esp_err_t moonshine_convert_bit_depth(
    const void* input,
    int16_t* output,
    int32_t num_samples,
    uint8_t input_bits,
    bool is_float,
    uint8_t num_channels
)
{
    if (!input || !output || num_samples <= 0 || num_channels == 0) {
        ESP_LOGE(TAG, "Invalid convert_bit_depth parameters");
        return ESP_ERR_INVALID_ARG;
    }

    int32_t total_samples = num_samples * num_channels;
    
    if (is_float && input_bits == 32) {
        // float32 → int16: scale from [-1.0, 1.0] to [-32768, 32767]
        const float* in_f32 = (const float*)input;
        for (int32_t i = 0; i < total_samples; i++) {
            float val = in_f32[i];
            // Clamp to [-1.0, 1.0]
            if (val > 1.0f) val = 1.0f;
            if (val < -1.0f) val = -1.0f;
            output[i] = (int16_t)(val * 32767.0f);
        }
    }
    else if (input_bits == 8) {
        // 8-bit unsigned → 16-bit signed: [0, 255] → [-32768, 32767]
        const uint8_t* in_u8 = (const uint8_t*)input;
        for (int32_t i = 0; i < total_samples; i++) {
            // Convert [0, 255] to [-128, 127] then scale up
            int16_t val = (int16_t)in_u8[i] - 128;
            output[i] = val * 256;
        }
    }
    else if (input_bits == 16) {
        // 16-bit → 16-bit: passthrough (just copy)
        const int16_t* in_i16 = (const int16_t*)input;
        memcpy(output, in_i16, total_samples * sizeof(int16_t));
    }
    else if (input_bits == 24) {
        // 24-bit signed → 16-bit signed: truncate LSB
        // Assumes packed 24-bit (3 bytes per sample)
        const uint8_t* in_bytes = (const uint8_t*)input;
        for (int32_t i = 0; i < total_samples; i++) {
            int32_t byte_offset = i * 3;
            // Read 24-bit little-endian signed integer
            int32_t val = (int32_t)(
                ((int32_t)in_bytes[byte_offset + 0]) |
                ((int32_t)in_bytes[byte_offset + 1] << 8) |
                ((int32_t)in_bytes[byte_offset + 2] << 16)
            );
            // Sign extend if negative (bit 23 set)
            if (val & 0x800000) {
                val |= 0xFF000000;
            }
            // Shift down by 8 bits (keep MSB 16 bits)
            output[i] = (int16_t)(val >> 8);
        }
    }
    else if (input_bits == 32 && !is_float) {
        // 32-bit signed → 16-bit signed: truncate
        const int32_t* in_i32 = (const int32_t*)input;
        for (int32_t i = 0; i < total_samples; i++) {
            // Keep MSB 16 bits
            output[i] = (int16_t)(in_i32[i] >> 16);
        }
    }
    else {
        ESP_LOGE(TAG, "Unsupported bit depth conversion: %d-bit %s",
                 input_bits, is_float ? "float" : "int");
        return ESP_ERR_INVALID_ARG;
    }
    
    return ESP_OK;
}

esp_err_t moonshine_downmix_to_mono(
    const int16_t* stereo_input,
    int16_t* mono_output,
    int32_t num_frames
)
{
    if (!stereo_input || !mono_output || num_frames <= 0) {
        ESP_LOGE(TAG, "Invalid downmix parameters");
        return ESP_ERR_INVALID_ARG;
    }

    // Average left and right channels: mono[i] = (L + R) / 2
    for (int32_t i = 0; i < num_frames; i++) {
        int32_t left = stereo_input[i * 2];
        int32_t right = stereo_input[i * 2 + 1];
        // Use int32 to avoid overflow, then divide by 2
        mono_output[i] = (int16_t)((left + right) / 2);
    }
    
    return ESP_OK;
}

esp_err_t moonshine_normalize_audio(
    int16_t* buffer,
    int32_t num_samples
)
{
    if (!buffer || num_samples <= 0) {
        ESP_LOGE(TAG, "Invalid normalize parameters");
        return ESP_ERR_INVALID_ARG;
    }

    // Find peak absolute value
    int16_t peak = 0;
    for (int32_t i = 0; i < num_samples; i++) {
        int16_t abs_val = buffer[i] < 0 ? -buffer[i] : buffer[i];
        if (abs_val > peak) {
            peak = abs_val;
        }
    }
    
    // If silence or already at max, no need to normalize
    if (peak == 0 || peak == 32767) {
        return ESP_OK;
    }
    
    // Calculate scaling factor to use full 16-bit range
    float scale = 32767.0f / (float)peak;
    
    // Apply scaling
    for (int32_t i = 0; i < num_samples; i++) {
        float scaled = (float)buffer[i] * scale;
        // Clamp to prevent overflow (should not happen, but safety first)
        if (scaled > 32767.0f) scaled = 32767.0f;
        if (scaled < -32768.0f) scaled = -32768.0f;
        buffer[i] = (int16_t)scaled;
    }
    
    ESP_LOGD(TAG, "Normalized audio: peak was %d, scale factor %.3f", peak, scale);
    
    return ESP_OK;
}

/* ========================================================================
 * HIGH-LEVEL API IMPLEMENTATION
 * ======================================================================== */

int32_t moonshine_calculate_output_size(
    int32_t input_samples,
    const moonshine_audio_format_t* input_format
)
{
    if (!input_format || input_samples <= 0) {
        ESP_LOGE(TAG, "Invalid calculate_output_size parameters");
        return -1;
    }

    // Account for channel reduction
    int32_t mono_samples = input_samples / input_format->num_channels;
    
    // Account for resampling
    int32_t resampled = (int32_t)((float)mono_samples * 16000.0f / (float)input_format->sample_rate);
    
    // Add safety margin
    return resampled + 128;
}

int32_t moonshine_preprocess_audio(
    const void* input,
    int32_t input_samples,
    const moonshine_audio_format_t* input_format,
    int16_t* output,
    int32_t output_max,
    bool normalize
)
{
    if (!input || !output || !input_format || input_samples <= 0 || output_max <= 0) {
        ESP_LOGE(TAG, "Invalid preprocess_audio parameters");
        return -ESP_ERR_INVALID_ARG;
    }

    ESP_LOGI(TAG, "Preprocessing: %lu Hz, %d ch, %d-bit → 16kHz mono 16-bit (normalize: %s)",
             input_format->sample_rate, input_format->num_channels,
             input_format->bits_per_sample, normalize ? "yes" : "no");
    
    esp_err_t ret;
    int16_t* working_buffer = NULL;
    int16_t* temp_buffer = NULL;
    int32_t working_samples = 0;
    
    // ===================================================================
    // STEP 1: Bit depth conversion
    // ===================================================================
    
    bool need_bit_convert = (input_format->bits_per_sample != 16);
    
    if (need_bit_convert) {
        // Allocate buffer for bit depth conversion
        temp_buffer = (int16_t*)heap_caps_malloc(
            input_samples * sizeof(int16_t),
            MALLOC_CAP_DEFAULT
        );
        
        if (!temp_buffer) {
            ESP_LOGE(TAG, "Failed to allocate temp buffer for bit conversion");
            return -ESP_ERR_NO_MEM;
        }
        
        // Convert to 16-bit
        ret = moonshine_convert_bit_depth(
            input,
            temp_buffer,
            input_samples / input_format->num_channels,  // samples per channel
            input_format->bits_per_sample,
            false,  // assume integer format (TODO: add float detection)
            input_format->num_channels
        );
        
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Bit depth conversion failed: %d", ret);
            free(temp_buffer);
            return -ret;
        }
        
        working_buffer = temp_buffer;
        working_samples = input_samples;
    } else {
        // Already 16-bit, use input directly
        working_buffer = (int16_t*)input;
        working_samples = input_samples;
    }
    
    // ===================================================================
    // STEP 2: Channel downmix (stereo → mono)
    // ===================================================================
    
    int16_t* mono_buffer = NULL;
    int32_t mono_samples = 0;
    
    if (input_format->num_channels == 2) {
        // Allocate mono buffer
        mono_samples = working_samples / 2;
        mono_buffer = (int16_t*)heap_caps_malloc(
            mono_samples * sizeof(int16_t),
            MALLOC_CAP_DEFAULT
        );
        
        if (!mono_buffer) {
            ESP_LOGE(TAG, "Failed to allocate mono buffer");
            if (temp_buffer) free(temp_buffer);
            return -ESP_ERR_NO_MEM;
        }
        
        // Downmix stereo to mono
        ret = moonshine_downmix_to_mono(working_buffer, mono_buffer, mono_samples);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Downmix failed: %d", ret);
            free(mono_buffer);
            if (temp_buffer) free(temp_buffer);
            return -ret;
        }
        
        // Free temp buffer if used
        if (temp_buffer) {
            free(temp_buffer);
            temp_buffer = NULL;
        }
        
        working_buffer = mono_buffer;
        working_samples = mono_samples;
    } else if (input_format->num_channels == 1) {
        // Already mono
        mono_samples = working_samples;
        // If we have temp_buffer, it's already mono, reuse it
        // Otherwise working_buffer points to input (no allocation needed)
    } else {
        ESP_LOGE(TAG, "Unsupported channel count: %d", input_format->num_channels);
        if (temp_buffer) free(temp_buffer);
        return -ESP_ERR_INVALID_ARG;
    }
    
    // ===================================================================
    // STEP 3: Sample rate conversion (any → 16kHz)
    // ===================================================================
    
    int16_t* resampled_buffer = NULL;
    int32_t resampled_samples = 0;
    
    if (input_format->sample_rate != 16000) {
        // Initialize resampler
        moonshine_resampler_t resampler;
        ret = moonshine_resampler_init(&resampler, input_format->sample_rate, 16000);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Resampler init failed: %d", ret);
            if (mono_buffer) free(mono_buffer);
            if (temp_buffer) free(temp_buffer);
            return -ret;
        }
        
        // Resample directly to output buffer
        resampled_samples = moonshine_resample(
            &resampler,
            working_buffer,
            working_samples,
            output,
            output_max
        );
        
        // Free resampler
        moonshine_resampler_free(&resampler);
        
        if (resampled_samples < 0) {
            ESP_LOGE(TAG, "Resampling failed: %ld", resampled_samples);
            if (mono_buffer) free(mono_buffer);
            if (temp_buffer) free(temp_buffer);
            return resampled_samples;
        }
        
        resampled_buffer = output;  // Already in output buffer
    } else {
        // Already 16kHz, copy to output
        resampled_samples = (working_samples < output_max) ? working_samples : output_max;
        memcpy(output, working_buffer, resampled_samples * sizeof(int16_t));
        resampled_buffer = output;
    }
    
    // Free intermediate buffers
    if (mono_buffer) free(mono_buffer);
    if (temp_buffer) free(temp_buffer);
    
    // ===================================================================
    // STEP 4: Normalization (optional)
    // ===================================================================
    
    if (normalize) {
        ret = moonshine_normalize_audio(resampled_buffer, resampled_samples);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "Normalization failed: %d (continuing anyway)", ret);
        }
    }
    
    ESP_LOGI(TAG, "Preprocessing complete: %ld samples output", resampled_samples);
    
    return resampled_samples;
}
