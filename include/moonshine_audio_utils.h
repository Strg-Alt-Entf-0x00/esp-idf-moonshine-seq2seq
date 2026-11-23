/**
 * @file moonshine_audio_utils.h
 * @brief Audio preprocessing utilities for Moonshine ASR
 *
 * Provides universal audio preprocessing to convert any audio format
 * to Moonshine's required format (16kHz, mono, 16-bit PCM).
 * 
 * Features:
 * - Sample rate conversion (any rate → 16kHz)
 * - Channel downmix (stereo → mono)
 * - Bit depth conversion (8/24/32-bit → 16-bit)
 * - Audio normalization (optional)
 * 
 * This file is only available when CONFIG_MOONSHINE_ENABLE_AUDIO_UTILS is enabled.
 * Enable it in menuconfig: Component config → Moonshine ASR Configuration
 *
 * @copyright 2026
 */

#ifndef MOONSHINE_AUDIO_UTILS_H
#define MOONSHINE_AUDIO_UTILS_H

#include "sdkconfig.h"

#ifndef CONFIG_MOONSHINE_ENABLE_AUDIO_UTILS
#error "Audio utilities are disabled; enable CONFIG_MOONSHINE_ENABLE_AUDIO_UTILS"
#endif

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * AUDIO FORMAT DESCRIPTION
 * ======================================================================== */

/**
 * @brief Audio format descriptor
 * 
 * Describes input audio format for preprocessing
 */
typedef struct {
    uint32_t sample_rate;      ///< Sample rate in Hz (e.g., 8000, 22050, 44100, 48000)
    uint8_t  num_channels;     ///< Number of channels: 1=mono, 2=stereo
    uint8_t  bits_per_sample;  ///< Bit depth: 8, 16, 24, or 32
} moonshine_audio_format_t;

/* ========================================================================
 * RESAMPLER
 * ======================================================================== */

/**
 * @brief Resampler state (opaque)
 * 
 * Internal state for sample-rate conversion
 */
typedef struct {
    uint32_t input_rate;       ///< Input sample rate
    uint32_t output_rate;      ///< Output sample rate (16000 Hz)
    float    resample_factor;  ///< Ratio: output_rate / input_rate
    void*    dsp_resampler;    ///< ESP-DSP state used for upsampling; NULL for downsampling
} moonshine_resampler_t;

/**
 * @brief Initialize audio resampler
 * 
 * Creates a sample-rate converter. Upsampling uses ESP-DSP cubic
 * interpolation; downsampling uses an anti-aliased windowed-sinc filter.
 * 
 * @param[out] resampler Resampler state to initialize
 * @param[in] input_rate Input sample rate (Hz)
 * @param[in] output_rate Output sample rate (Hz), typically 16000
 * 
 * @return ESP_OK on success
 *         ESP_ERR_NO_MEM if allocation fails
 *         ESP_ERR_INVALID_ARG if rates are invalid
 * 
 * @note Must call moonshine_resampler_free() when done
 */
esp_err_t moonshine_resampler_init(
    moonshine_resampler_t* resampler,
    uint32_t input_rate,
    uint32_t output_rate
);

/**
 * @brief Resample audio buffer
 * 
 * Converts audio from input sample rate to output sample rate.
 * Uses cubic interpolation for upsampling and a windowed-sinc filter for
 * downsampling.
 * 
 * @param[in] resampler Initialized resampler
 * @param[in] input Input audio buffer (16-bit PCM)
 * @param[in] input_length Number of input samples
 * @param[out] output Output audio buffer (16-bit PCM)
 * @param[in] output_max_length Maximum output buffer size (samples)
 * 
 * @return Number of output samples written (>= 0)
 *         Negative error code on failure
 * 
 * @note Output buffer size formula:
 *       output_samples ≈ (input_samples * output_rate) / input_rate + 16
 *       Allocate slightly more to handle rounding
 */
int32_t moonshine_resample(
    moonshine_resampler_t* resampler,
    const int16_t* input,
    int32_t input_length,
    int16_t* output,
    int32_t output_max_length
);

/**
 * @brief Free resampler resources
 * 
 * @param[in] resampler Resampler to free
 */
void moonshine_resampler_free(moonshine_resampler_t* resampler);

/* ========================================================================
 * FORMAT CONVERSION
 * ======================================================================== */

/**
 * @brief Convert bit depth to 16-bit signed PCM
 * 
 * Supported conversions:
 * - 8-bit unsigned → 16-bit signed
 * - 16-bit signed → 16-bit signed (passthrough)
 * - 24-bit signed → 16-bit signed (truncate LSB)
 * - 32-bit signed → 16-bit signed (truncate)
 * - 32-bit float → 16-bit signed (scale from [-1.0, 1.0])
 * 
 * @param[in] input Input audio buffer (any bit depth)
 * @param[out] output Output buffer (16-bit signed PCM)
 * @param[in] num_samples Number of samples (per channel)
 * @param[in] input_bits Input bit depth (8, 16, 24, 32)
 * @param[in] is_float True if input is float32, false if integer
 * @param[in] num_channels Number of channels (for interleaved data)
 * 
 * @return ESP_OK on success
 *         ESP_ERR_INVALID_ARG if unsupported bit depth
 * 
 * @note For interleaved stereo, num_samples = total_samples / 2
 */
esp_err_t moonshine_convert_bit_depth(
    const void* input,
    int16_t* output,
    int32_t num_samples,
    uint8_t input_bits,
    bool is_float,
    uint8_t num_channels
);

/**
 * @brief Downmix stereo to mono
 * 
 * Averages left and right channels:
 * mono[i] = (left[i] + right[i]) / 2
 * 
 * @param[in] stereo_input Interleaved stereo audio [L, R, L, R, ...]
 * @param[out] mono_output Mono audio output
 * @param[in] num_frames Number of stereo frames (not total samples!)
 * 
 * @return ESP_OK on success
 * 
 * @note num_frames = total_samples / 2
 *       Output buffer size = num_frames samples
 */
esp_err_t moonshine_downmix_to_mono(
    const int16_t* stereo_input,
    int16_t* mono_output,
    int32_t num_frames
);

/**
 * @brief Normalize audio amplitude
 * 
 * Scales audio to use full 16-bit range [-32768, 32767].
 * Finds peak absolute value and scales accordingly.
 * 
 * @param[in,out] buffer Audio buffer to normalize (modified in-place)
 * @param[in] num_samples Number of samples
 * 
 * @return ESP_OK on success
 * 
 * @note If peak == 0 (silence), buffer is unchanged
 */
esp_err_t moonshine_normalize_audio(
    int16_t* buffer,
    int32_t num_samples
);

/* ========================================================================
 * HIGH-LEVEL API
 * ======================================================================== */

/**
 * @brief Preprocess audio to Moonshine format (all-in-one)
 * 
 * Converts any audio format to Moonshine requirements:
 * - Target: 16kHz sample rate
 * - Target: Mono (1 channel)
 * - Target: 16-bit signed PCM
 * - Optional: Normalized amplitude
 * 
 * Processing pipeline:
 * 1. Bit depth conversion (8/24/32-bit → 16-bit)
 * 2. Channel downmix (stereo → mono)
 * 3. Sample rate conversion (any → 16kHz)
 * 4. Normalization (optional)
 * 
 * @param[in] input Input audio buffer (any format)
 * @param[in] input_samples Number of input samples (total, including all channels)
 * @param[in] input_format Input format descriptor
 * @param[out] output Output buffer (16kHz, mono, 16-bit) - must be pre-allocated
 * @param[in] output_max Maximum output buffer size (samples)
 * @param[in] normalize Whether to normalize amplitude
 * 
 * @return Number of output samples (>= 0)
 *         Negative error code on failure
 * 
 * @note Output buffer sizing:
 *       Mono samples = input_samples / num_channels
 *       Resampled = (mono_samples * 16000) / input_sample_rate
 *       Recommended: output_max = resampled + 128 (safety margin)
 * 
 * @example
 * // 44.1kHz stereo 16-bit → 16kHz mono
 * moonshine_audio_format_t fmt = {
 *     .sample_rate = 44100,
 *     .num_channels = 2,
 *     .bits_per_sample = 16
 * };
 * int16_t output[40000];  // ~2.5 seconds @ 16kHz
 * int32_t out_samples = moonshine_preprocess_audio(
 *     input, input_samples, &fmt, output, 40000, true
 * );
 */
int32_t moonshine_preprocess_audio(
    const void* input,
    int32_t input_samples,
    const moonshine_audio_format_t* input_format,
    int16_t* output,
    int32_t output_max,
    bool normalize
);

/**
 * @brief Calculate required output buffer size
 * 
 * Helper to estimate output buffer size for preprocessing
 * 
 * @param[in] input_samples Total input samples (including all channels)
 * @param[in] input_format Input format descriptor
 * 
 * @return Estimated output buffer size (samples)
 */
int32_t moonshine_calculate_output_size(
    int32_t input_samples,
    const moonshine_audio_format_t* input_format
);

#ifdef __cplusplus
}
#endif

#endif /* MOONSHINE_AUDIO_UTILS_H */
