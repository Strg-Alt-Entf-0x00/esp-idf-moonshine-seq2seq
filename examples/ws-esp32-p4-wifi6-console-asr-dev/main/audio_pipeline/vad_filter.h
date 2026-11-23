/**
 * @file vad_filter.h
 * @brief Voice Activity Detection (VAD) Filter for ASR
 * 
 * Uses PulseVAD (2.1k ultra-lightweight) as a pre-filter before ASR inference.
 * Rejects pure silence using an RMS threshold and uses neural VAD
 * to ensure speech is present before invoking heavy encoder models.
 */

#ifndef VAD_FILTER_H
#define VAD_FILTER_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Configuration for VAD Filter
 */
typedef struct {
    float rms_threshold;        ///< RMS energy threshold for early rejection (e.g. 0.005)
    float vad_threshold;        ///< Probability threshold for neural VAD (default 0.5)
    uint32_t min_speech_ms;     ///< Minimum speech duration to trigger (default 100ms)
    uint32_t min_silence_ms;    ///< Silence hangover duration (default 200ms)
    uint32_t chunk_size_ms;     ///< Processing chunk size in ms (default 200ms for PulseVAD)
} vad_filter_config_t;

/**
 * @brief Opaque handle for VAD filter
 */
typedef struct vad_filter_t* vad_filter_handle_t;

/**
 * @brief Initialize the VAD filter and load the model
 * 
 * @param config Configuration parameters
 * @param handle Output handle
 * @return ESP_OK on success
 */
esp_err_t vad_filter_init(const vad_filter_config_t* config, vad_filter_handle_t* handle);

/**
 * @brief Process an audio chunk to check for speech
 * 
 * Takes raw audio and determines if speech is present by:
 * 1. Checking RMS energy (fast pre-filter)
 * 2. Running PulseVAD neural inference
 * 3. Applying hysteresis (min_speech_ms / min_silence_ms)
 * 
 * NOTE: PulseVAD works best with 3200 samples (200ms @ 16kHz).
 *       Smaller/larger chunks will be buffered internally.
 * 
 * @param handle VAD filter handle
 * @param audio Raw audio (16kHz, 16-bit PCM)
 * @param audio_len Number of samples
 * @param has_speech Output boolean: true if speech detected
 * @return ESP_OK on success
 */
esp_err_t vad_filter_process(
    vad_filter_handle_t handle,
    const int16_t* audio,
    uint32_t audio_len,
    bool* has_speech
);

/**
 * @brief Deinitialize and free resources
 * 
 * @param handle VAD filter handle
 * @return ESP_OK on success
 */
esp_err_t vad_filter_deinit(vad_filter_handle_t handle);

/**
 * @brief Reset internal accumulator buffer and VAD state machine
 * 
 * @param handle VAD filter handle
 * @return ESP_OK on success
 */
esp_err_t vad_filter_reset(vad_filter_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif // VAD_FILTER_H
