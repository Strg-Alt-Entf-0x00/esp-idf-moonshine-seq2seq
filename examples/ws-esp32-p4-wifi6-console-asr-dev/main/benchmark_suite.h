#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Run an automated benchmark against a directory of WAV files.
 * 
 * @param model_dir Directory containing native .espm model files
 * @param audio_dir Path to the directory containing .wav files
 * @param output_csv Path to the CSV file where results will be appended
 * @param model_prefix Optional ESPM model-name prefix
 * @param audio_prefix Optional filename prefix used to select WAV files
 * @return esp_err_t 
 */
esp_err_t benchmark_run_suite(
    const char* model_dir,
    const char* audio_dir,
    const char* output_csv,
    const char* model_prefix,
    const char* audio_prefix);

#ifdef __cplusplus
}
#endif
