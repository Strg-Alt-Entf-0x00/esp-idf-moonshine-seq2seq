/**
 * @file main.c
 * @brief ESP32-P4 Automatic Speech Recognition - Main Application
 *
 * This is the entry point for the ESP32-P4 ASR system.
 * It initializes all components and starts the main ASR loop.
 *
 * @author Autonomous AI Development System
 * @date 2026-08-15
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <atomic>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_system.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_log.h"
#include "esp_console.h"
#include "esp_idf_version.h"
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "sdmmc_cmd.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include "soc/soc_caps.h"
#include "cmd_asr_cli.h"
#include "esp_heap_caps.h"
#include "esp_task_wdt.h"
#include "esp_cpu.h"
#include "nvs_flash.h"

#include "moonshine_tokenizer.h"
#include "moonshine_audio_utils.h"
#include "app_config.h"
#include "wifi_transfer.h"

// Component includes
#include "audio_manager.h"
#include "audio_wav.h"
#include "profiler.h"
#include "debug_log.h"            // Professional debug system
#include "debug_uart.h"           // Secondary UART for real-time debug output
#include "console_cli.h"          // Console command line interface
#include "moonshine.h"            // Moonshine inference engine
#include "model_manager.h"
#include "vad_filter.h"           // Voice Activity Detection filter
// Test functions (declared in test files)
extern "C" {
    void test_moonshine_encoder(void);
    void benchmark_moonshine_encoder(void);
}

static const char *TAG = "ASR_MAIN";

// Global handles (exported for console_cli and other components)



// Moonshine models (loaded from SD card)
moonshine_model_t g_encoder_model;
moonshine_model_t g_decoder_model;
bool g_models_loaded = false;
std::atomic<bool> g_listen_active{false};
std::atomic<bool> g_live_inference_active{false};

// VAD Filter handle
static vad_filter_handle_t g_vad_filter = NULL;

/**
 * @brief Print system information
 */
static void print_system_info(void)
{
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, "ESP32-P4 ASR System Starting...");
    ESP_LOGI(TAG, "================================");
    
    // Get chip information
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    
    ESP_LOGI(TAG, "Chip: ESP32-P4 with %d cores", chip_info.cores);
    ESP_LOGI(TAG, "Silicon revision: %d", chip_info.revision);
    
    uint32_t flash_size;
    esp_flash_get_size(NULL, &flash_size);
    ESP_LOGI(TAG, "Flash: %luMB %s", 
             flash_size / (1024 * 1024),
             (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "embedded" : "external");
    
    // Memory information
    size_t free_heap = esp_get_free_heap_size();
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t total_psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    
    ESP_LOGI(TAG, "Free heap: %d bytes", free_heap);
    ESP_LOGI(TAG, "PSRAM: %d / %d bytes free", free_psram, total_psram);
    
    ESP_LOGI(TAG, "================================");
}

/**
 * @brief Print configuration
 */
static void print_config(void)
{
    ESP_LOGI(TAG, "Configuration:");
    ESP_LOGI(TAG, "  Sample Rate: %d Hz", ASR_SAMPLE_RATE);
    ESP_LOGI(TAG, "  Channels: %d", ASR_CHANNELS);
}

/**
 * @brief Initialize NVS (Non-Volatile Storage)
 */
static esp_err_t init_nvs(void)
{
    ESP_LOGI(TAG, "Initializing NVS...");
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    return ret;
}

/**
 * @brief Test with WAV file
 */

/**
 * @brief Run Moonshine inference, optionally exporting the encoder states
 */
static esp_err_t run_inference_on_buffer(
    int16_t* audio_buffer,
    size_t num_samples,
    const char* source_name,
    const char* encoder_dump_path)
{
    if (!audio_buffer || !source_name) {
        ESP_LOGE(TAG, "Invalid audio input");
        return ESP_ERR_INVALID_ARG;
    }
    if (num_samples < MOONSHINE_MIN_AUDIO_SAMPLES) {
        ESP_LOGE(TAG, "Audio is too short: %u samples; need at least %u",
                 (unsigned)num_samples, (unsigned)MOONSHINE_MIN_AUDIO_SAMPLES);
        return ESP_ERR_INVALID_SIZE;
    }

    // Step 0.5: VAD Pre-filtering
    if (g_vad_filter && !encoder_dump_path) {
        ESP_LOGI(TAG, "Step 0.5: Running VAD filter...");
        vad_filter_reset(g_vad_filter);
        bool has_speech = false;
        PERF_START(vad_filter);
        esp_err_t vad_ret = vad_filter_process(g_vad_filter, audio_buffer, num_samples, &has_speech);
        PERF_END(vad_filter, TAG);
        
        if (vad_ret == ESP_OK) {
            if (!has_speech) {
                ESP_LOGI(TAG, "VAD detected no speech. Skipping ASR inference.");
                vad_filter_reset(g_vad_filter);
                return ESP_OK;
            }
        } else {
            ESP_LOGW(TAG, "VAD filter failed (%d). Proceeding anyway...", vad_ret);
        }
        vad_filter_reset(g_vad_filter);
    }
    
    // Step 1: Normalize audio to float [-1.0, 1.0]
    ESP_LOGI(TAG, "Step 1: Normalizing raw audio to float...");
    
    float* features = (float*)heap_caps_malloc(num_samples * sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (!features) {
        ESP_LOGE(TAG, "Failed to allocate audio feature buffer!");
        return ESP_ERR_NO_MEM;
    }
    
    for (size_t i = 0; i < num_samples; i++) {
        features[i] = (float)audio_buffer[i] / 32768.0f;
    }
    
    // Show first few feature values for debugging
    ESP_LOGI(TAG, "First frame audio values:");
    for (uint32_t i = 0; i < 10 && i < num_samples; i++) {
        ESP_LOGI(TAG, "  Audio[%lu] = %.6f", i, features[i]);
    }
    
    // Calculate final sequence length (number of frames) out of the encoder
    // All convolutions have padding=0 (verified from PyTorch model inspection)
    // Conv1: kernel=127, stride=64, padding=0
    uint32_t seq_len1 = ((num_samples - 127) / 64) + 1;
    // Conv2: kernel=7, stride=3, padding=0
    uint32_t seq_len2 = ((seq_len1 - 7) / 3) + 1;
    // Conv3: kernel=3, stride=2, padding=0
    uint32_t seq_len3 = ((seq_len2 - 3) / 2) + 1;
    
    uint32_t extracted_frames = seq_len3;
    
    ESP_LOGI(TAG, "✓ Audio normalization complete");
    ESP_LOGI(TAG, "  Raw samples: %lu", num_samples);
    ESP_LOGI(TAG, "  Encoder target frames: %lu", extracted_frames);
    
    // Step 2: Run encoder inference
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, "Step 2: Running encoder inference...");
    ESP_LOGI(TAG, "================================");
    
    if (!g_models_loaded) {
        ESP_LOGE(TAG, "Models not loaded!");
        heap_caps_free(features);
        return ESP_ERR_INVALID_STATE;
    }
    
    // Create encoder context
    moonshine_context_t encoder_ctx;
    memset(&encoder_ctx, 0, sizeof(moonshine_context_t));
    
    esp_err_t ret = moonshine_create_context(&g_encoder_model, extracted_frames, &encoder_ctx);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create encoder context!");
        heap_caps_free(features);
        return ret;
    }
    
    ESP_LOGI(TAG, "Encoder context created");
    
    // Calculate encoder output size: num_frames × hidden_size
    uint32_t hidden_size = g_encoder_model.header.hidden_size;  // 288 for moonshine-tiny
    size_t encoder_output_size = extracted_frames * hidden_size * sizeof(float);
    
    ESP_LOGI(TAG, "Allocating encoder output buffer:");
    ESP_LOGI(TAG, "  Frames: %lu", extracted_frames);
    ESP_LOGI(TAG, "  Hidden size: %lu", hidden_size);
    ESP_LOGI(TAG, "  Total size: %.2f KB", encoder_output_size / 1024.0f);
    
    float* encoder_output = (float*)heap_caps_malloc(encoder_output_size, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (!encoder_output) {
        ESP_LOGE(TAG, "Failed to allocate encoder output buffer!");
        heap_caps_free(features);
        moonshine_destroy_context(&encoder_ctx);
        return ESP_ERR_NO_MEM;
    }
    
    // Run encoder inference (pass num_samples!)
    int64_t encoder_start = esp_timer_get_time();
    PERF_START(encoder);
    ret = moonshine_encode(&encoder_ctx, features, num_samples, encoder_output);
    PERF_END(encoder, TAG);
    int64_t encoder_time_us = esp_timer_get_time() - encoder_start;
    (void)encoder_time_us; // Suppress unused warning
    
    if (ret != ESP_OK) {
        moonshine_destroy_context(&encoder_ctx);
        heap_caps_free(encoder_output);
        heap_caps_free(features);
        return ret;
    }
    
    ESP_LOGI(TAG, "✓ Encoder inference complete!");
    ESP_LOGI(TAG, "  Output shape: %lu × %lu", extracted_frames, hidden_size);
    ESP_LOGI(TAG, "  Output size: %.2f KB", encoder_output_size / 1024.0f);
    
    // Show first few encoder output values for debugging
    ESP_LOGI(TAG, "First frame encoder values (first 10 dims):");
    for (uint32_t i = 0; i < 10 && i < hidden_size; i++) {
        ESP_LOGI(TAG, "  Hidden[%lu] = %.6f", i, encoder_output[i]);
    }
    
    // Check for reasonable values (not NaN or Inf)
    bool values_ok = true;
    uint32_t check_count = extracted_frames * hidden_size;
    if (check_count > 100) check_count = 100;  // Check first 100 values
    
    for (uint32_t i = 0; i < check_count; i++) {
        if (isnan(encoder_output[i]) || isinf(encoder_output[i])) {
            ESP_LOGE(TAG, "Invalid encoder output at index %lu: %.6f", i, encoder_output[i]);
            values_ok = false;
            break;
        }
    }
    
    if (values_ok) {
        ESP_LOGI(TAG, "✓ Encoder output values are valid (no NaN/Inf)");
    } else {
        ESP_LOGE(TAG, "✗ Encoder output contains invalid values!");
        moonshine_destroy_context(&encoder_ctx);
        heap_caps_free(encoder_output);
        heap_caps_free(features);
        return ESP_ERR_INVALID_RESPONSE;
    }

    if (encoder_dump_path) {
        FILE* dump_file = fopen(encoder_dump_path, "wb");
        if (!dump_file) {
            ESP_LOGE(TAG, "Failed to open encoder dump file: %s", encoder_dump_path);
            moonshine_destroy_context(&encoder_ctx);
            heap_caps_free(encoder_output);
            heap_caps_free(features);
            return ESP_FAIL;
        }

        size_t output_values = (size_t)extracted_frames * hidden_size;
        size_t values_written = fwrite(encoder_output, sizeof(float), output_values, dump_file);
        int close_result = fclose(dump_file);
        if (values_written != output_values || close_result != 0) {
            ESP_LOGE(TAG, "Failed to write complete encoder dump to %s", encoder_dump_path);
            moonshine_destroy_context(&encoder_ctx);
            heap_caps_free(encoder_output);
            heap_caps_free(features);
            return ESP_FAIL;
        }

        ESP_LOGI(TAG, "Wrote %zu float32 encoder values (%lu x %lu) to %s",
                 output_values, extracted_frames, hidden_size, encoder_dump_path);
        moonshine_destroy_context(&encoder_ctx);
        heap_caps_free(encoder_output);
        heap_caps_free(features);
        return ESP_OK;
    }
    
    // Step 3: Run decoder inference
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, "Step 3: Running decoder inference...");
    ESP_LOGI(TAG, "================================");
    
    // Create decoder context
    moonshine_context_t decoder_ctx;
    memset(&decoder_ctx, 0, sizeof(moonshine_context_t));
    
    ret = moonshine_create_context(&g_decoder_model, 100, &decoder_ctx); // max 100 tokens
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create decoder context!");
        heap_caps_free(encoder_output);
        moonshine_destroy_context(&encoder_ctx);
        heap_caps_free(features);
        return ret;
    }
    
    uint32_t num_tokens = 100;
    
    int64_t decoder_start = esp_timer_get_time();
    PERF_START(decoder);
    ret = moonshine_decode(&decoder_ctx, encoder_output, extracted_frames, decoder_ctx.tokens, &num_tokens);
    PERF_END(decoder, TAG);
    int64_t decoder_time_us = esp_timer_get_time() - decoder_start;
    (void)decoder_time_us; // Suppress unused warning
    
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Decoder inference failed!");
        moonshine_destroy_context(&decoder_ctx);
        moonshine_destroy_context(&encoder_ctx);
        heap_caps_free(encoder_output);
        heap_caps_free(features);
        return ret;
    }
    ESP_LOGI(TAG, "✓ Decoder inference complete! Generated %lu tokens.", num_tokens);

    ESP_LOGI(TAG, "Generated tokens:");
    char token_str[512] = {0};
    int offset = 0;
    for (uint32_t i = 0; i < num_tokens; i++) {
        offset += snprintf(token_str + offset, sizeof(token_str) - offset, "%ld ", decoder_ctx.tokens[i]);
        if (offset >= sizeof(token_str) - 10) break;
    }
    ESP_LOGI(TAG, "[ %s]", token_str);
    
    // Step 4: Convert tokens to text
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, "Step 4: Token Decoding");
    ESP_LOGI(TAG, "================================");
    
    size_t decoded_text_size = (size_t)num_tokens * 16;
    char* decoded_text = (char*)heap_caps_calloc(1, decoded_text_size, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (!decoded_text) {
        ESP_LOGE(TAG, "Failed to allocate memory for decoded text");
        moonshine_destroy_context(&decoder_ctx);
        moonshine_destroy_context(&encoder_ctx);
        heap_caps_free(encoder_output);
        heap_caps_free(features);
        return ESP_ERR_NO_MEM;
    }
    ret = moonshine_tokens_to_text(decoder_ctx.tokens, num_tokens, decoded_text, decoded_text_size);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, " ");
        ESP_LOGI(TAG, ">>> RECOGNIZED TEXT <<<");
        ESP_LOGI(TAG, "%s", decoded_text);
        ESP_LOGI(TAG, ">>>>>>>>><<<<<<<<<<<<<<");
        ESP_LOGI(TAG, " ");
    } else {
        ESP_LOGE(TAG, "Token-to-text conversion failed: %d", ret);
    }
    heap_caps_free(decoded_text);
    // Cleanup
    moonshine_destroy_context(&decoder_ctx);
    moonshine_destroy_context(&encoder_ctx);
    heap_caps_free(encoder_output);
    heap_caps_free(features);
    
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, "Inference Complete for %s", source_name);
    ESP_LOGI(TAG, "================================");
    return ret;
}
esp_err_t test_with_wav_file(const char* wav_path, const char* encoder_dump_path)
{
    if (!wav_path || (encoder_dump_path && strcmp(wav_path, encoder_dump_path) == 0)) {
        return ESP_ERR_INVALID_ARG;
    }
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, "WAV File ASR Test");
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, "Loading WAV file: %s", wav_path);
    
    // Load WAV file
    int16_t* audio_buffer = NULL;
    size_t buffer_size = 0;
    audio_wav_info_t wav_info;
    
    PERF_START(wav_load);
    esp_err_t ret = audio_wav_load_file(wav_path, &audio_buffer, &buffer_size, &wav_info);
    PERF_END(wav_load, TAG);
    
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to load WAV file!");
        return ret;
    }
    
    ESP_LOGI(TAG, "✓ WAV file loaded successfully");
    ESP_LOGI(TAG, "  Sample rate: %u Hz", wav_info.sample_rate);
    ESP_LOGI(TAG, "  Channels: %d", wav_info.num_channels);
    ESP_LOGI(TAG, "  Bits/sample: %d", wav_info.bits_per_sample);
    ESP_LOGI(TAG, "  Duration: %.2f seconds", (float)wav_info.duration_ms / 1000.0f);
    ESP_LOGI(TAG, "  Data size: %u bytes", wav_info.data_size);
    
    // Validate format and determine if preprocessing is needed
    moonshine_audio_format_t fmt = {
        .sample_rate = wav_info.sample_rate,
        .num_channels = (uint8_t)wav_info.num_channels,
        .bits_per_sample = (uint8_t)wav_info.bits_per_sample
    };
    
    bool needs_preprocessing = (
        fmt.sample_rate != ASR_SAMPLE_RATE ||
        fmt.num_channels != ASR_CHANNELS ||
        fmt.bits_per_sample != ASR_BITS_PER_SAMPLE
    );
    
    if (needs_preprocessing) {
        ESP_LOGW(TAG, "Audio format mismatch - preprocessing required");
        ESP_LOGI(TAG, "Converting: %luHz %dch %dbit → %dHz %dch %dbit",
                 fmt.sample_rate, fmt.num_channels, fmt.bits_per_sample,
                 ASR_SAMPLE_RATE, ASR_CHANNELS, ASR_BITS_PER_SAMPLE);
    } else {
        ESP_LOGI(TAG, "✓ Audio format correct - no preprocessing needed");
    }
    
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, "Processing Audio...");
    ESP_LOGI(TAG, "================================");
    
    // Calculate number of samples
    size_t num_samples = buffer_size / sizeof(int16_t);
    ESP_LOGI(TAG, "Audio buffer: %lu samples", num_samples);
    
    int16_t* processed_buffer = NULL;
    size_t processed_samples = num_samples;
    
    if (needs_preprocessing) {
        // Calculate required output buffer size
        int32_t output_size = moonshine_calculate_output_size(num_samples, &fmt);
        
        if (output_size <= 0) {
            ESP_LOGE(TAG, "Failed to calculate output buffer size");
            free(audio_buffer);
            return ESP_ERR_INVALID_SIZE;
        }
        
        // Allocate preprocessing output buffer
        processed_buffer = (int16_t*)heap_caps_malloc(
            output_size * sizeof(int16_t),
            MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM
        );
        
        if (!processed_buffer) {
            ESP_LOGE(TAG, "Failed to allocate preprocessing buffer (%ld bytes)", output_size * sizeof(int16_t));
            free(audio_buffer);
            return ESP_ERR_NO_MEM;
        }
        
        ESP_LOGI(TAG, "Preprocessing audio...");
        PERF_START(preprocess);
        
        // Run preprocessing pipeline
        int32_t out_samples = moonshine_preprocess_audio(
            audio_buffer,
            num_samples,
            &fmt,
            processed_buffer,
            output_size,
            true  // normalize
        );
        
        PERF_END(preprocess, TAG);
        
        if (out_samples <= 0) {
            ESP_LOGE(TAG, "Preprocessing failed: %ld", out_samples);
            free(processed_buffer);
            free(audio_buffer);
            return out_samples < 0 ? (esp_err_t)(-out_samples) : ESP_ERR_INVALID_RESPONSE;
        }
        
        processed_samples = out_samples;
        ESP_LOGI(TAG, "✓ Preprocessing complete: %lu → %lu samples", num_samples, processed_samples);
        
        // Free original buffer, use preprocessed
        free(audio_buffer);
        audio_buffer = processed_buffer;
    } else {
        ESP_LOGI(TAG, "Using original audio buffer (no conversion needed)");
    }

    ret = run_inference_on_buffer(audio_buffer, processed_samples, wav_path, encoder_dump_path);
    
    free(audio_buffer);
    
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, "Audio preprocessing complete");
    ESP_LOGI(TAG, "================================");
    
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, "WAV Test Complete");
    ESP_LOGI(TAG, "================================");
    return ret;
}

/**
 * @brief Process live microphone audio for speech recognition.
 */

static void asr_main_task_i2s(void *pvParameters)
{
    ESP_LOGI(TAG, "ASR main task started (I2S mode with PulseVAD & Pre-Roll)");
    
    // Allocate 8 seconds buffer @ 16kHz for dynamic recording (in PSRAM)
    const size_t max_samples = ASR_SAMPLE_RATE * 8; 
    int16_t *audio_buffer = (int16_t*)heap_caps_malloc(max_samples * sizeof(int16_t), MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (!audio_buffer) {
        ESP_LOGE(TAG, "Failed to allocate mic buffer");
        vTaskDelete(NULL);
        return;
    }

    // 200ms processing chunks (3,200 samples @ 16kHz) matching PulseVAD's native window
    const size_t chunk_samples = ASR_SAMPLE_RATE / 5;
    constexpr int pre_roll_chunks = 2;
    const size_t work_buffer_samples = (pre_roll_chunks + 1) * chunk_samples;
    int16_t *work_buffer = (int16_t*)heap_caps_malloc(
        work_buffer_samples * sizeof(int16_t),
        MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (!work_buffer) {
        ESP_LOGE(TAG, "Failed to allocate I2S task buffers");
        free(audio_buffer);
        vTaskDelete(NULL);
        return;
    }

    int16_t *chunk = work_buffer;
    int16_t *pre_roll = work_buffer + chunk_samples;
    int pr_write_idx = 0;
    int pr_count = 0;

    size_t current_samples = 0;
    bool is_recording = false;
    int silence_chunks = 0;
    bool last_listen_active = false;
    
    while (1) {
        if (!g_listen_active.load()) {
            if (last_listen_active) {
                // Listen mode just deactivated: clean up
                is_recording = false;
                current_samples = 0;
                silence_chunks = 0;
                pr_count = 0;
                pr_write_idx = 0;
                if (g_vad_filter) vad_filter_reset(g_vad_filter);
                last_listen_active = false;
                if (audio_manager_is_capture_running()) {
                    const esp_err_t stop_error = audio_manager_stop_capture();
                    if (stop_error != ESP_OK) {
                        ESP_LOGE(TAG, "Failed to stop microphone capture: %s",
                                 esp_err_to_name(stop_error));
                    }
                }
            }
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        if (!last_listen_active) {
            // Listen mode just activated: clean start
            last_listen_active = true;
            is_recording = false;
            current_samples = 0;
            silence_chunks = 0;
            pr_count = 0;
            pr_write_idx = 0;
            if (g_vad_filter) vad_filter_reset(g_vad_filter);
            ESP_LOGI(TAG, "Listening active (200ms VAD window, 400ms pre-roll, 800ms hangover)...");
        }
        
        // Ensure models are loaded before listening
        if (!g_models_loaded) {
            ESP_LOGW(TAG, "Models not loaded. Cannot listen. Stopping listen mode.");
            g_listen_active.store(false);
            const esp_err_t stop_error = audio_manager_stop_capture();
            if (stop_error != ESP_OK) {
                ESP_LOGE(TAG, "Failed to stop microphone capture: %s",
                         esp_err_to_name(stop_error));
            }
            continue;
        }
        
        // NOTE: Microphone audio from Audio Manager is already in Moonshine format
        // (16kHz mono 16-bit). No preprocessing needed for live audio.
        int read = audio_manager_read(chunk, chunk_samples, 400);
        if (read <= 0) {
            vTaskDelay(pdMS_TO_TICKS(10));  // Prevent tight loop if no audio available
            continue;
        }

        if (!is_recording) {
            // IDLE LISTENING: buffer into pre-roll FIFO and run VAD
            memcpy(pre_roll + pr_write_idx * chunk_samples, chunk, read * sizeof(int16_t));
            pr_write_idx = (pr_write_idx + 1) % pre_roll_chunks;
            if (pr_count < pre_roll_chunks) {
                pr_count++;
            }

            bool has_speech = false;
            if (g_vad_filter) {
                vad_filter_process(g_vad_filter, chunk, read, &has_speech);
            } else {
                has_speech = true; // Fallback if no VAD
            }

            if (has_speech) {
                is_recording = true;
                current_samples = 0;
                silence_chunks = 0;

                // Prepend pre-roll buffer in chronological order so no leading phonemes are lost
                int oldest_idx = (pr_write_idx - pr_count + pre_roll_chunks) % pre_roll_chunks;
                for (int i = 0; i < pr_count; i++) {
                    int idx = (oldest_idx + i) % pre_roll_chunks;
                    if (current_samples + chunk_samples <= max_samples) {
                        memcpy(&audio_buffer[current_samples], pre_roll + idx * chunk_samples, chunk_samples * sizeof(int16_t));
                        current_samples += chunk_samples;
                    }
                }

                // Append current speech chunk
                if (current_samples + read <= max_samples) {
                    memcpy(&audio_buffer[current_samples], chunk, read * sizeof(int16_t));
                    current_samples += read;
                }

                ESP_LOGI(TAG, "Speech onset detected! Prepended %d ms pre-roll. Recording started (%zu samples)...",
                         pr_count * 200, current_samples);

                pr_count = 0;
                pr_write_idx = 0;
            }
        } else {
            // ACTIVE RECORDING: append chunk to audio buffer and monitor silence hangover
            if (current_samples + read <= max_samples) {
                memcpy(&audio_buffer[current_samples], chunk, read * sizeof(int16_t));
                current_samples += read;
            }

            bool has_speech = false;
            if (g_vad_filter) {
                vad_filter_process(g_vad_filter, chunk, read, &has_speech);
            } else {
                has_speech = true;
            }

            if (has_speech) {
                silence_chunks = 0;
            } else {
                silence_chunks++;
            }

            // End-of-utterance: 4 consecutive silence chunks = 800ms silence hangover
            // Minimum speech duration: at least 1.0s total audio
            if (silence_chunks >= 4 && current_samples >= ASR_SAMPLE_RATE) {
                ESP_LOGI(TAG, "Silence hangover met (800 ms). Finalizing utterance (%zu samples, %.2fs)...",
                         current_samples, (float)current_samples / ASR_SAMPLE_RATE);

                // Trim 600ms of trailing silence, keeping 200ms padding
                size_t trim_samples = 3 * chunk_samples;
                if (current_samples > trim_samples + ASR_SAMPLE_RATE / 2) {
                    current_samples -= trim_samples;
                }

                g_live_inference_active.store(true);
                esp_err_t infer_ret = run_inference_on_buffer(audio_buffer, current_samples, "mic", NULL);
                g_live_inference_active.store(false);
                if (infer_ret != ESP_OK) {
                    ESP_LOGE(TAG, "Microphone inference failed: %d", infer_ret);
                }

                is_recording = false;
                current_samples = 0;
                silence_chunks = 0;
                pr_count = 0;
                pr_write_idx = 0;
                if (g_vad_filter) vad_filter_reset(g_vad_filter);

                g_listen_active.store(false);
                const esp_err_t stop_error = audio_manager_stop_capture();
                if (stop_error != ESP_OK) {
                    ESP_LOGE(TAG, "Failed to stop microphone capture: %s",
                             esp_err_to_name(stop_error));
                }
                ESP_LOGI(TAG, "Listening paused. Type 'listen start' to resume.");
                continue;
            }

            // Buffer full protection
            if (current_samples + chunk_samples > max_samples) {
                ESP_LOGI(TAG, "Max recording buffer reached (%zu samples, %.2fs). Finalizing...",
                         current_samples, (float)current_samples / ASR_SAMPLE_RATE);

                g_live_inference_active.store(true);
                esp_err_t infer_ret = run_inference_on_buffer(audio_buffer, current_samples, "mic", NULL);
                g_live_inference_active.store(false);
                if (infer_ret != ESP_OK) {
                    ESP_LOGE(TAG, "Microphone inference failed: %d", infer_ret);
                }

                is_recording = false;
                current_samples = 0;
                silence_chunks = 0;
                pr_count = 0;
                pr_write_idx = 0;
                if (g_vad_filter) vad_filter_reset(g_vad_filter);

                g_listen_active.store(false);
                const esp_err_t stop_error = audio_manager_stop_capture();
                if (stop_error != ESP_OK) {
                    ESP_LOGE(TAG, "Failed to stop microphone capture: %s",
                             esp_err_to_name(stop_error));
                }
                ESP_LOGI(TAG, "Listening paused. Type 'listen start' to resume.");
                continue;
            }
        }
    }
}



#if defined(CONFIG_IDF_TARGET_ESP32P4) && \
    defined(CONFIG_ESP_HOSTED_HOST_TRANSPORT_BUS_SDIO) && \
    ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
static esp_err_t shared_sdmmc_host_init(void) { return ESP_OK; }
static esp_err_t shared_sdmmc_host_deinit(void) { return ESP_OK; }
#endif

static esp_err_t mount_sd_card(void)
{
#if SOC_SDMMC_HOST_SUPPORTED
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;
#if defined(CONFIG_IDF_TARGET_ESP32P4)
    host.slot = SDMMC_HOST_SLOT_0;
#endif
#if defined(CONFIG_IDF_TARGET_ESP32P4) && \
    defined(CONFIG_ESP_HOSTED_HOST_TRANSPORT_BUS_SDIO) && \
    ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
    host.init = shared_sdmmc_host_init;
    host.deinit = shared_sdmmc_host_deinit;
#endif

#if defined(CONFIG_IDF_TARGET_ESP32P4)
    sd_pwr_ctrl_ldo_config_t ldo_config = {.ldo_chan_id = 4};
    sd_pwr_ctrl_handle_t power_control = nullptr;
    const esp_err_t power_error =
        sd_pwr_ctrl_new_on_chip_ldo(&ldo_config, &power_control);
    if (power_error != ESP_OK) {
        return power_error;
    }
    host.pwr_ctrl_handle = power_control;
#endif

    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
#if defined(CONFIG_IDF_TARGET_ESP32P4)
    slot.clk = static_cast<gpio_num_t>(43);
    slot.cmd = static_cast<gpio_num_t>(44);
    slot.d0 = static_cast<gpio_num_t>(39);
    slot.d1 = static_cast<gpio_num_t>(40);
    slot.d2 = static_cast<gpio_num_t>(41);
    slot.d3 = static_cast<gpio_num_t>(42);
    slot.width = 4;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
#endif

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {};
    mount_config.format_if_mount_failed = false;
    mount_config.max_files = 10;
    mount_config.allocation_unit_size = 16 * 1024;
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
    mount_config.disk_status_check_enable = true;
    mount_config.use_one_fat = false;
#endif

    sdmmc_card_t* card = nullptr;
    const esp_err_t error = esp_vfs_fat_sdmmc_mount(
        "/sd", &host, &slot, &mount_config, &card);
    if (error != ESP_OK) {
        ESP_LOGE(TAG, "SD-card mount failed: %s", esp_err_to_name(error));
#if defined(CONFIG_IDF_TARGET_ESP32P4)
        sd_pwr_ctrl_del_on_chip_ldo(power_control);
#endif
        return error;
    }

    sdmmc_card_print_info(stdout, card);
    const char* sd_directories[] = {"/sd/models", "/sd/audio", "/sd/results"};
    for (const char* directory : sd_directories) {
        if (mkdir(directory, 0775) != 0 && errno != EEXIST) {
            ESP_LOGE(TAG, "Cannot create %s: errno=%d", directory, errno);
        }
    }
    ESP_LOGI(TAG, "SD card mounted at /sd");
    return ESP_OK;
#else
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

/**
 * @brief Load Moonshine models from SD card
 */
esp_err_t load_moonshine_models(const char* model_path)
{
    if (!model_path) {
        return ESP_ERR_INVALID_ARG;
    }
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Loading native Moonshine ESPM from SD Card");
    ESP_LOGI(TAG, "========================================");
    
    // Deinitialize VAD filter first to free internal RAM used by cache blocks
    if (g_vad_filter) {
        vad_filter_deinit(g_vad_filter);
        g_vad_filter = NULL;
    }
    
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Model: %s", model_path);
    ESP_LOGI(TAG, "========================================");
    
    // Zero out model structures
    memset(&g_encoder_model, 0, sizeof(moonshine_model_t));
    memset(&g_decoder_model, 0, sizeof(moonshine_model_t));
    
    // Initialize Moonshine inference engine
    ESP_LOGI(TAG, "Initializing Moonshine engine...");
    esp_err_t ret = moonshine_init();
    if (ret != ESP_OK) {
        ESP_LOGI(TAG, "ERROR: Failed to initialize Moonshine engine: %d", ret);
        ESP_LOGE(TAG, "Failed to initialize Moonshine engine");
        return ret;
    }
    ESP_LOGI(TAG, "✓ Moonshine engine initialized");
    
    ESP_LOGI(TAG, "Loading native ESPM tensors and embedded tokenizer...");
    ret = moonshine_load_espm(model_path, &g_encoder_model, &g_decoder_model, true);
    
    if (ret != ESP_OK) {
        ESP_LOGI(TAG, "ERROR: Failed to load models: %d", ret);
        ESP_LOGE(TAG, "Failed to load models");
        return ret;
    }
    ESP_LOGI(TAG, "✓ Models loaded successfully");
    
    // Verify models
    ESP_LOGI(TAG, "Verifying encoder...");
    ret = moonshine_verify_model(&g_encoder_model);
    if (ret != ESP_OK) {
        ESP_LOGI(TAG, "ERROR: Encoder verification failed: %d", ret);
        ESP_LOGE(TAG, "Encoder verification failed");
        moonshine_unload_model(&g_encoder_model);
        moonshine_unload_model(&g_decoder_model);
        moonshine_tokenizer_free();
        return ret;
    }
    ESP_LOGI(TAG, "✓ Encoder verified");
    
    ESP_LOGI(TAG, "Verifying decoder...");
    ret = moonshine_verify_model(&g_decoder_model);
    if (ret != ESP_OK) {
        ESP_LOGI(TAG, "ERROR: Decoder verification failed: %d", ret);
        ESP_LOGE(TAG, "Decoder verification failed");
        moonshine_unload_model(&g_encoder_model);
        moonshine_unload_model(&g_decoder_model);
        moonshine_tokenizer_free();
        return ret;
    }
    ESP_LOGI(TAG, "✓ Decoder verified");
    
    g_models_loaded = true;
    
    vad_filter_config_t vad_cfg = {
        .rms_threshold = 0.0005f,
        .vad_threshold = 0.5f,
        .min_speech_ms = 100,
        .min_silence_ms = 200,
        .chunk_size_ms = 200
    };
    
    ret = vad_filter_init(&vad_cfg, &g_vad_filter);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "✓ VAD filter initialized (PulseVAD 2.1k params)");
    } else {
        ESP_LOGE(TAG, "Failed to initialize VAD filter. Pre-filtering will be disabled.");
        ESP_LOGI(TAG, "WARNING: Failed to initialize VAD filter");
    }
    
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Models loaded and verified successfully!");
    ESP_LOGI(TAG, "Encoder: %.2f MB, %lu tensors", 
           g_encoder_model.total_size / 1024.0 / 1024.0,
           g_encoder_model.num_tensors);
    ESP_LOGI(TAG, "Decoder: %.2f MB, %lu tensors",
           g_decoder_model.total_size / 1024.0 / 1024.0,
           g_decoder_model.num_tensors);
    ESP_LOGI(TAG, "========================================");
    
    ESP_LOGI(TAG, "✓ Models loaded and verified successfully!");
    ESP_LOGI(TAG, "  Encoder: %.2f MB, %lu tensors", 
             g_encoder_model.total_size / 1024.0 / 1024.0,
             g_encoder_model.num_tensors);
    ESP_LOGI(TAG, "  Decoder: %.2f MB, %lu tensors",
             g_decoder_model.total_size / 1024.0 / 1024.0,
             g_decoder_model.num_tensors);
    ESP_LOGI(TAG, "========================================");
    
    return ESP_OK;
}

/**
 * @brief Application entry point
 */

#include "esp_chip_info.h"
#include "esp_heap_caps.h"

static void print_banner(void) {
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    
    size_t psram_size = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    size_t sram_size = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    
    int cpu_freq_mhz = 0;
#ifdef CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ
    cpu_freq_mhz = CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ;
#endif
    
    printf("\n");
    printf("  ASR CONSOLE\n");
    printf("  ESP32-P4 Edition\n");
    printf("  --------------------------------------------------\n");
    if (cpu_freq_mhz > 0) {
        printf("  Hardware : ESP32-P4 rev %d.%d (%d-Core @ %dMHz)\n", 
               chip_info.revision / 100, chip_info.revision % 100, chip_info.cores, cpu_freq_mhz);
    } else {
        printf("  Hardware : ESP32-P4 rev %d.%d (%d-Core)\n", 
               chip_info.revision / 100, chip_info.revision % 100, chip_info.cores);
    }
    printf("  Memory   : %zu MB PSRAM / %zu KB SRAM\n", 
           psram_size / (1024 * 1024), sram_size / 1024);
    printf("  Version  : v1.1.0\n\n");
}


#include <dirent.h>

extern "C" void moonshine_tokenizer_free(void);

void unload_moonshine_models(void)
{
    moonshine_unload_model(&g_encoder_model);
    moonshine_unload_model(&g_decoder_model);
    moonshine_tokenizer_free();
}



extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting ESP32-P4 ASR System...");
    
    // Initialize debug system
    debug_init();
    
    // Note: UART0 console is already active at 115200 baud
    // No need to initialize debug UART - using standard console
    ESP_LOGI(TAG, "Console: UART0 @ 115200 baud (GPIO%d/%d)", ASR_UART_TX_PIN, ASR_UART_RX_PIN);
    // Print system information
    print_system_info();
    
    // Initialize NVS
    ESP_ERROR_CHECK(init_nvs());
    
    const esp_err_t sd_error = mount_sd_card();
    if (sd_error != ESP_OK) {
        ESP_LOGW(TAG, "SD card unavailable; model and audio file commands will fail: %s",
                 esp_err_to_name(sd_error));
    } else {
        const esp_err_t manager_init_error = model_manager_init("/sd/models");
        if (manager_init_error != ESP_OK) {
            ESP_LOGE(TAG, "Model manager initialization failed: %s",
                     esp_err_to_name(manager_init_error));
        } else {
            const int model_count = model_manager_scan();
            if (model_count < 0) {
                ESP_LOGW(TAG, "Model scan failed: %s",
                         esp_err_to_name((esp_err_t)-model_count));
            } else {
                ESP_LOGI(TAG, "Model manager found %d valid ESPM model(s)", model_count);
            }
        }
    }
    const esp_err_t wifi_error = wifi_transfer_init();
    if (wifi_error != ESP_OK) {
        ESP_LOGW(TAG, "WLAN file transfer initialization failed: %s",
                 esp_err_to_name(wifi_error));
    }
    
    // Print configuration
    print_config();
    
    ESP_LOGI(TAG, "================================");
    // Initialize Audio Manager (I2S, I2C, ES8311 Codec)
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, "Initializing Audio Manager...");
    ESP_LOGI(TAG, "================================");
    esp_err_t audio_ret = audio_manager_init();
    if (audio_ret == ESP_OK) {
        audio_manager_play_boot_sequence();
        ESP_LOGI(TAG, "Audio Manager OK, boot chime played");
    } else {
        ESP_LOGE(TAG, "Audio Manager FAILED! err=0x%x", audio_ret);
    }

    ESP_LOGI(TAG, "Initialization Complete");
    
    // Create I2S Audio Task
    xTaskCreatePinnedToCore(asr_main_task_i2s, "asr_i2s_task", 16 * 1024, NULL, 5, NULL, 1);
    
// Start console REPL
    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_config.prompt = "ASR>";
    repl_config.max_cmdline_length = 256;
    repl_config.task_stack_size = 8192;
    
    // Register ASR console commands
    cmd_asr_cli_register();
    
    esp_console_dev_uart_config_t hw_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&hw_config, &repl_config, &repl));
    
    print_banner();
    printf("  Type 'help'  to see all commands.\n\n");
    
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}
