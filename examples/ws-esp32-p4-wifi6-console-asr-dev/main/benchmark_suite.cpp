#include "benchmark_suite.h"
#include <atomic>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <math.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "moonshine.h"
#include "audio_wav.h"
#include "model_manager.h"

static const char* TAG = "BENCHMARK";
static const char CSV_HEADER[] =
    "Schema_Version,Model_Name,Quantization,Audio_File,Model_Load_Time_ms,"
    "PSRAM_Free_After_Model_Load_Bytes,Audio_Duration_s,Wav_Load_Time_ms,"
    "Encoder_Time_ms,Decoder_Time_ms,RTF,Tokens_Gen,Text,"
    "PSRAM_Free_After_Inference_Bytes";
static constexpr char CSV_SCHEMA_VERSION[] = "2";

static void csv_write_field(FILE* csv_file, const char* value)
{
    fputc('"', csv_file);
    for (const char* ch = value ? value : ""; *ch; ch++) {
        if (*ch == '"') fputc('"', csv_file);
        fputc(*ch, csv_file);
    }
    fputc('"', csv_file);
}

// Extern from main.cpp
extern moonshine_model_t g_encoder_model;
extern moonshine_model_t g_decoder_model;
extern bool g_models_loaded;
extern std::atomic<bool> g_listen_active;
extern void unload_moonshine_models(void);

static esp_err_t benchmark_audio_file(
    const char* wav_path,
    const char* model_name,
    const char* quantization,
    double model_load_time_ms,
    size_t psram_free_after_model_load,
    FILE* csv_file)
{
    int16_t* audio_buffer = NULL;
    size_t buffer_size = 0;
    audio_wav_info_t wav_info;
    
    int64_t load_start = esp_timer_get_time();
    esp_err_t ret = audio_wav_load_file(wav_path, &audio_buffer, &buffer_size, &wav_info);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to load WAV: %s", wav_path);
        return ret;
    }
    double load_time_ms = (double)(esp_timer_get_time() - load_start) / 1000.0;

    if (wav_info.sample_rate != 16000 ||
        wav_info.num_channels != 1 ||
        wav_info.bits_per_sample != 16) {
        ESP_LOGE(TAG, "Benchmark requires 16 kHz mono 16-bit PCM WAV: %s", wav_path);
        free(audio_buffer);
        return ESP_ERR_NOT_SUPPORTED;
    }
    
    size_t num_samples = buffer_size / sizeof(int16_t);
    if (num_samples < MOONSHINE_MIN_AUDIO_SAMPLES) {
        ESP_LOGE(TAG, "WAV is too short for the encoder: %s", wav_path);
        free(audio_buffer);
        return ESP_ERR_INVALID_SIZE;
    }
    float duration_s = (float)num_samples / (float)wav_info.sample_rate;
    
    // Normalize audio
    float* features = (float*)heap_caps_malloc(num_samples * sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (!features) {
        free(audio_buffer);
        return ESP_ERR_NO_MEM;
    }
    
    for (size_t i = 0; i < num_samples; i++) {
        features[i] = (float)audio_buffer[i] / 32768.0f;
    }
    free(audio_buffer);
    
    uint32_t seq_len1 = ((num_samples - 127) / 64) + 1;
    uint32_t seq_len2 = ((seq_len1 - 7) / 3) + 1;
    uint32_t extracted_frames = ((seq_len2 - 3) / 2) + 1;
    
    // Encoder Inference
    moonshine_context_t encoder_ctx;
    memset(&encoder_ctx, 0, sizeof(moonshine_context_t));
    ret = moonshine_create_context(&g_encoder_model, extracted_frames, &encoder_ctx);
    if (ret != ESP_OK) {
        heap_caps_free(features);
        return ret;
    }
    
    uint32_t hidden_size = g_encoder_model.header.hidden_size;
    size_t encoder_output_size = extracted_frames * hidden_size * sizeof(float);
    float* encoder_output = (float*)heap_caps_malloc(encoder_output_size, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    
    if (!encoder_output) {
        moonshine_destroy_context(&encoder_ctx);
        heap_caps_free(features);
        return ESP_ERR_NO_MEM;
    }
    
    int64_t encoder_start = esp_timer_get_time();
    ret = moonshine_encode(&encoder_ctx, features, num_samples, encoder_output);
    double encoder_time_ms = (double)(esp_timer_get_time() - encoder_start) / 1000.0;
    
    if (ret != ESP_OK) {
        moonshine_destroy_context(&encoder_ctx);
        heap_caps_free(encoder_output);
        heap_caps_free(features);
        return ret;
    }
    
    // Decoder Inference
    moonshine_context_t decoder_ctx;
    memset(&decoder_ctx, 0, sizeof(moonshine_context_t));
    ret = moonshine_create_context(&g_decoder_model, 100, &decoder_ctx);
    if (ret != ESP_OK) {
        moonshine_destroy_context(&encoder_ctx);
        heap_caps_free(encoder_output);
        heap_caps_free(features);
        return ret;
    }
    
    uint32_t num_tokens = 100;
    int64_t decoder_start = esp_timer_get_time();
    ret = moonshine_decode(&decoder_ctx, encoder_output, extracted_frames, decoder_ctx.tokens, &num_tokens);
    double decoder_time_ms = (double)(esp_timer_get_time() - decoder_start) / 1000.0;
    if (ret != ESP_OK) {
        moonshine_destroy_context(&decoder_ctx);
        moonshine_destroy_context(&encoder_ctx);
        heap_caps_free(encoder_output);
        heap_caps_free(features);
        return ret;
    }
    
    // Text output
    char decoded_text[512] = {0};
    ret = moonshine_tokens_to_text(decoder_ctx.tokens, num_tokens, decoded_text, sizeof(decoded_text) - 1);
    if (ret != ESP_OK) {
        moonshine_destroy_context(&decoder_ctx);
        moonshine_destroy_context(&encoder_ctx);
        heap_caps_free(encoder_output);
        heap_caps_free(features);
        return ret;
    }
    
    // Clean up memory
    moonshine_destroy_context(&decoder_ctx);
    moonshine_destroy_context(&encoder_ctx);
    heap_caps_free(encoder_output);
    heap_caps_free(features);
    
    size_t final_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    double rtf = (encoder_time_ms + decoder_time_ms) / (duration_s * 1000.0);
    
    if (csv_file) {
        csv_write_field(csv_file, CSV_SCHEMA_VERSION);
        fputc(',', csv_file);
        csv_write_field(csv_file, model_name);
        fputc(',', csv_file);
        csv_write_field(csv_file, quantization);
        fputc(',', csv_file);
        csv_write_field(csv_file, wav_path);
        fprintf(csv_file, ",%.3f,%zu,%.2f,%.3f,%.3f,%.3f,%.6f,%lu,",
                model_load_time_ms, psram_free_after_model_load,
                duration_s, load_time_ms,
                encoder_time_ms, decoder_time_ms,
                rtf, num_tokens);
        csv_write_field(csv_file, decoded_text);
        fprintf(csv_file, ",%zu\n", final_psram);
        fflush(csv_file);
    }
    
    ESP_LOGI(TAG, "Benchmarked %s on %s: Enc=%.3fms Dec=%.3fms RTF=%.4f",
             model_name, wav_path, encoder_time_ms, decoder_time_ms, rtf);
    return ret;
}

esp_err_t benchmark_run_suite(
    const char* model_dir,
    const char* audio_dir,
    const char* output_csv,
    const char* model_prefix,
    const char* audio_prefix)
{
    if (!model_dir || !audio_dir || !output_csv) {
        return ESP_ERR_INVALID_ARG;
    }
    if (g_models_loaded || g_listen_active.load()) {
        ESP_LOGE(TAG, "Unload models and stop listening before running the benchmark");
        return ESP_ERR_INVALID_STATE;
    }

    ESP_LOGI(TAG, "Starting Benchmark Suite: Models=%s, Audio=%s, Model prefix=%s, Audio prefix=%s",
             model_dir, audio_dir,
             model_prefix ? model_prefix : "(all)",
             audio_prefix ? audio_prefix : "(all)");
    size_t successful_runs = 0;
    
    FILE* csv = fopen(output_csv, "a");
    if (!csv) {
        ESP_LOGE(TAG, "Failed to open CSV output: %s", output_csv);
        return ESP_FAIL;
    }
    
    // Keep an existing CSV's schema consistent rather than appending incompatible rows.
    struct stat st;
    if (stat(output_csv, &st) == 0 && st.st_size > 0) {
        FILE* existing = fopen(output_csv, "r");
        char header[sizeof(CSV_HEADER)] = {0};
        if (!existing || !fgets(header, sizeof(header), existing)) {
            if (existing) fclose(existing);
            ESP_LOGE(TAG, "Failed to read existing CSV header: %s", output_csv);
            fclose(csv);
            return ESP_FAIL;
        }
        fclose(existing);
        header[strcspn(header, "\r\n")] = '\0';
        if (strcmp(header, CSV_HEADER) != 0) {
            ESP_LOGE(TAG, "CSV schema mismatch; use a new output file: %s", output_csv);
            fclose(csv);
            return ESP_ERR_INVALID_STATE;
        }
    } else {
        fprintf(csv, "%s\n", CSV_HEADER);
        fflush(csv);
    }
    
    esp_err_t manager_error = model_manager_init(model_dir);
    if (manager_error != ESP_OK) {
        fclose(csv);
        return manager_error;
    }
    const int model_count = model_manager_scan();
    if (model_count < 0) {
        fclose(csv);
        return (esp_err_t)-model_count;
    }

    size_t selected_models = 0;
    for (int model_index = 0; model_index < model_count; ++model_index) {
        const model_entry_t* registered = model_manager_get_by_index((size_t)model_index);
        if (!registered || registered->type != MODEL_TYPE_ASR_SEQ2SEQ ||
            (model_prefix &&
             strncmp(registered->name, model_prefix, strlen(model_prefix)) != 0)) {
            continue;
        }
        ++selected_models;
        model_entry_t model = *registered;
        const char* quantization = model.quant == MODEL_QUANT_INT8
            ? "INT8" : model.quant == MODEL_QUANT_INT8_PER_CH
                ? "INT8_PER_CH" : "OTHER";
        const int64_t model_load_start = esp_timer_get_time();
        const esp_err_t load_error = moonshine_load_espm(
            model.file_path, &g_encoder_model, &g_decoder_model, true);
        const double model_load_time_ms =
            (double)(esp_timer_get_time() - model_load_start) / 1000.0;
        if (load_error != ESP_OK) {
            ESP_LOGE(TAG, "Skipping %s: ESPM load failed: %s",
                     model.name, esp_err_to_name(load_error));
            unload_moonshine_models();
            continue;
        }

        const size_t psram_free_after_model_load =
            heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
        ESP_LOGI(TAG, "BENCH_MODEL_LOAD_RESULT model=%s quant=%s model_ms=%.3f "
                 "psram_free_after_model_bytes=%zu",
                 model.name, quantization, model_load_time_ms,
                 psram_free_after_model_load);

        DIR* d_audio = opendir(audio_dir);
        if (!d_audio) {
            ESP_LOGE(TAG, "Failed to open audio dir: %s", audio_dir);
            unload_moonshine_models();
            continue;
        }
        struct dirent* ent_audio;
        while ((ent_audio = readdir(d_audio)) != NULL) {
            const char* extension = strrchr(ent_audio->d_name, '.');
            if (!extension || strcmp(extension, ".wav") != 0 ||
                (audio_prefix &&
                 strncmp(ent_audio->d_name, audio_prefix, strlen(audio_prefix)) != 0)) {
                continue;
            }
            char wav_path[1024];
            const int path_length = snprintf(
                wav_path, sizeof(wav_path), "%s/%s", audio_dir, ent_audio->d_name);
            if (path_length <= 0 || (size_t)path_length >= sizeof(wav_path)) {
                ESP_LOGE(TAG, "WAV path is too long: %s", ent_audio->d_name);
                continue;
            }

            const esp_err_t bench_error = benchmark_audio_file(
                wav_path, model.name, quantization, model_load_time_ms,
                psram_free_after_model_load, csv);
            if (bench_error == ESP_OK) {
                ++successful_runs;
            } else if (bench_error == ESP_ERR_NO_MEM) {
                ESP_LOGE(TAG, "Skipping %s on %s: out of memory",
                         model.name, ent_audio->d_name);
                fprintf(csv, "%s,", CSV_SCHEMA_VERSION);
                csv_write_field(csv, model.name);
                fputc(',', csv);
                csv_write_field(csv, quantization);
                fputc(',', csv);
                csv_write_field(csv, ent_audio->d_name);
                fprintf(csv, ",%.3f,%zu,0,0,0,0,0,0,\"OOM_DURING_INFERENCE\",%zu\n",
                        model_load_time_ms, psram_free_after_model_load,
                        heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
                fflush(csv);
            } else {
                ESP_LOGE(TAG, "Skipping %s on %s: %s",
                         model.name, ent_audio->d_name, esp_err_to_name(bench_error));
            }
        }
        closedir(d_audio);
        unload_moonshine_models();
        ESP_LOGI(TAG, "Unloaded ESPM model %s", model.name);
    }

    fclose(csv);
    
    if (selected_models == 0 || successful_runs == 0) {
        ESP_LOGE(TAG, "No benchmark samples completed successfully");
        return ESP_ERR_NOT_FOUND;
    }
    ESP_LOGI(TAG, "Benchmark Suite Completed!");
    return ESP_OK;
}
