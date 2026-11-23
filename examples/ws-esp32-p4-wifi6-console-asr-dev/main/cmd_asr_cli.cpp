#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <atomic>
#include "esp_log.h"
#include "esp_console.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "argtable3/argtable3.h"
#include "cmd_asr_cli.h"
#include "audio_manager.h"
#include "audio_wav.h"
#include "benchmark_suite.h"
#include "moonshine_audio_utils.h"
#include "wifi_transfer.h"
#include "model_manager.h"
#include "app_config.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "sdkconfig.h"

extern esp_err_t test_with_wav_file(const char* wav_path, const char* encoder_dump_path);
extern esp_err_t load_moonshine_models(const char* model_path);
extern void unload_moonshine_models(void);
extern "C" esp_err_t console_cli_list_directory(const char* path);
extern bool g_models_loaded;
extern std::atomic<bool> g_listen_active;
extern std::atomic<bool> g_live_inference_active;

static const char* TAG = "ASR_CLI";

static int cmd_ping(int argc, char **argv)
{
    ESP_LOGI(TAG, "PONG");
    return 0;
}

static int cmd_mem(int argc, char **argv)
{
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t free_heap = esp_get_free_heap_size();
    ESP_LOGI(TAG, "{\"memory\": {\"psram_free\": %zu, \"heap_free\": %zu}}", free_psram, free_heap);
    return 0;
}

static struct {
    struct arg_str *path;
    struct arg_end *end;
} ls_args;

static int cmd_ls(int argc, char **argv)
{
    int nerrors = arg_parse(argc, argv, (void **) &ls_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, ls_args.end, argv[0]);
        return 1;
    }
    
    const char* path = "/sd";
    if (ls_args.path->count > 0) {
        path = ls_args.path->sval[0];
    }
    const esp_err_t error = console_cli_list_directory(path);
    if (error != ESP_OK) {
        ESP_LOGE(TAG, "Could not list directory %s: %s",
                 path, esp_err_to_name(error));
        return 1;
    }
    return 0;
}

static int cmd_models(int argc, char **argv)
{
    const esp_err_t init_error = model_manager_init("/sd/models");
    if (init_error != ESP_OK) {
        ESP_LOGE(TAG, "Model manager initialization failed: %s",
                 esp_err_to_name(init_error));
        return 1;
    }
    const int count = model_manager_scan();
    if (count < 0) {
        ESP_LOGE(TAG, "Model scan failed: %s", esp_err_to_name((esp_err_t)-count));
        return 1;
    }
    model_manager_print_list();
    return 0;
}

static struct {
    struct arg_str *file;
    struct arg_end *end;
} infer_args;


static struct {
    struct arg_str *file;
    struct arg_str *output;
    struct arg_end *end;
} dump_args;

static int cmd_dump_golden(int argc, char **argv)
{
    int nerrors = arg_parse(argc, argv, (void **) &dump_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, dump_args.end, argv[0]);
        return 1;
    }
    
    if (!g_models_loaded) {
        ESP_LOGE(TAG, "Models not loaded. Send LOAD command first.");
        return 1;
    }
    if (g_listen_active.load() || g_live_inference_active.load()) {
        ESP_LOGE(TAG, "Stop live microphone recognition before running WAV inference");
        return 1;
    }
    
    ESP_LOGI(TAG, "Dumping encoder tensor from: %s", dump_args.file->sval[0]);
    esp_err_t ret = test_with_wav_file(dump_args.file->sval[0], dump_args.output->sval[0]);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "DUMP_GOLDEN_ERROR (%d)", ret);
        return 1;
    }
    ESP_LOGI(TAG, "DUMP_GOLDEN_DONE");
    return 0;
}
static int cmd_infer(int argc, char **argv)
{
    int nerrors = arg_parse(argc, argv, (void **) &infer_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, infer_args.end, argv[0]);
        return 1;
    }
    
    if (!g_models_loaded) {
        ESP_LOGE(TAG, "Models not loaded. Send LOAD command first.");
        ESP_LOGI(TAG, "INFER_ERROR");
        return 1;
    }
    if (g_listen_active.load() || g_live_inference_active.load()) {
        ESP_LOGE(TAG, "Stop live microphone recognition before running WAV inference");
        return 1;
    }
    
    ESP_LOGI(TAG, "Executing INFER on: %s", infer_args.file->sval[0]);
    esp_err_t ret = test_with_wav_file(infer_args.file->sval[0], NULL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "INFER_ERROR (%d)", ret);
        return 1;
    }
    ESP_LOGI(TAG, "INFER_DONE");
    return 0;
}

static struct {
    struct arg_str *model;
    struct arg_end *end;
} load_args;

static int cmd_load(int argc, char **argv)
{
    int nerrors = arg_parse(argc, argv, (void **) &load_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, load_args.end, argv[0]);
        return 1;
    }
    
    if (g_listen_active.load() || g_live_inference_active.load()) {
        ESP_LOGE(TAG, "Stop microphone recognition and wait for inference before loading another model");
        return 1;
    }
    if (g_models_loaded) {
        unload_moonshine_models();
        g_models_loaded = false;
    }

    const esp_err_t manager_init_error = model_manager_init("/sd/models");
    if (manager_init_error != ESP_OK) {
        ESP_LOGE(TAG, "Model manager initialization failed: %s",
                 esp_err_to_name(manager_init_error));
        return 1;
    }
    const int model_count = model_manager_scan();
    if (model_count < 0) {
        ESP_LOGE(TAG, "Model scan failed: %s",
                 esp_err_to_name((esp_err_t)-model_count));
        return 1;
    }
    const model_entry_t* model = model_manager_find_by_path(load_args.model->sval[0]);
    if (!model || model->type != MODEL_TYPE_ASR_SEQ2SEQ) {
        ESP_LOGE(TAG, "No ASR ESPM model registered at %s",
                 load_args.model->sval[0]);
        return 1;
    }
    
    ESP_LOGI(TAG, "Executing LOAD");
    size_t psram_free_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    int64_t load_start_us = esp_timer_get_time();
    esp_err_t ret = load_moonshine_models(model->file_path);
    double load_time_ms = (double)(esp_timer_get_time() - load_start_us) / 1000.0;
    size_t psram_free_after = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG,
             "MODEL_LOAD_RESULT status=%s elapsed_ms=%.3f psram_free_before_bytes=%zu psram_free_after_bytes=%zu",
             esp_err_to_name(ret), load_time_ms, psram_free_before, psram_free_after);
    if (ret == ESP_OK) {
        g_models_loaded = true;
        ESP_LOGI(TAG, "LOAD_DONE");
        return 0;
    } else {
        ESP_LOGE(TAG, "Failed to load models");
        ESP_LOGI(TAG, "LOAD_ERROR");
        return 1;
    }
}

static int cmd_unload(int argc, char **argv)
{
    if (g_listen_active.load() || g_live_inference_active.load()) {
        ESP_LOGE(TAG, "Stop microphone recognition and wait for inference before unloading models");
        return 1;
    }
    if (g_models_loaded) {
        unload_moonshine_models();
        g_models_loaded = false;
    }
    ESP_LOGI(TAG, "UNLOAD_DONE");
    return 0;
}


static int cmd_speaker_vol(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: speaker_vol <0-100>\n");
        return 1;
    }
    char *end = NULL;
    const long vol = strtol(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || vol < 0 || vol > 100) {
        printf("[ERROR] Volume must be 0-100\n");
        return 1;
    }
    const esp_err_t ret = audio_manager_set_speaker_vol((uint8_t)vol);
    if (ret == ESP_OK) {
        printf("[OK] Speaker volume set to %ld%% (saved across restarts)\n", vol);
    } else {
        ESP_LOGE(TAG, "Could not apply or save speaker volume: %s", esp_err_to_name(ret));
        return 1;
    }
    return 0;
}

static struct {
    struct arg_str *file;
    struct arg_end *end;
} play_wav_args;

static int cmd_play_wav(int argc, char **argv)
{
    const int nerrors = arg_parse(argc, argv, (void **)&play_wav_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, play_wav_args.end, argv[0]);
        return 1;
    }
    if (g_listen_active.load() || g_live_inference_active.load() ||
        audio_manager_is_capture_running()) {
        ESP_LOGE(TAG, "Stop microphone capture before WAV playback");
        return 1;
    }

    const char *path = play_wav_args.file->sval[0];
    audio_wav_handle_t wav = NULL;
    esp_err_t result = audio_wav_open(path, &wav);
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Could not open WAV file %s: %s", path, esp_err_to_name(result));
        return 1;
    }

    audio_wav_info_t info = {};
    result = audio_wav_get_info(wav, &info);
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Could not read WAV metadata: %s", esp_err_to_name(result));
        audio_wav_close(wav);
        return 1;
    }
    if (info.sample_rate != ASR_SAMPLE_RATE || info.num_channels != 1 ||
        info.bits_per_sample != 16 || info.data_size == 0 ||
        (info.data_size % sizeof(int16_t)) != 0) {
        ESP_LOGE(TAG,
                 "Playback requires non-empty %u Hz mono PCM16 WAV; got %u Hz, %u channel(s), %u-bit",
                 ASR_SAMPLE_RATE, (unsigned)info.sample_rate,
                 (unsigned)info.num_channels, (unsigned)info.bits_per_sample);
        audio_wav_close(wav);
        return 1;
    }

    result = audio_manager_start_playback();
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Could not start speaker playback: %s", esp_err_to_name(result));
        audio_wav_close(wav);
        return 1;
    }

    printf("Playing %s (%u Hz, mono PCM16, %.2f s)\n",
           path, (unsigned)info.sample_rate, (double)info.duration_ms / 1000.0);
    int16_t samples[1024];
    size_t remaining = info.data_size;
    while (remaining > 0) {
        const size_t requested = remaining < sizeof(samples) ? remaining : sizeof(samples);
        size_t bytes_read = 0;
        result = audio_wav_read(wav, samples, requested, &bytes_read);
        if (result != ESP_OK || bytes_read != requested) {
            if (result == ESP_OK) {
                result = ESP_ERR_INVALID_RESPONSE;
            }
            ESP_LOGE(TAG, "Failed reading WAV audio data: %s", esp_err_to_name(result));
            break;
        }
        const size_t sample_count = bytes_read / sizeof(int16_t);
        const int written = audio_manager_write(samples, sample_count, 1000);
        if (written != (int)sample_count) {
            ESP_LOGE(TAG, "Speaker audio write failed");
            result = ESP_FAIL;
            break;
        }
        remaining -= bytes_read;
    }

    const esp_err_t stop_error = audio_manager_stop_playback();
    if (result == ESP_OK && stop_error != ESP_OK) {
        result = stop_error;
    }
    const esp_err_t close_error = audio_wav_close(wav);
    if (result == ESP_OK && close_error != ESP_OK) {
        result = close_error;
    }
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "WAV playback failed: %s", esp_err_to_name(result));
        return 1;
    }
    printf("Playback finished.\n");
    return 0;
}

static struct {
    struct arg_str *file;
    struct arg_int *duration;
    struct arg_end *end;
} record_mic_args;

static esp_err_t write_pcm16_mono_wav_header(FILE *file, uint32_t data_size)
{
    return audio_wav_write_pcm16_header(file, ASR_SAMPLE_RATE, 1, data_size);
}

static int cmd_record_mic(int argc, char **argv)
{
    const int nerrors = arg_parse(argc, argv, (void **)&record_mic_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, record_mic_args.end, argv[0]);
        return 1;
    }
    if (g_listen_active.load() || g_live_inference_active.load() ||
        audio_manager_is_capture_running()) {
        ESP_LOGE(TAG, "Stop microphone capture before recording a WAV file");
        return 1;
    }
    if (!audio_manager_is_initialized()) {
        ESP_LOGE(TAG, "Audio manager is not initialized");
        return 1;
    }

    const char *path = record_mic_args.file->sval[0];
    const int duration_seconds = record_mic_args.duration->ival[0];
    if (duration_seconds < 1 || duration_seconds > 60) {
        ESP_LOGE(TAG, "Recording duration must be between 1 and 60 seconds");
        return 1;
    }
    if (strlen(path) >= MAX_FILE_PATH) {
        ESP_LOGE(TAG, "WAV output path is too long");
        return 1;
    }

    FILE *file = fopen(path, "wb");
    if (!file) {
        ESP_LOGE(TAG, "Could not create %s", path);
        return 1;
    }
    if (write_pcm16_mono_wav_header(file, 0) != ESP_OK) {
        ESP_LOGE(TAG, "Could not write WAV header to %s", path);
        fclose(file);
        return 1;
    }

    esp_err_t result = audio_manager_start_capture();
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Could not start microphone capture: %s", esp_err_to_name(result));
        fclose(file);
        return 1;
    }

    const uint32_t target_samples = (uint32_t)duration_seconds * ASR_SAMPLE_RATE;
    uint32_t samples_recorded = 0;
    const int64_t deadline_us = esp_timer_get_time() +
        ((int64_t)duration_seconds + 5) * 1000000;
    int16_t samples[1024];
    printf("Recording %d second(s) from %s to %s\n",
           duration_seconds, audio_manager_get_mic_name(), path);

    while (samples_recorded < target_samples) {
        const uint32_t remaining = target_samples - samples_recorded;
        const size_t requested = remaining < (uint32_t)(sizeof(samples) / sizeof(samples[0]))
            ? remaining : sizeof(samples) / sizeof(samples[0]);
        const int count = audio_manager_read(samples, requested, 1000);
        if (count < 0) {
            ESP_LOGE(TAG, "Microphone read failed");
            result = ESP_FAIL;
            break;
        }
        if (count == 0) {
            if (esp_timer_get_time() >= deadline_us) {
                ESP_LOGE(TAG, "Timed out waiting for microphone audio");
                result = ESP_ERR_TIMEOUT;
                break;
            }
            continue;
        }
        const size_t written = fwrite(samples, sizeof(samples[0]), (size_t)count, file);
        samples_recorded += (uint32_t)written;
        if (written != (size_t)count) {
            ESP_LOGE(TAG, "Failed writing audio samples to %s", path);
            result = ESP_FAIL;
            break;
        }
    }

    const esp_err_t stop_error = audio_manager_stop_capture();
    if (result == ESP_OK && stop_error != ESP_OK) {
        ESP_LOGE(TAG, "Could not stop microphone capture: %s", esp_err_to_name(stop_error));
        result = stop_error;
    }
    const uint32_t data_size = samples_recorded * sizeof(int16_t);
    const esp_err_t header_error = write_pcm16_mono_wav_header(file, data_size);
    if (header_error != ESP_OK) {
        ESP_LOGE(TAG, "Could not finalize WAV header for %s", path);
        if (result == ESP_OK) {
            result = header_error;
        }
    }
    if (fclose(file) != 0) {
        ESP_LOGE(TAG, "Could not close WAV file %s", path);
        if (result == ESP_OK) {
            result = ESP_FAIL;
        }
    }

    if (result != ESP_OK) {
        ESP_LOGE(TAG, "WAV recording failed after %u samples", (unsigned)samples_recorded);
        return 1;
    }
    printf("Saved %s: %u samples, %.2f seconds\n",
           path, (unsigned)samples_recorded,
           (double)samples_recorded / ASR_SAMPLE_RATE);
    return 0;
}

static struct {
    struct arg_str *action;
    struct arg_end *end;
} listen_args;

static int cmd_listen(int argc, char **argv)
{
    bool should_listen;
    if (argc == 1) {
        should_listen = !g_listen_active.load();
    } else {
        const int nerrors = arg_parse(argc, argv, (void **) &listen_args);
        if (nerrors != 0 || listen_args.action->count == 0) {
            printf("Usage: listen [start|stop]\n");
            return 1;
        }
        if (strcmp(listen_args.action->sval[0], "start") == 0) {
            should_listen = true;
        } else if (strcmp(listen_args.action->sval[0], "stop") == 0) {
            should_listen = false;
        } else {
            printf("Usage: listen [start|stop]\n");
            return 1;
        }
    }

    if (should_listen == g_listen_active.load()) {
        printf("[INFO] Microphone listening is already %s.\n",
               should_listen ? "active" : "stopped");
        return 0;
    }

    if (should_listen) {
        if (!g_models_loaded) {
            printf("[ERROR] Load an ESPM model before starting microphone recognition.\n");
            return 1;
        }
        if (g_live_inference_active.load()) {
            printf("[ERROR] Wait for the current inference to finish before listening.\n");
            return 1;
        }
        if (g_listen_active.load() || g_live_inference_active.load()) {
            ESP_LOGE(TAG, "Stop live microphone recognition before running WAV inference");
            return 1;
        }
        const esp_err_t ret = audio_manager_start_capture();
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Could not start microphone capture: %s", esp_err_to_name(ret));
            return 1;
        }
        g_listen_active.store(true);
        printf("[INFO] Microphone listening STARTED (%s).\n", audio_manager_get_mic_name());
    } else {
        g_listen_active.store(false);
        const esp_err_t ret = audio_manager_stop_capture();
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Could not stop microphone capture: %s", esp_err_to_name(ret));
            return 1;
        }
        printf("[INFO] Microphone listening STOPPED.\n");
    }
    return 0;
}

static int cmd_mic(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "status") == 0) {
        if (!audio_manager_is_initialized()) {
            printf("[ERROR] Audio manager is not initialized.\n");
            return 1;
        }
        float gain_db = 0.0f;
        const esp_err_t ret = audio_manager_get_mic_gain(&gain_db);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Could not read microphone gain: %s", esp_err_to_name(ret));
            return 1;
        }
        printf("Input: %s\nCapture: %s\nSample format: 16000 Hz, mono, signed PCM16\n"
               "Onboard ES8311 PGA gain: %.0f dB\n"
               "External I2S input: BCLK=%d WS=%d DIN=%d, %s slot, 32-bit slots / 24-bit samples\n",
               audio_manager_get_mic_name(),
               audio_manager_is_capture_running() ? "running" : "stopped",
               gain_db,
               CONFIG_ASR_INMP441_BCLK_GPIO,
               CONFIG_ASR_INMP441_WS_GPIO,
               CONFIG_ASR_INMP441_DIN_GPIO,
#if CONFIG_ASR_INMP441_SLOT_RIGHT
               "right");
#else
               "left");
#endif
        return 0;
    }

    if (argc == 3 && strcmp(argv[1], "select") == 0) {
        mic_type_t source;
        if (strcmp(argv[2], "onboard") == 0) {
            source = MIC_I2S0_ES8311;
        } else if (strcmp(argv[2], "i2s") == 0) {
            source = MIC_I2S1_INMP441;
        } else {
            printf("Usage: mic select <onboard|i2s>\n");
            return 1;
        }
        const esp_err_t ret = audio_manager_set_mic(source);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Could not select microphone: %s", esp_err_to_name(ret));
            return 1;
        }
        printf("[INFO] Selected %s (saved across restarts).\n", audio_manager_get_mic_name());
        return 0;
    }

    if (argc == 3 && strcmp(argv[1], "gain") == 0) {
        char *end = NULL;
        const float gain_db = strtof(argv[2], &end);
        if (end == argv[2] || *end != '\0') {
            printf("Gain must be one of 0, 6, 12, 18, 24, 30, 36, or 42 dB.\n");
            return 1;
        }
        const esp_err_t ret = audio_manager_set_mic_gain(gain_db);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Could not set microphone gain: %s", esp_err_to_name(ret));
            return 1;
        }
        printf("[INFO] Onboard microphone gain set to %.0f dB (saved across restarts).\n", gain_db);
        return 0;
    }

    if (argc == 2 && strcmp(argv[1], "level") == 0) {
        if (g_listen_active.load() || g_live_inference_active.load()) {
            printf("[ERROR] Stop listen before running the one-second level check.\n");
            return 1;
        }
        float rms = 0.0f;
        float peak = 0.0f;
        int clipping_count = 0;
        const esp_err_t ret = audio_manager_get_levels(&rms, &peak, &clipping_count);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Microphone level measurement failed: %s", esp_err_to_name(ret));
            return 1;
        }
        const float rms_dbfs = rms > 0.0f ? 20.0f * log10f(rms / 32768.0f) : -INFINITY;
        const float peak_dbfs = peak > 0.0f ? 20.0f * log10f(peak / 32768.0f) : -INFINITY;
        printf("1 s level: RMS=%.1f dBFS, peak=%.1f dBFS, clipped_samples=%d\n",
               rms_dbfs, peak_dbfs, clipping_count);
        return 0;
    }

    printf("Usage: mic status | mic select <onboard|i2s> | mic gain <0|6|12|18|24|30|36|42> | mic level\n");
    return 1;
}

static struct {
    struct arg_str *model_dir;
    struct arg_str *audio_dir;
    struct arg_str *csv_file;
    struct arg_str *model_prefix;
    struct arg_str *audio_prefix;
    struct arg_end *end;
} benchmark_args;

typedef struct {
    const char* model_dir;
    const char* audio_dir;
    const char* csv_file;
    const char* model_prefix;
    const char* audio_prefix;
    SemaphoreHandle_t completed;
    esp_err_t result;
} benchmark_job_t;

static void benchmark_task(void* arg)
{
    benchmark_job_t* job = (benchmark_job_t*)arg;
    job->result = benchmark_run_suite(
        job->model_dir, job->audio_dir, job->csv_file,
        job->model_prefix, job->audio_prefix);
    xSemaphoreGive(job->completed);
    vTaskDelete(NULL);
}

static int cmd_benchmark(int argc, char **argv)
{
    int nerrors = arg_parse(argc, argv, (void **) &benchmark_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, benchmark_args.end, argv[0]);
        return 1;
    }
    
    printf("Running benchmark suite...\n");
    printf("Models: %s\n", benchmark_args.model_dir->sval[0]);
    printf("Audio: %s\n", benchmark_args.audio_dir->sval[0]);
    printf("CSV: %s\n", benchmark_args.csv_file->sval[0]);
    if (benchmark_args.model_prefix->count > 0) {
        printf("Model prefix: %s\n", benchmark_args.model_prefix->sval[0]);
    }
    if (benchmark_args.audio_prefix->count > 0) {
        printf("Audio prefix: %s\n", benchmark_args.audio_prefix->sval[0]);
    }

    benchmark_job_t job = {
        .model_dir = benchmark_args.model_dir->sval[0],
        .audio_dir = benchmark_args.audio_dir->sval[0],
        .csv_file = benchmark_args.csv_file->sval[0],
        .model_prefix = benchmark_args.model_prefix->count > 0
            ? benchmark_args.model_prefix->sval[0] : NULL,
        .audio_prefix = benchmark_args.audio_prefix->count > 0
            ? benchmark_args.audio_prefix->sval[0] : NULL,
        .completed = xSemaphoreCreateBinary(),
        .result = ESP_FAIL,
    };
    if (!job.completed) {
        ESP_LOGE(TAG, "Failed to create benchmark completion semaphore");
        return 1;
    }

    const BaseType_t task_result = xTaskCreate(
        benchmark_task, "asr_benchmark", 32 * 1024, &job, tskIDLE_PRIORITY + 1, NULL);
    if (task_result != pdPASS) {
        vSemaphoreDelete(job.completed);
        ESP_LOGE(TAG, "Failed to create benchmark task");
        return 1;
    }

    if (xSemaphoreTake(job.completed, portMAX_DELAY) != pdTRUE) {
        vSemaphoreDelete(job.completed);
        ESP_LOGE(TAG, "Benchmark task completion wait failed");
        return 1;
    }
    vSemaphoreDelete(job.completed);
    if (job.result != ESP_OK) {
        ESP_LOGE(TAG, "BENCHMARK_ERROR (%s)", esp_err_to_name(job.result));
        return 1;
    }
    return 0;
}

void cmd_asr_cli_register(void)
{
    const esp_console_cmd_t cmd_ping_reg = {
        .command = "ping",
        .help = "Check if system is responsive",
        .hint = NULL,
        .func = &cmd_ping,
        .argtable = NULL,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_ping_reg));
    
    const esp_console_cmd_t cmd_mem_reg = {
        .command = "mem",
        .help = "Print memory statistics",
        .hint = NULL,
        .func = &cmd_mem,
        .argtable = NULL,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_mem_reg));
    
    ls_args.path = arg_str0(NULL, NULL, "<path>", "Directory path to list");
    ls_args.end = arg_end(1);
    const esp_console_cmd_t cmd_ls_reg = {
        .command = "ls",
        .help = "List files in an SD-card directory (default: /sd)",
        .hint = "<path>",
        .func = &cmd_ls,
        .argtable = &ls_args,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_ls_reg));

    const esp_console_cmd_t cmd_models_reg = {
        .command = "models",
        .help = "Scan and list valid ESPM models",
        .hint = NULL,
        .func = &cmd_models,
        .argtable = NULL,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_models_reg));
    
    infer_args.file = arg_str1(NULL, NULL, "<filename.wav>", "WAV file to run ASR on");
    infer_args.end = arg_end(1);
    const esp_console_cmd_t cmd_infer_reg = {
        .command = "infer",
        .help = "Run ASR inference on a WAV file",
        .hint = NULL,
        .func = &cmd_infer,
        .argtable = &infer_args,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_infer_reg));
    
    load_args.model = arg_str1(NULL, NULL, "<model.espm>", "Native ESPM model path");
    load_args.end = arg_end(1);
    const esp_console_cmd_t cmd_load_reg = {
        .command = "load",
        .help = "Load a native Moonshine ESPM model",
        .hint = NULL,
        .func = &cmd_load,
        .argtable = &load_args,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_load_reg));
    
    const esp_console_cmd_t cmd_unload_reg = {
        .command = "unload",
        .help = "Unload Moonshine ASR models to free memory",
        .hint = NULL,
        .func = &cmd_unload,
        .argtable = NULL,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_unload_reg));
    const esp_console_cmd_t cmd_sv_reg = {
        .command = "speaker_vol",
        .help = "Set speaker volume (0-100)",
        .hint = "<0-100>",
        .func = &cmd_speaker_vol,
        .argtable = NULL,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_sv_reg));
    dump_args.file = arg_str1(NULL, NULL, "<filename.wav>", "16-bit PCM WAV file");
    dump_args.output = arg_str1(NULL, NULL, "<encoder_out.bin>", "Output path for raw float32 encoder states");
    dump_args.end = arg_end(2);
    const esp_console_cmd_t cmd_dump = {
        .command = "dump_golden",
        .help = "Write raw float32 encoder states for tensor comparison",
        .hint = NULL,
        .func = &cmd_dump_golden,
        .argtable = &dump_args,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_dump));

    play_wav_args.file = arg_str1(NULL, NULL, "<file.wav>", "16 kHz mono PCM16 WAV file");
    play_wav_args.end = arg_end(1);
    const esp_console_cmd_t cmd_play_wav_reg = {
        .command = "play_wav",
        .help = "Play a WAV file through the speaker",
        .hint = NULL,
        .func = &cmd_play_wav,
        .argtable = &play_wav_args,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_play_wav_reg));

    record_mic_args.file = arg_str1(NULL, NULL, "<output.wav>", "WAV output path");
    record_mic_args.duration = arg_int1(NULL, NULL, "<seconds>", "Recording duration (1-60)");
    record_mic_args.end = arg_end(2);
    const esp_console_cmd_t cmd_record_mic_reg = {
        .command = "record_mic",
        .help = "Record the selected microphone to a PCM16 WAV file",
        .hint = NULL,
        .func = &cmd_record_mic,
        .argtable = &record_mic_args,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_record_mic_reg));

    listen_args.action = arg_str0(NULL, NULL, "<start|stop>", "Start or stop live microphone ASR");
    listen_args.end = arg_end(1);
    const esp_console_cmd_t cmd_listen_reg = {
        .command = "listen",
        .help = "Toggle real-time microphone ASR",
        .hint = NULL,
        .func = &cmd_listen,
        .argtable = &listen_args,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_listen_reg));

    const esp_console_cmd_t cmd_mic_reg = {
        .command = "mic",
        .help = "Show, select, or diagnose the microphone input",
        .hint = "status|select|gain|level",
        .func = &cmd_mic,
        .argtable = NULL,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_mic_reg));

    benchmark_args.model_dir = arg_str1(NULL, NULL, "<models_dir>", "Directory containing models");
    benchmark_args.audio_dir = arg_str1(NULL, NULL, "<audio_dir>", "Directory containing audio files");
    benchmark_args.csv_file = arg_str1(NULL, NULL, "<csv_file>", "CSV output file path");
    benchmark_args.model_prefix = arg_str0("m", "model-prefix", "<prefix>", "Only benchmark model pairs with this filename prefix");
    benchmark_args.audio_prefix = arg_str0("a", "audio-prefix", "<prefix>", "Only benchmark WAV files with this filename prefix");
    benchmark_args.end = arg_end(3);
    const esp_console_cmd_t cmd_benchmark_reg = {
        .command = "benchmark",
        .help = "Run the benchmark suite; optional prefixes isolate selected models and audio files",
        .hint = NULL,
        .func = &cmd_benchmark,
        .argtable = &benchmark_args,
        .func_w_context = NULL,
        .context = NULL
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&cmd_benchmark_reg));
    wifi_transfer_register_commands();
}

