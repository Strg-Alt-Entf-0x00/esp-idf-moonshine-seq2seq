#include "audio_manager.h"
#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "esp_err.h"
#include "driver/i2c_master.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"
#include "es8311_codec.h"
#include "nvs.h"
#include "board_audio_config.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "freertos/ringbuf.h"
#include <atomic>
#include <math.h>

#define ASR_CHUNK_BYTES_100MS (BOARD_AUDIO_SAMPLE_RATE / 10 * sizeof(int16_t))

#define READ_CHUNK_FRAMES 400
#define READ_TIMEOUT_MS   50

static const char* TAG = "AUDIO_MANAGER";

static std::atomic<bool> g_initialized{false};
static std::atomic<bool> g_capture_running{false};
static std::atomic<bool> g_playback_running{false};
static std::atomic<uint32_t> g_capture_epoch{0};
static SemaphoreHandle_t g_state_mutex = NULL;
static float g_current_mic_gain = (float)BOARD_AUDIO_DEFAULT_MIC_GAIN_DB;
static uint8_t g_current_speaker_vol = BOARD_AUDIO_DEFAULT_SPEAKER_VOLUME;
static std::atomic<esp_err_t> g_tx_error{ESP_OK};
static const audio_codec_if_t *g_codec_if = NULL;
#if CONFIG_ASR_MIC_DEFAULT_INMP441
static std::atomic<mic_type_t> g_active_mic{MIC_I2S1_INMP441};
#else
static std::atomic<mic_type_t> g_active_mic{MIC_I2S0_ES8311};
#endif

static const char *AUDIO_SETTINGS_NAMESPACE = "audio_cfg";
static const char *AUDIO_SETTING_SPEAKER_VOLUME = "speaker_vol";
static const char *AUDIO_SETTING_MIC_GAIN = "mic_gain";
static const char *AUDIO_SETTING_MIC_INPUT = "mic_input";

static bool is_valid_mic_gain(uint8_t gain_db)
{
    return gain_db <= 42 && gain_db % 6 == 0;
}

static void load_audio_settings(void)
{
    nvs_handle_t handle;
    esp_err_t ret = nvs_open(AUDIO_SETTINGS_NAMESPACE, NVS_READONLY, &handle);
    if (ret == ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGI(TAG, "No saved audio settings; using build defaults");
        return;
    }
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Could not open saved audio settings; using build defaults: %s",
                 esp_err_to_name(ret));
        return;
    }

    uint8_t value = 0;
    ret = nvs_get_u8(handle, AUDIO_SETTING_SPEAKER_VOLUME, &value);
    if (ret == ESP_OK) {
        if (value <= 100) {
            g_current_speaker_vol = value;
        } else {
            ESP_LOGW(TAG, "Ignoring invalid saved speaker volume %u", value);
        }
    } else if (ret != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGW(TAG, "Could not read saved speaker volume: %s", esp_err_to_name(ret));
    }

    ret = nvs_get_u8(handle, AUDIO_SETTING_MIC_GAIN, &value);
    if (ret == ESP_OK) {
        if (is_valid_mic_gain(value)) {
            g_current_mic_gain = (float)value;
        } else {
            ESP_LOGW(TAG, "Ignoring invalid saved microphone gain %u dB", value);
        }
    } else if (ret != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGW(TAG, "Could not read saved microphone gain: %s", esp_err_to_name(ret));
    }

    ret = nvs_get_u8(handle, AUDIO_SETTING_MIC_INPUT, &value);
    if (ret == ESP_OK) {
        if (value == 0) {
            g_active_mic.store(MIC_I2S0_ES8311);
        } else if (value == 1) {
            g_active_mic.store(MIC_I2S1_INMP441);
        } else {
            ESP_LOGW(TAG, "Ignoring invalid saved microphone input %u", value);
        }
    } else if (ret != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGW(TAG, "Could not read saved microphone input: %s", esp_err_to_name(ret));
    }

    nvs_close(handle);
}

static esp_err_t save_audio_setting(const char *key, uint8_t value)
{
    nvs_handle_t handle;
    esp_err_t ret = nvs_open(AUDIO_SETTINGS_NAMESPACE, NVS_READWRITE, &handle);
    if (ret != ESP_OK) {
        return ret;
    }
    ret = nvs_set_u8(handle, key, value);
    if (ret == ESP_OK) {
        ret = nvs_commit(handle);
    }
    nvs_close(handle);
    return ret;
}

static esp_err_t apply_speaker_volume(uint8_t vol_level)
{
    const float db_val = vol_level == 0
        ? -90.0f
        : 20.0f * log10f((float)vol_level / 100.0f);
    return g_codec_if->set_vol(g_codec_if, db_val);
}

static void record_tx_error(esp_err_t error)
{
    esp_err_t expected = ESP_OK;
    if (g_tx_error.compare_exchange_strong(expected, error)) {
        ESP_LOGE(TAG, "Speaker I2S transmit failed: %s", esp_err_to_name(error));
    }
}

// Ringbuffers
#define TX_RINGBUF_SIZE  32000
#define RX_RINGBUF_SIZE  32000
#define DSP_RINGBUF_SIZE 32000

static RingbufHandle_t g_tx_ringbuf = NULL;
static RingbufHandle_t g_rx_ringbuf = NULL;
static RingbufHandle_t g_dsp_ringbuf = NULL;

static TaskHandle_t g_tx_task_handle = NULL;
static TaskHandle_t g_rx_task_handle = NULL;
static TaskHandle_t g_dsp_task_handle = NULL;

static int16_t g_rx_stereo_16[READ_CHUNK_FRAMES * 2];
static int32_t g_rx_stereo_32[READ_CHUNK_FRAMES * 2];
static int16_t g_rx_mono[READ_CHUNK_FRAMES];

// I2C Master Bus Handle
static i2c_master_bus_handle_t g_i2c_bus_handle = NULL;

// I2S Handles
static i2s_chan_handle_t g_i2s_rx_handle = NULL;
static i2s_chan_handle_t g_i2s_tx_handle = NULL;
static i2s_chan_handle_t g_i2s1_rx_handle = NULL;

// Codec Interfaces
static const audio_codec_ctrl_if_t *g_codec_ctrl_if = NULL;
static const audio_codec_data_if_t *g_codec_data_if = NULL;
static const audio_codec_gpio_if_t *g_codec_gpio_if = NULL;

static esp_err_t audio_i2c_init(void) {
    i2c_master_bus_config_t i2c_mst_config = {};
    i2c_mst_config.clk_source = I2C_CLK_SRC_DEFAULT;
    i2c_mst_config.i2c_port = I2C_NUM_0;
    i2c_mst_config.scl_io_num = (gpio_num_t)BOARD_AUDIO_I2C_SCL_GPIO;
    i2c_mst_config.sda_io_num = (gpio_num_t)BOARD_AUDIO_I2C_SDA_GPIO;
    i2c_mst_config.glitch_ignore_cnt = 7;
    i2c_mst_config.flags.enable_internal_pullup = true;
    
    ESP_LOGI(TAG, "Initializing I2C Master: SCL=%d, SDA=%d, Freq=400000Hz", 
             BOARD_AUDIO_I2C_SCL_GPIO, BOARD_AUDIO_I2C_SDA_GPIO);
             
    return i2c_new_master_bus(&i2c_mst_config, &g_i2c_bus_handle);
}

static esp_err_t audio_i2s_init(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.auto_clear = true;
    
    esp_err_t ret = i2s_new_channel(&chan_cfg, &g_i2s_tx_handle, &g_i2s_rx_handle);
    if (ret != ESP_OK) return ret;
    
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(BOARD_AUDIO_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG((i2s_data_bit_width_t)BOARD_AUDIO_BITS_PER_SAMPLE, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = (gpio_num_t)BOARD_AUDIO_I2S_MCLK_GPIO,
            .bclk = (gpio_num_t)BOARD_AUDIO_I2S_BCLK_GPIO,
            .ws   = (gpio_num_t)BOARD_AUDIO_I2S_WS_GPIO,
            .dout = (gpio_num_t)BOARD_AUDIO_I2S_DOUT_GPIO,
            .din  = (gpio_num_t)BOARD_AUDIO_I2S_DIN_GPIO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    std_cfg.clk_cfg.clk_src = I2S_CLK_SRC_APLL;
    
    ret = i2s_channel_init_std_mode(g_i2s_rx_handle, &std_cfg);
    if (ret != ESP_OK) return ret;
    ret = i2s_channel_init_std_mode(g_i2s_tx_handle, &std_cfg);    
    return ret;
}

static esp_err_t audio_i2s1_init(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_MASTER);
    chan_cfg.auto_clear = true;
    
    esp_err_t ret = i2s_new_channel(&chan_cfg, NULL, &g_i2s1_rx_handle);
    if (ret != ESP_OK) return ret;
    
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(BOARD_AUDIO_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)BOARD_AUDIO_INMP441_BCLK_GPIO,
            .ws   = (gpio_num_t)BOARD_AUDIO_INMP441_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din  = (gpio_num_t)BOARD_AUDIO_INMP441_DIN_GPIO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    std_cfg.clk_cfg.clk_src = I2S_CLK_SRC_APLL;
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_BOTH;
    
    ret = i2s_channel_init_std_mode(g_i2s1_rx_handle, &std_cfg);
    if (ret != ESP_OK) return ret;
    return i2s_channel_enable(g_i2s1_rx_handle);
}

static esp_err_t audio_codec_init(void) {
    audio_codec_i2c_cfg_t i2c_cfg = {};
    i2c_cfg.port = I2C_NUM_0;
    i2c_cfg.addr = ES8311_CODEC_DEFAULT_ADDR;
    i2c_cfg.bus_handle = g_i2c_bus_handle;
    
    g_codec_ctrl_if = audio_codec_new_i2c_ctrl(&i2c_cfg);
    if (!g_codec_ctrl_if) return ESP_FAIL;
    
    audio_codec_i2s_cfg_t i2s_cfg = {};
    i2s_cfg.port = I2S_NUM_0;
    i2s_cfg.rx_handle = g_i2s_rx_handle;
    i2s_cfg.tx_handle = g_i2s_tx_handle;
    i2s_cfg.clk_src = I2S_CLK_SRC_APLL;
    
    g_codec_data_if = audio_codec_new_i2s_data(&i2s_cfg);
    if (!g_codec_data_if) return ESP_FAIL;
    
    if (BOARD_AUDIO_PA_PIN_GPIO >= 0) {
        g_codec_gpio_if = audio_codec_new_gpio();
        if (!g_codec_gpio_if) return ESP_ERR_NO_MEM;
    }
    
    es8311_codec_cfg_t es8311_cfg = {};
    es8311_cfg.ctrl_if = g_codec_ctrl_if;
    es8311_cfg.gpio_if = g_codec_gpio_if;
    es8311_cfg.codec_mode = ESP_CODEC_DEV_WORK_MODE_BOTH;
    es8311_cfg.pa_pin = BOARD_AUDIO_PA_PIN_GPIO;
    es8311_cfg.pa_reverted = false;
    es8311_cfg.master_mode = false;
    es8311_cfg.use_mclk = true;
    es8311_cfg.digital_mic = false;
    es8311_cfg.invert_mclk = false;
    es8311_cfg.invert_sclk = false;
    es8311_cfg.hw_gain.pa_voltage = 5.0;
    es8311_cfg.hw_gain.codec_dac_voltage = 3.3;
    
    g_codec_if = es8311_codec_new(&es8311_cfg);
    if (!g_codec_if) return ESP_FAIL;
    
    esp_codec_dev_sample_info_t fs = {};
    fs.bits_per_sample = BOARD_AUDIO_BITS_PER_SAMPLE;
    fs.channel = 2; 
    fs.sample_rate = BOARD_AUDIO_SAMPLE_RATE;
    
    esp_err_t ret = g_codec_if->set_fs(g_codec_if, &fs);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure ES8311 sample rate: %s", esp_err_to_name(ret));
        return ret;
    }
    ret = g_codec_if->set_mic_gain(g_codec_if, g_current_mic_gain);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set ES8311 microphone gain: %s", esp_err_to_name(ret));
        return ret;
    }
    ret = i2s_channel_enable(g_i2s_rx_handle);
    if (ret != ESP_OK) return ret;
    ret = i2s_channel_enable(g_i2s_tx_handle);
    return ret;
}

// -------------------------------------------------------------
// THREADS (TX, RX, DSP)
// -------------------------------------------------------------

static void audio_tx_task(void *pvParameters) {
    while (1) {
        size_t size = 0;
        int16_t *data = (int16_t *)xRingbufferReceiveUpTo(g_tx_ringbuf, &size, portMAX_DELAY, 4000);
        if (data) {
            int samples = size / sizeof(int16_t);
            int16_t* stereo_buf = (int16_t*)malloc(samples * 2 * sizeof(int16_t));
            if (stereo_buf) {
                for (int i = 0; i < samples; i++) {
                    stereo_buf[i * 2] = data[i];
                    stereo_buf[i * 2 + 1] = data[i];
                }
                const size_t total_bytes = samples * 2 * sizeof(int16_t);
                size_t total_written = 0;
                esp_err_t ret = ESP_OK;
                while (total_written < total_bytes) {
                    size_t bytes_written = 0;
                    ret = i2s_channel_write(
                        g_i2s_tx_handle,
                        (uint8_t *)stereo_buf + total_written,
                        total_bytes - total_written,
                        &bytes_written,
                        portMAX_DELAY);
                    if (ret != ESP_OK || bytes_written == 0) {
                        if (ret == ESP_OK) {
                            ret = ESP_ERR_TIMEOUT;
                        }
                        break;
                    }
                    total_written += bytes_written;
                }
                if (ret != ESP_OK) {
                    record_tx_error(ret);
                }
                free(stereo_buf);
            } else {
                record_tx_error(ESP_ERR_NO_MEM);
            }
            vRingbufferReturnItem(g_tx_ringbuf, (void *)data);
        }
    }
}

static void audio_rx_task(void *pvParameters) {
    uint32_t dropped_chunks = 0;
    int64_t last_error_log_us = 0;
    while (1) {
        size_t bytes_read = 0;
        const uint32_t read_epoch = g_capture_epoch.load();
        const mic_type_t source = g_active_mic.load();
        esp_err_t ret;
        size_t frame_count = 0;
        i2s_chan_handle_t read_handle;

        if (source == MIC_I2S1_INMP441) {
            read_handle = g_i2s1_rx_handle;
            ret = i2s_channel_read(
                read_handle, g_rx_stereo_32, sizeof(g_rx_stereo_32),
                &bytes_read, READ_TIMEOUT_MS);
            frame_count = bytes_read / (sizeof(int32_t) * 2);
            const size_t channel =
#if CONFIG_ASR_INMP441_SLOT_RIGHT
                1;
#else
                0;
#endif
            // INMP441 samples are 24-bit signed values left-aligned in 32-bit slots.
            for (size_t i = 0; i < frame_count; ++i) {
                g_rx_mono[i] = (int16_t)(g_rx_stereo_32[i * 2 + channel] >> 16);
            }
        } else {
            read_handle = g_i2s_rx_handle;
            ret = i2s_channel_read(
                read_handle, g_rx_stereo_16, sizeof(g_rx_stereo_16),
                &bytes_read, READ_TIMEOUT_MS);
            frame_count = bytes_read / (sizeof(int16_t) * 2);
            for (size_t i = 0; i < frame_count; ++i) {
                g_rx_mono[i] = g_rx_stereo_16[i * 2];
            }
        }

        if (ret == ESP_ERR_TIMEOUT) {
            continue;
        }
        if (ret != ESP_OK) {
            const int64_t now_us = esp_timer_get_time();
            if (now_us - last_error_log_us >= 1000000) {
                ESP_LOGE(TAG, "I2S microphone read failed: %s", esp_err_to_name(ret));
                last_error_log_us = now_us;
            }
            continue;
        }
        if (read_epoch != g_capture_epoch.load() ||
            source != g_active_mic.load() ||
            !g_capture_running.load() ||
            frame_count == 0) {
            continue;
        }

        xSemaphoreTake(g_state_mutex, portMAX_DELAY);
        const BaseType_t queued = g_capture_running.load() &&
            read_epoch == g_capture_epoch.load() &&
            source == g_active_mic.load()
            ? xRingbufferSend(g_rx_ringbuf, g_rx_mono,
                              frame_count * sizeof(int16_t), 0)
            : pdFALSE;
        xSemaphoreGive(g_state_mutex);
        if (queued != pdTRUE &&
            g_capture_running.load() &&
            read_epoch == g_capture_epoch.load()) {
            if (++dropped_chunks % 100 == 1) {
                ESP_LOGW(TAG, "Microphone queue full; dropped %lu audio chunks",
                         (unsigned long)dropped_chunks);
            }
        }
    }
}

static void audio_dsp_task(void *pvParameters) {
    while (1) {
        size_t size = 0;
        // Wait for mono audio from RX task
        int16_t *data = (int16_t *)xRingbufferReceiveUpTo(
            g_rx_ringbuf, &size, portMAX_DELAY, ASR_CHUNK_BYTES_100MS);
        if (data) {
            xSemaphoreTake(g_state_mutex, portMAX_DELAY);
            const BaseType_t queued = g_capture_running.load()
                ? xRingbufferSend(g_dsp_ringbuf, data, size, 0)
                : pdFALSE;
            xSemaphoreGive(g_state_mutex);
            if (queued != pdTRUE && g_capture_running.load()) {
                ESP_LOGW(TAG, "Processed-audio queue full; dropping microphone data");
            }
            vRingbufferReturnItem(g_rx_ringbuf, (void *)data);
        }
    }
}

// -------------------------------------------------------------
// PUBLIC API
// -------------------------------------------------------------

static void drain_ringbuffer(RingbufHandle_t ringbuf) {
    if (!ringbuf) return;
    size_t size = 0;
    void *item = NULL;
    while ((item = xRingbufferReceive(ringbuf, &size, 0)) != NULL) {
        vRingbufferReturnItem(ringbuf, item);
    }
}

static void audio_manager_cleanup(void) {
    g_initialized.store(false);
    g_capture_running.store(false);
    g_playback_running.store(false);

    if (g_tx_task_handle) {
        vTaskDelete(g_tx_task_handle);
        g_tx_task_handle = NULL;
    }
    if (g_rx_task_handle) {
        vTaskDelete(g_rx_task_handle);
        g_rx_task_handle = NULL;
    }
    if (g_dsp_task_handle) {
        vTaskDelete(g_dsp_task_handle);
        g_dsp_task_handle = NULL;
    }

    if (g_codec_if) {
        const esp_err_t disable_error = g_codec_if->enable(g_codec_if, false);
        if (disable_error != ESP_OK) {
            ESP_LOGW(TAG, "Codec disable during cleanup failed: %s",
                     esp_err_to_name(disable_error));
        }
        const int delete_error = audio_codec_delete_codec_if(g_codec_if);
        if (delete_error != 0) {
            ESP_LOGW(TAG, "Codec cleanup returned %d", delete_error);
        }
        g_codec_if = NULL;
    }
    if (g_codec_gpio_if) {
        const int delete_error = audio_codec_delete_gpio_if(g_codec_gpio_if);
        if (delete_error != 0) ESP_LOGW(TAG, "Codec GPIO cleanup returned %d", delete_error);
        g_codec_gpio_if = NULL;
    }
    if (g_codec_data_if) {
        const int delete_error = audio_codec_delete_data_if(g_codec_data_if);
        if (delete_error != 0) ESP_LOGW(TAG, "Codec data cleanup returned %d", delete_error);
        g_codec_data_if = NULL;
    }
    if (g_codec_ctrl_if) {
        const int delete_error = audio_codec_delete_ctrl_if(g_codec_ctrl_if);
        if (delete_error != 0) ESP_LOGW(TAG, "Codec control cleanup returned %d", delete_error);
        g_codec_ctrl_if = NULL;
    }
    if (g_i2s1_rx_handle) {
        const esp_err_t delete_error = i2s_del_channel(g_i2s1_rx_handle);
        if (delete_error != ESP_OK) ESP_LOGW(TAG, "I2S1 cleanup failed: %s", esp_err_to_name(delete_error));
        g_i2s1_rx_handle = NULL;
    }
    if (g_i2s_rx_handle) {
        const esp_err_t delete_error = i2s_del_channel(g_i2s_rx_handle);
        if (delete_error != ESP_OK) ESP_LOGW(TAG, "I2S0 RX cleanup failed: %s", esp_err_to_name(delete_error));
        g_i2s_rx_handle = NULL;
    }
    if (g_i2s_tx_handle) {
        const esp_err_t delete_error = i2s_del_channel(g_i2s_tx_handle);
        if (delete_error != ESP_OK) ESP_LOGW(TAG, "I2S0 TX cleanup failed: %s", esp_err_to_name(delete_error));
        g_i2s_tx_handle = NULL;
    }
    if (g_i2c_bus_handle) {
        const esp_err_t delete_error = i2c_del_master_bus(g_i2c_bus_handle);
        if (delete_error != ESP_OK) ESP_LOGW(TAG, "I2C bus cleanup failed: %s", esp_err_to_name(delete_error));
        g_i2c_bus_handle = NULL;
    }
    if (g_tx_ringbuf) {
        vRingbufferDelete(g_tx_ringbuf);
        g_tx_ringbuf = NULL;
    }
    if (g_rx_ringbuf) {
        vRingbufferDelete(g_rx_ringbuf);
        g_rx_ringbuf = NULL;
    }
    if (g_dsp_ringbuf) {
        vRingbufferDelete(g_dsp_ringbuf);
        g_dsp_ringbuf = NULL;
    }
    if (g_state_mutex) {
        vSemaphoreDelete(g_state_mutex);
        g_state_mutex = NULL;
    }
}

bool audio_manager_is_initialized(void) { return g_initialized.load(); }

esp_err_t audio_manager_init(void) {
    if (g_initialized.load()) return ESP_OK;

    esp_err_t ret = ESP_OK;
    g_state_mutex = xSemaphoreCreateMutex();
    if (!g_state_mutex) return ESP_ERR_NO_MEM;

    load_audio_settings();
    ret = audio_i2c_init();
    if (ret != ESP_OK) goto fail;
    ret = audio_i2s_init();
    if (ret != ESP_OK) goto fail;
    ret = audio_i2s1_init();
    if (ret != ESP_OK) goto fail;
    ret = audio_codec_init();
    if (ret != ESP_OK) goto fail;

    ret = apply_speaker_volume(g_current_speaker_vol);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Could not set startup speaker volume: %s", esp_err_to_name(ret));
    }
    
    g_tx_ringbuf = xRingbufferCreate(TX_RINGBUF_SIZE, RINGBUF_TYPE_BYTEBUF);
    g_rx_ringbuf = xRingbufferCreate(RX_RINGBUF_SIZE, RINGBUF_TYPE_BYTEBUF);
    g_dsp_ringbuf = xRingbufferCreate(DSP_RINGBUF_SIZE, RINGBUF_TYPE_BYTEBUF);
    
    if (!g_tx_ringbuf || !g_rx_ringbuf || !g_dsp_ringbuf) {
        ret = ESP_ERR_NO_MEM;
        goto fail;
    }
    
    if (xTaskCreatePinnedToCore(audio_tx_task, "audio_tx", 4096, NULL, 15, &g_tx_task_handle, 1) != pdPASS ||
        xTaskCreatePinnedToCore(audio_rx_task, "audio_rx", 4096, NULL, 22, &g_rx_task_handle, 1) != pdPASS ||
        xTaskCreatePinnedToCore(audio_dsp_task, "audio_dsp", 8192, NULL, 18, &g_dsp_task_handle, 1) != pdPASS) {
        ret = ESP_ERR_NO_MEM;
        goto fail;
    }
    
    g_initialized.store(true);
    ESP_LOGI(TAG, "Audio initialized; microphone: %s, ES8311 gain: %.0f dB, speaker volume: %u%%",
             audio_manager_get_mic_name(), g_current_mic_gain, g_current_speaker_vol);
    return ESP_OK;

fail:
    ESP_LOGE(TAG, "Audio initialization failed: %s", esp_err_to_name(ret));
    audio_manager_cleanup();
    return ret;
}

esp_err_t audio_manager_set_mic_gain(float gain_db) {
    if (!isfinite(gain_db) || gain_db < 0.0f || gain_db > 42.0f ||
        fmodf(gain_db, 6.0f) != 0.0f) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!g_initialized.load() || !g_state_mutex || !g_codec_if) {
        return ESP_ERR_INVALID_STATE;
    }
    xSemaphoreTake(g_state_mutex, portMAX_DELAY);
    esp_err_t ret = g_codec_if->set_mic_gain(g_codec_if, gain_db);
    if (ret == ESP_OK) {
        g_current_mic_gain = gain_db;
        ret = save_audio_setting(AUDIO_SETTING_MIC_GAIN, (uint8_t)gain_db);
        if (ret == ESP_OK) {
            ESP_LOGI(TAG, "Mic gain set to %.1f dB and saved", gain_db);
        } else {
            ESP_LOGE(TAG, "Mic gain applied but could not save it: %s", esp_err_to_name(ret));
        }
    }
    xSemaphoreGive(g_state_mutex);
    return ret;
}

esp_err_t audio_manager_get_mic_gain(float* gain_db) {
    if (!gain_db) return ESP_ERR_INVALID_ARG;
    if (!g_initialized.load() || !g_state_mutex) return ESP_ERR_INVALID_STATE;
    xSemaphoreTake(g_state_mutex, portMAX_DELAY);
    *gain_db = g_current_mic_gain;
    xSemaphoreGive(g_state_mutex);
    return ESP_OK;
}

esp_err_t audio_manager_set_mic(mic_type_t mic) {
    if (mic != MIC_I2S1_INMP441 && mic != MIC_I2S0_ES8311) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!g_initialized.load() || !g_state_mutex) return ESP_ERR_INVALID_STATE;

    xSemaphoreTake(g_state_mutex, portMAX_DELAY);
    if (g_capture_running.load()) {
        xSemaphoreGive(g_state_mutex);
        return ESP_ERR_INVALID_STATE;
    }
    g_capture_epoch.fetch_add(1);
    g_active_mic.store(mic);
    drain_ringbuffer(g_rx_ringbuf);
    drain_ringbuffer(g_dsp_ringbuf);
    const uint8_t saved_input = mic == MIC_I2S0_ES8311 ? 0 : 1;
    esp_err_t ret = save_audio_setting(AUDIO_SETTING_MIC_INPUT, saved_input);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Active microphone set to %s and saved", audio_manager_get_mic_name());
    } else {
        ESP_LOGE(TAG, "Microphone selected but could not save it: %s", esp_err_to_name(ret));
    }
    xSemaphoreGive(g_state_mutex);
    return ret;
}

mic_type_t audio_manager_get_mic(void) {
    return g_active_mic.load();
}

const char* audio_manager_get_mic_name(void) {
    return g_active_mic.load() == MIC_I2S1_INMP441
        ? "external I2S microphone"
        : "onboard ES8311 microphone";
}

bool audio_manager_is_capture_running(void) {
    return g_capture_running.load();
}

esp_err_t audio_manager_set_speaker_vol(uint8_t vol_level) {
    if (vol_level > 100) return ESP_ERR_INVALID_ARG;
    if (!g_initialized.load() || !g_state_mutex || !g_codec_if) {
        return ESP_ERR_INVALID_STATE;
    }
    xSemaphoreTake(g_state_mutex, portMAX_DELAY);
    esp_err_t ret = apply_speaker_volume(vol_level);
    if (ret == ESP_OK) {
        g_current_speaker_vol = vol_level;
        ret = save_audio_setting(AUDIO_SETTING_SPEAKER_VOLUME, vol_level);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Speaker volume applied but could not save it: %s", esp_err_to_name(ret));
        }
    }
    xSemaphoreGive(g_state_mutex);
    return ret;
}

esp_err_t audio_manager_start_capture(void) {
    if (!g_initialized.load() || !g_state_mutex) return ESP_ERR_INVALID_STATE;
    xSemaphoreTake(g_state_mutex, portMAX_DELAY);
    if (g_capture_running.load()) {
        xSemaphoreGive(g_state_mutex);
        return ESP_OK;
    }

    esp_err_t ret = ESP_OK;
    if (g_active_mic.load() == MIC_I2S0_ES8311) {
        ret = g_codec_if->enable(g_codec_if, true);
        if (ret != ESP_OK) {
            xSemaphoreGive(g_state_mutex);
            return ret;
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }
    drain_ringbuffer(g_rx_ringbuf);
    drain_ringbuffer(g_dsp_ringbuf);
    g_capture_epoch.fetch_add(1);
    g_capture_running.store(true);
    xSemaphoreGive(g_state_mutex);
    ESP_LOGI(TAG, "Audio capture started");
    return ESP_OK;
}

esp_err_t audio_manager_stop_capture(void) {
    if (!g_initialized.load() || !g_state_mutex) return ESP_ERR_INVALID_STATE;
    xSemaphoreTake(g_state_mutex, portMAX_DELAY);
    if (!g_capture_running.load()) {
        xSemaphoreGive(g_state_mutex);
        return ESP_OK;
    }
    g_capture_epoch.fetch_add(1);
    g_capture_running.store(false);
    esp_err_t ret = ESP_OK;
    if (g_active_mic.load() == MIC_I2S0_ES8311 && !g_playback_running.load()) {
        ret = g_codec_if->enable(g_codec_if, false);
    }
    drain_ringbuffer(g_rx_ringbuf);
    drain_ringbuffer(g_dsp_ringbuf);
    xSemaphoreGive(g_state_mutex);
    ESP_LOGI(TAG, "Audio capture stopped");
    return ret;
}

int audio_manager_read(int16_t* buffer, size_t samples, uint32_t timeout_ms) {
    if (!buffer || samples == 0 || !g_capture_running.load()) return 0;
    
    size_t requested_bytes = samples * sizeof(int16_t);
    size_t total_received_bytes = 0;
    uint32_t start_time = xTaskGetTickCount();
    
    while (total_received_bytes < requested_bytes) {
        if (!g_capture_running.load()) break;
        
        uint32_t elapsed = (xTaskGetTickCount() - start_time) * portTICK_PERIOD_MS;
        if (elapsed >= timeout_ms && timeout_ms != portMAX_DELAY) {
            break; // Timeout
        }
        
        uint32_t wait_ticks = (timeout_ms == portMAX_DELAY) ? portMAX_DELAY : pdMS_TO_TICKS(timeout_ms - elapsed);
        if (wait_ticks == 0 && timeout_ms != 0) wait_ticks = 1;
        
        size_t received_bytes = 0;
        void *data = xRingbufferReceiveUpTo(g_dsp_ringbuf, &received_bytes, wait_ticks, requested_bytes - total_received_bytes);
        
        if (data) {
            memcpy((uint8_t*)buffer + total_received_bytes, data, received_bytes);
            vRingbufferReturnItem(g_dsp_ringbuf, data);
            total_received_bytes += received_bytes;
        }
    }
    
    return total_received_bytes / sizeof(int16_t);
}

esp_err_t audio_manager_get_levels(float *rms_out, float *peak_out, int *clipping_count) {
    if (!rms_out || !peak_out || !clipping_count) return ESP_ERR_INVALID_ARG;
    if (!g_initialized.load()) return ESP_ERR_INVALID_STATE;

    const size_t target_samples = BOARD_AUDIO_SAMPLE_RATE;
    int16_t chunk_buf[READ_CHUNK_FRAMES];
    size_t total_samples = 0;
    int total_clipping = 0;
    float max_peak = 0.0f;
    double sum_sq = 0.0;

    const bool was_running = g_capture_running.load();
    if (!was_running) {
        const esp_err_t start_error = audio_manager_start_capture();
        if (start_error != ESP_OK) return start_error;
    }

    const int64_t deadline_us = esp_timer_get_time() + 3000000;
    esp_err_t result = ESP_OK;
    while (total_samples < target_samples) {
        if (esp_timer_get_time() >= deadline_us) {
            result = ESP_ERR_TIMEOUT;
            break;
        }
        const size_t request_samples = target_samples - total_samples < READ_CHUNK_FRAMES
            ? target_samples - total_samples
            : READ_CHUNK_FRAMES;
        const int read = audio_manager_read(chunk_buf, request_samples, READ_TIMEOUT_MS * 4);
        if (read <= 0) {
            if (esp_timer_get_time() >= deadline_us) {
                result = ESP_ERR_TIMEOUT;
                break;
            }
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        for (int j = 0; j < read; ++j) {
            float val = (float)chunk_buf[j];
            sum_sq += (val * val);
            float abs_val = fabsf(val);
            if (abs_val > max_peak) max_peak = abs_val;
            if (abs_val >= 32700.0f) total_clipping++;
        }
        total_samples += (size_t)read;
    }

    if (!was_running) {
        const esp_err_t stop_error = audio_manager_stop_capture();
        if (result == ESP_OK && stop_error != ESP_OK) result = stop_error;
    }

    if (result != ESP_OK) return result;
    if (total_samples == 0) return ESP_ERR_TIMEOUT;
    *rms_out = sqrtf((float)(sum_sq / (double)total_samples));
    *peak_out = max_peak;
    *clipping_count = total_clipping;
    return ESP_OK;
}

esp_err_t audio_manager_start_playback(void) {
    if (!g_initialized.load() || !g_state_mutex) return ESP_ERR_INVALID_STATE;
    xSemaphoreTake(g_state_mutex, portMAX_DELAY);
    if (g_playback_running.load()) {
        xSemaphoreGive(g_state_mutex);
        return ESP_OK;
    }
    if (xRingbufferGetCurFreeSize(g_tx_ringbuf) < TX_RINGBUF_SIZE) {
        xSemaphoreGive(g_state_mutex);
        ESP_LOGE(TAG, "Cannot start playback while previous speaker audio is still queued");
        return ESP_ERR_INVALID_STATE;
    }
    esp_err_t ret = g_codec_if->enable(g_codec_if, true);
    if (ret != ESP_OK) {
        xSemaphoreGive(g_state_mutex);
        return ret;
    }
    ret = apply_speaker_volume(g_current_speaker_vol);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Could not restore speaker volume %u%% for playback: %s",
                 g_current_speaker_vol, esp_err_to_name(ret));
        const esp_err_t disable_error = g_codec_if->enable(g_codec_if, false);
        if (disable_error != ESP_OK) {
            ESP_LOGE(TAG, "Could not disable codec after playback setup failed: %s",
                     esp_err_to_name(disable_error));
        }
        xSemaphoreGive(g_state_mutex);
        return ret;
    }
    g_tx_error.store(ESP_OK);
    g_playback_running.store(true);
    xSemaphoreGive(g_state_mutex);
    return ESP_OK;
}

esp_err_t audio_manager_stop_playback(void) {
    if (!g_initialized.load() || !g_state_mutex) return ESP_ERR_INVALID_STATE;
    xSemaphoreTake(g_state_mutex, portMAX_DELAY);
    if (!g_playback_running.load()) {
        xSemaphoreGive(g_state_mutex);
        return ESP_OK;
    }
    uint32_t wait_ticks = 3000 / 10;
    while (xRingbufferGetCurFreeSize(g_tx_ringbuf) < TX_RINGBUF_SIZE && wait_ticks > 0) {
        vTaskDelay(pdMS_TO_TICKS(10));
        wait_ticks--;
    }
    esp_err_t ret = ESP_OK;
    if (xRingbufferGetCurFreeSize(g_tx_ringbuf) < TX_RINGBUF_SIZE) {
        ESP_LOGE(TAG, "Timed out waiting for speaker audio to drain");
        ret = ESP_ERR_TIMEOUT;
    }
    g_playback_running.store(false);
    const bool onboard_capture =
        g_capture_running.load() && g_active_mic.load() == MIC_I2S0_ES8311;
    if (!onboard_capture) {
        const esp_err_t disable_error = g_codec_if->enable(g_codec_if, false);
        if (ret == ESP_OK) {
            ret = disable_error;
        }
    }
    const esp_err_t tx_error = g_tx_error.load();
    if (ret == ESP_OK && tx_error != ESP_OK) {
        ret = tx_error;
    }
    xSemaphoreGive(g_state_mutex);
    return ret;
}

int audio_manager_write(const int16_t* buffer, size_t samples, uint32_t timeout_ms) {
    if (!buffer || samples == 0 || !g_playback_running.load()) return 0;
    size_t size = samples * sizeof(int16_t);
    BaseType_t res = xRingbufferSend(g_tx_ringbuf, buffer, size, pdMS_TO_TICKS(timeout_ms));
    if (res != pdTRUE) return -1;
    return samples;
}

void audio_manager_play_boot_sequence(void) {
    if (!g_initialized.load()) return;
    const float freqs[] = {261.63f, 329.63f, 392.00f, 523.25f};
    const int duration_ms = 150; 
    const int sample_rate = BOARD_AUDIO_SAMPLE_RATE;
    const float amplitude = 8000.0f; 
    
    if (audio_manager_start_playback() != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start boot chime playback");
        return;
    }
    for (int note = 0; note < 4; note++) {
        int total_samples = (sample_rate * duration_ms) / 1000;
        int16_t* buffer = (int16_t*)malloc(total_samples * sizeof(int16_t));
        if (!buffer) break;
        for (int i = 0; i < total_samples; i++) {
            float t = (float)i / sample_rate;
            buffer[i] = (int16_t)(amplitude * sin(2.0f * M_PI * freqs[note] * t)); 
        }
        audio_manager_write(buffer, total_samples, 1000);
        free(buffer);
    }
    const esp_err_t stop_error = audio_manager_stop_playback();
    if (stop_error != ESP_OK) {
        ESP_LOGE(TAG, "Failed to stop boot chime playback: %s", esp_err_to_name(stop_error));
    }
}
