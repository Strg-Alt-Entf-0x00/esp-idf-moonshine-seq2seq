/**
 * @file moonshine_parallel.c
 * @brief Dual-core worker implementation using FreeRTOS task notifications
 */

#include "moonshine_parallel.h"
#include "esp_log.h"
#include "sdkconfig.h"

#if defined(CONFIG_FREERTOS_NUMBER_OF_CORES) && (CONFIG_FREERTOS_NUMBER_OF_CORES > 1)

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

static const char* TAG = "moonshine_par";

static TaskHandle_t s_worker = NULL;
static SemaphoreHandle_t s_done = NULL;
static volatile moonshine_par_fn s_job_fn = NULL;
static void *volatile s_job_ctx = NULL;

static void worker_task(void *arg) {
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (s_job_fn) {
            s_job_fn(s_job_ctx, 1, MOONSHINE_PAR_PARTS);
        }
        xSemaphoreGive(s_done);
    }
}

void moonshine_parallel_init(void) {
    if (!s_worker) {
        s_done = xSemaphoreCreateBinary();
        if (!s_done) {
            ESP_LOGE(TAG, "Failed to create completion semaphore");
            return;
        }
        BaseType_t ret = xTaskCreatePinnedToCore(
            worker_task,
            "moon_par1",
            16384,
            NULL,
            configMAX_PRIORITIES - 2,
            &s_worker,
            1
        );
        if (ret == pdPASS) {
            ESP_LOGI(TAG, "Dual-core parallel worker started on Core 1");
        } else {
            ESP_LOGE(TAG, "Failed to create worker task on Core 1");
            vSemaphoreDelete(s_done);
            s_done = NULL;
            s_worker = NULL;
        }
    }
}

void moonshine_parallel_run(moonshine_par_fn fn, void *ctx) {
    if (!s_worker) {
        fn(ctx, 0, MOONSHINE_PAR_PARTS);
        fn(ctx, 1, MOONSHINE_PAR_PARTS);
        return;
    }
    s_job_fn = fn;
    s_job_ctx = ctx;
    xTaskNotifyGive(s_worker);
    fn(ctx, 0, MOONSHINE_PAR_PARTS);
    xSemaphoreTake(s_done, portMAX_DELAY);
}

#else

void moonshine_parallel_init(void) {
    // Single core: no worker needed
}

void moonshine_parallel_run(moonshine_par_fn fn, void *ctx) {
    fn(ctx, 0, MOONSHINE_PAR_PARTS);
    fn(ctx, 1, MOONSHINE_PAR_PARTS);
}

#endif
