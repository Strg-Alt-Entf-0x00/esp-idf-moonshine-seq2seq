/**
 * @file moonshine_loader_vfs.c
 * @brief VFS path adapter for native ESPM loading.
 */

#include "moonshine.h"

#include <limits.h>
#include <stdio.h>

#include "esp_log.h"

static const char *TAG = "moonshine_vfs";

typedef struct {
    FILE *file;
} vfs_source_t;

static esp_err_t vfs_read_at(void *context, uint64_t offset,
                             void *buffer, size_t size)
{
    vfs_source_t *source = context;
    if (!source || !source->file || (!buffer && size)) {
        return ESP_ERR_INVALID_ARG;
    }
    if (offset > (uint64_t)LONG_MAX ||
        fseek(source->file, (long)offset, SEEK_SET) != 0) {
        return ESP_ERR_INVALID_SIZE;
    }
    return fread(buffer, 1, size, source->file) == size
        ? ESP_OK : ESP_ERR_INVALID_SIZE;
}

static esp_err_t vfs_get_size(void *context, uint64_t *out_size)
{
    vfs_source_t *source = context;
    if (!source || !source->file || !out_size) {
        return ESP_ERR_INVALID_ARG;
    }
    const long current = ftell(source->file);
    if (current < 0 || fseek(source->file, 0, SEEK_END) != 0) {
        return ESP_ERR_INVALID_SIZE;
    }
    const long end = ftell(source->file);
    if (end < 0 || fseek(source->file, current, SEEK_SET) != 0) {
        return ESP_ERR_INVALID_SIZE;
    }
    *out_size = (uint64_t)end;
    return ESP_OK;
}

static esp_err_t vfs_close(void *context)
{
    vfs_source_t *source = context;
    if (!source || !source->file) {
        return ESP_OK;
    }
    FILE *file = source->file;
    source->file = NULL;
    return fclose(file) == 0 ? ESP_OK : ESP_FAIL;
}

esp_err_t moonshine_load_espm(const char *filepath, moonshine_model_t *encoder,
                              moonshine_model_t *decoder, bool load_tokenizer)
{
    if (!filepath) {
        return ESP_ERR_INVALID_ARG;
    }
    vfs_source_t source = {.file = fopen(filepath, "rb")};
    if (!source.file) {
        ESP_LOGE(TAG, "Cannot open ESPM model %s", filepath);
        return ESP_ERR_NOT_FOUND;
    }

    const moonshine_io_t io = {
        .read_at = vfs_read_at,
        .get_size = vfs_get_size,
        .close = vfs_close,
        .context = &source,
    };
    return moonshine_load_espm_io(&io, encoder, decoder, load_tokenizer);
}
