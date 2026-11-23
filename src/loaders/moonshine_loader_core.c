/**
 * @file moonshine_loader_core.c
 * @brief Validated native ESPM parser using random-access I/O callbacks.
 */

#include "moonshine.h"
#include "esp_model_format.h"
#include "moonshine_tokenizer.h"

#include <math.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "moonshine_espm";

#define MODEL_ARENA_SIZE (26U * 1024U * 1024U + 256U * 1024U)
#define FILE_READ_CHUNK 4096U

static uint8_t *s_model_arena;

typedef struct {
    const moonshine_io_t *io;
} format_io_context_t;

typedef struct {
    uint64_t start;
    uint64_t size;
} tensor_range_t;

static esp_err_t read_at(const moonshine_io_t *io, uint64_t offset,
                         void *buffer, size_t size)
{
    return io->read_at(io->context, offset, buffer, size);
}

static esp_err_t format_read_at(void *context, uint64_t offset,
                                void *buffer, size_t size)
{
    const format_io_context_t *source = context;
    if (!source || !source->io) {
        return ESP_ERR_INVALID_ARG;
    }
    return read_at(source->io, offset, buffer, size);
}

static bool all_zero(const uint8_t *data, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        if (data[i] != 0) {
            return false;
        }
    }
    return true;
}

static bool valid_utf8(const uint8_t *data, size_t size)
{
    for (size_t i = 0; i < size;) {
        const uint8_t first = data[i++];
        if (first <= 0x7FU) {
            continue;
        }
        if (first >= 0xC2U && first <= 0xDFU) {
            if (i >= size || (data[i++] & 0xC0U) != 0x80U) {
                return false;
            }
            continue;
        }
        if (first >= 0xE0U && first <= 0xEFU) {
            if (size - i < 2) {
                return false;
            }
            const uint8_t second = data[i++];
            const uint8_t third = data[i++];
            if ((third & 0xC0U) != 0x80U ||
                (first == 0xE0U && (second < 0xA0U || second > 0xBFU)) ||
                (first == 0xEDU && (second < 0x80U || second > 0x9FU)) ||
                ((first != 0xE0U && first != 0xEDU) &&
                 (second & 0xC0U) != 0x80U)) {
                return false;
            }
            continue;
        }
        if (first >= 0xF0U && first <= 0xF4U) {
            if (size - i < 3) {
                return false;
            }
            const uint8_t second = data[i++];
            const uint8_t third = data[i++];
            const uint8_t fourth = data[i++];
            if ((third & 0xC0U) != 0x80U || (fourth & 0xC0U) != 0x80U ||
                (first == 0xF0U && (second < 0x90U || second > 0xBFU)) ||
                (first == 0xF4U && (second < 0x80U || second > 0x8FU)) ||
                (first != 0xF0U && first != 0xF4U &&
                 (second & 0xC0U) != 0x80U)) {
                return false;
            }
            continue;
        }
        return false;
    }
    return true;
}

static bool fixed_string_is_valid(const char *value, size_t size)
{
    const char *terminator = memchr(value, '\0', size);
    if (!terminator || terminator == value) {
        return false;
    }
    const size_t string_size = (size_t)(terminator - value);
    return valid_utf8((const uint8_t *)value, string_size) &&
        all_zero((const uint8_t *)(terminator + 1),
                 size - (size_t)(terminator + 1 - value));
}

static const esp_model_section_t *find_section(
    const esp_model_section_t *sections, uint16_t count, uint16_t type)
{
    for (uint16_t i = 0; i < count; ++i) {
        if (sections[i].section_type == type) {
            return &sections[i];
        }
    }
    return NULL;
}

static bool ranges_overlap(const tensor_range_t *left, const tensor_range_t *right)
{
    return left->size != 0 && right->size != 0 &&
           left->start < right->start + right->size &&
           right->start < left->start + left->size;
}

static esp_err_t validate_scale_values(const moonshine_io_t *io,
                                       const esp_model_section_t *data_section,
                                       const esp_model_tensor_record_t *record)
{
    uint8_t bytes[FILE_READ_CHUNK];
    uint64_t position = 0;
    if (record->scales_offset > data_section->size ||
        record->scales_size > data_section->size - record->scales_offset) {
        return ESP_ERR_INVALID_SIZE;
    }
    esp_err_t error = read_at(
        io, data_section->offset + record->scales_offset, bytes,
        record->scales_size < sizeof(bytes) ? (size_t)record->scales_size : sizeof(bytes));
    if (error != ESP_OK) {
        return error;
    }
    while (position < record->scales_size) {
        const size_t count = record->scales_size - position < sizeof(bytes)
            ? (size_t)(record->scales_size - position) : sizeof(bytes);
        if (position != 0 &&
            (error = read_at(io, data_section->offset + record->scales_offset + position,
                             bytes, count)) != ESP_OK) {
            return error;
        }
        for (size_t offset = 0; offset < count; offset += sizeof(float)) {
            float scale;
            memcpy(&scale, bytes + offset, sizeof(scale));
            if (!isfinite(scale) || scale <= 0.0f) {
                ESP_LOGE(TAG, "Tensor scale is non-finite or non-positive");
                return ESP_ERR_INVALID_ARG;
            }
        }
        position += count;
    }
    return ESP_OK;
}

static esp_err_t validate_tensor_index(
    const moonshine_io_t *io,
    const esp_model_header_t *header,
    const esp_model_section_t *data_section,
    const uint8_t *index_bytes,
    size_t index_size,
    esp_model_tensor_index_header_t *out_index_header,
    const esp_model_tensor_record_t **out_records)
{
    if (index_size < sizeof(*out_index_header)) {
        return ESP_ERR_INVALID_SIZE;
    }
    memcpy(out_index_header, index_bytes, sizeof(*out_index_header));
    const esp_model_tensor_index_header_t *index_header = out_index_header;
    if (memcmp(index_header->magic, ESPM_TENSOR_INDEX_MAGIC, 4) != 0 ||
        index_header->version != ESPM_TENSOR_INDEX_LEGACY_VERSION ||
        index_header->header_size != sizeof(*index_header) ||
        index_header->record_size != sizeof(esp_model_tensor_record_t) ||
        index_header->reserved != 0 ||
        index_header->tensor_count == 0 ||
        index_header->tensor_count > ESPM_MAX_TENSORS) {
        ESP_LOGE(TAG, "Invalid tensor index header");
        return ESP_ERR_INVALID_ARG;
    }

    const uint64_t required_size = sizeof(*index_header) +
        (uint64_t)index_header->tensor_count * sizeof(esp_model_tensor_record_t);
    if (required_size != index_size ||
        index_header->vocab_size == 0 ||
        index_header->vocab_size > MOONSHINE_MAX_VOCAB_SIZE ||
        index_header->hidden_size == 0 ||
        index_header->hidden_size > MOONSHINE_MAX_HIDDEN_SIZE ||
        index_header->encoder_layers == 0 ||
        index_header->encoder_layers > MOONSHINE_MAX_LAYERS ||
        index_header->decoder_layers == 0 ||
        index_header->decoder_layers > MOONSHINE_MAX_LAYERS ||
        index_header->attention_heads == 0 ||
        index_header->hidden_size % index_header->attention_heads != 0 ||
        index_header->ffn_size == 0 ||
        index_header->encoder_tensor_count == 0 ||
        index_header->decoder_tensor_count == 0) {
        ESP_LOGE(TAG, "Invalid tensor index dimensions or size");
        return ESP_ERR_INVALID_SIZE;
    }

    const esp_model_tensor_record_t *records =
        (const esp_model_tensor_record_t *)(index_bytes + sizeof(*index_header));
    uint32_t part_counts[3] = {0};
    uint64_t total_elements = 0;
    for (uint32_t i = 0; i < index_header->tensor_count; ++i) {
        const esp_model_tensor_record_t *record = &records[i];
        if (!fixed_string_is_valid(record->name, sizeof(record->name)) ||
            !all_zero(record->reserved, sizeof(record->reserved)) ||
            record->part < ESPM_TENSOR_PART_ENCODER ||
            record->part > ESPM_TENSOR_PART_DECODER ||
            record->dtype != ESPM_DTYPE_INT8 ||
            record->rank == 0 || record->rank > 4 ||
            (header->quant_type == MODEL_QUANT_INT8 && record->quant_type != MODEL_QUANT_INT8) ||
            (record->quant_type != MODEL_QUANT_INT8 &&
             record->quant_type != MODEL_QUANT_INT8_PER_CH)) {
            ESP_LOGE(TAG, "Invalid tensor record %lu", (unsigned long)i);
            return ESP_ERR_INVALID_ARG;
        }

        uint64_t element_count = 1;
        for (uint8_t axis = 0; axis < 4; ++axis) {
            if (axis < record->rank) {
                if (record->shape[axis] == 0 ||
                    element_count > UINT32_MAX / record->shape[axis]) {
                    return ESP_ERR_INVALID_SIZE;
                }
                element_count *= record->shape[axis];
            } else if (record->shape[axis] != 0) {
                return ESP_ERR_INVALID_ARG;
            }
        }
        if (element_count != record->num_elements ||
            record->data_size != record->num_elements ||
            record->num_channels == 0 ||
            record->num_channels > UINT32_MAX / sizeof(float) ||
            record->scales_size != record->num_channels * sizeof(float) ||
            record->scales_size % sizeof(float) != 0 ||
            record->scales_offset % sizeof(float) != 0 ||
            record->data_offset % 16U != 0 ||
            (record->quant_type == MODEL_QUANT_INT8 &&
             (record->channel_axis != 0xFF || record->num_channels != 1)) ||
            (record->quant_type == MODEL_QUANT_INT8_PER_CH &&
             record->channel_axis == 0xFF) ||
            (record->channel_axis != 0xFF &&
             (record->channel_axis >= record->rank ||
              record->shape[record->channel_axis] != record->num_channels)) ||
            record->scales_offset > data_section->size ||
            record->scales_size > data_section->size - record->scales_offset ||
            record->data_offset > data_section->size ||
            record->data_size > data_section->size - record->data_offset) {
            ESP_LOGE(TAG, "Invalid tensor dimensions or payload bounds at record %lu",
                     (unsigned long)i);
            return ESP_ERR_INVALID_SIZE;
        }

        const char *terminator = memchr(record->name, '\0', sizeof(record->name));
        if (moonshine_crc32((const uint8_t *)record->name,
                            (size_t)(terminator - record->name)) != record->name_hash) {
            ESP_LOGE(TAG, "Tensor name hash mismatch at record %lu", (unsigned long)i);
            return ESP_ERR_INVALID_ARG;
        }
        tensor_range_t ranges[2] = {
            {record->scales_offset, record->scales_size},
            {record->data_offset, record->data_size},
        };
        if (ranges_overlap(&ranges[0], &ranges[1])) {
            ESP_LOGE(TAG, "Tensor scales and data overlap at record %lu", (unsigned long)i);
            return ESP_ERR_INVALID_ARG;
        }
        for (uint32_t previous = 0; previous < i; ++previous) {
            const esp_model_tensor_record_t *other = &records[previous];
            if (record->part == other->part &&
                (record->name_hash == other->name_hash ||
                 strcmp(record->name, other->name) == 0)) {
                ESP_LOGE(TAG, "Duplicate tensor identity: %s", record->name);
                return ESP_ERR_INVALID_ARG;
            }
            tensor_range_t other_ranges[2] = {
                {other->scales_offset, other->scales_size},
                {other->data_offset, other->data_size},
            };
            if (ranges_overlap(&ranges[0], &other_ranges[0]) ||
                ranges_overlap(&ranges[0], &other_ranges[1]) ||
                ranges_overlap(&ranges[1], &other_ranges[0]) ||
                ranges_overlap(&ranges[1], &other_ranges[1])) {
                ESP_LOGE(TAG, "Tensor payloads overlap at record %lu", (unsigned long)i);
                return ESP_ERR_INVALID_ARG;
            }
        }

        const esp_err_t scale_error = validate_scale_values(io, data_section, record);
        if (scale_error != ESP_OK) {
            return scale_error;
        }
        ++part_counts[record->part];
        total_elements += element_count;
        if (total_elements > UINT32_MAX) {
            return ESP_ERR_INVALID_SIZE;
        }
    }
    if (part_counts[ESPM_TENSOR_PART_ENCODER] != index_header->encoder_tensor_count ||
        part_counts[ESPM_TENSOR_PART_DECODER] != index_header->decoder_tensor_count ||
        total_elements != index_header->total_params) {
        ESP_LOGE(TAG, "Tensor index summary does not match tensor records");
        return ESP_ERR_INVALID_ARG;
    }
    *out_records = records;
    return ESP_OK;
}

static esp_err_t required_metadata_u32(const esp_model_metadata_view_t *metadata,
                                       const char *key, uint32_t *out_value)
{
    esp_model_metadata_value_t value;
    const esp_err_t error = esp_model_metadata_find(metadata, key, &value);
    if (error != ESP_OK) {
        return error;
    }
    if (value.type != ESPM_METADATA_UINT32 ||
        (value.flags & ESPM_METADATA_FLAG_REQUIRED) == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    return esp_model_metadata_get_u32(metadata, key, out_value);
}

static esp_err_t required_metadata_architecture(
    const esp_model_metadata_view_t *metadata, const char *expected)
{
    esp_model_metadata_value_t value;
    const esp_err_t error = esp_model_metadata_find(
        metadata, "general.architecture", &value);
    if (error != ESP_OK) {
        return error;
    }
    const size_t expected_size = strlen(expected);
    if (value.type != ESPM_METADATA_UTF8 ||
        (value.flags & ESPM_METADATA_FLAG_REQUIRED) == 0 ||
        value.value_size != expected_size ||
        memcmp(value.value, expected, expected_size) != 0) {
        return ESP_ERR_NOT_SUPPORTED;
    }
    return ESP_OK;
}

static esp_err_t prepare_v3_tensor_index(
    const uint8_t *index_data, size_t index_size, uint64_t tensor_data_size,
    const esp_model_header_t *container_header,
    const esp_model_metadata_view_t *metadata,
    uint8_t **out_legacy_index, size_t *out_legacy_index_size)
{
    static const char *const known_keys[] = {
        "general.architecture",
        "general.architecture_version",
        "moonshine.vocab_size",
        "moonshine.hidden_size",
        "moonshine.encoder_layers",
        "moonshine.decoder_layers",
        "moonshine.attention_heads",
        "moonshine.ffn_size",
        "moonshine.encoder_tensor_count",
        "moonshine.decoder_tensor_count",
        "moonshine.total_params",
    };
    if (!index_data || !container_header || !metadata ||
        !out_legacy_index || !out_legacy_index_size) {
        return ESP_ERR_INVALID_ARG;
    }

    esp_err_t error = esp_model_metadata_validate_required_keys(
        metadata, known_keys, sizeof(known_keys) / sizeof(known_keys[0]));
    if (error != ESP_OK) {
        return error;
    }
    error = required_metadata_architecture(metadata, "moonshine.seq2seq");
    if (error != ESP_OK) {
        return error;
    }

    uint32_t architecture_version;
    esp_model_tensor_index_header_v2_t source_header;
    const esp_model_tensor_record_v2_t *source_records = NULL;
    error = esp_model_tensor_index_v2_validate(
        index_data, index_size, tensor_data_size, &source_header, &source_records);
    if (error != ESP_OK) {
        return error;
    }
    if (source_header.scope_count != 2 ||
        source_header.total_elements > UINT32_MAX) {
        return ESP_ERR_INVALID_SIZE;
    }

    esp_model_tensor_index_header_t legacy_header = {0};
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t encoder_layers;
    uint32_t decoder_layers;
    uint32_t attention_heads;
    uint32_t ffn_size;
    uint32_t encoder_tensor_count;
    uint32_t decoder_tensor_count;
    uint32_t total_params;
    memcpy(legacy_header.magic, ESPM_TENSOR_INDEX_MAGIC, 4);
    legacy_header.version = ESPM_TENSOR_INDEX_LEGACY_VERSION;
    legacy_header.header_size = sizeof(legacy_header);
    legacy_header.record_size = sizeof(esp_model_tensor_record_t);
    legacy_header.tensor_count = source_header.tensor_count;
    if ((error = required_metadata_u32(metadata, "general.architecture_version",
                                       &architecture_version)) != ESP_OK ||
        architecture_version != 1 ||
        (error = required_metadata_u32(metadata, "moonshine.vocab_size",
                                       &vocab_size)) != ESP_OK ||
        (error = required_metadata_u32(metadata, "moonshine.hidden_size",
                                       &hidden_size)) != ESP_OK ||
        (error = required_metadata_u32(metadata, "moonshine.encoder_layers",
                                       &encoder_layers)) != ESP_OK ||
        (error = required_metadata_u32(metadata, "moonshine.decoder_layers",
                                       &decoder_layers)) != ESP_OK ||
        (error = required_metadata_u32(metadata, "moonshine.attention_heads",
                                       &attention_heads)) != ESP_OK ||
        (error = required_metadata_u32(metadata, "moonshine.ffn_size",
                                       &ffn_size)) != ESP_OK ||
        (error = required_metadata_u32(metadata, "moonshine.encoder_tensor_count",
                                       &encoder_tensor_count)) != ESP_OK ||
        (error = required_metadata_u32(metadata, "moonshine.decoder_tensor_count",
                                       &decoder_tensor_count)) != ESP_OK ||
        (error = required_metadata_u32(metadata, "moonshine.total_params",
                                       &total_params)) != ESP_OK) {
        return error == ESP_OK ? ESP_ERR_NOT_SUPPORTED : error;
    }
    legacy_header.vocab_size = vocab_size;
    legacy_header.hidden_size = hidden_size;
    legacy_header.encoder_layers = encoder_layers;
    legacy_header.decoder_layers = decoder_layers;
    legacy_header.attention_heads = attention_heads;
    legacy_header.ffn_size = ffn_size;
    legacy_header.encoder_tensor_count = encoder_tensor_count;
    legacy_header.decoder_tensor_count = decoder_tensor_count;
    legacy_header.total_params = total_params;
    if (total_params != source_header.total_elements) {
        return ESP_ERR_INVALID_ARG;
    }

    const uint64_t converted_size = sizeof(legacy_header) +
        (uint64_t)source_header.tensor_count * sizeof(esp_model_tensor_record_t);
    if (converted_size > SIZE_MAX) {
        return ESP_ERR_INVALID_SIZE;
    }
    uint8_t *converted = heap_caps_calloc(
        1, (size_t)converted_size, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (!converted) {
        return ESP_ERR_NO_MEM;
    }
    memcpy(converted, &legacy_header, sizeof(legacy_header));
    esp_model_tensor_record_t *records =
        (esp_model_tensor_record_t *)(converted + sizeof(legacy_header));
    uint32_t scope_counts[2] = {0};
    bool has_per_tensor = false;
    bool has_per_axis = false;
    for (uint32_t i = 0; i < source_header.tensor_count; ++i) {
        const esp_model_tensor_record_v2_t *source = &source_records[i];
        if (source->dtype != ESPM_TENSOR_DTYPE_INT8 ||
            source->rank > 4 ||
            source->quant_type == ESPM_TENSOR_QUANT_NONE) {
            heap_caps_free(converted);
            return ESP_ERR_NOT_SUPPORTED;
        }
        esp_model_tensor_record_t *record = &records[i];
        record->name_hash = source->name_hash;
        record->part = (uint8_t)(source->scope_id + 1U);
        record->dtype = source->dtype;
        record->rank = source->rank;
        record->quant_type = source->quant_type == ESPM_TENSOR_QUANT_SYMMETRIC
            ? MODEL_QUANT_INT8 : MODEL_QUANT_INT8_PER_CH;
        record->num_elements = source->num_elements;
        memcpy(record->shape, source->shape, sizeof(record->shape));
        record->num_channels = source->num_channels;
        record->channel_axis = source->channel_axis;
        record->scales_offset = source->scales_offset;
        record->scales_size = source->scales_size;
        record->data_offset = source->data_offset;
        record->data_size = source->data_size;
        memcpy(record->name, source->name, sizeof(record->name));
        ++scope_counts[source->scope_id];
        has_per_tensor |= record->quant_type == MODEL_QUANT_INT8;
        has_per_axis |= record->quant_type == MODEL_QUANT_INT8_PER_CH;
    }
    if (scope_counts[0] != legacy_header.encoder_tensor_count ||
        scope_counts[1] != legacy_header.decoder_tensor_count) {
        heap_caps_free(converted);
        return ESP_ERR_INVALID_ARG;
    }
    const uint16_t observed_quant = has_per_tensor && has_per_axis
        ? MODEL_QUANT_MIXED
        : has_per_axis ? MODEL_QUANT_INT8_PER_CH : MODEL_QUANT_INT8;
    if (container_header->quant_type != observed_quant) {
        heap_caps_free(converted);
        return ESP_ERR_INVALID_ARG;
    }
    *out_legacy_index = converted;
    *out_legacy_index_size = (size_t)converted_size;
    return ESP_OK;
}

static void set_model_header(moonshine_model_t *model,
                             const esp_model_tensor_index_header_t *index_header,
                             uint16_t quant_type, uint32_t part)
{
    model->header.quant_type = quant_type;
    model->header.part = part == ESPM_TENSOR_PART_ENCODER
        ? MOONSHINE_PART_ENCODER : MOONSHINE_PART_DECODER;
    model->header.vocab_size = index_header->vocab_size;
    model->header.hidden_size = index_header->hidden_size;
    model->header.encoder_layers = index_header->encoder_layers;
    model->header.decoder_layers = index_header->decoder_layers;
    model->header.attention_heads = index_header->attention_heads;
    model->header.ffn_size = index_header->ffn_size;
}

static esp_err_t populate_model(moonshine_model_t *model,
                                const esp_model_tensor_record_t *records,
                                uint32_t record_count, uint32_t part,
                                uint16_t quant_type,
                                const esp_model_tensor_index_header_t *index_header)
{
    uint32_t tensor_count = 0;
    for (uint32_t i = 0; i < record_count; ++i) {
        tensor_count += records[i].part == part;
    }
    model->tensors = heap_caps_calloc(tensor_count, sizeof(*model->tensors),
                                      MALLOC_CAP_8BIT);
    if (!model->tensors) {
        return ESP_ERR_NO_MEM;
    }
    model->num_tensors = tensor_count;
    set_model_header(model, index_header, quant_type, part);

    uint32_t output = 0;
    for (uint32_t i = 0; i < record_count; ++i) {
        const esp_model_tensor_record_t *record = &records[i];
        if (record->part != part) {
            continue;
        }
        moonshine_tensor_t *tensor = &model->tensors[output++];
        tensor->name_hash = record->name_hash;
        tensor->num_elements = record->num_elements;
        tensor->num_channels = record->num_channels;
        tensor->scales = (float *)(s_model_arena + record->scales_offset);
        tensor->data = s_model_arena + record->data_offset;
        tensor->data_size = record->data_size;
        model->total_size += record->data_size + record->scales_size;
    }
    model->psram_buffer = s_model_arena;
    model->owns_tensor_data = true;
    model->scales_in_buffer = true;
    model->is_static_arena = true;
    model->loaded = true;
    model->header.total_params = 0;
    for (uint32_t i = 0; i < model->num_tensors; ++i) {
        if (model->header.total_params > UINT32_MAX - model->tensors[i].num_elements) {
            return ESP_ERR_INVALID_SIZE;
        }
        model->header.total_params += model->tensors[i].num_elements;
    }
    return ESP_OK;
}

static esp_err_t load_espm_io_internal(const moonshine_io_t *io,
                                       moonshine_model_t *encoder,
                                       moonshine_model_t *decoder,
                                       bool load_tokenizer)
{
    if (!io || !io->read_at || !io->get_size || !encoder || !decoder ||
        encoder == decoder ||
        encoder->loaded || decoder->loaded || encoder->tensors || decoder->tensors) {
        return ESP_ERR_INVALID_ARG;
    }
    if (load_tokenizer && moonshine_tokenizer_get_vocab_size() != 0) {
        ESP_LOGE(TAG, "Tokenizer is already initialized for another model");
        return ESP_ERR_INVALID_STATE;
    }

    esp_err_t error = ESP_FAIL;
    uint8_t *index_bytes = NULL;
    uint8_t *legacy_index_bytes = NULL;
    uint8_t *metadata_bytes = NULL;
    uint8_t *vocab_bytes = NULL;
    bool tokenizer_loaded = false;
    uint64_t actual_file_size = 0;
    error = io->get_size(io->context, &actual_file_size);
    if (error != ESP_OK) {
        goto cleanup;
    }
    if (actual_file_size < ESPM_HEADER_SIZE) {
        error = ESP_ERR_INVALID_SIZE;
        goto cleanup;
    }

    format_io_context_t format_context = {.io = io};
    const esp_model_io_t format_io = {
        .context = &format_context,
        .read_at = format_read_at,
        .file_size = actual_file_size,
    };
    esp_model_container_t container;
    error = esp_model_container_open(&format_io, &container);
    if (error != ESP_OK) {
        ESP_LOGE(TAG, "Invalid ESPM container: %s", esp_err_to_name(error));
        goto cleanup;
    }
    const esp_model_header_t header = container.header;
    if (header.format_version != ESPM_CURRENT_VERSION ||
        header.model_type != MODEL_TYPE_ASR_SEQ2SEQ ||
        (header.quant_type != MODEL_QUANT_INT8 &&
         header.quant_type != MODEL_QUANT_INT8_PER_CH &&
         header.quant_type != MODEL_QUANT_MIXED) ||
        header.sample_rate != 16000) {
        ESP_LOGE(TAG, "Unsupported Moonshine ESPM model header");
        error = ESP_ERR_NOT_SUPPORTED;
        goto cleanup;
    }
    const esp_model_section_t *sections = container.sections;

    const esp_model_section_t *vocab_section =
        find_section(sections, header.num_sections, SECTION_TYPE_VOCAB);
    const esp_model_section_t *index_section =
        find_section(sections, header.num_sections, SECTION_TYPE_TENSOR_INDEX);
    const esp_model_section_t *data_section =
        find_section(sections, header.num_sections, SECTION_TYPE_TENSOR_DATA);
    const esp_model_section_t *metadata_section =
        find_section(sections, header.num_sections, SECTION_TYPE_METADATA);
    const esp_model_section_t *config_json_section =
        find_section(sections, header.num_sections, SECTION_TYPE_CONFIG_JSON);
    const uint64_t max_index_size =
        sizeof(esp_model_tensor_index_header_v2_t) +
        (uint64_t)ESPM_MAX_TENSORS * sizeof(esp_model_tensor_record_v2_t);
    if (!vocab_section || !index_section || !data_section || !metadata_section ||
        index_section->size > max_index_size ||
        data_section->size > MODEL_ARENA_SIZE ||
        index_section->size > SIZE_MAX || vocab_section->size > SIZE_MAX ||
        metadata_section->size > ESPM_MAX_METADATA_SIZE ||
        metadata_section->size > SIZE_MAX ||
        ((vocab_section->flags | index_section->flags |
          data_section->flags | metadata_section->flags) &
         ESPM_SECTION_FLAG_OPTIONAL) != 0 ||
        (config_json_section &&
         (config_json_section->flags & ESPM_SECTION_FLAG_OPTIONAL) == 0)) {
        ESP_LOGE(TAG, "Missing native ESPM sections or section exceeds runtime limits");
        error = ESP_ERR_INVALID_SIZE;
        goto cleanup;
    }

    esp_model_metadata_view_t metadata_view = {0};
    if (metadata_section) {
        metadata_bytes = heap_caps_malloc((size_t)metadata_section->size,
                                          MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
        if (!metadata_bytes) {
            error = ESP_ERR_NO_MEM;
            goto cleanup;
        }
        error = read_at(io, metadata_section->offset, metadata_bytes,
                        (size_t)metadata_section->size);
        if (error != ESP_OK) {
            goto cleanup;
        }
        error = esp_model_container_verify_section_buffer(
            metadata_section, metadata_bytes, (size_t)metadata_section->size);
        if (error == ESP_OK) {
            error = esp_model_metadata_open(metadata_bytes,
                                            (size_t)metadata_section->size,
                                            &metadata_view);
        }
        if (error != ESP_OK) {
            goto cleanup;
        }
    }

    for (uint16_t i = 0; i < header.num_sections; ++i) {
        if (sections[i].section_type == SECTION_TYPE_TENSOR_DATA ||
            sections[i].section_type == SECTION_TYPE_METADATA ||
            sections[i].section_type == SECTION_TYPE_TENSOR_INDEX ||
            (load_tokenizer &&
             sections[i].section_type == SECTION_TYPE_VOCAB)) {
            continue;
        }
        error = esp_model_container_verify_section(&container, &sections[i]);
        if (error != ESP_OK) {
            goto cleanup;
        }
    }

    index_bytes = heap_caps_malloc((size_t)index_section->size,
                                   MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (!index_bytes) {
        error = ESP_ERR_NO_MEM;
        goto cleanup;
    }
    error = read_at(io, index_section->offset, index_bytes,
                    (size_t)index_section->size);
    if (error != ESP_OK) {
        goto cleanup;
    }
    error = esp_model_container_verify_section_buffer(
        index_section, index_bytes, (size_t)index_section->size);
    if (error != ESP_OK) {
        goto cleanup;
    }
    esp_model_tensor_index_header_t index_header;
    const esp_model_tensor_record_t *records = NULL;
    size_t converted_index_size = 0;
    error = prepare_v3_tensor_index(
        index_bytes, (size_t)index_section->size, data_section->size,
        &header, &metadata_view, &legacy_index_bytes,
        &converted_index_size);
    if (error == ESP_OK) {
        error = validate_tensor_index(
            io, &header, data_section, legacy_index_bytes,
            converted_index_size, &index_header, &records);
    }
    if (error != ESP_OK) {
        goto cleanup;
    }
    heap_caps_free(metadata_bytes);
    metadata_bytes = NULL;

    const uint64_t header_requirement = (uint64_t)header.psram_required_kb * 1024U;
    const uint64_t computed_requirement = data_section->size + vocab_section->size +
        (uint64_t)index_header.vocab_size * (sizeof(char *) + 5U) +
        index_section->size + converted_index_size +
        (uint64_t)index_header.tensor_count * sizeof(moonshine_tensor_t);
    if (header.psram_required_kb == 0 ||
        header_requirement < data_section->size ||
        computed_requirement > SIZE_MAX) {
        ESP_LOGE(TAG, "Invalid ESPM PSRAM requirement metadata");
        error = ESP_ERR_INVALID_SIZE;
        goto cleanup;
    }
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    uint64_t available_psram = free_psram;
    if (s_model_arena) {
        available_psram += MODEL_ARENA_SIZE;
    }
    const uint64_t required_psram = computed_requirement;
    if (available_psram < required_psram) {
        ESP_LOGE(TAG, "Insufficient PSRAM: need %llu bytes, available %llu bytes",
                 (unsigned long long)required_psram,
                 (unsigned long long)available_psram);
        error = ESP_ERR_NO_MEM;
        goto cleanup;
    }

    if (!s_model_arena) {
        s_model_arena = heap_caps_aligned_alloc(
            ESPM_ALIGNMENT, MODEL_ARENA_SIZE, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
        if (!s_model_arena) {
            error = ESP_ERR_NO_MEM;
            goto cleanup;
        }
    }
    error = read_at(io, data_section->offset, s_model_arena,
                    (size_t)data_section->size);
    if (error != ESP_OK) {
        goto cleanup;
    }
    if (esp_model_container_verify_section_buffer(
            data_section, s_model_arena, (size_t)data_section->size) != ESP_OK) {
        ESP_LOGE(TAG, "ESPM tensor-data section CRC mismatch");
        error = ESP_ERR_INVALID_CRC;
        goto cleanup;
    }

    error = populate_model(encoder, records, index_header.tensor_count,
                           ESPM_TENSOR_PART_ENCODER, header.quant_type, &index_header);
    if (error != ESP_OK) {
        goto cleanup;
    }
    error = populate_model(decoder, records, index_header.tensor_count,
                           ESPM_TENSOR_PART_DECODER, header.quant_type, &index_header);
    if (error != ESP_OK) {
        goto cleanup;
    }

    if (load_tokenizer) {
        vocab_bytes = heap_caps_malloc((size_t)vocab_section->size,
                                       MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
        if (!vocab_bytes) {
            error = ESP_ERR_NO_MEM;
            goto cleanup;
        }
        error = read_at(io, vocab_section->offset, vocab_bytes,
                        (size_t)vocab_section->size);
        if (error != ESP_OK) {
            goto cleanup;
        }
        error = esp_model_container_verify_section_buffer(
            vocab_section, vocab_bytes, (size_t)vocab_section->size);
        if (error != ESP_OK) {
            goto cleanup;
        }
        error = moonshine_tokenizer_init_from_buffer(
            vocab_bytes, (size_t)vocab_section->size);
        if (error != ESP_OK) {
            goto cleanup;
        }
        tokenizer_loaded = true;
        if (moonshine_tokenizer_get_vocab_size() != index_header.vocab_size) {
            ESP_LOGE(TAG, "Embedded vocabulary count does not match tensor index");
            error = ESP_ERR_INVALID_SIZE;
            goto cleanup;
        }
    }

    ESP_LOGI(TAG, "Loaded native ESPM model %s: %lu encoder + %lu decoder tensors",
             header.model_name, (unsigned long)encoder->num_tensors,
             (unsigned long)decoder->num_tensors);
    ESP_LOGI(TAG, "Tensor data %llu bytes; PSRAM free %u bytes",
             (unsigned long long)data_section->size,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    error = ESP_OK;

cleanup:
    if (error != ESP_OK) {
        if (decoder->tensors) {
            moonshine_unload_model(decoder);
        }
        if (encoder->tensors) {
            moonshine_unload_model(encoder);
        }
        if (tokenizer_loaded) {
            moonshine_tokenizer_free();
        }
    }
    heap_caps_free(vocab_bytes);
    heap_caps_free(index_bytes);
    heap_caps_free(legacy_index_bytes);
    heap_caps_free(metadata_bytes);
    return error;
}

esp_err_t moonshine_load_espm_io(const moonshine_io_t *io,
                                 moonshine_model_t *encoder,
                                 moonshine_model_t *decoder,
                                 bool load_tokenizer)
{
    if (!io) {
        return ESP_ERR_INVALID_ARG;
    }
    esp_err_t error = load_espm_io_internal(io, encoder, decoder, load_tokenizer);
    if (io->close) {
        const esp_err_t close_error = io->close(io->context);
        if (close_error != ESP_OK) {
            ESP_LOGE(TAG, "ESPM source close failed: %s", esp_err_to_name(close_error));
            if (error == ESP_OK) {
                if (decoder && decoder->tensors) {
                    moonshine_unload_model(decoder);
                }
                if (encoder && encoder->tensors) {
                    moonshine_unload_model(encoder);
                }
                if (load_tokenizer) {
                    moonshine_tokenizer_free();
                }
                error = close_error;
            }
        }
    }
    return error;
}

typedef struct {
    const uint8_t *data;
    size_t size;
} memory_source_t;

static esp_err_t memory_read_at(void *context, uint64_t offset,
                                void *buffer, size_t size)
{
    const memory_source_t *source = context;
    if (!source || (!buffer && size) || offset > source->size ||
        size > source->size - (size_t)offset) {
        return ESP_ERR_INVALID_SIZE;
    }
    memcpy(buffer, source->data + (size_t)offset, size);
    return ESP_OK;
}

static esp_err_t memory_get_size(void *context, uint64_t *out_size)
{
    const memory_source_t *source = context;
    if (!source || !out_size) {
        return ESP_ERR_INVALID_ARG;
    }
    *out_size = source->size;
    return ESP_OK;
}

esp_err_t moonshine_load_espm_mem(const uint8_t *buffer, size_t buffer_size,
                                  moonshine_model_t *encoder,
                                  moonshine_model_t *decoder,
                                  bool load_tokenizer)
{
    if (!buffer || buffer_size == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    memory_source_t source = {.data = buffer, .size = buffer_size};
    moonshine_io_t io = {
        .read_at = memory_read_at,
        .get_size = memory_get_size,
        .close = NULL,
        .context = &source,
    };
    return moonshine_load_espm_io(&io, encoder, decoder, load_tokenizer);
}
