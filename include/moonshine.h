/**
 * @file moonshine.h
 * @brief Moonshine ASR Model Inference for ESP32-P4
 * 
 * Native ESPM model loader and inference engine with INT8 and
 * INT8-per-channel quantized inference using C.
 * 
 * @copyright 2026
 */

#ifndef MOONSHINE_H
#define MOONSHINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * CONSTANTS & CONFIGURATION
 * ======================================================================== */

/* Runtime model metadata */
#define MOONSHINE_PART_ENCODER         1
#define MOONSHINE_PART_DECODER         2
#define MOONSHINE_QUANT_INT8           1
#define MOONSHINE_QUANT_INT8_PER_CH    2
#define MOONSHINE_QUANT_MIXED          5

/* Maximum Sizes */
#define MOONSHINE_MAX_HIDDEN_SIZE  512
#define MOONSHINE_MAX_LAYERS       12
#define MOONSHINE_MAX_VOCAB_SIZE   32768
#define MOONSHINE_MIN_AUDIO_SAMPLES 895

/* ========================================================================
 * DATA STRUCTURES
 * ======================================================================== */

/**
 * @brief Runtime model configuration populated from the ESPM tensor index.
 */
typedef struct {
    uint32_t quant_type;        ///< INT8 or INT8 per-channel
    uint32_t part;              ///< Encoder or decoder
    uint32_t total_params;      ///< Total parameter count
    uint32_t vocab_size;        ///< Vocabulary size
    uint32_t hidden_size;       ///< Hidden dimension size
    uint32_t encoder_layers;    ///< Number of encoder layers
    uint32_t decoder_layers;    ///< Number of decoder layers
    uint32_t attention_heads;   ///< Number of attention heads
    uint32_t ffn_size;          ///< Feed-forward network size
} __attribute__((packed)) moonshine_header_t;

/**
 * @brief Tensor Metadata
 * 
 * Metadata for each tensor in the binary
 */
typedef struct {
    uint32_t name_hash;         ///< CRC32 hash of tensor name
    uint32_t num_elements;      ///< Total elements
    uint32_t num_channels;      ///< 0=scalar, 1=per-tensor, N=per-channel
    float* scales;              ///< Scale factors (size: num_channels)
    void* data;                 ///< INT8 tensor data referenced from the ESPM data section
    size_t data_size;           ///< Size of data in bytes
} moonshine_tensor_t;

/**
 * @brief Moonshine Model Structure
 * 
 * Main model container
 */
typedef struct {
    moonshine_header_t header;  ///< Model header
    moonshine_tensor_t* tensors;///< Array of tensors
    uint32_t num_tensors;       ///< Number of tensors
    size_t total_size;          ///< Total model size in bytes
    bool loaded;                ///< Model loaded flag
    void* psram_buffer;         ///< PSRAM buffer (owns tensor data)
    bool owns_tensor_data;      ///< True if tensors point into psram_buffer
    bool scales_in_buffer;      ///< True if tensor scales also point into psram_buffer
    bool is_static_arena;       ///< True if psram_buffer points to static arena (do not free)
} moonshine_model_t;

/**
 * @brief Inference Context
 * 
 * Runtime state for inference
 */
typedef struct {
    moonshine_model_t* model;   ///< Pointer to loaded model
    float* encoder_output;      ///< Encoder output buffer
    float* decoder_output;      ///< Decoder output buffer
    int32_t* tokens;            ///< Token buffer
    uint32_t max_seq_len;       ///< Maximum sequence length
    bool initialized;           ///< Context initialized
} moonshine_context_t;

/* ========================================================================
 * MODEL LOADING & MANAGEMENT
 * ======================================================================== */

/**
 * @brief Initialize Moonshine inference engine
 * 
 * Must be called once before any other operations.
 * Checks that PSRAM is available.
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t moonshine_init(void);

/**
 * @brief Random-access source for native ESPM loading.
 *
 * read_at and get_size must be provided. The optional close callback is called
 * once after loading; it should release only resources owned by the source.
 */
typedef struct {
    esp_err_t (*read_at)(void *context, uint64_t offset, void *buffer, size_t size);
    esp_err_t (*get_size)(void *context, uint64_t *out_size);
    esp_err_t (*close)(void *context);
    void *context;
} moonshine_io_t;

/**
 * @brief Load full Moonshine ASR model from a file (VFS-based)
 * 
 * Works with any mounted VFS filesystem (SD-Card, SPIFFS, LittleFS, etc.)
 * Validates the complete ESPM container and tensor index before allocating
 * model storage, then initializes the tokenizer from the embedded vocabulary.
 * 
 * Examples:
 *   - SD-Card:   "/sdcard/models/moonshine_tiny_en.espm"
 *   - SPIFFS:    "/spiffs/moonshine_tiny_en.espm"
 *   - LittleFS:  "/littlefs/moonshine_tiny_en.espm"
 * 
 * @param[in] filepath Path to .espm file (must be mounted via VFS)
 * @param[out] encoder Encoder model structure
 * @param[out] decoder Decoder model structure
 * @param[in] load_tokenizer If true, initializes tokenizer from the embedded vocabulary section
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t moonshine_load_espm(
    const char* filepath,
    moonshine_model_t* encoder,
    moonshine_model_t* decoder,
    bool load_tokenizer
);

/**
 * @brief Load model using custom I/O interface (advanced)
 * 
 * For custom loading scenarios:
 *   - Direct partition access (bypass VFS)
 *   - Encrypted storage
 *   - Network streaming
 *   - Memory-mapped files
 * 
 * @param[in] io Custom I/O interface implementation
 * @param[out] encoder Encoder model structure
 * @param[out] decoder Decoder model structure
 * @param[in] load_tokenizer If true, initializes tokenizer from vocabulary
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t moonshine_load_espm_io(
    const moonshine_io_t *io,
    moonshine_model_t *encoder,
    moonshine_model_t *decoder,
    bool load_tokenizer
);

/**
 * @brief Load model from memory buffer (advanced)
 * 
 * Useful when model is already loaded into RAM/PSRAM
 * 
 * @param[in] buffer Pointer to model data in memory
 * @param[in] buffer_size Size of model data
 * @param[out] encoder Encoder model structure
 * @param[out] decoder Decoder model structure
 * @param[in] load_tokenizer If true, initializes tokenizer from vocabulary
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t moonshine_load_espm_mem(
    const uint8_t* buffer,
    size_t buffer_size,
    moonshine_model_t* encoder,
    moonshine_model_t* decoder,
    bool load_tokenizer
);

/**
 * @brief Unload model and free memory
 * 
 * @param[in] model Model to unload
 * 
 * @return ESP_OK on success
 */
esp_err_t moonshine_unload_model(moonshine_model_t* model);

/**
 * @brief Verify model integrity
 * 
 * Checks header ranges and tensor pointers/scale metadata. It does not
 * checksum tensor contents.
 * 
 * @param[in] model Model to verify
 * 
 * @return ESP_OK if valid, error code otherwise
 */
esp_err_t moonshine_verify_model(const moonshine_model_t* model);

/* ========================================================================
 * INFERENCE CONTEXT
 * ======================================================================== */

/**
 * @brief Create inference context
 * 
 * Allocates buffers for inference operations
 * 
 * @param[in] model Loaded model
 * @param[in] max_seq_len Maximum sequence length
 * @param[out] context Context to initialize
 * 
 * @return ESP_OK on success
 */
esp_err_t moonshine_create_context(
    moonshine_model_t* model,
    uint32_t max_seq_len,
    moonshine_context_t* context
);

/**
 * @brief Destroy inference context
 * 
 * Frees all context buffers
 * 
 * @param[in] context Context to destroy
 * 
 * @return ESP_OK on success
 */
esp_err_t moonshine_destroy_context(moonshine_context_t* context);

/* ========================================================================
 * ENCODER INFERENCE
 * ======================================================================== */

/**
 * @brief Run encoder inference
 * 
 * Processes normalized waveform samples through the encoder's convolutional
 * audio front end and transformer layers.
 *
 * Input: mono, 16 kHz waveform as float32 samples, normally in [-1, 1]
 * Output: hidden states [encoded_frames × hidden_size]
 * 
 * @param[in] context Inference context
 * @param[in] features Normalized waveform samples [num_samples]
 * @param[in] num_samples Number of waveform samples (minimum 895)
 * @param[out] output Encoder output [encoded_frames × hidden_size]
 * 
 * @return ESP_OK on success
 */
esp_err_t moonshine_encode(
    moonshine_context_t* context,
    const float* features,
    uint32_t num_samples,
    float* output
);

/* ========================================================================
 * DECODER INFERENCE
 * ======================================================================== */

/**
 * @brief Run decoder inference (autoregressive)
 * 
 * Generates tokens from encoder output.
 * 
 * @param[in] context Inference context
 * @param[in] encoder_output Encoder output
 * @param[in] num_frames Number of encoder frames
 * @param[out] tokens Output token IDs
 * @param[in,out] num_tokens In: max tokens, Out: actual tokens
 * 
 * @return ESP_OK on success
 */
esp_err_t moonshine_decode(
    moonshine_context_t* context,
    const float* encoder_output,
    uint32_t num_frames,
    int32_t* tokens,
    uint32_t* num_tokens
);

/**
 * @brief Convert tokens to text
 * 
 * Uses tokenizer to decode token IDs to text
 * 
 * @param[in] tokens Token IDs
 * @param[in] num_tokens Number of tokens
 * @param[out] text Output text buffer
 * @param[in] text_len Size of text buffer
 * 
 * @return ESP_OK on success
 */
esp_err_t moonshine_tokens_to_text(
    const int32_t* tokens,
    uint32_t num_tokens,
    char* text,
    size_t text_len
);

/* ========================================================================
 * UTILITIES
 * ======================================================================== */

/**
 * @brief Get model information string
 * 
 * Returns human-readable model info
 * 
 * @param[in] model Model
 * @param[out] buffer Output buffer
 * @param[in] buffer_len Buffer size
 * 
 * @return ESP_OK on success
 */
esp_err_t moonshine_get_model_info(
    const moonshine_model_t* model,
    char* buffer,
    size_t buffer_len
);

/**
 * @brief Calculate CRC32 hash
 * 
 * Used for tensor name hashing
 * 
 * @param[in] data Data to hash
 * @param[in] len Data length
 * 
 * @return CRC32 hash value
 */
uint32_t moonshine_crc32(const uint8_t* data, size_t len);
uint32_t moonshine_crc32_update(uint32_t crc, const uint8_t* data, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* MOONSHINE_H */
