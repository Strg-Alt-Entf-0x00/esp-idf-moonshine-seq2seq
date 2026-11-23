/**
 * @file moonshine_encoder.c
 * @brief Moonshine Encoder Inference Implementation
 *
 * Implements the Moonshine transformer encoder:
 * - Conv1d preprocessing (3 layers + GroupNorm)
 *   * conv1: [frames×1] → [frames×288], kernel=127
 *   * conv2: [frames×288] → [frames×576], kernel=7
 *   * conv3: [frames×576] → [frames×288], kernel=3
 *   * groupnorm: GroupNormalization on 288 channels
 * - Multi-head self-attention (6 layers)
 *   * 8 heads, 36 dims per head
 *   * RoPE position encoding
 *   * RMSNorm before attention
 * - Feed-forward networks (FFN: 288 → 1152 → 288)
 * - RMSNorm (NOT LayerNorm - no bias!)
 * - Residual connections
 *
 * Based on Moonshine-tiny architecture (27M params):
 * - Hidden size: 288
 * - Attention heads: 8 (36 dims per head)
 * - FFN size: 1152
 * - Encoder layers: 6
 *
 * IMPORTANT: Tensor names have NO "encoder." prefix!
 * - Correct: "conv1.weight", "layers.0.self_attn.q_proj.weight"
 * - Wrong: "encoder.conv1.weight", "encoder.layers.0.self_attn.q_proj.weight"
 */

#include "moonshine.h"
#include "moonshine_ops.h"
#include "moonshine_parallel.h"
#define LOG_LOCAL_LEVEL ESP_LOG_INFO
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <inttypes.h> // For PRIu32
#include <math.h>
#include <string.h>

#define PERF_START(name) int64_t _perf_start_##name = esp_timer_get_time()
#define PERF_END(name, tag)                                                    \
  do {                                                                         \
    int64_t _perf_duration = esp_timer_get_time() - _perf_start_##name;        \
    ESP_LOGI(tag, "[PERF] %s: %lld us", #name, _perf_duration);                \
  } while (0)

static const char *TAG = "moonshine_encoder";

static int64_t s_enc_qkv_us = 0;
static int64_t s_enc_rope_us = 0;
static int64_t s_enc_self_attn_us = 0;
static int64_t s_enc_proj_us = 0;
static int64_t s_enc_fc1_us = 0;
static int64_t s_enc_gelu_us = 0;
static int64_t s_enc_fc2_us = 0;
static int64_t s_enc_norm_us = 0;

/* ========================================================================
 * MEMORY DEBUGGING
 * ======================================================================== */

/**
 * @brief Log detailed memory state for debugging
 */
static void log_memory_state(const char *context) {
  size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  size_t largest_block = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
  size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  float fragmentation =
      (free_psram > 0) ? (1.0f - ((float)largest_block / (float)free_psram))
                       : 0.0f;

  ESP_LOGI(TAG, "[MEM] %s:", context);
  ESP_LOGI(TAG, "  PSRAM free: %zu bytes (%.2f MB)", free_psram,
           free_psram / 1048576.0f);
  ESP_LOGI(TAG, "  Largest block: %zu bytes (%.2f MB)", largest_block,
           largest_block / 1048576.0f);
  ESP_LOGI(TAG, "  Fragmentation: %.1f%%", fragmentation * 100.0f);
  ESP_LOGI(TAG, "  Internal RAM free: %zu bytes (%.2f KB)", free_internal,
           free_internal / 1024.0f);
}

/* ========================================================================
 * HELPER FUNCTIONS
 * ======================================================================== */

/**
 * @brief Find tensor by name hash
 */
static moonshine_tensor_t *find_tensor_by_hash(moonshine_model_t *model,
                                               uint32_t hash) {
  for (uint32_t i = 0; i < model->num_tensors; i++) {
    if (model->tensors[i].name_hash == hash) {
      return &model->tensors[i];
    }
  }
  ESP_LOGW(TAG, "Tensor with hash 0x%08x not found", hash);
  return NULL;
}

/**
 * @brief Dequantize tensor into buffer
 */
static esp_err_t dequantize_tensor(moonshine_tensor_t *tensor, float *buffer) {
  if (tensor->num_channels == 0 || tensor->num_channels == 1) {
    // Scalar or per-tensor quantization
    moonshine_dequantize_int8((const int8_t *)tensor->data, tensor->scales, 1,
                              tensor->num_elements, buffer);
  } else {
    // Per-channel quantization
    uint32_t elements_per_channel = tensor->num_elements / tensor->num_channels;
    moonshine_dequantize_int8((const int8_t *)tensor->data, tensor->scales,
                              tensor->num_channels, elements_per_channel,
                              buffer);
  }
  return ESP_OK;
}

/* ========================================================================
 * CONV1D PREPROCESSING LAYERS
 * ======================================================================== */

/**
 * @brief Apply Conv1d preprocessing layers with sequential loading
 *
 * Architecture (from debug JSON):
 *   Conv1: [1 × 80] → [288 × 80], kernel=127, stride=1, padding=63, NO bias
 *   Conv2: [288 × 80] → [576 × 80], kernel=7, stride=1, padding=3, WITH bias
 *   Conv3: [576 × 80] → [288 × 80], kernel=3, stride=1, padding=1, WITH bias
 *   GroupNorm: 288 channels, num_groups=1 (or 32?)
 *
 * Memory strategy: Sequential loading (load → dequantize → process → free)
 *
 * @param model Model containing weights
 * @param input Normalized waveform samples [num_samples]
 * @param num_samples Number of waveform samples
 * @param temp1 Temp buffer [288 × seq_len1]
 * @param temp2 Temp buffer [576 × seq_len2]
 * @param out_features Output buffer [seq_len3 × 288]
 * @param out_seq_len Output sequence length
 */
static esp_err_t
apply_conv_preprocessing(moonshine_model_t *model, const float *input,
                         uint32_t num_samples, float *output,
                         float *temp1, // conv_out1 [288 × seq_len1]
                         float *temp2, // conv_out2 [576 × seq_len2]
                         uint32_t seq_len1, uint32_t seq_len2,
                         uint32_t final_seq_len) {
  PERF_START(conv_total);
  ESP_LOGI(TAG, "Conv preprocessing: %" PRIu32 " samples", num_samples);
  esp_err_t ret;

  // ========================================================================
  // Conv1: 1 → 288, kernel=127, stride=1, padding=63 (NO BIAS)
  // Input: [1 × num_samples]
  // Weight: [288, 1, 127] = 36,576 elements (142.5 KB)
  // Output: [288 × seq_len1]
  // ========================================================================

  ESP_LOGI(TAG, "Conv1: 1 → 288, kernel=127");

  // Find conv1 weight
  uint32_t conv1_w_hash = moonshine_crc32((const uint8_t *)"conv1.weight", 12);
  moonshine_tensor_t *conv1_w = find_tensor_by_hash(model, conv1_w_hash);

  if (!conv1_w) {
    ESP_LOGE(TAG, "conv1.weight not found");
    return ESP_ERR_NOT_FOUND;
  }

  // Apply conv1d: [1 × num_samples] → [288 × seq_len1]
  // kernel_size = 127, stride = 64, padding = 63, NO BIAS
  moonshine_conv1d_int8(input, // [1 × num_samples]
                        (const int8_t *)conv1_w->data, conv1_w->scales,
                        conv1_w->num_channels,
                        NULL,        // NO BIAS
                        temp1,       // [288 × seq_len1]
                        1,           // in_channels
                        288,         // out_channels
                        num_samples, // sequence length
                        127,         // kernel size
                        64,          // stride
                        0            // padding (HF Moonshine uses no padding)
  );
  ESP_LOGI(TAG, "Conv1 complete");
  ESP_LOGI(TAG, "Conv1[0..4]: %f, %f, %f, %f, %f", temp1[0], temp1[1], temp1[2],
           temp1[3], temp1[4]);

  // Apply TANH activation
  moonshine_tanh(temp1, 288 * seq_len1);
  ESP_LOGI(TAG, "Conv1_act[0..4]: %f, %f, %f, %f, %f", temp1[0], temp1[1],
           temp1[2], temp1[3], temp1[4]);

  // ========================================================================
  // GroupNorm: 288 channels, num_groups=1 (effectively LayerNorm per time step)
  // ========================================================================

  ESP_LOGI(TAG, "Applying GroupNorm");

  // Find groupnorm weight and bias
  uint32_t gn_w_hash = moonshine_crc32((const uint8_t *)"groupnorm.weight", 16);
  uint32_t gn_b_hash = moonshine_crc32((const uint8_t *)"groupnorm.bias", 14);

  moonshine_tensor_t *gn_w = find_tensor_by_hash(model, gn_w_hash);
  moonshine_tensor_t *gn_b = find_tensor_by_hash(model, gn_b_hash);

  if (!gn_w || !gn_b) {
    ESP_LOGE(TAG, "groupnorm weights/bias not found");
    return ESP_ERR_NOT_FOUND;
  }

  // Dequantize groupnorm weight and bias into stack buffers
  float gn_weight[288];
  float gn_bias[288];

  ret = dequantize_tensor(gn_w, gn_weight);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to dequantize groupnorm weight");
    return ret;
  }

  ret = dequantize_tensor(gn_b, gn_bias);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to dequantize groupnorm bias");
    return ret;
  }

  // Debug: print first 5 values of gn_weight and gn_bias
  ESP_LOGI(TAG, "gn_weight[0..4]: %f, %f, %f, %f, %f", gn_weight[0],
           gn_weight[1], gn_weight[2], gn_weight[3], gn_weight[4]);
  ESP_LOGI(TAG, "gn_bias[0..4]: %f, %f, %f, %f, %f", gn_bias[0], gn_bias[1],
           gn_bias[2], gn_bias[3], gn_bias[4]);

  // Apply GroupNorm (num_groups=1 means it's like LayerNorm)
  // Data layout: [288 × num_frames] (channels × spatial)
  moonshine_group_norm(temp1,     // [288 × seq_len1]
                       gn_weight, // [288]
                       gn_bias,   // [288]
                       288,       // num_channels
                       1,         // num_groups (1 = like LayerNorm)
                       seq_len1,  // spatial_size
                       1e-5f      // eps
  );

  // ========================================================================
  // Conv2: 288 → 576, kernel=7, stride=1, padding=3 (WITH BIAS)
  // Input: [seq_len1 × 288] (treat as [288 × seq_len1])
  // Weight: [576, 288, 7] = 1,161,216 elements (4.5 MB!)
  // Bias: [576]
  // Output: [seq_len2 × 576]
  // ========================================================================

  ESP_LOGI(TAG, "Conv2: 288 → 576, kernel=7");

  // Find conv2 weight and bias
  uint32_t conv2_w_hash = moonshine_crc32((const uint8_t *)"conv2.weight", 12);
  uint32_t conv2_b_hash = moonshine_crc32((const uint8_t *)"conv2.bias", 10);

  moonshine_tensor_t *conv2_w = find_tensor_by_hash(model, conv2_w_hash);
  moonshine_tensor_t *conv2_b = find_tensor_by_hash(model, conv2_b_hash);

  if (!conv2_w || !conv2_b) {
    ESP_LOGE(TAG, "conv2 weights/bias not found");
    return ESP_ERR_NOT_FOUND;
  }

  // Dequantize conv2 bias into stack buffer (weights stay compressed)
  float conv2_bias[576];

  ret = dequantize_tensor(conv2_b, conv2_bias);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to dequantize conv2 bias");
    return ret;
  }

  // Apply conv2 with on-the-fly dequantization
  moonshine_conv1d_int8(temp1, // [288 × seq_len1]
                        (const int8_t *)conv2_w->data, conv2_w->scales,
                        conv2_w->num_channels,
                        conv2_bias, // [576]
                        temp2,      // [576 × seq_len2]
                        288,        // in_channels
                        576,        // out_channels
                        seq_len1,   // sequence length
                        7,          // kernel size
                        3,          // stride
                        0           // padding
  );

  ESP_LOGI(TAG, "Conv2 complete");
  ESP_LOGI(TAG, "Conv2[0..4]: %f, %f, %f, %f, %f", temp2[0], temp2[1], temp2[2],
           temp2[3], temp2[4]);

  // Apply GELU activation
  moonshine_gelu(temp2, 576 * seq_len2);

  // ========================================================================
  // Conv3: 576 → 288, kernel=3, stride=1, padding=1 (WITH BIAS)
  // Input: [seq_len2 × 576]
  // Weight: [288, 576, 3] = 497,664 elements (1.9 MB)
  // Bias: [288]
  // Output: [final_seq_len × 288]
  // ========================================================================

  ESP_LOGI(TAG, "Conv3: 576 → 288, kernel=3");

  // Find conv3 weight and bias
  uint32_t conv3_w_hash = moonshine_crc32((const uint8_t *)"conv3.weight", 12);
  uint32_t conv3_b_hash = moonshine_crc32((const uint8_t *)"conv3.bias", 10);

  moonshine_tensor_t *conv3_w = find_tensor_by_hash(model, conv3_w_hash);
  moonshine_tensor_t *conv3_b = find_tensor_by_hash(model, conv3_b_hash);

  if (!conv3_w || !conv3_b) {
    ESP_LOGE(TAG, "conv3 weights/bias not found");
    return ESP_ERR_NOT_FOUND;
  }

  // Dequantize conv3 bias into stack buffer (weights stay compressed)
  float conv3_bias[288];

  ret = dequantize_tensor(conv3_b, conv3_bias);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to dequantize conv3 bias");
    return ret;
  }

  // Apply conv3 with on-the-fly dequantization
  moonshine_conv1d_int8(temp2, // [576 × seq_len2]
                        (const int8_t *)conv3_w->data, conv3_w->scales,
                        conv3_w->num_channels,
                        conv3_bias, // [288]
                        output,     // [288 × final_seq_len]
                        576,        // in_channels
                        288,        // out_channels
                        seq_len2,   // sequence length
                        3,          // kernel size
                        2,          // stride
                        0           // padding (HF Moonshine uses no padding)
  );

  ESP_LOGI(TAG, "Conv3 complete");
  ESP_LOGI(TAG, "Conv3[0..4]: %f, %f, %f, %f, %f", output[0], output[1],
           output[2], output[3], output[4]);

  // Apply GELU activation
  moonshine_gelu(output, 288 * final_seq_len);

  PERF_END(conv_total, TAG);
  ESP_LOGI(TAG, "Conv preprocessing complete");
  return ESP_OK;
}

/* ========================================================================
 * MULTI-HEAD SELF-ATTENTION
 * ======================================================================== */

typedef struct {
  const float *Q;
  const float *K;
  const float *V;
  float *attn_output;
  float *s_row[MOONSHINE_PAR_PARTS];
  float *k_head[MOONSHINE_PAR_PARTS];
  float *v_head[MOONSHINE_PAR_PARTS];
  uint32_t seq_len;
  uint32_t hidden_size;
  uint32_t num_heads;
  uint32_t head_dim;
  float scale;
} encoder_self_attn_job_t;

#ifdef ESP_PLATFORM
static void encoder_self_attn_part(void *ctx, int part, int parts) {
  encoder_self_attn_job_t *job = (encoder_self_attn_job_t *)ctx;
  int h_start, h_end;
  moonshine_par_range((int)job->num_heads, part, parts, 1, &h_start, &h_end);

  const uint32_t seq_len = job->seq_len;
  const uint32_t hidden_size = job->hidden_size;
  const uint32_t head_dim = job->head_dim;
  const float scale = job->scale;
  const float *Q = job->Q;
  const float *K = job->K;
  const float *V = job->V;
  float *attn_output = job->attn_output;
  float *s_row = job->s_row[part];
  float *k_head = job->k_head[part];
  float *v_head = job->v_head[part];

  for (int h = h_start; h < h_end; h++) {
    uint32_t head_offset = h * head_dim;

    // Linearize K and V for head h into contiguous buffer (in internal fast SRAM)
    for (uint32_t j = 0; j < seq_len; j++) {
      memcpy(k_head + j * head_dim, K + j * hidden_size + head_offset, head_dim * sizeof(float));
      memcpy(v_head + j * head_dim, V + j * hidden_size + head_offset, head_dim * sizeof(float));
    }

    for (uint32_t i = 0; i < seq_len; i++) {
      float q_local[64];
      const float *q_src = Q + i * hidden_size + head_offset;
      for (uint32_t d = 0; d < head_dim; d++) {
        q_local[d] = q_src[d] * scale;
      }

      // 1. Compute attention scores: Q @ K^T
      if (head_dim == 36) {
        for (uint32_t j = 0; j < seq_len; j++) {
          const float *k_row = k_head + j * 36;
          float s0 = q_local[0] * k_row[0] + q_local[1] * k_row[1] + q_local[2] * k_row[2] + q_local[3] * k_row[3];
          float s1 = q_local[4] * k_row[4] + q_local[5] * k_row[5] + q_local[6] * k_row[6] + q_local[7] * k_row[7];
          float s2 = q_local[8] * k_row[8] + q_local[9] * k_row[9] + q_local[10] * k_row[10] + q_local[11] * k_row[11];
          float s3 = q_local[12] * k_row[12] + q_local[13] * k_row[13] + q_local[14] * k_row[14] + q_local[15] * k_row[15];
          float s4 = q_local[16] * k_row[16] + q_local[17] * k_row[17] + q_local[18] * k_row[18] + q_local[19] * k_row[19];
          float s5 = q_local[20] * k_row[20] + q_local[21] * k_row[21] + q_local[22] * k_row[22] + q_local[23] * k_row[23];
          float s6 = q_local[24] * k_row[24] + q_local[25] * k_row[25] + q_local[26] * k_row[26] + q_local[27] * k_row[27];
          float s7 = q_local[28] * k_row[28] + q_local[29] * k_row[29] + q_local[30] * k_row[30] + q_local[31] * k_row[31];
          float s8 = q_local[32] * k_row[32] + q_local[33] * k_row[33] + q_local[34] * k_row[34] + q_local[35] * k_row[35];
          s_row[j] = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7) + s8;
        }
      } else {
        for (uint32_t j = 0; j < seq_len; j++) {
          const float *k_row = k_head + j * head_dim;
          float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
          uint32_t d = 0;
          for (; d + 4 <= head_dim; d += 4) {
            s0 += q_local[d] * k_row[d];
            s1 += q_local[d + 1] * k_row[d + 1];
            s2 += q_local[d + 2] * k_row[d + 2];
            s3 += q_local[d + 3] * k_row[d + 3];
          }
          float score = (s0 + s1) + (s2 + s3);
          for (; d < head_dim; d++) {
            score += q_local[d] * k_row[d];
          }
          s_row[j] = score;
        }
      }

      // 2. Softmax in L1 cache
      moonshine_softmax(s_row, seq_len);

      // 3. Accumulate V in L1 stack buffer and write out_ptr once
      float accum[64] = {0};
      if (head_dim == 36) {
        for (uint32_t j = 0; j < seq_len; j++) {
          float s = s_row[j];
          if (s <= 1e-7f)
            continue;
          const float *v_row = v_head + j * 36;
          accum[0]  += s * v_row[0];  accum[1]  += s * v_row[1];
          accum[2]  += s * v_row[2];  accum[3]  += s * v_row[3];
          accum[4]  += s * v_row[4];  accum[5]  += s * v_row[5];
          accum[6]  += s * v_row[6];  accum[7]  += s * v_row[7];
          accum[8]  += s * v_row[8];  accum[9]  += s * v_row[9];
          accum[10] += s * v_row[10]; accum[11] += s * v_row[11];
          accum[12] += s * v_row[12]; accum[13] += s * v_row[13];
          accum[14] += s * v_row[14]; accum[15] += s * v_row[15];
          accum[16] += s * v_row[16]; accum[17] += s * v_row[17];
          accum[18] += s * v_row[18]; accum[19] += s * v_row[19];
          accum[20] += s * v_row[20]; accum[21] += s * v_row[21];
          accum[22] += s * v_row[22]; accum[23] += s * v_row[23];
          accum[24] += s * v_row[24]; accum[25] += s * v_row[25];
          accum[26] += s * v_row[26]; accum[27] += s * v_row[27];
          accum[28] += s * v_row[28]; accum[29] += s * v_row[29];
          accum[30] += s * v_row[30]; accum[31] += s * v_row[31];
          accum[32] += s * v_row[32]; accum[33] += s * v_row[33];
          accum[34] += s * v_row[34]; accum[35] += s * v_row[35];
        }
      } else {
        for (uint32_t j = 0; j < seq_len; j++) {
          float s = s_row[j];
          if (s <= 1e-7f)
            continue;
          const float *v_row = v_head + j * head_dim;
          uint32_t d = 0;
          for (; d + 4 <= head_dim; d += 4) {
            accum[d] += s * v_row[d];
            accum[d + 1] += s * v_row[d + 1];
            accum[d + 2] += s * v_row[d + 2];
            accum[d + 3] += s * v_row[d + 3];
          }
          for (; d < head_dim; d++) {
            accum[d] += s * v_row[d];
          }
        }
      }

      float *out_ptr = attn_output + i * hidden_size + head_offset;
      memcpy(out_ptr, accum, head_dim * sizeof(float));
    }
  }
}
#endif

/**
 * @brief Multi-head self-attention layer
 *
 * @param input Input [seq_len × hidden_size]
 * @param q_weight Query weight [hidden_size × hidden_size]
 * @param k_weight Key weight [hidden_size × hidden_size]
 * @param v_weight Value weight [hidden_size × hidden_size]
 * @param out_weight Output projection weight [hidden_size × hidden_size]
 * @param q_bias Query bias [hidden_size] (can be NULL)
 * @param k_bias Key bias [hidden_size] (can be NULL)
 * @param v_bias Value bias [hidden_size] (can be NULL)
 * @param out_bias Output bias [hidden_size] (can be NULL)
 * @param output Output [seq_len × hidden_size]
 * @param seq_len Sequence length
 * @param hidden_size Hidden dimension
 * @param num_heads Number of attention heads
 * @param temp_buffers Temporary buffers (Q, K, V, s_row, attn_output)
 */
static esp_err_t multi_head_self_attention(
    const float *input, const moonshine_tensor_t *q_weight,
    const moonshine_tensor_t *k_weight, const moonshine_tensor_t *v_weight,
    const moonshine_tensor_t *out_weight, const float *q_bias,
    const float *k_bias, const float *v_bias, const float *out_bias,
    float *output, uint32_t seq_len, uint32_t hidden_size, uint32_t num_heads,
    float *Q, // [seq_len × hidden_size]
    float *K, // [seq_len × hidden_size]
    float *V, // [seq_len × hidden_size]
    float *s_row_parts[MOONSHINE_PAR_PARTS],
    float *k_head_parts[MOONSHINE_PAR_PARTS],
    float *v_head_parts[MOONSHINE_PAR_PARTS],
    float *attn_output, // [seq_len × hidden_size]
    int8_t *in_q, float *in_scales) {
  uint32_t head_dim = hidden_size / num_heads;
  float scale = 1.0f / sqrtf((float)head_dim);

  // Q, K, V projections: quantize input ONCE into fast internal memory and
  // reuse across Q, K, V
  int64_t t_qkv0 = esp_timer_get_time();
  if (in_q && in_scales) {
    moonshine_quantize_int8_rows(input, in_q, in_scales, seq_len, hidden_size);
    moonshine_gemm_int8_prequantized(
        in_q, in_scales, (const int8_t *)q_weight->data, q_weight->scales,
        q_weight->num_channels, Q, seq_len, hidden_size, hidden_size);
    moonshine_gemm_int8_prequantized(
        in_q, in_scales, (const int8_t *)k_weight->data, k_weight->scales,
        k_weight->num_channels, K, seq_len, hidden_size, hidden_size);
    moonshine_gemm_int8_prequantized(
        in_q, in_scales, (const int8_t *)v_weight->data, v_weight->scales,
        v_weight->num_channels, V, seq_len, hidden_size, hidden_size);
  } else {
    moonshine_gemm_int8(input, (const int8_t *)q_weight->data, q_weight->scales,
                        q_weight->num_channels, Q, seq_len, hidden_size,
                        hidden_size, true);
    moonshine_gemm_int8(input, (const int8_t *)k_weight->data, k_weight->scales,
                        k_weight->num_channels, K, seq_len, hidden_size,
                        hidden_size, true);
    moonshine_gemm_int8(input, (const int8_t *)v_weight->data, v_weight->scales,
                        v_weight->num_channels, V, seq_len, hidden_size,
                        hidden_size, true);
  }

  // Add biases if present
  if (q_bias) {
    for (uint32_t i = 0; i < seq_len; i++) {
      moonshine_add_inplace(Q + i * hidden_size, q_bias, hidden_size);
    }
  }
  if (k_bias) {
    for (uint32_t i = 0; i < seq_len; i++) {
      moonshine_add_inplace(K + i * hidden_size, k_bias, hidden_size);
    }
  }
  if (v_bias) {
    for (uint32_t i = 0; i < seq_len; i++) {
      moonshine_add_inplace(V + i * hidden_size, v_bias, hidden_size);
    }
  }
  s_enc_qkv_us += (esp_timer_get_time() - t_qkv0);

  // Apply Rotary Positional Embeddings (RoPE) to Q and K
  int64_t t_rope0 = esp_timer_get_time();
  moonshine_apply_rope(Q, seq_len, hidden_size, num_heads, 10000.0f);
  moonshine_apply_rope(K, seq_len, hidden_size, num_heads, 10000.0f);
  s_enc_rope_us += (esp_timer_get_time() - t_rope0);

  // Compute attention for each head
  int64_t t_sa0 = esp_timer_get_time();
#ifdef ESP_PLATFORM
  encoder_self_attn_job_t job = {.Q = Q,
                                 .K = K,
                                 .V = V,
                                 .attn_output = attn_output,
                                 .s_row = {s_row_parts[0], s_row_parts[1]},
                                 .k_head = {k_head_parts[0], k_head_parts[1]},
                                 .v_head = {v_head_parts[0], v_head_parts[1]},
                                 .seq_len = seq_len,
                                 .hidden_size = hidden_size,
                                 .num_heads = num_heads,
                                 .head_dim = head_dim,
                                 .scale = scale};
  moonshine_parallel_run(encoder_self_attn_part, &job);
#else
  for (uint32_t h = 0; h < num_heads; h++) {
    uint32_t head_offset = h * head_dim;
    float *s_row = s_row_parts[0];
    for (uint32_t i = 0; i < seq_len; i++) {
      const float *q_row = Q + i * hidden_size + head_offset;
      for (uint32_t j = 0; j < seq_len; j++) {
        const float *k_row = K + j * hidden_size + head_offset;
        float score = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
          score += q_row[d] * k_row[d];
        }
        s_row[j] = score * scale;
      }
      moonshine_softmax(s_row, seq_len);
      float accum[64] = {0};
      for (uint32_t j = 0; j < seq_len; j++) {
        float s = s_row[j];
        const float *v_row = V + j * hidden_size + head_offset;
        for (uint32_t d = 0; d < head_dim; d++) {
          accum[d] += s * v_row[d];
        }
      }
      float *out_ptr = attn_output + i * hidden_size + head_offset;
      memcpy(out_ptr, accum, head_dim * sizeof(float));
    }
  }
#endif
  s_enc_self_attn_us += (esp_timer_get_time() - t_sa0);

  // Output projection
  int64_t t_proj0 = esp_timer_get_time();
  if (in_q && in_scales) {
    moonshine_quantize_int8_rows(attn_output, in_q, in_scales, seq_len,
                                 hidden_size);
    moonshine_gemm_int8_prequantized(
        in_q, in_scales, (const int8_t *)out_weight->data, out_weight->scales,
        out_weight->num_channels, output, seq_len, hidden_size, hidden_size);
  } else {
    moonshine_gemm_int8(attn_output, (const int8_t *)out_weight->data,
                        out_weight->scales, out_weight->num_channels, output,
                        seq_len, hidden_size, hidden_size, true);
  }

  if (out_bias) {
    for (uint32_t i = 0; i < seq_len; i++) {
      moonshine_add_inplace(output + i * hidden_size, out_bias, hidden_size);
    }
  }
  s_enc_proj_us += (esp_timer_get_time() - t_proj0);

  return ESP_OK;
}

/* ========================================================================
 * FEED-FORWARD NETWORK
 * ======================================================================== */

static esp_err_t feed_forward_network(const float *input,
                                      const moonshine_tensor_t *fc1_weight,
                                      const float *fc1_bias,
                                      const moonshine_tensor_t *fc2_weight,
                                      const float *fc2_bias, float *output,
                                      uint32_t seq_len, uint32_t hidden_size,
                                      uint32_t ffn_size, float *temp,
                                      int8_t *in_q, float *in_scales) {
  // First linear layer with INT8 on-the-fly dequantization
  int64_t t_fc1_0 = esp_timer_get_time();
  if (in_q && in_scales) {
    moonshine_quantize_int8_rows(input, in_q, in_scales, seq_len, hidden_size);
    moonshine_gemm_int8_prequantized(
        in_q, in_scales, (const int8_t *)fc1_weight->data, fc1_weight->scales,
        fc1_weight->num_channels, temp, seq_len, ffn_size, hidden_size);
  } else {
    moonshine_gemm_int8(input, (const int8_t *)fc1_weight->data,
                        fc1_weight->scales, fc1_weight->num_channels, temp,
                        seq_len, ffn_size, hidden_size, true);
  }

  // Add bias
  if (fc1_bias) {
    for (uint32_t i = 0; i < seq_len; i++) {
      moonshine_add_inplace(temp + i * ffn_size, fc1_bias, ffn_size);
    }
  }
  s_enc_fc1_us += (esp_timer_get_time() - t_fc1_0);

  // GELU activation
  int64_t t_gelu0 = esp_timer_get_time();
  moonshine_gelu(temp, seq_len * ffn_size);
  s_enc_gelu_us += (esp_timer_get_time() - t_gelu0);

  // Second linear layer with INT8 prequantized
  int64_t t_fc2_0 = esp_timer_get_time();
  if (in_q && in_scales) {
    moonshine_quantize_int8_rows(temp, in_q, in_scales, seq_len, ffn_size);
    moonshine_gemm_int8_prequantized(
        in_q, in_scales, (const int8_t *)fc2_weight->data, fc2_weight->scales,
        fc2_weight->num_channels, output, seq_len, hidden_size, ffn_size);
  } else {
    moonshine_gemm_int8(temp, (const int8_t *)fc2_weight->data,
                        fc2_weight->scales, fc2_weight->num_channels, output,
                        seq_len, hidden_size, ffn_size, true);
  }

  // Add bias
  if (fc2_bias) {
    for (uint32_t i = 0; i < seq_len; i++) {
      moonshine_add_inplace(output + i * hidden_size, fc2_bias, hidden_size);
    }
  }
  s_enc_fc2_us += (esp_timer_get_time() - t_fc2_0);

  return ESP_OK;
}

/* ========================================================================
 * ENCODER LAYER & WORKSPACE
 * ======================================================================== */

typedef struct {
  float *residual;
  float *attn_final;
  float *attn_output;
  float *Q;
  float *K;
  float *V;
  float *s_row[MOONSHINE_PAR_PARTS];
  float *k_head[MOONSHINE_PAR_PARTS];
  float *v_head[MOONSHINE_PAR_PARTS];
  float *ffn_temp;
  float *ffn_output;
  int8_t *in_q_raw;
  int8_t *in_q;
  float *in_scales;
} encoder_workspace_t;

typedef struct {
  moonshine_tensor_t *q_w;
  moonshine_tensor_t *k_w;
  moonshine_tensor_t *v_w;
  moonshine_tensor_t *out_w;
  moonshine_tensor_t *fc1_w;
  moonshine_tensor_t *fc2_w;
  float *fc1_b;
  float *fc2_b;
  float *ln1_w;
  float *ln2_w;
} encoder_layer_weights_t;

static float *load_dequantize_bias(moonshine_model_t *model, const char *name) {
  uint32_t hash = moonshine_crc32((const uint8_t *)name, strlen(name));
  moonshine_tensor_t *tensor = find_tensor_by_hash(model, hash);
  if (!tensor)
    return NULL;

  float *bias =
      heap_caps_malloc(tensor->num_elements * sizeof(float), MALLOC_CAP_SPIRAM);
  if (!bias)
    return NULL;

  if (dequantize_tensor(tensor, bias) != ESP_OK) {
    free(bias);
    return NULL;
  }
  return bias;
}

static moonshine_tensor_t *get_tensor(moonshine_model_t *model,
                                      const char *name) {
  return find_tensor_by_hash(
      model, moonshine_crc32((const uint8_t *)name, strlen(name)));
}

typedef struct {
  float *data;
  const float *weight;
  uint32_t hidden_size;
  uint32_t seq_len;
} enc_ln_job_t;

#ifdef ESP_PLATFORM
static void enc_ln_part(void *ctx, int part, int parts) {
  enc_ln_job_t *job = (enc_ln_job_t *)ctx;
  int i_start, i_end;
  moonshine_par_range((int)job->seq_len, part, parts, 1, &i_start, &i_end);
  const uint32_t h = job->hidden_size;
  for (int i = i_start; i < i_end; i++) {
    moonshine_layer_norm(job->data + (size_t)i * h, job->weight, NULL, h, 1e-5f);
  }
}
#endif

static esp_err_t encoder_layer(const encoder_layer_weights_t *lw,
                               encoder_workspace_t *ws, const float *input,
                               float *output, uint32_t seq_len,
                               uint32_t hidden_size, uint32_t num_heads,
                               uint32_t ffn_size) {
  int64_t t_n0 = esp_timer_get_time();
  // Pre-LN: attn_final = input (avoiding redundant residual copy)
  moonshine_copy(ws->attn_final, input, seq_len * hidden_size);

  // Apply layer norm (RMSNorm) across dual cores
#ifdef ESP_PLATFORM
  enc_ln_job_t ln1_job = {.data = ws->attn_final,
                          .weight = lw->ln1_w,
                          .hidden_size = hidden_size,
                          .seq_len = seq_len};
  moonshine_parallel_run(enc_ln_part, &ln1_job);
#else
  for (uint32_t i = 0; i < seq_len; i++) {
    moonshine_layer_norm(ws->attn_final + i * hidden_size, lw->ln1_w, NULL,
                         hidden_size, 1e-5f);
  }
#endif
  s_enc_norm_us += (esp_timer_get_time() - t_n0);

  // Self-attention
  esp_err_t ret = multi_head_self_attention(
      ws->attn_final, lw->q_w, lw->k_w, lw->v_w, lw->out_w, NULL, NULL, NULL,
      NULL, ws->attn_output, seq_len, hidden_size, num_heads, ws->Q, ws->K,
      ws->V, ws->s_row, ws->k_head, ws->v_head, ws->attn_output, ws->in_q, ws->in_scales);
  if (ret != ESP_OK)
    return ret;

  int64_t t_n1 = esp_timer_get_time();
  // Add residual: residual = input + attn_output (single pass, zero redundant copy)
  moonshine_add(input, ws->attn_output, ws->residual, seq_len * hidden_size);

  // Pre-LN for FFN
  moonshine_copy(ws->attn_final, ws->residual, seq_len * hidden_size);
#ifdef ESP_PLATFORM
  enc_ln_job_t ln2_job = {.data = ws->attn_final,
                          .weight = lw->ln2_w,
                          .hidden_size = hidden_size,
                          .seq_len = seq_len};
  moonshine_parallel_run(enc_ln_part, &ln2_job);
#else
  for (uint32_t i = 0; i < seq_len; i++) {
    moonshine_layer_norm(ws->attn_final + i * hidden_size, lw->ln2_w, NULL,
                         hidden_size, 1e-5f);
  }
#endif
  s_enc_norm_us += (esp_timer_get_time() - t_n1);

  // FFN
  ret = feed_forward_network(ws->attn_final, lw->fc1_w, lw->fc1_b, lw->fc2_w,
                             lw->fc2_b, ws->ffn_output, seq_len, hidden_size,
                             ffn_size, ws->ffn_temp, ws->in_q, ws->in_scales);
  if (ret != ESP_OK)
    return ret;

  int64_t t_n2 = esp_timer_get_time();
  // Add residual: output = residual + ffn_output (single pass, zero redundant copy)
  moonshine_add(ws->residual, ws->ffn_output, output, seq_len * hidden_size);
  s_enc_norm_us += (esp_timer_get_time() - t_n2);

  return ESP_OK;
}

static esp_err_t init_encoder_workspace(encoder_workspace_t *ws,
                                        uint32_t seq_len, uint32_t hidden_size,
                                        uint32_t num_heads, uint32_t ffn_size) {
  memset(ws, 0, sizeof(encoder_workspace_t));
  ws->residual = heap_caps_malloc(seq_len * hidden_size * sizeof(float),
                                  MALLOC_CAP_SPIRAM);
  ws->attn_final = heap_caps_malloc(seq_len * hidden_size * sizeof(float),
                                    MALLOC_CAP_SPIRAM);
  ws->attn_output = heap_caps_malloc(seq_len * hidden_size * sizeof(float),
                                     MALLOC_CAP_SPIRAM);
  ws->Q = heap_caps_malloc(seq_len * hidden_size * sizeof(float),
                           MALLOC_CAP_SPIRAM);
  ws->K = heap_caps_malloc(seq_len * hidden_size * sizeof(float),
                           MALLOC_CAP_SPIRAM);
  ws->V = heap_caps_malloc(seq_len * hidden_size * sizeof(float),
                           MALLOC_CAP_SPIRAM);

  // Top Priority for fast Internal SRAM: k_head, v_head, and s_row
  uint32_t head_dim = hidden_size / num_heads;
  for (int p = 0; p < MOONSHINE_PAR_PARTS; p++) {
    ws->s_row[p] = (float *)heap_caps_malloc(
        seq_len * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!ws->s_row[p]) {
      ws->s_row[p] =
          (float *)heap_caps_malloc(seq_len * sizeof(float), MALLOC_CAP_SPIRAM);
    }
    ws->k_head[p] = (float *)heap_caps_malloc(
        seq_len * head_dim * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!ws->k_head[p]) {
      ws->k_head[p] = (float *)heap_caps_malloc(
          seq_len * head_dim * sizeof(float), MALLOC_CAP_SPIRAM);
    }
    ws->v_head[p] = (float *)heap_caps_malloc(
        seq_len * head_dim * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!ws->v_head[p]) {
      ws->v_head[p] = (float *)heap_caps_malloc(
          seq_len * head_dim * sizeof(float), MALLOC_CAP_SPIRAM);
    }
  }

  ws->ffn_temp =
      heap_caps_malloc(seq_len * ffn_size * sizeof(float), MALLOC_CAP_SPIRAM);
  ws->ffn_output = heap_caps_malloc(seq_len * hidden_size * sizeof(float),
                                    MALLOC_CAP_SPIRAM);

  // in_q and in_scales
  size_t max_row_bytes = (ffn_size > hidden_size) ? ffn_size : hidden_size;
  ws->in_q_raw = (int8_t *)heap_caps_malloc(
      seq_len * max_row_bytes + 16, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!ws->in_q_raw) {
    ws->in_q_raw = (int8_t *)heap_caps_malloc(
        seq_len * max_row_bytes + 16, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
  }
  ws->in_q =
      ws->in_q_raw ? (int8_t *)(((uintptr_t)ws->in_q_raw + 15) & ~15) : NULL;

  ws->in_scales = (float *)heap_caps_malloc(
      seq_len * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!ws->in_scales) {
    ws->in_scales = (float *)heap_caps_malloc(
        seq_len * sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
  }

  if (!ws->residual || !ws->attn_final || !ws->attn_output || !ws->Q ||
      !ws->K || !ws->V || !ws->s_row[0] || !ws->s_row[1] ||
      !ws->k_head[0] || !ws->k_head[1] || !ws->v_head[0] || !ws->v_head[1] ||
      !ws->ffn_temp || !ws->ffn_output || !ws->in_q || !ws->in_scales) {
    return ESP_ERR_NO_MEM;
  }
  return ESP_OK;
}

static void free_encoder_workspace(encoder_workspace_t *ws) {
  if (ws->residual)
    free(ws->residual);
  if (ws->attn_final)
    free(ws->attn_final);
  if (ws->attn_output)
    free(ws->attn_output);
  if (ws->Q)
    free(ws->Q);
  if (ws->K)
    free(ws->K);
  if (ws->V)
    free(ws->V);
  for (int p = 0; p < MOONSHINE_PAR_PARTS; p++) {
    if (ws->s_row[p])
      free(ws->s_row[p]);
    if (ws->k_head[p])
      free(ws->k_head[p]);
    if (ws->v_head[p])
      free(ws->v_head[p]);
  }
  if (ws->ffn_temp)
    free(ws->ffn_temp);
  if (ws->ffn_output)
    free(ws->ffn_output);
  if (ws->in_q_raw)
    free(ws->in_q_raw);
  if (ws->in_scales)
    free(ws->in_scales);
  memset(ws, 0, sizeof(encoder_workspace_t));
}

static esp_err_t preload_encoder_weights(moonshine_model_t *model,
                                         uint32_t num_layers,
                                         encoder_layer_weights_t *layers) {
  char name_buf[64];
  for (uint32_t l = 0; l < num_layers; l++) {
    encoder_layer_weights_t *lw = &layers[l];
    memset(lw, 0, sizeof(encoder_layer_weights_t));

    snprintf(name_buf, sizeof(name_buf),
             "layers.%" PRIu32 ".self_attn.q_proj.weight", l);
    lw->q_w = get_tensor(model, name_buf);
    snprintf(name_buf, sizeof(name_buf),
             "layers.%" PRIu32 ".self_attn.k_proj.weight", l);
    lw->k_w = get_tensor(model, name_buf);
    snprintf(name_buf, sizeof(name_buf),
             "layers.%" PRIu32 ".self_attn.v_proj.weight", l);
    lw->v_w = get_tensor(model, name_buf);
    snprintf(name_buf, sizeof(name_buf),
             "layers.%" PRIu32 ".self_attn.o_proj.weight", l);
    lw->out_w = get_tensor(model, name_buf);

    snprintf(name_buf, sizeof(name_buf), "layers.%" PRIu32 ".mlp.fc1.weight",
             l);
    lw->fc1_w = get_tensor(model, name_buf);
    snprintf(name_buf, sizeof(name_buf), "layers.%" PRIu32 ".mlp.fc2.weight",
             l);
    lw->fc2_w = get_tensor(model, name_buf);

    if (!lw->q_w || !lw->k_w || !lw->v_w || !lw->out_w || !lw->fc1_w ||
        !lw->fc2_w) {
      ESP_LOGE(TAG, "Missing weights in layer %" PRIu32, l);
      return ESP_ERR_NOT_FOUND;
    }

    snprintf(name_buf, sizeof(name_buf), "layers.%" PRIu32 ".mlp.fc1.bias", l);
    lw->fc1_b = load_dequantize_bias(model, name_buf);
    snprintf(name_buf, sizeof(name_buf), "layers.%" PRIu32 ".mlp.fc2.bias", l);
    lw->fc2_b = load_dequantize_bias(model, name_buf);

    snprintf(name_buf, sizeof(name_buf),
             "layers.%" PRIu32 ".input_layernorm.weight", l);
    lw->ln1_w = load_dequantize_bias(model, name_buf);
    snprintf(name_buf, sizeof(name_buf),
             "layers.%" PRIu32 ".post_attention_layernorm.weight", l);
    lw->ln2_w = load_dequantize_bias(model, name_buf);

    if (!lw->ln1_w || !lw->ln2_w) {
      ESP_LOGE(TAG, "Missing layernorm in layer %" PRIu32, l);
      return ESP_ERR_NOT_FOUND;
    }
  }
  return ESP_OK;
}

static void free_encoder_weights(uint32_t num_layers,
                                 encoder_layer_weights_t *layers) {
  for (uint32_t l = 0; l < num_layers; l++) {
    encoder_layer_weights_t *lw = &layers[l];
    if (lw->fc1_b)
      free(lw->fc1_b);
    if (lw->fc2_b)
      free(lw->fc2_b);
    if (lw->ln1_w)
      free(lw->ln1_w);
    if (lw->ln2_w)
      free(lw->ln2_w);
  }
}

/* ========================================================================
 * MAIN ENCODER FUNCTION
 * ======================================================================== */

esp_err_t moonshine_encode(moonshine_context_t *context, const float *features,
                           uint32_t num_samples, float *output) {
  if (!context || !context->initialized || !features || !output) {
    ESP_LOGE(TAG, "Invalid parameters");
    return ESP_ERR_INVALID_ARG;
  }
  if (num_samples < MOONSHINE_MIN_AUDIO_SAMPLES) {
    ESP_LOGE(TAG, "Audio is too short: %" PRIu32 " samples; need at least %u",
             num_samples, (unsigned)MOONSHINE_MIN_AUDIO_SAMPLES);
    return ESP_ERR_INVALID_SIZE;
  }

  moonshine_model_t *model = context->model;
  uint32_t hidden_size = model->header.hidden_size;
  uint32_t num_heads = model->header.attention_heads;
  uint32_t ffn_size = model->header.ffn_size;
  uint32_t num_layers = model->header.encoder_layers;

  ESP_LOGI(TAG, "Starting encoder inference: %" PRIu32 " samples", num_samples);
  ESP_LOGI(TAG,
           "Model: hidden=%" PRIu32 ", heads=%" PRIu32 ", ffn=%" PRIu32
           ", layers=%" PRIu32 "",
           hidden_size, num_heads, ffn_size, num_layers);

  log_memory_state("Before encoder - initial state");

  // ===========================================================================
  // PHASE 1: CONV PREPROCESSING
  // Allocate only what's needed for conv, then free immediately
  // ===========================================================================
  // Calculate sequence lengths for each conv layer
  uint32_t seq_len1 = ((num_samples - 127) / 64) + 1;
  uint32_t seq_len2 = ((seq_len1 - 7) / 3) + 1;
  uint32_t seq_len3 = ((seq_len2 - 3) / 2) + 1;

  // The final encoder sequence length is seq_len3
  uint32_t final_seq_len = seq_len3;

  // ===========================================================================
  // PHASE 1: CONV PREPROCESSING
  // ===========================================================================

  // We need buffers for the intermediate conv outputs
  // Max size needed is after Conv1: 288 * seq_len1
  // Conv2 output is 576 * seq_len2, Conv3 output is 288 * seq_len3

  float *conv_out1 =
      heap_caps_malloc(288 * seq_len1 * sizeof(float), MALLOC_CAP_SPIRAM);
  float *conv_out2 =
      heap_caps_malloc(576 * seq_len2 * sizeof(float), MALLOC_CAP_SPIRAM);
  float *layer_input = heap_caps_malloc(
      final_seq_len * hidden_size * sizeof(float), MALLOC_CAP_SPIRAM);

  if (!conv_out1 || !conv_out2 || !layer_input) {
    ESP_LOGE(TAG, "Failed to allocate conv preprocessing buffers");
    if (conv_out1)
      free(conv_out1);
    if (conv_out2)
      free(conv_out2);
    if (layer_input)
      free(layer_input);
    return ESP_ERR_NO_MEM;
  }

  // Step 1: Conv preprocessing
  esp_err_t ret = apply_conv_preprocessing(model, features, num_samples,
                                           layer_input, conv_out1, conv_out2,
                                           seq_len1, seq_len2, final_seq_len);

  // FREE CONV BUFFERS IMMEDIATELY
  free(conv_out1);
  free(conv_out2);
  conv_out1 = NULL;
  conv_out2 = NULL;

  log_memory_state("After conv preprocessing (freed conv buffers)");

  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Conv preprocessing failed");
    free(layer_input);
    return ret;
  }

  // Debug output: print first 10 values of layer_input (Conv output) BEFORE
  // transpose
  ESP_LOGI(TAG, "First 10 values of Conv preprocessing BEFORE transpose:");
  for (int i = 0; i < 10; i++) {
    ESP_LOGI(TAG, "  pre_enc[%d] = %f", i, layer_input[i]);
  }

  // ===========================================================================
  // TRANSPOSE LAYER INPUT: [hidden_size × final_seq_len] -> [final_seq_len ×
  // hidden_size]
  // ===========================================================================
  float *temp_transpose = heap_caps_malloc(
      hidden_size * final_seq_len * sizeof(float), MALLOC_CAP_SPIRAM);
  if (temp_transpose) {
    for (uint32_t c = 0; c < hidden_size; c++) {
      for (uint32_t t = 0; t < final_seq_len; t++) {
        temp_transpose[t * hidden_size + c] =
            layer_input[c * final_seq_len + t];
      }
    }
    memcpy(layer_input, temp_transpose,
           hidden_size * final_seq_len * sizeof(float));
    free(temp_transpose);
    ESP_LOGI(TAG, "layer_input transposed to [seq_len x hidden_size]");
  } else {
    ESP_LOGE(TAG, "Failed to allocate memory for transpose");
    free(layer_input);
    return ESP_ERR_NO_MEM;
  }

  // Debug output: print first 10 values of layer_input AFTER transpose
  ESP_LOGI(TAG, "First 10 values of Conv preprocessing AFTER transpose:");
  for (int i = 0; i < 10; i++) {
    ESP_LOGI(TAG, "  pre_enc_t[%d] = %f", i, layer_input[i]);
  }

  // ===========================================================================
  // PHASE 2: TRANSFORMER LAYERS
  // Now allocate transformer buffers - conv buffers already freed
  // ===========================================================================

  uint32_t num_frames = final_seq_len;

  float *layer_output = heap_caps_malloc(
      num_frames * hidden_size * sizeof(float), MALLOC_CAP_SPIRAM);

  if (!layer_output) {
    ESP_LOGE(TAG, "Failed to allocate layer_output buffer");
    if (layer_input)
      free(layer_input);
    return ESP_ERR_NO_MEM;
  }

  log_memory_state("After allocating layer_output buffer");

  // Step 2: Preload encoder weights and initialize persistent transformer
  // workspace
  encoder_layer_weights_t *layer_weights =
      (encoder_layer_weights_t *)heap_caps_malloc(
          num_layers * sizeof(encoder_layer_weights_t), MALLOC_CAP_SPIRAM);
  encoder_workspace_t ws;
  memset(&ws, 0, sizeof(encoder_workspace_t));

  if (!layer_weights) {
    ESP_LOGE(TAG, "Failed to allocate layer_weights array");
    free(layer_input);
    free(layer_output);
    return ESP_ERR_NO_MEM;
  }

  ret = preload_encoder_weights(model, num_layers, layer_weights);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to preload encoder weights");
    free_encoder_weights(num_layers, layer_weights);
    free(layer_weights);
    free(layer_input);
    free(layer_output);
    return ret;
  }

  ret =
      init_encoder_workspace(&ws, num_frames, hidden_size, num_heads, ffn_size);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to initialize persistent encoder workspace");
    free_encoder_workspace(&ws);
    free_encoder_weights(num_layers, layer_weights);
    free(layer_weights);
    free(layer_input);
    free(layer_output);
    return ret;
  }

  // Step 3: Encoder layers
  s_enc_qkv_us = 0;
  s_enc_rope_us = 0;
  s_enc_self_attn_us = 0;
  s_enc_proj_us = 0;
  s_enc_fc1_us = 0;
  s_enc_gelu_us = 0;
  s_enc_fc2_us = 0;
  s_enc_norm_us = 0;
  for (uint32_t layer = 0; layer < num_layers; layer++) {
    ret = encoder_layer(&layer_weights[layer], &ws, layer_input, layer_output,
                        num_frames, hidden_size, num_heads, ffn_size);

    if (ret != ESP_OK) {
      ESP_LOGE(TAG, "Encoder layer %" PRIu32 " failed", layer);
      goto cleanup;
    }

    // Yield between layers to feed the IDLE watchdog
    vTaskDelay(1);

    // Swap buffers for next layer
    float *temp = layer_input;
    layer_input = layer_output;
    layer_output = temp;
  }

  // Copy final output
  moonshine_copy(output, layer_input, num_frames * hidden_size);

  // Apply final layer norm (layer_norm.weight)
  float *final_ln_w = load_dequantize_bias(model, "layer_norm.weight");
  if (final_ln_w) {
    for (uint32_t i = 0; i < num_frames; i++) {
      moonshine_layer_norm(output + i * hidden_size, final_ln_w, NULL,
                           hidden_size, 1e-5f);
    }
    free(final_ln_w);
    ESP_LOGI(TAG, "Final layer norm applied");
  } else {
    ESP_LOGW(TAG, "Final layer_norm.weight not found, skipping");
  }

  ESP_LOGI(TAG,
           "Encoder profiling (total over %" PRIu32 " layers):", num_layers);
  ESP_LOGI(TAG, "  Norm/Resid : %lld us", s_enc_norm_us);
  ESP_LOGI(TAG, "  QKV Proj   : %lld us", s_enc_qkv_us);
  ESP_LOGI(TAG, "  RoPE       : %lld us", s_enc_rope_us);
  ESP_LOGI(TAG, "  Self-Attn  : %lld us", s_enc_self_attn_us);
  ESP_LOGI(TAG, "  Out Proj   : %lld us", s_enc_proj_us);
  ESP_LOGI(TAG, "  FFN fc1    : %lld us", s_enc_fc1_us);
  ESP_LOGI(TAG, "  FFN GELU   : %lld us", s_enc_gelu_us);
  ESP_LOGI(TAG, "  FFN fc2    : %lld us", s_enc_fc2_us);

  ESP_LOGI(TAG, "Encoder inference complete");
  ret = ESP_OK;

cleanup:
  free_encoder_workspace(&ws);
  free_encoder_weights(num_layers, layer_weights);
  free(layer_weights);
  free(layer_input);
  free(layer_output);

  log_memory_state("After encoder cleanup");

  return ret;
}
