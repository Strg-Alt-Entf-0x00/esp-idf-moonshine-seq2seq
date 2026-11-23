/**
 * @file moonshine_decoder.c
 * @brief Moonshine Decoder implementation
 */

#include "moonshine.h"
#include "moonshine_ops.h"
#include "moonshine_parallel.h"
#define LOG_LOCAL_LEVEL ESP_LOG_INFO
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <math.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "moonshine_decoder";

// Helper to find a tensor by hash
static moonshine_tensor_t *find_tensor_by_hash(moonshine_model_t *model,
                                               uint32_t hash) {
  for (uint32_t i = 0; i < model->num_tensors; i++) {
    if (model->tensors[i].name_hash == hash) {
      return &model->tensors[i];
    }
  }
  return NULL;
}

// Helper to dequantize a full tensor (used for biases/layernorms)
static esp_err_t dequantize_tensor(moonshine_tensor_t *tensor, float *output) {
  if (!tensor || !output)
    return ESP_ERR_INVALID_ARG;

  if (tensor->num_channels == 0 || tensor->num_channels == 1) {
    moonshine_dequantize_int8((const int8_t *)tensor->data, tensor->scales, 1,
                              tensor->num_elements, output);
  } else {
    uint32_t elements_per_channel = tensor->num_elements / tensor->num_channels;
    moonshine_dequantize_int8((const int8_t *)tensor->data, tensor->scales,
                              tensor->num_channels, elements_per_channel,
                              output);
  }

  return ESP_OK;
}

#include "esp_timer.h"

#define PERF_START(name) int64_t name##_start = esp_timer_get_time()
#define PERF_END(name, acc) acc += (esp_timer_get_time() - name##_start)

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

typedef struct {
  float *input_ln_w;
  const moonshine_tensor_t *q_w;
  const moonshine_tensor_t *k_w;
  const moonshine_tensor_t *v_w;
  const moonshine_tensor_t *o_w;
  float *post_attn_ln_w;
  const moonshine_tensor_t *cq_w;
  const moonshine_tensor_t *ck_w;
  const moonshine_tensor_t *cv_w;
  const moonshine_tensor_t *co_w;
  float *final_ln_w;
  const moonshine_tensor_t *fc1_w;
  float *fc1_b;
  const moonshine_tensor_t *fc2_w;
  float *fc2_b;
} decoder_layer_weights_t;

static esp_err_t preload_decoder_weights(moonshine_model_t *model,
                                         uint32_t num_layers,
                                         decoder_layer_weights_t *layers,
                                         float **final_norm_w) {
  char name[64];
  for (uint32_t l = 0; l < num_layers; l++) {
    decoder_layer_weights_t *lw = &layers[l];
    memset(lw, 0, sizeof(decoder_layer_weights_t));

    snprintf(name, sizeof(name), "layers.%lu.input_layernorm.weight", l);
    lw->input_ln_w = load_dequantize_bias(model, name);

    snprintf(name, sizeof(name), "layers.%lu.self_attn.q_proj.weight", l);
    lw->q_w = find_tensor_by_hash(
        model, moonshine_crc32((const uint8_t *)name, strlen(name)));
    snprintf(name, sizeof(name), "layers.%lu.self_attn.k_proj.weight", l);
    lw->k_w = find_tensor_by_hash(
        model, moonshine_crc32((const uint8_t *)name, strlen(name)));
    snprintf(name, sizeof(name), "layers.%lu.self_attn.v_proj.weight", l);
    lw->v_w = find_tensor_by_hash(
        model, moonshine_crc32((const uint8_t *)name, strlen(name)));
    snprintf(name, sizeof(name), "layers.%lu.self_attn.o_proj.weight", l);
    lw->o_w = find_tensor_by_hash(
        model, moonshine_crc32((const uint8_t *)name, strlen(name)));

    snprintf(name, sizeof(name), "layers.%lu.post_attention_layernorm.weight",
             l);
    lw->post_attn_ln_w = load_dequantize_bias(model, name);

    snprintf(name, sizeof(name), "layers.%lu.encoder_attn.q_proj.weight", l);
    lw->cq_w = find_tensor_by_hash(
        model, moonshine_crc32((const uint8_t *)name, strlen(name)));
    snprintf(name, sizeof(name), "layers.%lu.encoder_attn.k_proj.weight", l);
    lw->ck_w = find_tensor_by_hash(
        model, moonshine_crc32((const uint8_t *)name, strlen(name)));
    snprintf(name, sizeof(name), "layers.%lu.encoder_attn.v_proj.weight", l);
    lw->cv_w = find_tensor_by_hash(
        model, moonshine_crc32((const uint8_t *)name, strlen(name)));
    snprintf(name, sizeof(name), "layers.%lu.encoder_attn.o_proj.weight", l);
    lw->co_w = find_tensor_by_hash(
        model, moonshine_crc32((const uint8_t *)name, strlen(name)));

    snprintf(name, sizeof(name), "layers.%lu.final_layernorm.weight", l);
    lw->final_ln_w = load_dequantize_bias(model, name);

    snprintf(name, sizeof(name), "layers.%lu.mlp.fc1.weight", l);
    lw->fc1_w = find_tensor_by_hash(
        model, moonshine_crc32((const uint8_t *)name, strlen(name)));
    snprintf(name, sizeof(name), "layers.%lu.mlp.fc1.bias", l);
    lw->fc1_b = load_dequantize_bias(model, name);

    snprintf(name, sizeof(name), "layers.%lu.mlp.fc2.weight", l);
    lw->fc2_w = find_tensor_by_hash(
        model, moonshine_crc32((const uint8_t *)name, strlen(name)));
    snprintf(name, sizeof(name), "layers.%lu.mlp.fc2.bias", l);
    lw->fc2_b = load_dequantize_bias(model, name);

    if (!lw->input_ln_w || !lw->q_w || !lw->k_w || !lw->v_w || !lw->o_w ||
        !lw->post_attn_ln_w || !lw->cq_w || !lw->ck_w || !lw->cv_w ||
        !lw->co_w || !lw->final_ln_w || !lw->fc1_w || !lw->fc2_w) {
      ESP_LOGE(TAG, "Failed to preload weights for layer %lu", l);
      return ESP_ERR_NOT_FOUND;
    }
  }

  *final_norm_w = load_dequantize_bias(model, "norm.weight");
  if (!*final_norm_w) {
    ESP_LOGE(TAG, "Failed to preload norm.weight");
    return ESP_ERR_NOT_FOUND;
  }
  return ESP_OK;
}

static void free_decoder_weights(uint32_t num_layers,
                                 decoder_layer_weights_t *layers,
                                 float *final_norm_w) {
  for (uint32_t l = 0; l < num_layers; l++) {
    decoder_layer_weights_t *lw = &layers[l];
    if (lw->input_ln_w)
      free(lw->input_ln_w);
    if (lw->post_attn_ln_w)
      free(lw->post_attn_ln_w);
    if (lw->final_ln_w)
      free(lw->final_ln_w);
    if (lw->fc1_b)
      free(lw->fc1_b);
    if (lw->fc2_b)
      free(lw->fc2_b);
  }
  if (final_norm_w)
    free(final_norm_w);
}

typedef struct {
  const float *q_input; // [1 × hidden_size]
  const float *layer_k; // [num_heads, num_frames, head_dim]
  const float *layer_v; // [num_heads, head_dim, num_frames]
  float *attn_out;      // [1 × hidden_size]
  uint32_t num_frames;
  uint32_t hidden_size;
  uint32_t num_heads;
  uint32_t head_dim;
  float scale;
  float *scores_ws[MOONSHINE_PAR_PARTS]; // [num_frames] per part
} cross_attn_fast_job_t;

#ifdef ESP_PLATFORM
static void cross_attention_fast_part(void *ctx, int part, int parts) {
  cross_attn_fast_job_t *job = (cross_attn_fast_job_t *)ctx;
  int h_start, h_end;
  moonshine_par_range((int)job->num_heads, part, parts, 1, &h_start, &h_end);

  const uint32_t head_dim = job->head_dim;
  const uint32_t num_frames = job->num_frames;
  const float scale = job->scale;
  const float *q_input = job->q_input;
  const float *layer_k = job->layer_k;
  const float *layer_v = job->layer_v;
  float *attn_out = job->attn_out;
  float *scores = job->scores_ws[part];

  for (int h = h_start; h < h_end; h++) {
    uint32_t head_offset = h * head_dim;
    const float *q_h = q_input + head_offset;
    const float *k_h = layer_k + (size_t)h * num_frames * head_dim;
    const float *v_h = layer_v + (size_t)h * head_dim * num_frames;

    // 1. Scores_h = Q_h @ K_h^T -> [1 × num_frames]
    if (head_dim == 36) {
      float q_scaled[36];
      for (uint32_t d = 0; d < 36; d++) {
        q_scaled[d] = q_h[d] * scale;
      }
      for (uint32_t f = 0; f < num_frames; f++) {
        const float *k_f = k_h + f * 36;
        float s0 = q_scaled[0] * k_f[0] + q_scaled[1] * k_f[1] + q_scaled[2] * k_f[2] + q_scaled[3] * k_f[3];
        float s1 = q_scaled[4] * k_f[4] + q_scaled[5] * k_f[5] + q_scaled[6] * k_f[6] + q_scaled[7] * k_f[7];
        float s2 = q_scaled[8] * k_f[8] + q_scaled[9] * k_f[9] + q_scaled[10] * k_f[10] + q_scaled[11] * k_f[11];
        float s3 = q_scaled[12] * k_f[12] + q_scaled[13] * k_f[13] + q_scaled[14] * k_f[14] + q_scaled[15] * k_f[15];
        float s4 = q_scaled[16] * k_f[16] + q_scaled[17] * k_f[17] + q_scaled[18] * k_f[18] + q_scaled[19] * k_f[19];
        float s5 = q_scaled[20] * k_f[20] + q_scaled[21] * k_f[21] + q_scaled[22] * k_f[22] + q_scaled[23] * k_f[23];
        float s6 = q_scaled[24] * k_f[24] + q_scaled[25] * k_f[25] + q_scaled[26] * k_f[26] + q_scaled[27] * k_f[27];
        float s7 = q_scaled[28] * k_f[28] + q_scaled[29] * k_f[29] + q_scaled[30] * k_f[30] + q_scaled[31] * k_f[31];
        float s8 = q_scaled[32] * k_f[32] + q_scaled[33] * k_f[33] + q_scaled[34] * k_f[34] + q_scaled[35] * k_f[35];
        scores[f] = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7) + s8;
      }
    } else {
      for (uint32_t f = 0; f < num_frames; f++) {
        const float *k_f = k_h + f * head_dim;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        uint32_t d = 0;
        for (; d + 4 <= head_dim; d += 4) {
          s0 += q_h[d] * k_f[d];
          s1 += q_h[d + 1] * k_f[d + 1];
          s2 += q_h[d + 2] * k_f[d + 2];
          s3 += q_h[d + 3] * k_f[d + 3];
        }
        float sum = (s0 + s1) + (s2 + s3);
        for (; d < head_dim; d++) {
          sum += q_h[d] * k_f[d];
        }
        scores[f] = sum * scale;
      }
    }

    // 2. Softmax
    moonshine_softmax(scores, num_frames);

    // 3. Attn_out_h = Scores_h @ V_h -> [1 × head_dim]
    // For each d, v_h[d] is contiguous num_frames floats!
    for (uint32_t d = 0; d < head_dim; d++) {
      const float *v_d = v_h + d * num_frames;
      float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
      float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
      uint32_t f = 0;
      for (; f + 8 <= num_frames; f += 8) {
        s0 += scores[f] * v_d[f];
        s1 += scores[f + 1] * v_d[f + 1];
        s2 += scores[f + 2] * v_d[f + 2];
        s3 += scores[f + 3] * v_d[f + 3];
        s4 += scores[f + 4] * v_d[f + 4];
        s5 += scores[f + 5] * v_d[f + 5];
        s6 += scores[f + 6] * v_d[f + 6];
        s7 += scores[f + 7] * v_d[f + 7];
      }
      float sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
      for (; f < num_frames; f++) {
        sum += scores[f] * v_d[f];
      }
      attn_out[head_offset + d] = sum;
    }
  }
}
#endif

static esp_err_t multi_head_cross_attention_fast(
    const float *q_input,                 // [1 × hidden_size]
    const float *layer_k,                 // [num_heads, num_frames, head_dim]
    const float *layer_v,                 // [num_heads, head_dim, num_frames]
    const moonshine_tensor_t *out_weight, // [hidden_size × hidden_size] INT8
    float *output,                        // [1 × hidden_size]
    uint32_t num_frames, uint32_t hidden_size, uint32_t num_heads,
    float *attn_out, // [hidden_size]
    float *fast_ws   // 2 * num_frames workspace in SRAM
) {
  uint32_t head_dim = hidden_size / num_heads;
  float scale = 1.0f / sqrtf((float)head_dim);

#ifdef ESP_PLATFORM
  float *ws0 = fast_ws;
  float *ws1 = fast_ws + num_frames;

  cross_attn_fast_job_t job = {.q_input = q_input,
                               .layer_k = layer_k,
                               .layer_v = layer_v,
                               .attn_out = attn_out,
                               .num_frames = num_frames,
                               .hidden_size = hidden_size,
                               .num_heads = num_heads,
                               .head_dim = head_dim,
                               .scale = scale,
                               .scores_ws = {ws0, ws1}};

  moonshine_parallel_run(cross_attention_fast_part, &job);
#else
  float scores[num_frames];
  for (uint32_t h = 0; h < num_heads; h++) {
    uint32_t head_offset = h * head_dim;
    const float *q_h = q_input + head_offset;
    const float *k_h = layer_k + (size_t)h * num_frames * head_dim;
    const float *v_h = layer_v + (size_t)h * head_dim * num_frames;

    for (uint32_t f = 0; f < num_frames; f++) {
      const float *k_f = k_h + f * head_dim;
      float sum = 0.0f;
      for (uint32_t d = 0; d < head_dim; d++) {
        sum += q_h[d] * k_f[d];
      }
      scores[f] = sum * scale;
    }

    moonshine_softmax(scores, num_frames);

    for (uint32_t d = 0; d < head_dim; d++) {
      const float *v_d = v_h + d * num_frames;
      float sum = 0.0f;
      for (uint32_t f = 0; f < num_frames; f++) {
        sum += scores[f] * v_d[f];
      }
      attn_out[head_offset + d] = sum;
    }
  }
#endif

  // Output projection: output = attn_out * W_o^T
  int8_t q_attn_out[288] __attribute__((aligned(16)));
  float max_ao = 0.0f;
  for (uint32_t k = 0; k < hidden_size; k++) {
    float val = fabsf(attn_out[k]);
    if (val > max_ao)
      max_ao = val;
  }
  float ao_scale = (max_ao > 0.0f) ? (max_ao / 127.0f) : 1.0f;
  float inv_ao_scale = 1.0f / ao_scale;
  for (uint32_t k = 0; k < hidden_size; k++) {
    float val = attn_out[k] * inv_ao_scale;
    int32_t q = (int32_t)(val + (val >= 0.0f ? 0.5f : -0.5f));
    if (q > 127)
      q = 127;
    if (q < -128)
      q = -128;
    q_attn_out[k] = (int8_t)q;
  }
  moonshine_gemm_int8_prequantized(q_attn_out, &ao_scale,
                                   (const int8_t *)out_weight->data,
                                   out_weight->scales, out_weight->num_channels,
                                   output, 1, hidden_size, hidden_size);

  return ESP_OK;
}

// Causal Self Attention for Autoregressive Decoding
static esp_err_t causal_self_attention(
    const float *q_input, // [1 × hidden_size]
    const moonshine_tensor_t *q_weight, const moonshine_tensor_t *k_weight,
    const moonshine_tensor_t *v_weight, const moonshine_tensor_t *out_weight,
    float *output, // [1 × hidden_size]
    uint32_t pos,  // Current token position
    uint32_t hidden_size, uint32_t num_heads, const float *cos_t,
    const float *sin_t, uint32_t half_rotary,
    float *k_cache,    // [max_seq_len × hidden_size]
    float *v_cache,    // [max_seq_len × hidden_size]
    float *temp_buffer // Buffer of size: 3*hidden_size + max_seq_len
) {
  uint32_t head_dim = hidden_size / num_heads;
  float scale = 1.0f / sqrtf((float)head_dim);

  float *Q = temp_buffer;
  float *K_new = Q + hidden_size;
  float *V_new = K_new + hidden_size;
  float *scores = V_new + hidden_size;

  // 1. Q, K, V projections: quantize q_input ONCE
  int8_t q_in[288] __attribute__((aligned(16)));
  float max_in = 0.0f;
  for (uint32_t k = 0; k < hidden_size; k++) {
    float val = fabsf(q_input[k]);
    if (val > max_in)
      max_in = val;
  }
  float in_scale = (max_in > 0.0f) ? (max_in / 127.0f) : 1.0f;
  float inv_scale = 1.0f / in_scale;
  for (uint32_t k = 0; k < hidden_size; k++) {
    float val = q_input[k] * inv_scale;
    int32_t q = (int32_t)(val + (val >= 0.0f ? 0.5f : -0.5f));
    if (q > 127)
      q = 127;
    if (q < -128)
      q = -128;
    q_in[k] = (int8_t)q;
  }

  moonshine_gemm_int8_prequantized(
      q_in, &in_scale, (const int8_t *)q_weight->data, q_weight->scales,
      q_weight->num_channels, Q, 1, hidden_size, hidden_size);
  moonshine_gemm_int8_prequantized(
      q_in, &in_scale, (const int8_t *)k_weight->data, k_weight->scales,
      k_weight->num_channels, K_new, 1, hidden_size, hidden_size);
  moonshine_gemm_int8_prequantized(
      q_in, &in_scale, (const int8_t *)v_weight->data, v_weight->scales,
      v_weight->num_channels, V_new, 1, hidden_size, hidden_size);

  // 2. Apply RoPE to Q and K_new at current position `pos`
  for (uint32_t h = 0; h < num_heads; h++) {
    uint32_t head_offset = h * head_dim;
    for (uint32_t d = 0; d < half_rotary; d++) {
      uint32_t idx1 = head_offset + 2 * d;
      uint32_t idx2 = idx1 + 1;

      float q1 = Q[idx1];
      float q2 = Q[idx2];
      Q[idx1] = q1 * cos_t[d] - q2 * sin_t[d];
      Q[idx2] = q1 * sin_t[d] + q2 * cos_t[d];

      float k1 = K_new[idx1];
      float k2 = K_new[idx2];
      K_new[idx1] = k1 * cos_t[d] - k2 * sin_t[d];
      K_new[idx2] = k1 * sin_t[d] + k2 * cos_t[d];
    }
  }

  // 3. Update Cache
  memcpy(k_cache + pos * hidden_size, K_new, hidden_size * sizeof(float));
  memcpy(v_cache + pos * hidden_size, V_new, hidden_size * sizeof(float));

  // 4. Compute Attention
  float *attn_out = K_new; // reuse memory
  memset(attn_out, 0, hidden_size * sizeof(float));

  for (uint32_t h = 0; h < num_heads; h++) {
    uint32_t head_offset = h * head_dim;

    // Scores = Q @ K_cache^T for this head
    if (head_dim == 36) {
      float q_scaled[36];
      const float *q_row = Q + head_offset;
      for (uint32_t d = 0; d < 36; d++) {
        q_scaled[d] = q_row[d] * scale;
      }
      for (uint32_t i = 0; i <= pos; i++) {
        const float *k_row = k_cache + i * hidden_size + head_offset;
        float s0 = q_scaled[0] * k_row[0] + q_scaled[1] * k_row[1] + q_scaled[2] * k_row[2] + q_scaled[3] * k_row[3];
        float s1 = q_scaled[4] * k_row[4] + q_scaled[5] * k_row[5] + q_scaled[6] * k_row[6] + q_scaled[7] * k_row[7];
        float s2 = q_scaled[8] * k_row[8] + q_scaled[9] * k_row[9] + q_scaled[10] * k_row[10] + q_scaled[11] * k_row[11];
        float s3 = q_scaled[12] * k_row[12] + q_scaled[13] * k_row[13] + q_scaled[14] * k_row[14] + q_scaled[15] * k_row[15];
        float s4 = q_scaled[16] * k_row[16] + q_scaled[17] * k_row[17] + q_scaled[18] * k_row[18] + q_scaled[19] * k_row[19];
        float s5 = q_scaled[20] * k_row[20] + q_scaled[21] * k_row[21] + q_scaled[22] * k_row[22] + q_scaled[23] * k_row[23];
        float s6 = q_scaled[24] * k_row[24] + q_scaled[25] * k_row[25] + q_scaled[26] * k_row[26] + q_scaled[27] * k_row[27];
        float s7 = q_scaled[28] * k_row[28] + q_scaled[29] * k_row[29] + q_scaled[30] * k_row[30] + q_scaled[31] * k_row[31];
        float s8 = q_scaled[32] * k_row[32] + q_scaled[33] * k_row[33] + q_scaled[34] * k_row[34] + q_scaled[35] * k_row[35];
        scores[i] = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7) + s8;
      }
    } else {
      for (uint32_t i = 0; i <= pos; i++) {
        const float *k_row = k_cache + i * hidden_size + head_offset;
        const float *q_row = Q + head_offset;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        uint32_t d = 0;
        for (; d + 4 <= head_dim; d += 4) {
          s0 += q_row[d] * k_row[d];
          s1 += q_row[d + 1] * k_row[d + 1];
          s2 += q_row[d + 2] * k_row[d + 2];
          s3 += q_row[d + 3] * k_row[d + 3];
        }
        float sum = (s0 + s1) + (s2 + s3);
        for (; d < head_dim; d++) {
          sum += q_row[d] * k_row[d];
        }
        scores[i] = sum * scale;
      }
    }

    moonshine_softmax(scores, pos + 1);

    // Attn = Scores @ V_cache
    if (head_dim == 36) {
      for (uint32_t d = 0; d < 36; d++) {
        attn_out[head_offset + d] = 0.0f;
      }
      float *ao_ptr = attn_out + head_offset;
      for (uint32_t i = 0; i <= pos; i++) {
        float s = scores[i];
        if (s <= 1e-7f)
          continue;
        const float *v_row = v_cache + i * hidden_size + head_offset;
        ao_ptr[0]  += s * v_row[0];  ao_ptr[1]  += s * v_row[1];
        ao_ptr[2]  += s * v_row[2];  ao_ptr[3]  += s * v_row[3];
        ao_ptr[4]  += s * v_row[4];  ao_ptr[5]  += s * v_row[5];
        ao_ptr[6]  += s * v_row[6];  ao_ptr[7]  += s * v_row[7];
        ao_ptr[8]  += s * v_row[8];  ao_ptr[9]  += s * v_row[9];
        ao_ptr[10] += s * v_row[10]; ao_ptr[11] += s * v_row[11];
        ao_ptr[12] += s * v_row[12]; ao_ptr[13] += s * v_row[13];
        ao_ptr[14] += s * v_row[14]; ao_ptr[15] += s * v_row[15];
        ao_ptr[16] += s * v_row[16]; ao_ptr[17] += s * v_row[17];
        ao_ptr[18] += s * v_row[18]; ao_ptr[19] += s * v_row[19];
        ao_ptr[20] += s * v_row[20]; ao_ptr[21] += s * v_row[21];
        ao_ptr[22] += s * v_row[22]; ao_ptr[23] += s * v_row[23];
        ao_ptr[24] += s * v_row[24]; ao_ptr[25] += s * v_row[25];
        ao_ptr[26] += s * v_row[26]; ao_ptr[27] += s * v_row[27];
        ao_ptr[28] += s * v_row[28]; ao_ptr[29] += s * v_row[29];
        ao_ptr[30] += s * v_row[30]; ao_ptr[31] += s * v_row[31];
        ao_ptr[32] += s * v_row[32]; ao_ptr[33] += s * v_row[33];
        ao_ptr[34] += s * v_row[34]; ao_ptr[35] += s * v_row[35];
      }
    } else {
      for (uint32_t d = 0; d < head_dim; d++) {
        attn_out[head_offset + d] = 0.0f;
      }
      for (uint32_t i = 0; i <= pos; i++) {
        float s = scores[i];
        if (s <= 1e-7f)
          continue;
        const float *v_row = v_cache + i * hidden_size + head_offset;
        float *ao_ptr = attn_out + head_offset;
        uint32_t d = 0;
        for (; d + 4 <= head_dim; d += 4) {
          ao_ptr[d] += s * v_row[d];
          ao_ptr[d + 1] += s * v_row[d + 1];
          ao_ptr[d + 2] += s * v_row[d + 2];
          ao_ptr[d + 3] += s * v_row[d + 3];
        }
        for (; d < head_dim; d++) {
          ao_ptr[d] += s * v_row[d];
        }
      }
    }
  }

  // Output projection
  int8_t q_attn_out[288] __attribute__((aligned(16)));
  float max_ao = 0.0f;
  for (uint32_t k = 0; k < hidden_size; k++) {
    float val = fabsf(attn_out[k]);
    if (val > max_ao)
      max_ao = val;
  }
  float ao_scale = (max_ao > 0.0f) ? (max_ao / 127.0f) : 1.0f;
  float inv_ao_scale = 1.0f / ao_scale;
  for (uint32_t k = 0; k < hidden_size; k++) {
    float val = attn_out[k] * inv_ao_scale;
    int32_t q = (int32_t)(val + (val >= 0.0f ? 0.5f : -0.5f));
    if (q > 127)
      q = 127;
    if (q < -128)
      q = -128;
    q_attn_out[k] = (int8_t)q;
  }
  moonshine_gemm_int8_prequantized(q_attn_out, &ao_scale,
                                   (const int8_t *)out_weight->data,
                                   out_weight->scales, out_weight->num_channels,
                                   output, 1, hidden_size, hidden_size);

  return ESP_OK;
}

static esp_err_t feed_forward_network(const float *input,
                                      const moonshine_tensor_t *fc1_weight,
                                      const float *fc1_bias,
                                      const moonshine_tensor_t *fc2_weight,
                                      const float *fc2_bias, float *output,
                                      uint32_t hidden_size, uint32_t ffn_size,
                                      float *temp) {
  uint32_t gate_up_size = ffn_size * 2; // 2304

  // 1. fc1 projection: quantize input once
  int8_t q_in[288] __attribute__((aligned(16)));
  float max_in = 0.0f;
  for (uint32_t k = 0; k < hidden_size; k++) {
    float val = fabsf(input[k]);
    if (val > max_in)
      max_in = val;
  }
  float in_scale = (max_in > 0.0f) ? (max_in / 127.0f) : 1.0f;
  float inv_scale = 1.0f / in_scale;
  for (uint32_t k = 0; k < hidden_size; k++) {
    float val = input[k] * inv_scale;
    int32_t q = (int32_t)(val + (val >= 0.0f ? 0.5f : -0.5f));
    if (q > 127)
      q = 127;
    if (q < -128)
      q = -128;
    q_in[k] = (int8_t)q;
  }

  moonshine_gemm_int8_prequantized(
      q_in, &in_scale, (const int8_t *)fc1_weight->data, fc1_weight->scales,
      fc1_weight->num_channels, temp, 1, gate_up_size, hidden_size);

  if (fc1_bias) {
    moonshine_add_inplace(temp, fc1_bias, gate_up_size);
  }

  // 2. MoonshineDecoderMLP gating:
  float *hidden_half = temp;     // first half
  float *gate = temp + ffn_size; // second half

  // Apply SiLU to gate (second half)
  moonshine_silu(gate, ffn_size);

  // Element-wise multiply: hidden_half *= silu(gate) with 4-way unrolling
  uint32_t i = 0;
  for (; i + 4 <= ffn_size; i += 4) {
    hidden_half[i] *= gate[i];
    hidden_half[i + 1] *= gate[i + 1];
    hidden_half[i + 2] *= gate[i + 2];
    hidden_half[i + 3] *= gate[i + 3];
  }
  for (; i < ffn_size; i++) {
    hidden_half[i] *= gate[i];
  }

  // 3. fc2 projection: [ffn_size] -> [hidden_size]
  int8_t q_temp[1152] __attribute__((aligned(16)));
  float max_temp = 0.0f;
  for (uint32_t k = 0; k < ffn_size; k++) {
    float val = fabsf(temp[k]);
    if (val > max_temp)
      max_temp = val;
  }
  float temp_scale = (max_temp > 0.0f) ? (max_temp / 127.0f) : 1.0f;
  float inv_temp_scale = 1.0f / temp_scale;
  for (uint32_t k = 0; k < ffn_size; k++) {
    float val = temp[k] * inv_temp_scale;
    int32_t q = (int32_t)(val + (val >= 0.0f ? 0.5f : -0.5f));
    if (q > 127)
      q = 127;
    if (q < -128)
      q = -128;
    q_temp[k] = (int8_t)q;
  }
  moonshine_gemm_int8_prequantized(
      q_temp, &temp_scale, (const int8_t *)fc2_weight->data, fc2_weight->scales,
      fc2_weight->num_channels, output, 1, hidden_size, ffn_size);

  if (fc2_bias) {
    moonshine_add_inplace(output, fc2_bias, hidden_size);
  }

  return ESP_OK;
}

typedef struct {
  const int8_t *q_x;
  const moonshine_tensor_t *embed_tokens;
  uint32_t hidden_size;
  uint32_t vocab_size;
  float in_scale;
  uint32_t best_token[MOONSHINE_PAR_PARTS];
  float best_score[MOONSHINE_PAR_PARTS];
} decoder_logits_job_t;

#ifdef ESP_PLATFORM
static void decoder_logits_worker(void *ctx, int part, int parts) {
  decoder_logits_job_t *job = (decoder_logits_job_t *)ctx;
  int v_start, v_end;
  moonshine_par_range((int)job->vocab_size, part, parts, 1, &v_start, &v_end);

  uint32_t b_token = 0;
  float b_score = -INFINITY;
  const int8_t *q_x = job->q_x;
  const moonshine_tensor_t *embed = job->embed_tokens;
  const uint32_t h_size = job->hidden_size;
  const float in_scale = job->in_scale;
  const bool per_channel = (embed->num_channels > 1);
  const float common_scale = embed->scales[0];

  if (!per_channel) {
    int32_t b_sum = INT32_MIN;
    if (h_size == 288) {
      int v = v_start;
      const int8_t *emb_ptr =
          (const int8_t *)embed->data + (size_t)v_start * 288;
      for (; v + 8 <= v_end; v += 8) {
        int32_t s0 = moonshine_dot_s8_288_pie(q_x, emb_ptr);
        int32_t s1 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 288);
        int32_t s2 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 576);
        int32_t s3 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 864);
        int32_t s4 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 1152);
        int32_t s5 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 1440);
        int32_t s6 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 1728);
        int32_t s7 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 2016);
        emb_ptr += 2304;
        if (s0 > b_sum) { b_sum = s0; b_token = (uint32_t)(v + 0); }
        if (s1 > b_sum) { b_sum = s1; b_token = (uint32_t)(v + 1); }
        if (s2 > b_sum) { b_sum = s2; b_token = (uint32_t)(v + 2); }
        if (s3 > b_sum) { b_sum = s3; b_token = (uint32_t)(v + 3); }
        if (s4 > b_sum) { b_sum = s4; b_token = (uint32_t)(v + 4); }
        if (s5 > b_sum) { b_sum = s5; b_token = (uint32_t)(v + 5); }
        if (s6 > b_sum) { b_sum = s6; b_token = (uint32_t)(v + 6); }
        if (s7 > b_sum) { b_sum = s7; b_token = (uint32_t)(v + 7); }
      }
      for (; v < v_end; v++) {
        int32_t s = moonshine_dot_s8_288_pie(q_x, emb_ptr);
        emb_ptr += 288;
        if (s > b_sum) {
          b_sum = s;
          b_token = (uint32_t)v;
        }
      }
    } else {
      for (int v = v_start; v < v_end; v++) {
        const int8_t *emb_q = (const int8_t *)embed->data + (size_t)v * h_size;
        int32_t sum = moonshine_fc_dot_s8_pie(q_x, emb_q, (int32_t)h_size);
        if (sum > b_sum) {
          b_sum = sum;
          b_token = (uint32_t)v;
        }
      }
    }
    b_score = (float)b_sum * (in_scale * common_scale);
  } else {
    if (h_size == 288) {
      int v = v_start;
      const int8_t *emb_ptr =
          (const int8_t *)embed->data + (size_t)v_start * 288;
      const float *scales_ptr = embed->scales + v_start;
      for (; v + 8 <= v_end; v += 8) {
        int32_t s0 = moonshine_dot_s8_288_pie(q_x, emb_ptr);
        int32_t s1 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 288);
        int32_t s2 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 576);
        int32_t s3 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 864);
        int32_t s4 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 1152);
        int32_t s5 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 1440);
        int32_t s6 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 1728);
        int32_t s7 = moonshine_dot_s8_288_pie(q_x, emb_ptr + 2016);
        emb_ptr += 2304;

        float sc0 = (float)s0 * scales_ptr[0];
        float sc1 = (float)s1 * scales_ptr[1];
        float sc2 = (float)s2 * scales_ptr[2];
        float sc3 = (float)s3 * scales_ptr[3];
        float sc4 = (float)s4 * scales_ptr[4];
        float sc5 = (float)s5 * scales_ptr[5];
        float sc6 = (float)s6 * scales_ptr[6];
        float sc7 = (float)s7 * scales_ptr[7];
        scales_ptr += 8;

        if (sc0 > b_score) { b_score = sc0; b_token = (uint32_t)(v + 0); }
        if (sc1 > b_score) { b_score = sc1; b_token = (uint32_t)(v + 1); }
        if (sc2 > b_score) { b_score = sc2; b_token = (uint32_t)(v + 2); }
        if (sc3 > b_score) { b_score = sc3; b_token = (uint32_t)(v + 3); }
        if (sc4 > b_score) { b_score = sc4; b_token = (uint32_t)(v + 4); }
        if (sc5 > b_score) { b_score = sc5; b_token = (uint32_t)(v + 5); }
        if (sc6 > b_score) { b_score = sc6; b_token = (uint32_t)(v + 6); }
        if (sc7 > b_score) { b_score = sc7; b_token = (uint32_t)(v + 7); }
      }
      for (; v < v_end; v++) {
        int32_t s = moonshine_dot_s8_288_pie(q_x, emb_ptr);
        emb_ptr += 288;
        float sc = (float)s * (*scales_ptr++);
        if (sc > b_score) {
          b_score = sc;
          b_token = (uint32_t)v;
        }
      }
      b_score *= in_scale;
    } else {
      for (int v = v_start; v < v_end; v++) {
        const int8_t *emb_q = (const int8_t *)embed->data + (size_t)v * h_size;
        float scale = embed->scales[v];
        int32_t sum = moonshine_fc_dot_s8_pie(q_x, emb_q, (int32_t)h_size);
        float score = (float)sum * (in_scale * scale);

        if (score > b_score) {
          b_score = score;
          b_token = (uint32_t)v;
        }
      }
    }
  }

  job->best_token[part] = b_token;
  job->best_score[part] = b_score;
}
#endif

esp_err_t moonshine_decode(moonshine_context_t *context,
                           const float *encoder_output, uint32_t num_frames,
                           int32_t *tokens, uint32_t *num_tokens) {
  if (!context || !encoder_output || !tokens || !num_tokens)
    return ESP_ERR_INVALID_ARG;

  moonshine_model_t *model = context->model;
  uint32_t hidden_size = model->header.hidden_size;
  uint32_t num_heads = 8; // standard for tiny
  uint32_t head_dim = hidden_size / num_heads;
  uint32_t num_layers = model->header.decoder_layers;
  uint32_t ffn_size = hidden_size * 4;

  // Max sequence length
  uint32_t max_seq_len = *num_tokens;
  if (max_seq_len > context->max_seq_len) {
    max_seq_len = context->max_seq_len;
  }

  ESP_LOGI(TAG, "Starting decoder inference (max_tokens=%lu)", max_seq_len);

  // 1. Preload decoder layer weights & final layernorm
  decoder_layer_weights_t *layer_weights =
      (decoder_layer_weights_t *)heap_caps_malloc(
          num_layers * sizeof(decoder_layer_weights_t), MALLOC_CAP_SPIRAM);
  float *final_norm_w = NULL;
  if (!layer_weights) {
    ESP_LOGE(TAG, "Failed to allocate layer_weights array");
    return ESP_ERR_NO_MEM;
  }
  esp_err_t ret =
      preload_decoder_weights(model, num_layers, layer_weights, &final_norm_w);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to preload decoder weights");
    free_decoder_weights(num_layers, layer_weights, final_norm_w);
    free(layer_weights);
    return ret;
  }

  // 2. Check if encoder_output can fit into fast Internal HP SRAM
  size_t enc_bytes = (size_t)num_frames * hidden_size * sizeof(float);
  float *fast_enc_out = (float *)heap_caps_malloc(
      enc_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
  const float *effective_enc_out = encoder_output;
  if (fast_enc_out) {
    memcpy(fast_enc_out, encoder_output, enc_bytes);
    effective_enc_out = fast_enc_out;
    ESP_LOGI(TAG,
             "Mirrored encoder_output (%.1f KB) into fast Internal HP SRAM",
             enc_bytes / 1024.0f);
  }

  // 3. Preallocate internal fast workspace for cross-attention
  size_t cross_ws_floats = 2 * num_frames;
  float *internal_cross_ws = (float *)heap_caps_malloc(
      cross_ws_floats * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
  if (!internal_cross_ws) {
    internal_cross_ws = (float *)heap_caps_malloc(
        cross_ws_floats * sizeof(float), MALLOC_CAP_SPIRAM);
  }

  // 4. Precompute Cross-Attention K and V caches via PIE SIMD GEMM
  size_t cross_kv_bytes =
      (size_t)num_layers * num_frames * hidden_size * sizeof(float);
  float *cross_k =
      (float *)heap_caps_aligned_alloc(16, cross_kv_bytes, MALLOC_CAP_SPIRAM);
  float *cross_v =
      (float *)heap_caps_aligned_alloc(16, cross_kv_bytes, MALLOC_CAP_SPIRAM);
  if (!cross_k || !cross_v) {
    ESP_LOGE(TAG, "Failed to allocate Cross KV cache (%.2f MB)",
             (2.0f * cross_kv_bytes) / (1024.0f * 1024.0f));
    if (cross_k)
      free(cross_k);
    if (cross_v)
      free(cross_v);
    if (fast_enc_out)
      free(fast_enc_out);
    if (internal_cross_ws)
      free(internal_cross_ws);
    free_decoder_weights(num_layers, layer_weights, final_norm_w);
    free(layer_weights);
    return ESP_ERR_NO_MEM;
  }

  int64_t cross_precompute_start = esp_timer_get_time();
  int8_t *enc_q = (int8_t *)heap_caps_malloc(
      num_frames * hidden_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
  if (!enc_q)
    enc_q =
        (int8_t *)heap_caps_malloc(num_frames * hidden_size, MALLOC_CAP_SPIRAM);
  float *enc_scales = (float *)heap_caps_malloc(
      num_frames * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
  if (!enc_scales)
    enc_scales = (float *)heap_caps_malloc(num_frames * sizeof(float),
                                           MALLOC_CAP_SPIRAM);
  float *gemm_buf =
      (float *)heap_caps_malloc(num_frames * hidden_size * sizeof(float),
                                MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
  if (!gemm_buf)
    gemm_buf = (float *)heap_caps_malloc(
        num_frames * hidden_size * sizeof(float), MALLOC_CAP_SPIRAM);

  if (enc_q && enc_scales && gemm_buf) {
    moonshine_quantize_int8_rows(effective_enc_out, enc_q, enc_scales,
                                 num_frames, hidden_size);
    for (uint32_t l = 0; l < num_layers; l++) {
      const decoder_layer_weights_t *lw = &layer_weights[l];
      float *layer_k = cross_k + (size_t)l * num_frames * hidden_size;
      float *layer_v = cross_v + (size_t)l * num_frames * hidden_size;

      // K = enc_q @ W_k^T -> gemm_buf [num_frames × hidden_size]
      moonshine_gemm_int8_prequantized(
          enc_q, enc_scales, (const int8_t *)lw->ck_w->data, lw->ck_w->scales,
          lw->ck_w->num_channels, gemm_buf, num_frames, hidden_size,
          hidden_size);
      for (uint32_t h = 0; h < num_heads; h++) {
        for (uint32_t f = 0; f < num_frames; f++) {
          const float *src = gemm_buf + f * hidden_size + h * head_dim;
          float *dst = layer_k + (h * num_frames + f) * head_dim;
          memcpy(dst, src, head_dim * sizeof(float));
        }
      }

      // V = enc_q @ W_v^T -> gemm_buf [num_frames × hidden_size]
      moonshine_gemm_int8_prequantized(
          enc_q, enc_scales, (const int8_t *)lw->cv_w->data, lw->cv_w->scales,
          lw->cv_w->num_channels, gemm_buf, num_frames, hidden_size,
          hidden_size);
      for (uint32_t h = 0; h < num_heads; h++) {
        for (uint32_t f = 0; f < num_frames; f++) {
          const float *src = gemm_buf + f * hidden_size + h * head_dim;
          for (uint32_t d = 0; d < head_dim; d++) {
            layer_v[(h * head_dim + d) * num_frames + f] = src[d];
          }
        }
      }
    }
  }
  if (enc_q)
    free(enc_q);
  if (enc_scales)
    free(enc_scales);
  if (gemm_buf)
    free(gemm_buf);

  int64_t cross_precompute_us = esp_timer_get_time() - cross_precompute_start;
  ESP_LOGI(TAG, "Precomputed Cross KV cache in %lld us (%.2f MB)",
           cross_precompute_us, (2.0f * cross_kv_bytes) / (1024.0f * 1024.0f));

  // 5. Precalculate RoPE base frequencies
  uint32_t rotary_dim = (uint32_t)(head_dim * 0.9f);
  uint32_t half_rotary = rotary_dim / 2;
  if (half_rotary > 64)
    half_rotary = 64;

  float inv_freq[64];
  for (uint32_t d = 0; d < half_rotary; d++) {
    inv_freq[d] = 1.0f / powf(10000.0f, 2.0f * (float)d / (float)rotary_dim);
  }

  // 6. Allocate KV cache for self-attention
  // [num_layers × 2 × max_seq_len × hidden_size]
  size_t kv_cache_size =
      num_layers * 2 * max_seq_len * hidden_size * sizeof(float);
  float *kv_cache = heap_caps_malloc(kv_cache_size, MALLOC_CAP_SPIRAM);
  if (!kv_cache) {
    ESP_LOGE(TAG, "Failed to allocate KV cache (%.2f MB)",
             kv_cache_size / 1024.0 / 1024.0);
    if (cross_k)
      free(cross_k);
    if (cross_v)
      free(cross_v);
    if (fast_enc_out)
      free(fast_enc_out);
    if (internal_cross_ws)
      free(internal_cross_ws);
    free_decoder_weights(num_layers, layer_weights, final_norm_w);
    free(layer_weights);
    return ESP_ERR_NO_MEM;
  }
  memset(kv_cache, 0, kv_cache_size);

  // 7. Allocate Temp buffers
  size_t temp_size = (3 * hidden_size + max_seq_len) * sizeof(float);
  if (temp_size < (6 * hidden_size + 2 * num_frames) * sizeof(float)) {
    temp_size = (6 * hidden_size + 2 * num_frames) * sizeof(float);
  }
  if (temp_size < ffn_size * 2 * sizeof(float)) {
    temp_size = ffn_size * 2 * sizeof(float);
  }
  temp_size = (temp_size + 15) & ~15;

  float *temp_buffer =
      (float *)heap_caps_aligned_alloc(16, temp_size, MALLOC_CAP_SPIRAM);
  float *x = heap_caps_aligned_alloc(16, hidden_size * sizeof(float),
                                     MALLOC_CAP_SPIRAM);
  float *x_norm = heap_caps_aligned_alloc(16, hidden_size * sizeof(float),
                                          MALLOC_CAP_SPIRAM);
  float *residual =
      heap_caps_malloc(hidden_size * sizeof(float), MALLOC_CAP_SPIRAM);
  int8_t *q_x = (int8_t *)heap_caps_aligned_alloc(
      16, hidden_size * sizeof(int8_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
  if (!q_x) {
    q_x = (int8_t *)heap_caps_aligned_alloc(16, hidden_size * sizeof(int8_t),
                                            MALLOC_CAP_SPIRAM);
  }

  if (!temp_buffer || !x || !x_norm || !residual || !q_x) {
    ESP_LOGE(TAG, "Failed to allocate temp buffers");
    if (cross_k)
      free(cross_k);
    if (cross_v)
      free(cross_v);
    if (kv_cache)
      free(kv_cache);
    if (temp_buffer)
      free(temp_buffer);
    if (x)
      free(x);
    if (x_norm)
      free(x_norm);
    if (residual)
      free(residual);
    if (q_x)
      free(q_x);
    if (fast_enc_out)
      free(fast_enc_out);
    if (internal_cross_ws)
      free(internal_cross_ws);
    free_decoder_weights(num_layers, layer_weights, final_norm_w);
    free(layer_weights);
    return ESP_ERR_NO_MEM;
  }

  // Profiling accumulators
  int64_t total_self_attn_us = 0;
  int64_t total_cross_attn_us = 0;
  int64_t total_mlp_us = 0;
  int64_t total_logits_us = 0;

  // Load embed_tokens.weight
  uint32_t embed_hash =
      moonshine_crc32((const uint8_t *)"embed_tokens.weight", 19);
  moonshine_tensor_t *embed_tokens = find_tensor_by_hash(model, embed_hash);

  // Initialize tokens
  tokens[0] = 1; // BOS token
  uint32_t pos = 0;

  while (pos < max_seq_len - 1) {
    if (pos % 5 == 0) {
      ESP_LOGI(TAG, "Generating token %lu...", pos);
    }
    vTaskDelay(1); // Yield watchdog

    // Compute RoPE cos_t and sin_t once for all layers at this pos
    float cos_t[64];
    float sin_t[64];
    for (uint32_t d = 0; d < half_rotary; d++) {
      float theta = (float)pos * inv_freq[d];
      cos_t[d] = cosf(theta);
      sin_t[d] = sinf(theta);
    }

    int32_t current_token = tokens[pos];

    // 1. Token Embedding lookup
    const int8_t *embed_q =
        (const int8_t *)embed_tokens->data + current_token * hidden_size;
    float scale = (embed_tokens->num_channels > 1)
                      ? embed_tokens->scales[current_token]
                      : embed_tokens->scales[0];
    for (uint32_t i = 0; i < hidden_size; i++) {
      x[i] = (float)embed_q[i] * scale;
    }

    if (pos == 0) {
      ESP_LOGI(
          TAG,
          "POS 0: embed_tokens scale = %f, embed_q[0..4] = %d, %d, %d, %d, %d",
          scale, embed_q[0], embed_q[1], embed_q[2], embed_q[3], embed_q[4]);
      ESP_LOGI(TAG, "POS 0: x after embed[0..4] = %f, %f, %f, %f, %f", x[0],
               x[1], x[2], x[3], x[4]);
    }

    // 2. Layers
    for (uint32_t l = 0; l < num_layers; l++) {
      const decoder_layer_weights_t *lw = &layer_weights[l];

      // Input Layernorm
      memcpy(x_norm, x, hidden_size * sizeof(float));
      moonshine_layer_norm(x_norm, lw->input_ln_w, NULL, hidden_size, 1e-5f);

      // Self Attention
      float *k_cache_l = kv_cache + (l * 2) * max_seq_len * hidden_size;
      float *v_cache_l = kv_cache + (l * 2 + 1) * max_seq_len * hidden_size;

      PERF_START(self_attn);
      causal_self_attention(x_norm, lw->q_w, lw->k_w, lw->v_w, lw->o_w,
                            residual, pos, hidden_size, num_heads, cos_t, sin_t,
                            half_rotary, k_cache_l, v_cache_l, temp_buffer);
      PERF_END(self_attn, total_self_attn_us);

      moonshine_add_inplace(x, residual, hidden_size);

      // Post Attention Layernorm
      memcpy(x_norm, x, hidden_size * sizeof(float));
      moonshine_layer_norm(x_norm, lw->post_attn_ln_w, NULL, hidden_size,
                           1e-5f);

      // Cross Attention
      PERF_START(cross_attn);
      int8_t q_cq[288] __attribute__((aligned(16)));
      float max_cq = 0.0f;
      for (uint32_t k = 0; k < hidden_size; k++) {
        float val = fabsf(x_norm[k]);
        if (val > max_cq)
          max_cq = val;
      }
      float cq_scale = (max_cq > 0.0f) ? (max_cq / 127.0f) : 1.0f;
      float inv_cq_scale = 1.0f / cq_scale;
      for (uint32_t k = 0; k < hidden_size; k++) {
        float val = x_norm[k] * inv_cq_scale;
        int32_t q = (int32_t)(val + (val >= 0.0f ? 0.5f : -0.5f));
        if (q > 127)
          q = 127;
        if (q < -128)
          q = -128;
        q_cq[k] = (int8_t)q;
      }
      moonshine_gemm_int8_prequantized(
          q_cq, &cq_scale, (const int8_t *)lw->cq_w->data, lw->cq_w->scales,
          lw->cq_w->num_channels, temp_buffer, 1, hidden_size, hidden_size);
      const float *layer_k = cross_k + (size_t)l * num_frames * hidden_size;
      const float *layer_v = cross_v + (size_t)l * num_frames * hidden_size;
      multi_head_cross_attention_fast(
          temp_buffer, layer_k, layer_v, lw->co_w, residual, num_frames,
          hidden_size, num_heads, temp_buffer + hidden_size, internal_cross_ws);
      PERF_END(cross_attn, total_cross_attn_us);

      moonshine_add_inplace(x, residual, hidden_size);

      // Final Layernorm
      memcpy(x_norm, x, hidden_size * sizeof(float));
      moonshine_layer_norm(x_norm, lw->final_ln_w, NULL, hidden_size, 1e-5f);

      // MLP
      PERF_START(mlp);
      feed_forward_network(x_norm, lw->fc1_w, lw->fc1_b, lw->fc2_w, lw->fc2_b,
                           residual, hidden_size, ffn_size, temp_buffer);
      PERF_END(mlp, total_mlp_us);

      moonshine_add_inplace(x, residual, hidden_size);
    }

    // Apply final LayerNorm (self.norm in PyTorch MoonshineDecoder)
    memcpy(x_norm, x, hidden_size * sizeof(float));
    moonshine_layer_norm(x_norm, final_norm_w, NULL, hidden_size, 1e-5f);

    PERF_START(logits);
    uint32_t best_token = 0;
    float best_score = -INFINITY;

    bool can_use_pie = false;
#ifdef ESP_PLATFORM
    if (((uintptr_t)embed_tokens->data % 16 == 0) && (hidden_size % 16 == 0)) {
      can_use_pie = true;
    }
#endif

    if (can_use_pie) {
#ifdef ESP_PLATFORM
      // Quantize x_norm with 4-way unrolling
      float max_val = 0.0f;
      uint32_t i = 0;
      for (; i + 4 <= hidden_size; i += 4) {
        float v0 = fabsf(x_norm[i]);
        float v1 = fabsf(x_norm[i + 1]);
        float v2 = fabsf(x_norm[i + 2]);
        float v3 = fabsf(x_norm[i + 3]);
        if (v0 > max_val)
          max_val = v0;
        if (v1 > max_val)
          max_val = v1;
        if (v2 > max_val)
          max_val = v2;
        if (v3 > max_val)
          max_val = v3;
      }
      for (; i < hidden_size; i++) {
        float abs_val = fabsf(x_norm[i]);
        if (abs_val > max_val)
          max_val = abs_val;
      }
      float in_scale = (max_val > 0.0f) ? (max_val / 127.0f) : 1.0f;
      float inv_scale = 1.0f / in_scale;

      i = 0;
      for (; i + 4 <= hidden_size; i += 4) {
        float val0 = x_norm[i] * inv_scale;
        float val1 = x_norm[i + 1] * inv_scale;
        float val2 = x_norm[i + 2] * inv_scale;
        float val3 = x_norm[i + 3] * inv_scale;

        int32_t q0 = (int32_t)(val0 + (val0 >= 0.0f ? 0.5f : -0.5f));
        int32_t q1 = (int32_t)(val1 + (val1 >= 0.0f ? 0.5f : -0.5f));
        int32_t q2 = (int32_t)(val2 + (val2 >= 0.0f ? 0.5f : -0.5f));
        int32_t q3 = (int32_t)(val3 + (val3 >= 0.0f ? 0.5f : -0.5f));

        if (q0 > 127)
          q0 = 127;
        else if (q0 < -128)
          q0 = -128;
        if (q1 > 127)
          q1 = 127;
        else if (q1 < -128)
          q1 = -128;
        if (q2 > 127)
          q2 = 127;
        else if (q2 < -128)
          q2 = -128;
        if (q3 > 127)
          q3 = 127;
        else if (q3 < -128)
          q3 = -128;

        q_x[i] = (int8_t)q0;
        q_x[i + 1] = (int8_t)q1;
        q_x[i + 2] = (int8_t)q2;
        q_x[i + 3] = (int8_t)q3;
      }
      for (; i < hidden_size; i++) {
        float val = x_norm[i] * inv_scale;
        int32_t q = (int32_t)(val + (val >= 0.0f ? 0.5f : -0.5f));
        if (q > 127)
          q = 127;
        if (q < -128)
          q = -128;
        q_x[i] = (int8_t)q;
      }

      decoder_logits_job_t job = {.q_x = q_x,
                                  .embed_tokens = embed_tokens,
                                  .hidden_size = hidden_size,
                                  .vocab_size = model->header.vocab_size,
                                  .in_scale = in_scale,
                                  .best_token = {0, 0},
                                  .best_score = {-INFINITY, -INFINITY}};
      moonshine_parallel_run(decoder_logits_worker, &job);

      if (job.best_score[1] > job.best_score[0]) {
        best_score = job.best_score[1];
        best_token = job.best_token[1];
      } else {
        best_score = job.best_score[0];
        best_token = job.best_token[0];
      }
#endif
    } else {
      // Fallback C implementation
      for (uint32_t v = 0; v < model->header.vocab_size; v++) {
        const int8_t *emb_q =
            (const int8_t *)embed_tokens->data + v * hidden_size;
        float scale = (embed_tokens->num_channels > 1)
                          ? embed_tokens->scales[v]
                          : embed_tokens->scales[0];
        float score = 0.0f;

        for (uint32_t i = 0; i < hidden_size; i++) {
          score += x_norm[i] * ((float)emb_q[i] * scale);
        }

        if (score > best_score) {
          best_score = score;
          best_token = v;
        }
      }
    }

    if (pos == 0) {
      ESP_LOGI(TAG, "POS 0: x_norm[0..4] = %f, %f, %f, %f, %f", x_norm[0],
               x_norm[1], x_norm[2], x_norm[3], x_norm[4]);
      ESP_LOGI(TAG, "POS 0: best_token = %lu, best_score = %f", best_token,
               best_score);
      ESP_LOGI(TAG, "POS 0: can_use_pie = %d", can_use_pie);
    }

    PERF_END(logits, total_logits_us);

    pos++;
    tokens[pos] = best_token;

    // Stop on EOS token (2 = </s>)
    if (best_token == 2) {
      break;
    }
  }

  *num_tokens = pos + 1;

  ESP_LOGI(TAG, "Decoder profiling (total over %lu tokens):", pos);
  ESP_LOGI(TAG, "  Self-Attn  : %lld us", total_self_attn_us);
  ESP_LOGI(TAG, "  Cross-Attn : %lld us", total_cross_attn_us);
  ESP_LOGI(TAG, "  MLP        : %lld us", total_mlp_us);
  ESP_LOGI(TAG, "  Logits     : %lld us", total_logits_us);

  free(kv_cache);
  free(temp_buffer);
  free(x);
  free(x_norm);
  free(residual);
  free(q_x);
  if (cross_k)
    free(cross_k);
  if (cross_v)
    free(cross_v);
  if (fast_enc_out)
    free(fast_enc_out);
  if (internal_cross_ws)
    free(internal_cross_ws);
  free_decoder_weights(num_layers, layer_weights, final_norm_w);
  free(layer_weights);

  ESP_LOGI(TAG, "Decoder inference complete. Generated %lu tokens.",
           *num_tokens);
  return ESP_OK;
}
