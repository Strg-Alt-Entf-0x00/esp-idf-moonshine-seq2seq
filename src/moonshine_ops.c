/**
 * @file moonshine_ops.c
 * @brief Core Mathematical Operations for Moonshine Inference
 *
 * Implements INT8 quantized operations optimized for ESP32-P4
 * with per-channel quantization.
 */

#include "moonshine_ops.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "moonshine_parallel.h"
#include <math.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG __attribute__((unused)) = "moonshine_ops";

/* ========================================================================
 * QUANTIZATION UTILITIES
 * ======================================================================== */

void moonshine_dequantize_int8(const int8_t *quantized, const float *scales,
                               uint32_t num_channels,
                               uint32_t elements_per_channel, float *output) {
  if (num_channels == 1) {
    // Per-tensor quantization
    const float scale = scales[0];
    for (uint32_t i = 0; i < elements_per_channel; i++) {
      output[i] = (float)quantized[i] * scale;
    }
  } else {
    // Per-channel quantization
    for (uint32_t c = 0; c < num_channels; c++) {
      const float scale = scales[c];
      const int8_t *q_ptr = quantized + c * elements_per_channel;
      float *out_ptr = output + c * elements_per_channel;

      for (uint32_t i = 0; i < elements_per_channel; i++) {
        out_ptr[i] = (float)q_ptr[i] * scale;
      }
    }
  }
}

/* ========================================================================
 * MATRIX OPERATIONS
 * ======================================================================== */

void moonshine_gemm_fp32(const float *A, // [M × K]
                         const float *B, // [K × N]
                         float *C,       // [M × N]
                         uint32_t M, uint32_t N, uint32_t K,
                         bool transpose_B // If true, B is [N × K]
) {
  // Simple GEMM: C = A × B (or A × B^T)

  if (!transpose_B) {
    // C = A × B
    for (uint32_t m = 0; m < M; m++) {
      for (uint32_t n = 0; n < N; n++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
          sum += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = sum;
      }
    }
  } else {
    // C = A × B^T
    for (uint32_t m = 0; m < M; m++) {
      for (uint32_t n = 0; n < N; n++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
          sum += A[m * K + k] * B[n * K + k];
        }
        C[m * N + n] = sum;
      }
    }
  }
}

// Scratch buffer for dynamic activation quantization. Must be 16-byte aligned.
// Max K in tiny is 1152, base is 2304, large is 4096. 4096 is safe.
static __attribute__((aligned(16))) int8_t s_gemm_scratch_q[4096];

#ifdef ESP_PLATFORM

typedef struct {
  const int8_t *A_q;
  const float *A_scales;
  const int8_t *B_q;
  const float *B_scales;
  float *C;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t num_channels_B;
} gemm_matrix_job_t;

#define GEMM_N_BLOCK_SIZE 16

static void gemm_matrix_part(void *ctx, int part, int parts) {
  gemm_matrix_job_t *job = (gemm_matrix_job_t *)ctx;
  int n_start, n_end;
  moonshine_par_range((int)job->N, part, parts, 16, &n_start, &n_end);

  const uint32_t M = job->M;
  const uint32_t K = job->K;
  const uint32_t N = job->N;
  const int8_t *A_q = job->A_q;
  const float *A_scales = job->A_scales;
  const int8_t *B_q = job->B_q;
  const float *B_scales = job->B_scales;
  float *C = job->C;
  const uint32_t num_channels_B = job->num_channels_B;

  if (K == 288) {
    if (M == 1) {
      const int8_t *a_row = A_q;
      float a_scale = A_scales[0];
      int n = n_start;
      const int8_t *b_ptr = B_q + (size_t)n_start * 288;
      for (; n + 8 <= n_end; n += 8) {
        float bs0 = (num_channels_B == 1) ? B_scales[0] : B_scales[n];
        float bs1 = (num_channels_B == 1) ? B_scales[0] : B_scales[n + 1];
        float bs2 = (num_channels_B == 1) ? B_scales[0] : B_scales[n + 2];
        float bs3 = (num_channels_B == 1) ? B_scales[0] : B_scales[n + 3];
        float bs4 = (num_channels_B == 1) ? B_scales[0] : B_scales[n + 4];
        float bs5 = (num_channels_B == 1) ? B_scales[0] : B_scales[n + 5];
        float bs6 = (num_channels_B == 1) ? B_scales[0] : B_scales[n + 6];
        float bs7 = (num_channels_B == 1) ? B_scales[0] : B_scales[n + 7];
        int32_t s0 = moonshine_dot_s8_288_pie(a_row, b_ptr);
        int32_t s1 = moonshine_dot_s8_288_pie(a_row, b_ptr + 288);
        int32_t s2 = moonshine_dot_s8_288_pie(a_row, b_ptr + 576);
        int32_t s3 = moonshine_dot_s8_288_pie(a_row, b_ptr + 864);
        int32_t s4 = moonshine_dot_s8_288_pie(a_row, b_ptr + 1152);
        int32_t s5 = moonshine_dot_s8_288_pie(a_row, b_ptr + 1440);
        int32_t s6 = moonshine_dot_s8_288_pie(a_row, b_ptr + 1728);
        int32_t s7 = moonshine_dot_s8_288_pie(a_row, b_ptr + 2016);
        b_ptr += 2304;
        C[n] = (float)s0 * (a_scale * bs0);
        C[n + 1] = (float)s1 * (a_scale * bs1);
        C[n + 2] = (float)s2 * (a_scale * bs2);
        C[n + 3] = (float)s3 * (a_scale * bs3);
        C[n + 4] = (float)s4 * (a_scale * bs4);
        C[n + 5] = (float)s5 * (a_scale * bs5);
        C[n + 6] = (float)s6 * (a_scale * bs6);
        C[n + 7] = (float)s7 * (a_scale * bs7);
      }
      for (; n < n_end; n++) {
        float b_scale = (num_channels_B == 1) ? B_scales[0] : B_scales[n];
        int32_t sum = moonshine_dot_s8_288_pie(a_row, b_ptr);
        b_ptr += 288;
        C[n] = (float)sum * (a_scale * b_scale);
      }
    } else {
      #define GEMM_M_TILE_288 32
      for (uint32_t m_tile = 0; m_tile < M; m_tile += GEMM_M_TILE_288) {
        uint32_t m_end = m_tile + GEMM_M_TILE_288;
        if (m_end > M) m_end = M;
        for (int n = n_start; n < n_end; n++) {
          float b_scale = (num_channels_B == 1) ? B_scales[0] : B_scales[n];
          const int8_t *b_row = B_q + (size_t)n * 288;
          uint32_t m = m_tile;
          for (; m + 2 <= m_end; m += 2) {
            const int8_t *a0 = A_q + (size_t)m * 288;
            const int8_t *a1 = A_q + (size_t)(m + 1) * 288;
            int32_t sum0 = moonshine_dot_s8_288_pie(a0, b_row);
            int32_t sum1 = moonshine_dot_s8_288_pie(a1, b_row);
            C[m * N + n] = (float)sum0 * (A_scales[m] * b_scale);
            C[(m + 1) * N + n] = (float)sum1 * (A_scales[m + 1] * b_scale);
          }
          for (; m < m_end; m++) {
            const int8_t *a_row = A_q + (size_t)m * 288;
            int32_t sum = moonshine_dot_s8_288_pie(a_row, b_row);
            C[m * N + n] = (float)sum * (A_scales[m] * b_scale);
          }
        }
      }
    }
  } else if (K == 1152) {
    if (M == 1) {
      const int8_t *a_row = A_q;
      float a_scale = A_scales[0];
      int n = n_start;
      const int8_t *b_ptr = B_q + (size_t)n_start * 1152;
      for (; n + 4 <= n_end; n += 4) {
        float bs0 = (num_channels_B == 1) ? B_scales[0] : B_scales[n];
        float bs1 = (num_channels_B == 1) ? B_scales[0] : B_scales[n + 1];
        float bs2 = (num_channels_B == 1) ? B_scales[0] : B_scales[n + 2];
        float bs3 = (num_channels_B == 1) ? B_scales[0] : B_scales[n + 3];
        int32_t s0 = moonshine_dot_s8_1152_pie(a_row, b_ptr);
        int32_t s1 = moonshine_dot_s8_1152_pie(a_row, b_ptr + 1152);
        int32_t s2 = moonshine_dot_s8_1152_pie(a_row, b_ptr + 2304);
        int32_t s3 = moonshine_dot_s8_1152_pie(a_row, b_ptr + 3456);
        b_ptr += 4608;
        C[n] = (float)s0 * (a_scale * bs0);
        C[n + 1] = (float)s1 * (a_scale * bs1);
        C[n + 2] = (float)s2 * (a_scale * bs2);
        C[n + 3] = (float)s3 * (a_scale * bs3);
      }
      for (; n < n_end; n++) {
        float b_scale = (num_channels_B == 1) ? B_scales[0] : B_scales[n];
        int32_t sum = moonshine_dot_s8_1152_pie(a_row, b_ptr);
        b_ptr += 1152;
        C[n] = (float)sum * (a_scale * b_scale);
      }
    } else {
      #define GEMM_M_TILE_1152 16
      for (uint32_t m_tile = 0; m_tile < M; m_tile += GEMM_M_TILE_1152) {
        uint32_t m_end = m_tile + GEMM_M_TILE_1152;
        if (m_end > M) m_end = M;
        for (int n = n_start; n < n_end; n++) {
          float b_scale = (num_channels_B == 1) ? B_scales[0] : B_scales[n];
          const int8_t *b_row = B_q + (size_t)n * 1152;
          for (uint32_t m = m_tile; m < m_end; m++) {
            const int8_t *a_row = A_q + (size_t)m * 1152;
            int32_t sum = moonshine_dot_s8_1152_pie(a_row, b_row);
            C[m * N + n] = (float)sum * (A_scales[m] * b_scale);
          }
        }
      }
    }
  } else {
    for (uint32_t m_tile = 0; m_tile < M; m_tile += 16) {
      uint32_t m_end = m_tile + 16;
      if (m_end > M) m_end = M;
      for (int n = n_start; n < n_end; n++) {
        float b_scale = (num_channels_B == 1) ? B_scales[0] : B_scales[n];
        const int8_t *b_row = B_q + (size_t)n * K;
        for (uint32_t m = m_tile; m < m_end; m++) {
          const int8_t *a_row = A_q + (size_t)m * K;
          int32_t sum = moonshine_fc_dot_s8_pie(a_row, b_row, (int32_t)K);
          C[m * N + n] = (float)sum * (A_scales[m] * b_scale);
        }
      }
    }
  }
}

typedef struct {
  const int8_t *a_q;
  float a_scale;
  const int8_t *B_q;
  const float *B_scales;
  float *C;
  uint32_t N;
  uint32_t K;
  uint32_t num_channels_B;
} gemv_matrix_job_t;

static void gemv_matrix_part(void *ctx, int part, int parts) {
  gemv_matrix_job_t *job = (gemv_matrix_job_t *)ctx;
  int n_start, n_end;
  moonshine_par_range((int)job->N, part, parts, 16, &n_start, &n_end);

  const uint32_t K = job->K;
  const int8_t *a_q = job->a_q;
  float a_scale = job->a_scale;
  const int8_t *B_q = job->B_q;
  const float *B_scales = job->B_scales;
  float *C = job->C;
  const uint32_t num_channels_B = job->num_channels_B;

  for (int n = n_start; n < n_end; n++) {
    float b_scale = (num_channels_B == 1) ? B_scales[0] : B_scales[n];
    const int8_t *b_row = B_q + n * K;

    int32_t sum;
    if (K == 288) {
      sum = moonshine_dot_s8_288_pie(a_q, b_row);
    } else if (K == 1152) {
      sum = moonshine_dot_s8_1152_pie(a_q, b_row);
    } else {
      sum = moonshine_fc_dot_s8_pie(a_q, b_row, (int32_t)K);
    }
    C[n] = (float)sum * (a_scale * b_scale);
  }
}
#endif

void moonshine_quantize_int8_rows(const float *A, int8_t *A_q, float *A_scales,
                                  uint32_t M, uint32_t K) {
  for (uint32_t m = 0; m < M; m++) {
    const float *a_row = A + (size_t)m * K;
    int8_t *a_q_row = A_q + (size_t)m * K;

    float max_in = 0.0f;
    uint32_t k = 0;
    for (; k + 4 <= K; k += 4) {
      float v0 = fabsf(a_row[k]);
      float v1 = fabsf(a_row[k + 1]);
      float v2 = fabsf(a_row[k + 2]);
      float v3 = fabsf(a_row[k + 3]);
      if (v0 > max_in)
        max_in = v0;
      if (v1 > max_in)
        max_in = v1;
      if (v2 > max_in)
        max_in = v2;
      if (v3 > max_in)
        max_in = v3;
    }
    for (; k < K; k++) {
      float val = fabsf(a_row[k]);
      if (val > max_in)
        max_in = val;
    }
    float in_scale = (max_in > 0.0f) ? (max_in / 127.0f) : 1.0f;
    float inv_scale = 1.0f / in_scale;
    A_scales[m] = in_scale;

    k = 0;
    for (; k + 4 <= K; k += 4) {
      float val0 = a_row[k] * inv_scale;
      float val1 = a_row[k + 1] * inv_scale;
      float val2 = a_row[k + 2] * inv_scale;
      float val3 = a_row[k + 3] * inv_scale;

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

      a_q_row[k] = (int8_t)q0;
      a_q_row[k + 1] = (int8_t)q1;
      a_q_row[k + 2] = (int8_t)q2;
      a_q_row[k + 3] = (int8_t)q3;
    }
    for (; k < K; k++) {
      float val = a_row[k] * inv_scale;
      int32_t q = (int32_t)(val + (val >= 0.0f ? 0.5f : -0.5f));
      if (q > 127)
        q = 127;
      if (q < -128)
        q = -128;
      a_q_row[k] = (int8_t)q;
    }
  }
}

void moonshine_gemm_int8_prequantized(const int8_t *A_q, const float *A_scales,
                                      const int8_t *B_q, const float *B_scales,
                                      uint32_t num_channels_B, float *C,
                                      uint32_t M, uint32_t N, uint32_t K) {
#ifdef ESP_PLATFORM
  if (N >= 32) {
    gemm_matrix_job_t job = {.A_q = A_q,
                             .A_scales = A_scales,
                             .B_q = B_q,
                             .B_scales = B_scales,
                             .C = C,
                             .M = M,
                             .N = N,
                             .K = K,
                             .num_channels_B = num_channels_B};
    moonshine_parallel_run(gemm_matrix_part, &job);
  } else {
    for (uint32_t n = 0; n < N; n++) {
      float b_scale = (num_channels_B == 1) ? B_scales[0] : B_scales[n];
      const int8_t *b_row = B_q + (size_t)n * K;
      if (K == 288) {
        for (uint32_t m = 0; m < M; m++) {
          const int8_t *a_row = A_q + (size_t)m * 288;
          int32_t sum = moonshine_dot_s8_288_pie(a_row, b_row);
          C[m * N + n] = (float)sum * (A_scales[m] * b_scale);
        }
      } else if (K == 1152) {
        for (uint32_t m = 0; m < M; m++) {
          const int8_t *a_row = A_q + (size_t)m * 1152;
          int32_t sum = moonshine_dot_s8_1152_pie(a_row, b_row);
          C[m * N + n] = (float)sum * (A_scales[m] * b_scale);
        }
      } else {
        for (uint32_t m = 0; m < M; m++) {
          const int8_t *a_row = A_q + (size_t)m * K;
          int32_t sum = moonshine_fc_dot_s8_pie(a_row, b_row, (int32_t)K);
          C[m * N + n] = (float)sum * (A_scales[m] * b_scale);
        }
      }
    }
  }
#else
  for (uint32_t m = 0; m < M; m++) {
    const int8_t *a_row = A_q + m * K;
    float a_scale = A_scales[m];
    float *c_row = C + m * N;
    for (uint32_t n = 0; n < N; n++) {
      float b_scale = (num_channels_B == 1) ? B_scales[0] : B_scales[n];
      const int8_t *b_row = B_q + n * K;
      int32_t sum = 0;
      for (uint32_t k = 0; k < K; k++) {
        sum += (int32_t)a_row[k] * (int32_t)b_row[k];
      }
      c_row[n] = (float)sum * (a_scale * b_scale);
    }
  }
#endif
}

void moonshine_gemm_int8(const float *A, const int8_t *B_q,
                         const float *B_scales, uint32_t num_channels_B,
                         float *C, uint32_t M, uint32_t N, uint32_t K,
                         bool transpose_B) {
  if (transpose_B) {
    // C = A × B^T -> B is [N × K] in memory.

    bool can_use_pie = false;
#ifdef ESP_PLATFORM
    // Verify 16-byte alignment of B_q and K for PIE SIMD
    if (((uintptr_t)B_q % 16 == 0) && (K % 16 == 0) &&
        (K <= sizeof(s_gemm_scratch_q))) {
      can_use_pie = true;
    }
#endif

    if (can_use_pie && M > 1 && M <= 256) {
      // Memory Bandwidth Optimization: Cache Tiling for Matrix-Matrix product
      // Prefer fast internal SRAM; fallback to PSRAM if heap is tight
#ifdef ESP_PLATFORM
      int8_t *A_q_unaligned = (int8_t *)heap_caps_malloc(
          M * K + 16, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
      if (!A_q_unaligned) {
        A_q_unaligned = (int8_t *)heap_caps_malloc(
            M * K + 16, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
      }
      int8_t *A_q = A_q_unaligned
                        ? (int8_t *)(((uintptr_t)A_q_unaligned + 15) & ~15)
                        : NULL;
      float *A_scales = (float *)heap_caps_malloc(
          M * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
      if (!A_scales) {
        A_scales = (float *)heap_caps_malloc(
            M * sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
      }

      if (A_q && A_scales) {
        moonshine_quantize_int8_rows(A, A_q, A_scales, M, K);
        moonshine_gemm_int8_prequantized(A_q, A_scales, B_q, B_scales,
                                         num_channels_B, C, M, N, K);

        heap_caps_free(A_q_unaligned);
        heap_caps_free(A_scales);
      } else {
        if (A_q_unaligned)
          heap_caps_free(A_q_unaligned);
        if (A_scales)
          heap_caps_free(A_scales);
        can_use_pie = false;
      }
#endif
    }

    if (!can_use_pie || M <= 1 || M > 256) {
      for (uint32_t m = 0; m < M; m++) {
        const float *a_row = A + m * K;
        float in_scale = 1.0f;

        if (can_use_pie) {
#ifdef ESP_PLATFORM
          // Dynamic Activation Quantization (A -> int8)
          float max_in = 0.0f;
          for (uint32_t k = 0; k < K; k++) {
            float val = fabsf(a_row[k]);
            if (val > max_in)
              max_in = val;
          }

          in_scale = (max_in > 0.0f) ? (max_in / 127.0f) : 1.0f;
          float inv_scale = 1.0f / in_scale;

          for (uint32_t k = 0; k < K; k++) {
            float val = a_row[k] * inv_scale;
            int32_t q = (int32_t)(val + (val >= 0.0f ? 0.5f : -0.5f));
            if (q > 127)
              q = 127;
            if (q < -128)
              q = -128;
            s_gemm_scratch_q[k] = (int8_t)q;
          }

          if (N >= 64) {
            gemv_matrix_job_t job = {.a_q = s_gemm_scratch_q,
                                     .a_scale = in_scale,
                                     .B_q = B_q,
                                     .B_scales = B_scales,
                                     .C = C + m * N,
                                     .N = N,
                                     .K = K,
                                     .num_channels_B = num_channels_B};
            moonshine_parallel_run(gemv_matrix_part, &job);
            continue;
          }
#endif
        }

        for (uint32_t n = 0; n < N; n++) {
          float scale = (num_channels_B == 1) ? B_scales[0] : B_scales[n];
          const int8_t *b_row = B_q + n * K;

          if (can_use_pie) {
#ifdef ESP_PLATFORM
            int32_t sum;
            if (K == 288) {
              sum = moonshine_dot_s8_288_pie(s_gemm_scratch_q, b_row);
            } else if (K == 1152) {
              sum = moonshine_dot_s8_1152_pie(s_gemm_scratch_q, b_row);
            } else {
              sum =
                  moonshine_fc_dot_s8_pie(s_gemm_scratch_q, b_row, (int32_t)K);
            }
            C[m * N + n] = (float)sum * (in_scale * scale);
#endif
          } else {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
              sum += a_row[k] * ((float)b_row[k] * scale);
            }
            C[m * N + n] = sum;
          }
        }
      }
    }
  } else {
    // C = A × B -> B is [K × N] in memory.
    for (uint32_t m = 0; m < M; m++) {
      for (uint32_t n = 0; n < N; n++) {
        float sum = 0.0f;
        float scale = (num_channels_B == 1) ? B_scales[0] : B_scales[n];

        for (uint32_t k = 0; k < K; k++) {
          float a_val = A[m * K + k];
          int8_t b_val_q = B_q[k * N + n];
          sum += a_val * ((float)b_val_q * scale);
        }
        C[m * N + n] = sum;
      }
    }
  }
}

void moonshine_gemv_fp32(const float *A, // [M × N] matrix
                         const float *x, // [N] vector
                         float *y,       // [M] output vector
                         uint32_t M, uint32_t N) {
  // Matrix-vector multiplication: y = A × x
  for (uint32_t m = 0; m < M; m++) {
    float sum = 0.0f;
    for (uint32_t n = 0; n < N; n++) {
      sum += A[m * N + n] * x[n];
    }
    y[m] = sum;
  }
}

/* ========================================================================
 * FAST IEEE-754 TRANSCENDENTALS (Degree-6 polynomial minimax, max rel err < 1.6e-7)
 * ======================================================================== */

static inline float moonshine_fast_expf(float x) {
  if (x < -16.0f)
    return 0.0f;
  if (x > 16.0f)
    x = 16.0f;
  float z = x * 1.4426950408889634f;
  int32_t k = (int32_t)(z + (z >= 0.0f ? 0.5f : -0.5f));
  float r = z - (float)k;
  const float c1 = 0.69314718056f;
  const float c2 = 0.24022650696f;
  const float c3 = 0.05550410866f;
  const float c4 = 0.00961812911f;
  const float c5 = 0.00133335581f;
  const float c6 = 0.00015403530f;
  float poly = 1.0f + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * c6)))));
  union {
    float f;
    int32_t i;
  } scale;
  scale.i = (k + 127) << 23;
  return poly * scale.f;
}

/* ========================================================================
 * ACTIVATION FUNCTIONS
 * ======================================================================== */

void moonshine_relu(float *data, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    if (data[i] < 0.0f) {
      data[i] = 0.0f;
    }
  }
}

typedef struct {
  float *data;
  uint32_t size;
} act_job_t;

#ifdef ESP_PLATFORM
static void gelu_part(void *ctx, int part, int parts) {
  act_job_t *job = (act_job_t *)ctx;
  int lo, hi;
  moonshine_par_range((int)job->size, part, parts, 1, &lo, &hi);
  float *data = job->data;
  for (int i = lo; i < hi; i++) {
    float x = data[i];
    float x3 = x * x * x;
    float arg = -1.5957691216f * (x + 0.044715f * x3);
    if (arg > 16.0f) {
      data[i] = 0.0f;
    } else if (arg < -16.0f) {
      data[i] = x;
    } else {
      data[i] = x / (1.0f + moonshine_fast_expf(arg));
    }
  }
}

static void tanh_part(void *ctx, int part, int parts) {
  act_job_t *job = (act_job_t *)ctx;
  int lo, hi;
  moonshine_par_range((int)job->size, part, parts, 1, &lo, &hi);
  float *data = job->data;
  for (int i = lo; i < hi; i++) {
    float x = data[i];
    if (x >= 8.0f) {
      data[i] = 1.0f;
    } else if (x <= -8.0f) {
      data[i] = -1.0f;
    } else {
      data[i] = 2.0f / (1.0f + moonshine_fast_expf(-2.0f * x)) - 1.0f;
    }
  }
}
#endif

void moonshine_gelu(float *data, uint32_t size) {
#ifdef ESP_PLATFORM
  if (size >= 1024) {
    act_job_t job = {.data = data, .size = size};
    moonshine_parallel_run(gelu_part, &job);
    return;
  }
#endif
  for (uint32_t i = 0; i < size; i++) {
    float x = data[i];
    float x3 = x * x * x;
    float arg = -1.5957691216f * (x + 0.044715f * x3);
    if (arg > 16.0f) {
      data[i] = 0.0f;
    } else if (arg < -16.0f) {
      data[i] = x;
    } else {
      data[i] = x / (1.0f + moonshine_fast_expf(arg));
    }
  }
}

void moonshine_silu(float *data, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    float x = data[i];
    data[i] = x / (1.0f + moonshine_fast_expf(-x));
  }
}

void moonshine_tanh(float *data, uint32_t size) {
#ifdef ESP_PLATFORM
  if (size >= 1024) {
    act_job_t job = {.data = data, .size = size};
    moonshine_parallel_run(tanh_part, &job);
    return;
  }
#endif
  for (uint32_t i = 0; i < size; i++) {
    float x = data[i];
    if (x >= 8.0f) {
      data[i] = 1.0f;
    } else if (x <= -8.0f) {
      data[i] = -1.0f;
    } else {
      data[i] = 2.0f / (1.0f + moonshine_fast_expf(-2.0f * x)) - 1.0f;
    }
  }
}

void moonshine_softmax(float *data, uint32_t size) {
  if (size == 0)
    return;

  // Find max for numerical stability
  float max_val = data[0];
  for (uint32_t i = 1; i < size; i++) {
    if (data[i] > max_val) {
      max_val = data[i];
    }
  }

  // Compute exp(x - max) and sum with fast IEEE-754 exponent
  float sum = 0.0f;
  for (uint32_t i = 0; i < size; i++) {
    float v = moonshine_fast_expf(data[i] - max_val);
    data[i] = v;
    sum += v;
  }

  // Normalize with reciprocal multiplication
  float inv_sum = 1.0f / sum;
  for (uint32_t i = 0; i < size; i++) {
    data[i] *= inv_sum;
  }
}

/* ========================================================================
 * NORMALIZATION
 * ======================================================================== */

void moonshine_layer_norm(float *data,         // [N] input/output
                          const float *weight, // [N] gamma
                          const float *bias,   // [N] beta
                          uint32_t size, float eps) {
  // Compute mean with 4 accumulators
  float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
  uint32_t i = 0;
  for (; i + 4 <= size; i += 4) {
    m0 += data[i];
    m1 += data[i + 1];
    m2 += data[i + 2];
    m3 += data[i + 3];
  }
  float sum_m = (m0 + m1) + (m2 + m3);
  for (; i < size; i++) {
    sum_m += data[i];
  }
  float mean = sum_m / (float)size;

  // Compute variance with 4 accumulators
  float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
  i = 0;
  for (; i + 4 <= size; i += 4) {
    float d0 = data[i] - mean;
    float d1 = data[i + 1] - mean;
    float d2 = data[i + 2] - mean;
    float d3 = data[i + 3] - mean;
    v0 += d0 * d0;
    v1 += d1 * d1;
    v2 += d2 * d2;
    v3 += d3 * d3;
  }
  float sum_v = (v0 + v1) + (v2 + v3);
  for (; i < size; i++) {
    float diff = data[i] - mean;
    sum_v += diff * diff;
  }
  float variance = sum_v / (float)size;

  // Normalize with 4 accumulators
  float inv_std = 1.0f / sqrtf(variance + eps);
  i = 0;
  if (weight && !bias) {
    for (; i + 4 <= size; i += 4) {
      data[i] = (data[i] - mean) * inv_std * weight[i];
      data[i + 1] = (data[i + 1] - mean) * inv_std * weight[i + 1];
      data[i + 2] = (data[i + 2] - mean) * inv_std * weight[i + 2];
      data[i + 3] = (data[i + 3] - mean) * inv_std * weight[i + 3];
    }
  } else if (weight && bias) {
    for (; i + 4 <= size; i += 4) {
      data[i] = (data[i] - mean) * inv_std * weight[i] + bias[i];
      data[i + 1] =
          (data[i + 1] - mean) * inv_std * weight[i + 1] + bias[i + 1];
      data[i + 2] =
          (data[i + 2] - mean) * inv_std * weight[i + 2] + bias[i + 2];
      data[i + 3] =
          (data[i + 3] - mean) * inv_std * weight[i + 3] + bias[i + 3];
    }
  } else {
    for (; i + 4 <= size; i += 4) {
      data[i] = (data[i] - mean) * inv_std;
      data[i + 1] = (data[i + 1] - mean) * inv_std;
      data[i + 2] = (data[i + 2] - mean) * inv_std;
      data[i + 3] = (data[i + 3] - mean) * inv_std;
    }
  }
  for (; i < size; i++) {
    data[i] = (data[i] - mean) * inv_std;
    if (weight)
      data[i] *= weight[i];
    if (bias)
      data[i] += bias[i];
  }
}

void moonshine_rms_norm(float *data, const float *weight, uint32_t size,
                        float eps) {
  // Compute RMS with 4 accumulators
  float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;
  uint32_t i = 0;
  for (; i + 4 <= size; i += 4) {
    sq0 += data[i] * data[i];
    sq1 += data[i + 1] * data[i + 1];
    sq2 += data[i + 2] * data[i + 2];
    sq3 += data[i + 3] * data[i + 3];
  }
  float sum_squares = (sq0 + sq1) + (sq2 + sq3);
  for (; i < size; i++) {
    sum_squares += data[i] * data[i];
  }
  float rms = sqrtf(sum_squares / (float)size + eps);

  // Normalize and scale
  float inv_rms = 1.0f / rms;
  i = 0;
  for (; i + 4 <= size; i += 4) {
    data[i] = data[i] * inv_rms * weight[i];
    data[i + 1] = data[i + 1] * inv_rms * weight[i + 1];
    data[i + 2] = data[i + 2] * inv_rms * weight[i + 2];
    data[i + 3] = data[i + 3] * inv_rms * weight[i + 3];
  }
  for (; i < size; i++) {
    data[i] = data[i] * inv_rms * weight[i];
  }
}

typedef struct {
  float *data;
  const float *weight;
  const float *bias;
  uint32_t num_channels;
  uint32_t spatial_size;
  float mean;
  float inv_std;
} gn_norm_job_t;

#ifdef ESP_PLATFORM
static void gn_norm_part(void *ctx, int part, int parts) {
  gn_norm_job_t *job = (gn_norm_job_t *)ctx;
  int c_start, c_end;
  moonshine_par_range((int)job->num_channels, part, parts, 1, &c_start, &c_end);
  const uint32_t S = job->spatial_size;
  float *data = job->data;
  const float *weight = job->weight;
  const float *bias = job->bias;
  const float mean = job->mean;
  const float inv_std = job->inv_std;

  for (int c = c_start; c < c_end; c++) {
    float w = weight ? weight[c] : 1.0f;
    float b = bias ? bias[c] : 0.0f;
    float scale = inv_std * w;
    float *row = data + (size_t)c * S;
    uint32_t s = 0;
    for (; s + 4 <= S; s += 4) {
      row[s] = (row[s] - mean) * scale + b;
      row[s + 1] = (row[s + 1] - mean) * scale + b;
      row[s + 2] = (row[s + 2] - mean) * scale + b;
      row[s + 3] = (row[s + 3] - mean) * scale + b;
    }
    for (; s < S; s++) {
      row[s] = (row[s] - mean) * scale + b;
    }
  }
}
#endif

void moonshine_group_norm(float *data, const float *weight, const float *bias,
                          uint32_t num_channels, uint32_t num_groups,
                          uint32_t spatial_size, float eps) {
  if (num_groups == 1) {
    uint32_t total = num_channels * spatial_size;
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    uint32_t i = 0;
    for (; i + 4 <= total; i += 4) {
      sum0 += data[i];
      sum1 += data[i + 1];
      sum2 += data[i + 2];
      sum3 += data[i + 3];
    }
    float sum_m = (sum0 + sum1) + (sum2 + sum3);
    for (; i < total; i++) {
      sum_m += data[i];
    }
    float mean = sum_m / (float)total;

    float var0 = 0.0f, var1 = 0.0f, var2 = 0.0f, var3 = 0.0f;
    i = 0;
    for (; i + 4 <= total; i += 4) {
      float d0 = data[i] - mean;
      float d1 = data[i + 1] - mean;
      float d2 = data[i + 2] - mean;
      float d3 = data[i + 3] - mean;
      var0 += d0 * d0;
      var1 += d1 * d1;
      var2 += d2 * d2;
      var3 += d3 * d3;
    }
    float sum_v = (var0 + var1) + (var2 + var3);
    for (; i < total; i++) {
      float d = data[i] - mean;
      sum_v += d * d;
    }
    float variance = sum_v / (float)total;
    float inv_std = 1.0f / sqrtf(variance + eps);

#ifdef ESP_PLATFORM
    gn_norm_job_t job = {
        .data = data,
        .weight = weight,
        .bias = bias,
        .num_channels = num_channels,
        .spatial_size = spatial_size,
        .mean = mean,
        .inv_std = inv_std,
    };
    moonshine_parallel_run(gn_norm_part, &job);
#else
    for (uint32_t c = 0; c < num_channels; c++) {
      float w = weight ? weight[c] : 1.0f;
      float b = bias ? bias[c] : 0.0f;
      float scale = inv_std * w;
      float *row = data + (size_t)c * spatial_size;
      for (uint32_t s = 0; s < spatial_size; s++) {
        row[s] = (row[s] - mean) * scale + b;
      }
    }
#endif
    return;
  }

  uint32_t channels_per_group = num_channels / num_groups;
  uint32_t group_size = channels_per_group * spatial_size;

  for (uint32_t g = 0; g < num_groups; g++) {
    float mean = 0.0f;
    for (uint32_t c = 0; c < channels_per_group; c++) {
      uint32_t channel_idx = g * channels_per_group + c;
      for (uint32_t s = 0; s < spatial_size; s++) {
        mean += data[channel_idx * spatial_size + s];
      }
    }
    mean /= (float)group_size;

    float variance = 0.0f;
    for (uint32_t c = 0; c < channels_per_group; c++) {
      uint32_t channel_idx = g * channels_per_group + c;
      for (uint32_t s = 0; s < spatial_size; s++) {
        float diff = data[channel_idx * spatial_size + s] - mean;
        variance += diff * diff;
      }
    }
    variance /= (float)group_size;

    float inv_std = 1.0f / sqrtf(variance + eps);
    for (uint32_t c = 0; c < channels_per_group; c++) {
      uint32_t channel_idx = g * channels_per_group + c;
      float w = weight ? weight[channel_idx] : 1.0f;
      float b = bias ? bias[channel_idx] : 0.0f;
      float scale = inv_std * w;
      for (uint32_t s = 0; s < spatial_size; s++) {
        uint32_t idx = channel_idx * spatial_size + s;
        data[idx] = (data[idx] - mean) * scale + b;
      }
    }
  }
}

void moonshine_apply_rope(float *data, uint32_t seq_len, uint32_t hidden_size,
                          uint32_t num_heads, float base) {
  uint32_t head_dim = hidden_size / num_heads;
  uint32_t rotary_dim = (uint32_t)(head_dim * 0.9f);
  uint32_t half_rotary = rotary_dim / 2;
  if (half_rotary == 0 || seq_len == 0 || num_heads == 0)
    return;

  if (half_rotary > 64)
    half_rotary = 64;

  // Precompute inverse frequencies once for all dimensions
  static float s_cached_base = 0.0f;
  static uint32_t s_cached_rotary_dim = 0;
  static float s_cached_inv_freq[64];

  if (base != s_cached_base || rotary_dim != s_cached_rotary_dim) {
    for (uint32_t d = 0; d < half_rotary; d++) {
      s_cached_inv_freq[d] =
          1.0f / powf(base, (2.0f * (float)d) / (float)rotary_dim);
    }
    s_cached_base = base;
    s_cached_rotary_dim = rotary_dim;
  }

  float cos_table[64];
  float sin_table[64];

  // RoPE is applied to pairs of dimensions within each head
  for (uint32_t pos = 0; pos < seq_len; pos++) {
    // Precompute cos and sin for this position once across all heads
    for (uint32_t d = 0; d < half_rotary; d++) {
      float angle = (float)pos * s_cached_inv_freq[d];
      cos_table[d] = cosf(angle);
      sin_table[d] = sinf(angle);
    }

    float *pos_data = data + pos * hidden_size;
    for (uint32_t h = 0; h < num_heads; h++) {
      float *head_data = pos_data + h * head_dim;

      // Apply rotation to each pair of dimensions (up to rotary_dim)
      for (uint32_t d = 0; d < half_rotary; d++) {
        float cos_a = cos_table[d];
        float sin_a = sin_table[d];

        uint32_t idx1 = 2 * d;
        uint32_t idx2 = idx1 + 1;
        float x = head_data[idx1];
        float y = head_data[idx2];

        // Apply 2D rotation
        head_data[idx1] = x * cos_a - y * sin_a;
        head_data[idx2] = x * sin_a + y * cos_a;
      }
    }
  }
}

/* ========================================================================
 * ELEMENT-WISE OPERATIONS
 * ======================================================================== */

void moonshine_add(const float *a, const float *b, float *out, uint32_t size) {
  uint32_t i = 0;
  for (; i + 4 <= size; i += 4) {
    out[i] = a[i] + b[i];
    out[i + 1] = a[i + 1] + b[i + 1];
    out[i + 2] = a[i + 2] + b[i + 2];
    out[i + 3] = a[i + 3] + b[i + 3];
  }
  for (; i < size; i++) {
    out[i] = a[i] + b[i];
  }
}

void moonshine_add_inplace(float *a, const float *b, uint32_t size) {
  uint32_t i = 0;
  for (; i + 4 <= size; i += 4) {
    a[i] += b[i];
    a[i + 1] += b[i + 1];
    a[i + 2] += b[i + 2];
    a[i + 3] += b[i + 3];
  }
  for (; i < size; i++) {
    a[i] += b[i];
  }
}

void moonshine_multiply(const float *a, const float *b, float *out,
                        uint32_t size) {
  uint32_t i = 0;
  for (; i + 4 <= size; i += 4) {
    out[i] = a[i] * b[i];
    out[i + 1] = a[i + 1] * b[i + 1];
    out[i + 2] = a[i + 2] * b[i + 2];
    out[i + 3] = a[i + 3] * b[i + 3];
  }
  for (; i < size; i++) {
    out[i] = a[i] * b[i];
  }
}

void moonshine_scale(float *data, float scale, uint32_t size) {
  uint32_t i = 0;
  for (; i + 4 <= size; i += 4) {
    data[i] *= scale;
    data[i + 1] *= scale;
    data[i + 2] *= scale;
    data[i + 3] *= scale;
  }
  for (; i < size; i++) {
    data[i] *= scale;
  }
}

/* ========================================================================
 * CONVOLUTION (1D)
 * ======================================================================== */

void moonshine_conv1d(
    const float *input,  // [in_channels × seq_len]
    const float *kernel, // [out_channels × in_channels × kernel_size]
    const float *bias,   // [out_channels]
    float *output,       // [out_channels × out_seq_len]
    uint32_t in_channels, uint32_t out_channels, uint32_t seq_len,
    uint32_t kernel_size, uint32_t stride, uint32_t padding) {
  // Calculate output sequence length
  uint32_t out_seq_len = ((seq_len + 2 * padding - kernel_size) / stride) + 1;

  // For each output channel
  for (uint32_t oc = 0; oc < out_channels; oc++) {
    // For each output position
    for (uint32_t pos = 0; pos < out_seq_len; pos++) {
      float sum = 0.0f;

      // Convolve across input channels
      for (uint32_t ic = 0; ic < in_channels; ic++) {
        // Convolve across kernel
        for (uint32_t k = 0; k < kernel_size; k++) {
          int32_t input_pos = (int32_t)(pos * stride + k) - (int32_t)padding;

          if (input_pos >= 0 && input_pos < (int32_t)seq_len) {
            float in_val = input[ic * seq_len + input_pos];
            float kernel_val =
                kernel[(oc * in_channels + ic) * kernel_size + k];
            sum += in_val * kernel_val;
          }
        }
      }

      // Add bias
      if (bias) {
        sum += bias[oc];
      }

      output[oc * out_seq_len + pos] = sum;
    }
  }
}

typedef struct {
  const float *input;
  int8_t *A_q;
  float *A_scales;
  float *temp_buf[MOONSHINE_PAR_PARTS];
  uint32_t in_channels;
  uint32_t seq_len;
  uint32_t kernel_size;
  uint32_t stride;
  uint32_t padding;
  uint32_t chunk_pos_start;
  uint32_t chunk_len;
  uint32_t K;
  uint32_t K_padded;
} conv1d_pack_job_t;

#ifdef ESP_PLATFORM
static void conv1d_pack_part(void *ctx, int part, int parts) {
  conv1d_pack_job_t *job = (conv1d_pack_job_t *)ctx;
  int p_start, p_end;
  moonshine_par_range((int)job->chunk_len, part, parts, 1, &p_start, &p_end);

  const float *input = job->input;
  int8_t *A_q = job->A_q;
  float *A_scales = job->A_scales;
  float *temp_buf = job->temp_buf[part];
  const uint32_t in_channels = job->in_channels;
  const uint32_t seq_len = job->seq_len;
  const uint32_t kernel_size = job->kernel_size;
  const uint32_t stride = job->stride;
  const uint32_t padding = job->padding;
  const uint32_t K = job->K;
  const uint32_t K_padded = job->K_padded;
  const uint32_t chunk_pos_start = job->chunk_pos_start;
  const bool needs_padding = (K != K_padded);

  for (int p = p_start; p < p_end; p++) {
    uint32_t pos = chunk_pos_start + p;
    int8_t *a_q_pos = A_q + (size_t)p * K_padded;
    float max_in = 0.0f;
    uint32_t dst_idx = 0;

    int32_t base_pos = (int32_t)(pos * stride) - (int32_t)padding;
    if (base_pos >= 0 && (base_pos + (int32_t)kernel_size <= (int32_t)seq_len)) {
      for (uint32_t ic = 0; ic < in_channels; ic++) {
        const float *in_ptr = input + (size_t)ic * seq_len + base_pos;
        for (uint32_t k = 0; k < kernel_size; k++) {
          float v = in_ptr[k];
          float a = fabsf(v);
          if (a > max_in)
            max_in = a;
          temp_buf[dst_idx++] = v;
        }
      }
    } else {
      for (uint32_t ic = 0; ic < in_channels; ic++) {
        const float *in_ic = input + (size_t)ic * seq_len;
        for (uint32_t k = 0; k < kernel_size; k++) {
          int32_t input_pos = base_pos + (int32_t)k;
          float in_val = (input_pos >= 0 && input_pos < (int32_t)seq_len)
                             ? in_ic[input_pos]
                             : 0.0f;
          float a = fabsf(in_val);
          if (a > max_in)
            max_in = a;
          temp_buf[dst_idx++] = in_val;
        }
      }
    }

    float in_scale = (max_in > 0.0f) ? (max_in / 127.0f) : 1.0f;
    float inv_scale = 1.0f / in_scale;
    A_scales[p] = in_scale;

    uint32_t i = 0;
    for (; i + 4 <= K; i += 4) {
      float v0 = temp_buf[i] * inv_scale;
      float v1 = temp_buf[i + 1] * inv_scale;
      float v2 = temp_buf[i + 2] * inv_scale;
      float v3 = temp_buf[i + 3] * inv_scale;
      int32_t q0 = (int32_t)(v0 + (v0 >= 0.0f ? 0.5f : -0.5f));
      int32_t q1 = (int32_t)(v1 + (v1 >= 0.0f ? 0.5f : -0.5f));
      int32_t q2 = (int32_t)(v2 + (v2 >= 0.0f ? 0.5f : -0.5f));
      int32_t q3 = (int32_t)(v3 + (v3 >= 0.0f ? 0.5f : -0.5f));
      if (q0 > 127) q0 = 127; else if (q0 < -128) q0 = -128;
      if (q1 > 127) q1 = 127; else if (q1 < -128) q1 = -128;
      if (q2 > 127) q2 = 127; else if (q2 < -128) q2 = -128;
      if (q3 > 127) q3 = 127; else if (q3 < -128) q3 = -128;
      a_q_pos[i] = (int8_t)q0;
      a_q_pos[i + 1] = (int8_t)q1;
      a_q_pos[i + 2] = (int8_t)q2;
      a_q_pos[i + 3] = (int8_t)q3;
    }
    for (; i < K; i++) {
      float val = temp_buf[i] * inv_scale;
      int32_t q = (int32_t)(val + (val >= 0.0f ? 0.5f : -0.5f));
      if (q > 127)
        q = 127;
      if (q < -128)
        q = -128;
      a_q_pos[i] = (int8_t)q;
    }

    if (needs_padding) {
      for (uint32_t pi = K; pi < K_padded; pi++) {
        a_q_pos[pi] = 0;
      }
    }
  }
}

typedef struct {
  const int8_t *A_q;
  const float *A_scales;
  const int8_t *k_base;
  const float *kernel_scales;
  const float *bias;
  float *output;
  uint32_t out_channels;
  uint32_t out_seq_len;
  uint32_t chunk_pos_start;
  uint32_t chunk_len;
  uint32_t K_padded;
  uint32_t num_channels_q;
} conv1d_gemm_job_t;

static void conv1d_gemm_part(void *ctx, int part, int parts) {
  conv1d_gemm_job_t *job = (conv1d_gemm_job_t *)ctx;
  int oc_start, oc_end;
  moonshine_par_range((int)job->out_channels, part, parts, 1, &oc_start,
                      &oc_end);

  const int8_t *A_q = job->A_q;
  const float *A_scales = job->A_scales;
  const int8_t *k_base = job->k_base;
  const float *kernel_scales = job->kernel_scales;
  const float *bias = job->bias;
  float *output = job->output;
  const uint32_t out_seq_len = job->out_seq_len;
  const uint32_t chunk_pos_start = job->chunk_pos_start;
  const uint32_t chunk_len = job->chunk_len;
  const uint32_t K_padded = job->K_padded;
  const uint32_t num_channels_q = job->num_channels_q;

  for (int oc = oc_start; oc < oc_end; oc++) {
    float b_scale =
        (num_channels_q == 1) ? kernel_scales[0] : kernel_scales[oc];
    const int8_t *b_row = k_base + (size_t)oc * K_padded;
    float b_bias = bias ? bias[oc] : 0.0f;
    float *out_row = output + (size_t)oc * out_seq_len + chunk_pos_start;

    uint32_t p = 0;
    for (; p + 4 <= chunk_len; p += 4) {
      const int8_t *a_row0 = A_q + (size_t)p * K_padded;
      const int8_t *a_row1 = A_q + (size_t)(p + 1) * K_padded;
      const int8_t *a_row2 = A_q + (size_t)(p + 2) * K_padded;
      const int8_t *a_row3 = A_q + (size_t)(p + 3) * K_padded;
      int32_t sum0 = moonshine_fc_dot_s8_pie(a_row0, b_row, (int32_t)K_padded);
      int32_t sum1 = moonshine_fc_dot_s8_pie(a_row1, b_row, (int32_t)K_padded);
      int32_t sum2 = moonshine_fc_dot_s8_pie(a_row2, b_row, (int32_t)K_padded);
      int32_t sum3 = moonshine_fc_dot_s8_pie(a_row3, b_row, (int32_t)K_padded);
      out_row[p]     = (float)sum0 * (A_scales[p]     * b_scale) + b_bias;
      out_row[p + 1] = (float)sum1 * (A_scales[p + 1] * b_scale) + b_bias;
      out_row[p + 2] = (float)sum2 * (A_scales[p + 2] * b_scale) + b_bias;
      out_row[p + 3] = (float)sum3 * (A_scales[p + 3] * b_scale) + b_bias;
    }
    for (; p + 2 <= chunk_len; p += 2) {
      const int8_t *a_row0 = A_q + (size_t)p * K_padded;
      const int8_t *a_row1 = A_q + (size_t)(p + 1) * K_padded;
      int32_t sum0 = moonshine_fc_dot_s8_pie(a_row0, b_row, (int32_t)K_padded);
      int32_t sum1 = moonshine_fc_dot_s8_pie(a_row1, b_row, (int32_t)K_padded);
      out_row[p]     = (float)sum0 * (A_scales[p]     * b_scale) + b_bias;
      out_row[p + 1] = (float)sum1 * (A_scales[p + 1] * b_scale) + b_bias;
    }
    for (; p < chunk_len; p++) {
      const int8_t *a_row = A_q + (size_t)p * K_padded;
      float a_scale = A_scales[p];
      int32_t sum = moonshine_fc_dot_s8_pie(a_row, b_row, (int32_t)K_padded);
      out_row[p] = (float)sum * (a_scale * b_scale) + b_bias;
    }
  }
}
#endif

void moonshine_conv1d_int8(const float *input, const int8_t *kernel_q,
                           const float *kernel_scales, uint32_t num_channels_q,
                           const float *bias, float *output,
                           uint32_t in_channels, uint32_t out_channels,
                           uint32_t seq_len, uint32_t kernel_size,
                           uint32_t stride, uint32_t padding) {
  uint32_t out_seq_len = ((seq_len + 2 * padding - kernel_size) / stride) + 1;
  uint32_t K = in_channels * kernel_size;
  // Round K up to next multiple of 16 for PIE SIMD alignment
  uint32_t K_padded = (K + 15) & ~15u;

  bool can_use_pie = false;
  bool needs_padding = (K != K_padded);
#ifdef ESP_PLATFORM
  if ((uintptr_t)kernel_q % 16 == 0) {
    can_use_pie = true;
  }
#endif

  ESP_LOGD("Conv1D", "can_use_pie: %d, K: %lu, K_padded: %lu, align: %d",
           can_use_pie, K, K_padded, (int)((uintptr_t)kernel_q % 16));

  if (can_use_pie) {
#ifdef ESP_PLATFORM
    int8_t *kernel_padded_all = NULL;
    if (needs_padding) {
      size_t total_padded_bytes = (size_t)out_channels * K_padded;
      kernel_padded_all = (int8_t *)heap_caps_aligned_alloc(
          16, total_padded_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
      if (!kernel_padded_all) {
        kernel_padded_all = (int8_t *)heap_caps_aligned_alloc(
            16, total_padded_bytes, MALLOC_CAP_SPIRAM);
      }
      if (!kernel_padded_all) {
        ESP_LOGE("Conv1D", "Failed to allocate padded kernel buffer");
        goto fallback;
      }
      memset(kernel_padded_all, 0, total_padded_bytes);
      for (uint32_t oc = 0; oc < out_channels; oc++) {
        memcpy(kernel_padded_all + (size_t)oc * K_padded,
               kernel_q + (size_t)oc * K, K);
      }
    }

    const int8_t *k_base = needs_padding ? kernel_padded_all : kernel_q;

    // Choose chunk size so A_q fits entirely in fast internal HP SRAM (~64 KB max)
    uint32_t max_pos_per_chunk = (64 * 1024) / K_padded;
    if (max_pos_per_chunk == 0)
      max_pos_per_chunk = 1;
    if (max_pos_per_chunk > out_seq_len)
      max_pos_per_chunk = out_seq_len;

    size_t a_q_bytes = (size_t)max_pos_per_chunk * K_padded;
    int8_t *A_q = (int8_t *)heap_caps_aligned_alloc(
        16, a_q_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
    if (!A_q) {
      A_q = (int8_t *)heap_caps_aligned_alloc(16, a_q_bytes, MALLOC_CAP_SPIRAM);
    }
    float *A_scales = (float *)heap_caps_malloc(
        max_pos_per_chunk * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
    if (!A_scales) {
      A_scales = (float *)heap_caps_malloc(max_pos_per_chunk * sizeof(float),
                                           MALLOC_CAP_SPIRAM);
    }
    float *temp_pack = (float *)heap_caps_malloc(
        2 * K_padded * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
    if (!temp_pack) {
      temp_pack = (float *)heap_caps_malloc(
          2 * K_padded * sizeof(float), MALLOC_CAP_SPIRAM);
    }

    if (!A_q || !A_scales || !temp_pack) {
      if (A_q)
        heap_caps_free(A_q);
      if (A_scales)
        heap_caps_free(A_scales);
      if (temp_pack)
        heap_caps_free(temp_pack);
      if (kernel_padded_all)
        heap_caps_free(kernel_padded_all);
      goto fallback;
    }

    for (uint32_t pos_start = 0; pos_start < out_seq_len;
         pos_start += max_pos_per_chunk) {
      uint32_t chunk_len = (pos_start + max_pos_per_chunk > out_seq_len)
                               ? (out_seq_len - pos_start)
                               : max_pos_per_chunk;

      // Step 1: Parallel pack + quantize input chunk into A_q (HP SRAM)
      conv1d_pack_job_t pack_job = {
          .input = input,
          .A_q = A_q,
          .A_scales = A_scales,
          .temp_buf = {temp_pack, temp_pack + K_padded},
          .in_channels = in_channels,
          .seq_len = seq_len,
          .kernel_size = kernel_size,
          .stride = stride,
          .padding = padding,
          .chunk_pos_start = pos_start,
          .chunk_len = chunk_len,
          .K = K,
          .K_padded = K_padded,
      };
      moonshine_parallel_run(conv1d_pack_part, &pack_job);

      // Step 2: Parallel GEMM across output channels with weights read
      // sequentially
      conv1d_gemm_job_t gemm_job = {
          .A_q = A_q,
          .A_scales = A_scales,
          .k_base = k_base,
          .kernel_scales = kernel_scales,
          .bias = bias,
          .output = output,
          .out_channels = out_channels,
          .out_seq_len = out_seq_len,
          .chunk_pos_start = pos_start,
          .chunk_len = chunk_len,
          .K_padded = K_padded,
          .num_channels_q = num_channels_q,
      };
      moonshine_parallel_run(conv1d_gemm_part, &gemm_job);
    }

    heap_caps_free(A_q);
    heap_caps_free(A_scales);
    heap_caps_free(temp_pack);
    if (kernel_padded_all)
      heap_caps_free(kernel_padded_all);
    return;
#endif
  }

fallback:
  // Fallback C implementation
  for (uint32_t oc = 0; oc < out_channels; oc++) {
    float scale = (num_channels_q == 1) ? kernel_scales[0] : kernel_scales[oc];

    for (uint32_t pos = 0; pos < out_seq_len; pos++) {
      float sum = 0.0f;

      for (uint32_t ic = 0; ic < in_channels; ic++) {
        for (uint32_t k = 0; k < kernel_size; k++) {
          int32_t input_pos = (int32_t)(pos * stride + k) - (int32_t)padding;

          if (input_pos >= 0 && input_pos < (int32_t)seq_len) {
            float in_val = input[ic * seq_len + input_pos];
            int8_t q_val = kernel_q[(oc * in_channels + ic) * kernel_size + k];
            sum += in_val * ((float)q_val * scale);
          }
        }
      }

      if (bias) {
        sum += bias[oc];
      }

      output[oc * out_seq_len + pos] = sum;
    }
  }
}

/* ========================================================================
 * UTILITIES
 * ======================================================================== */

void moonshine_copy(float *dst, const float *src, uint32_t size) {
  memcpy(dst, src, size * sizeof(float));
}

void moonshine_zero(float *data, uint32_t size) {
  memset(data, 0, size * sizeof(float));
}

float moonshine_dot_product(const float *a, const float *b, uint32_t size) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < size; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}
