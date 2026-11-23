/**
 * @file moonshine_ops.h
 * @brief Core Mathematical Operations for Moonshine Inference
 */

#ifndef MOONSHINE_OPS_H
#define MOONSHINE_OPS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * QUANTIZATION OPERATIONS
 * ======================================================================== */

/**
 * @brief Dequantize INT8 data to FP32
 *
 * @param quantized INT8 quantized data
 * @param scales Scale factors (1 for per-tensor, N for per-channel)
 * @param num_channels Number of channels (1=per-tensor, N=per-channel)
 * @param elements_per_channel Elements per channel
 * @param output FP32 output buffer
 */
void moonshine_dequantize_int8(const int8_t *quantized, const float *scales,
                               uint32_t num_channels,
                               uint32_t elements_per_channel, float *output);

/* ========================================================================
 * MATRIX OPERATIONS
 * ======================================================================== */

/**
 * @brief General Matrix Multiplication (GEMM): C = A × B
 *
 * @param A Matrix A [M × K]
 * @param B Matrix B [K × N] or [N × K] if transpose_B
 * @param C Output matrix C [M × N]
 * @param M Number of rows in A
 * @param N Number of columns in B (or rows if transposed)
 * @param K Inner dimension
 * @param transpose_B If true, B is transposed
 */
void moonshine_gemm_fp32(const float *A, const float *B, float *C, uint32_t M,
                         uint32_t N, uint32_t K, bool transpose_B);

/**
 * @brief Matrix Multiplication with on-the-fly INT8 dequantization: C = A × B
 */
void moonshine_gemm_int8(const float *A, const int8_t *B_q,
                         const float *B_scales, uint32_t num_channels_B,
                         float *C, uint32_t M, uint32_t N, uint32_t K,
                         bool transpose_B);

/**
 * @brief Quantize rows of float matrix A to INT8 with per-row scaling.
 */
void moonshine_quantize_int8_rows(const float *A, int8_t *A_q, float *A_scales,
                                  uint32_t M, uint32_t K);

/**
 * @brief Matrix Multiplication with pre-quantized INT8 matrix A: C = A_q ×
 * B_q^T
 */
void moonshine_gemm_int8_prequantized(const int8_t *A_q, const float *A_scales,
                                      const int8_t *B_q, const float *B_scales,
                                      uint32_t num_channels_B, float *C,
                                      uint32_t M, uint32_t N, uint32_t K);

/**
 * @brief Matrix-Vector Multiplication: y = A × x
 *
 * @param A Matrix [M × N]
 * @param x Input vector [N]
 * @param y Output vector [M]
 * @param M Number of rows
 * @param N Number of columns
 */
void moonshine_gemv_fp32(const float *A, const float *x, float *y, uint32_t M,
                         uint32_t N);

/* ========================================================================
 * ACTIVATION FUNCTIONS
 * ======================================================================== */

void moonshine_relu(float *data, uint32_t size);
void moonshine_gelu(float *data, uint32_t size);
void moonshine_silu(float *data, uint32_t size);
void moonshine_tanh(float *data, uint32_t size);
void moonshine_softmax(float *data, uint32_t size);

/* ========================================================================
 * NORMALIZATION
 * ======================================================================== */

/**
 * @brief Layer Normalization
 *
 * Normalizes data to zero mean and unit variance,
 * then applies learned affine transformation.
 *
 * @param data Input/output data [N]
 * @param weight Gamma weights [N] (can be NULL)
 * @param bias Beta bias [N] (can be NULL)
 * @param size Number of elements
 * @param eps Small constant for numerical stability
 */
void moonshine_layer_norm(float *data, const float *weight, const float *bias,
                          uint32_t size, float eps);

/**
 * @brief RMS Normalization (Root Mean Square Normalization)
 *
 * RMSNorm: output = input / RMS(input) * weight
 * where RMS = sqrt(mean(input^2) + eps)
 *
 * NOTE: RMSNorm has NO bias term (unlike LayerNorm)!
 *
 * @param data Input/output data [N]
 * @param weight Scale weights [N]
 * @param size Number of elements
 * @param eps Small constant for numerical stability
 */
void moonshine_rms_norm(float *data, const float *weight, uint32_t size,
                        float eps);

/**
 * @brief Group Normalization
 *
 * Normalizes groups of channels to zero mean and unit variance,
 * then applies learned affine transformation.
 *
 * @param data Input/output data [num_channels × spatial_size]
 * @param weight Gamma weights [num_channels]
 * @param bias Beta bias [num_channels]
 * @param num_channels Number of channels
 * @param num_groups Number of groups
 * @param spatial_size Spatial dimension size
 * @param eps Small constant for numerical stability
 */
void moonshine_group_norm(float *data, const float *weight, const float *bias,
                          uint32_t num_channels, uint32_t num_groups,
                          uint32_t spatial_size, float eps);

/**
 * @brief Apply Rotary Position Embedding (RoPE)
 *
 * Applies rotary position embeddings to query or key tensors.
 * RoPE applies 2D rotations to consecutive pairs of dimensions.
 *
 * @param data Input/output data [seq_len × hidden_size]
 * @param seq_len Sequence length
 * @param hidden_size Hidden dimension
 * @param num_heads Number of attention heads
 * @param base Frequency base (typically 10000.0)
 */
void moonshine_apply_rope(float *data, uint32_t seq_len, uint32_t hidden_size,
                          uint32_t num_heads, float base);

/* ========================================================================
 * ELEMENT-WISE OPERATIONS
 * ======================================================================== */

void moonshine_add(const float *a, const float *b, float *out, uint32_t size);
void moonshine_add_inplace(float *a, const float *b, uint32_t size);
void moonshine_multiply(const float *a, const float *b, float *out,
                        uint32_t size);
void moonshine_scale(float *data, float scale, uint32_t size);

/* ========================================================================
 * CONVOLUTION
 * ======================================================================== */

/**
 * @brief 1D Convolution
 *
 * @param input Input tensor [in_channels × seq_len]
 * @param kernel Convolution kernel [out_channels × in_channels × kernel_size]
 * @param bias Bias [out_channels] (can be NULL)
 * @param output Output tensor [out_channels × out_seq_len]
 * @param in_channels Number of input channels
 * @param out_channels Number of output channels
 * @param seq_len Input sequence length
 * @param kernel_size Kernel size
 * @param stride Stride
 * @param padding Padding
 */
void moonshine_conv1d(const float *input, const float *kernel,
                      const float *bias, float *output, uint32_t in_channels,
                      uint32_t out_channels, uint32_t seq_len,
                      uint32_t kernel_size, uint32_t stride, uint32_t padding);

/**
 * @brief 1D Convolution with on-the-fly INT8 dequantization
 */
void moonshine_conv1d_int8(const float *input, const int8_t *kernel_q,
                           const float *kernel_scales, uint32_t num_channels_q,
                           const float *bias, float *output,
                           uint32_t in_channels, uint32_t out_channels,
                           uint32_t seq_len, uint32_t kernel_size,
                           uint32_t stride, uint32_t padding);

/* ========================================================================
 * UTILITIES
 * ======================================================================== */

void moonshine_copy(float *dst, const float *src, uint32_t size);
void moonshine_zero(float *data, uint32_t size);
float moonshine_dot_product(const float *a, const float *b, uint32_t size);

#ifdef ESP_PLATFORM
static inline int32_t moonshine_fc_dot_s8_pie(const int8_t *input,
                                              const int8_t *filter,
                                              int32_t row_len) {
  int32_t result = 0;
  int32_t idx = 0;
  if (row_len >= 32) {
    asm volatile("esp.zero.xacc                          \n\t"
                 "mv     x30, %[in]                      \n\t"
                 "mv     x31, %[flt]                     \n\t"
                 "li     %[idx], 32                      \n\t"
                 "addi   t0, %[len], -31                 \n\t"
                 "esp.vld.128.ip  q0, x30, 16            \n\t"
                 "esp.vld.128.ip  q2, x30, 16            \n\t"
                 "esp.vld.128.ip  q1, x31, 16            \n\t"
                 "esp.vld.128.ip  q3, x31, 16            \n\t"
                 "j      2f                              \n\t"
                 "1:                                     \n\t"
                 "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
                 "esp.vld.128.ip  q1, x31, 16            \n\t"
                 "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
                 "esp.vld.128.ip  q3, x31, 16            \n\t"
                 "addi   %[idx], %[idx], 32              \n\t"
                 "2:                                     \n\t"
                 "blt    %[idx], t0, 1b                  \n\t"
                 "esp.vmulas.s8.xacc  q0, q1             \n\t"
                 "esp.vmulas.s8.xacc  q2, q3             \n\t"
                 "addi   t0, %[len], -15                 \n\t"
                 "bge    %[idx], t0, 3f                  \n\t"
                 "esp.vld.128.ip  q0, x30, 16            \n\t"
                 "esp.vld.128.ip  q1, x31, 16            \n\t"
                 "esp.vmulas.s8.xacc  q0, q1             \n\t"
                 "addi   %[idx], %[idx], 16              \n\t"
                 "3:                                     \n\t"
                 "esp.movx.r.xacc.l   x30                \n\t"
                 "mv     %[res], x30                     \n\t"
                 : [idx] "+r"(idx), [res] "=r"(result)
                 : [in] "r"(input), [flt] "r"(filter), [len] "r"(row_len)
                 : "x30", "x31", "t0");
  } else if (row_len >= 16) {
    asm volatile("esp.zero.xacc                          \n\t"
                 "mv     x30, %[in]                      \n\t"
                 "mv     x31, %[flt]                     \n\t"
                 "li     %[idx], 16                      \n\t"
                 "addi   t0, %[len], -15                 \n\t"
                 "esp.vld.128.ip  q0, x30, 16            \n\t"
                 "esp.vld.128.ip  q1, x31, 16            \n\t"
                 "j      5f                              \n\t"
                 "4:                                     \n\t"
                 "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
                 "esp.vld.128.ip  q1, x31, 16            \n\t"
                 "addi   %[idx], %[idx], 16              \n\t"
                 "5:                                     \n\t"
                 "blt    %[idx], t0, 4b                  \n\t"
                 "esp.vmulas.s8.xacc  q0, q1             \n\t"
                 "esp.movx.r.xacc.l   x30                \n\t"
                 "mv     %[res], x30                     \n\t"
                 : [idx] "+r"(idx), [res] "=r"(result)
                 : [in] "r"(input), [flt] "r"(filter), [len] "r"(row_len)
                 : "x30", "x31", "t0");
  }
  for (; idx < row_len; idx++) {
    result += (int32_t)input[idx] * (int32_t)filter[idx];
  }
  return result;
}

static inline int32_t moonshine_dot_s8_288_pie(const int8_t *input,
                                               const int8_t *filter) {
  int32_t result;
  asm volatile("esp.zero.xacc                          \n\t"
               "mv     x30, %[in]                      \n\t"
               "mv     x31, %[flt]                     \n\t"
               // 1 (bytes 0..31)
               "esp.vld.128.ip  q0, x30, 16            \n\t"
               "esp.vld.128.ip  q2, x30, 16            \n\t"
               "esp.vld.128.ip  q1, x31, 16            \n\t"
               "esp.vld.128.ip  q3, x31, 16            \n\t"
               // 2 (bytes 32..63)
               "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
               "esp.vld.128.ip  q1, x31, 16            \n\t"
               "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
               "esp.vld.128.ip  q3, x31, 16            \n\t"
               // 3 (bytes 64..95)
               "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
               "esp.vld.128.ip  q1, x31, 16            \n\t"
               "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
               "esp.vld.128.ip  q3, x31, 16            \n\t"
               // 4 (bytes 96..127)
               "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
               "esp.vld.128.ip  q1, x31, 16            \n\t"
               "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
               "esp.vld.128.ip  q3, x31, 16            \n\t"
               // 5 (bytes 128..159)
               "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
               "esp.vld.128.ip  q1, x31, 16            \n\t"
               "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
               "esp.vld.128.ip  q3, x31, 16            \n\t"
               // 6 (bytes 160..191)
               "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
               "esp.vld.128.ip  q1, x31, 16            \n\t"
               "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
               "esp.vld.128.ip  q3, x31, 16            \n\t"
               // 7 (bytes 192..223)
               "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
               "esp.vld.128.ip  q1, x31, 16            \n\t"
               "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
               "esp.vld.128.ip  q3, x31, 16            \n\t"
               // 8 (bytes 224..255)
               "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
               "esp.vld.128.ip  q1, x31, 16            \n\t"
               "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
               "esp.vld.128.ip  q3, x31, 16            \n\t"
               // 9 (bytes 256..287)
               "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
               "esp.vld.128.ip  q1, x31, 16            \n\t"
               "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
               "esp.vld.128.ip  q3, x31, 16            \n\t"
               // finish the 9th 32 bytes
               "esp.vmulas.s8.xacc  q0, q1             \n\t"
               "esp.vmulas.s8.xacc  q2, q3             \n\t"
               "esp.movx.r.xacc.l   x30                \n\t"
               "mv     %[res], x30                     \n\t"
               : [res] "=r"(result)
               : [in] "r"(input), [flt] "r"(filter)
               : "x30", "x31");
  return result;
}

static inline int32_t moonshine_dot_s8_1152_pie(const int8_t *input,
                                                const int8_t *filter) {
  int32_t result;
  asm volatile("esp.zero.xacc                          \n\t"
               "mv     x30, %[in]                      \n\t"
               "mv     x31, %[flt]                     \n\t"
               "li     t0, 35                          \n\t"
               "esp.vld.128.ip  q0, x30, 16            \n\t"
               "esp.vld.128.ip  q2, x30, 16            \n\t"
               "esp.vld.128.ip  q1, x31, 16            \n\t"
               "esp.vld.128.ip  q3, x31, 16            \n\t"
               "1:                                     \n\t"
               "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
               "esp.vld.128.ip  q1, x31, 16            \n\t"
               "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
               "esp.vld.128.ip  q3, x31, 16            \n\t"
               "addi   t0, t0, -1                      \n\t"
               "bnez   t0, 1b                          \n\t"
               "esp.vmulas.s8.xacc  q0, q1             \n\t"
               "esp.vmulas.s8.xacc  q2, q3             \n\t"
               "esp.movx.r.xacc.l   x30                \n\t"
               "mv     %[res], x30                     \n\t"
               : [res] "=r"(result)
               : [in] "r"(input), [flt] "r"(filter)
               : "x30", "x31", "t0");
  return result;
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* MOONSHINE_OPS_H */
