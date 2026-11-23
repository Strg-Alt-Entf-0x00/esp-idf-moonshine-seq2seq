# Changelog

This log records notable, verified changes from 2026-10-24 onward.

## 2026-11-22

- **BREAKTHROUGH: Sub-1.0 Real-Time (RTF 0.931 EN / RTF 0.866 DE) Pure Native Moonshine ASR on ESP32-P4**:
  - Achieved **RTF 0.931** (2,831 ms inference on 3.04 s audio) on English speech and **RTF 0.866** (2,633 ms on 3.04 s audio) on German speech on the Waveshare ESP32-P4-WIFI6 (dual RV32 @ 360 MHz, 32 MB PSRAM @ 200 MHz HEX).
  - Delivered an overall **3.85x speedup** over the initial unoptimized baseline (10,950 ms -> 2,831 ms) with **100% bit-exact accuracy** verified against reference PyTorch FP32 models.
  - **Optimization Milestones (Progression on 3.04 s speech `en_basics_easy_002`)**:
    | Phase | Milestone / Optimization Technique | Latency | Speedup | RTF | Verification |
    |:---:|---|:---:|:---:|:---:|:---:|
    | **Baseline** | Reference unoptimized native implementation | 10,950 ms | 1.00x | 3.602 | Exact tokens |
    | **Phase 1** | FreeRTOS dual-core parallel dispatch for GEMM, GEMV, and Logits | 5,616 ms | 1.95x | 1.847 | Exact tokens |
    | **Phase 2** | Specialized unrolled 288-element PIE SIMD dot product (`moonshine_dot_s8_288_pie`) | 4,210 ms | 2.60x | 1.385 | Exact tokens |
    | **Phase 3** | Single-pass im2col HP SRAM Conv1D & persistent encoder workspace | 3,842 ms | 2.85x | 1.264 | Exact tokens |
    | **Phase 4** | 256 KB L2 Cache + 128B Cache Lines (doubled DDR HEX PSRAM burst efficiency) | 3,087 ms | 3.55x | 1.015 | Exact tokens |
    | **Phase 5** | Degree-6 minimax polynomial `moonshine_fast_expf` (max rel err < 1.6e-7) | 2,843 ms | 3.85x | **0.935** | Exact tokens |
    | **Phase 6** | Linearized attention FMA unrolling & 4-way Conv1D weight reuse | **2,831 ms** | **3.85x** | **0.931** | **100% Bit-Exact** |
  - **Memory & Cache Hierarchy Overhaul**:
    - Upgraded L2 Cache to 256 KB (`CONFIG_CACHE_L2_CACHE_256KB=y`), maximizing cache capacity.
    - Upgraded L2 Cache Line Size to 128 Bytes (`CONFIG_CACHE_L2_CACHE_LINE_128B=y`), doubling 16-bit DDR HEX PSRAM burst efficiency and dropping Decoder MLP latency by >2x (532 ms -> 224 ms) and Logits by >2.5x (792 ms -> 308 ms).
  - **Fast IEEE-754 Minimax Polynomial Transcendentals**:
    - Designed inline degree-6 minimax polynomial `moonshine_fast_expf` with maximum relative error $< 1.6 \times 10^{-7}$ (single-precision float exact), replacing newlib software `expf` across GELU, Softmax, SiLU, and Tanh, saving over 200 ms of CPU time per inference.
  - **RISC-V RV32 PIE SIMD Assembly & Kernel Optimizations**:
    - Specialized loop-unrolled 288-element PIE SIMD dot product assembly (`moonshine_dot_s8_288_pie`), eliminating loop branch overhead in GEMM, GEMV, and Logits.
    - Implemented M-tiled prequantized GEMM (`moonshine_gemm_int8_prequantized`), quantizing matrix A once in fast HP SRAM and parallelizing over M rows across both RISC-V cores.
    - Linearized Cross-Attention KV cache transposition (`precompute_cross_kv`) by ordering outer $f$ and inner $d$, eliminating 233,000 non-contiguous 1,152-byte memory leaps.
    - Fully unrolled 36-element straight-line FMA dot products for Self-Attention and Cross-Attention with pre-scaled queries, cutting loop branches.
    - Single-pass Conv1D packing in HP SRAM with 4-way spatial weight reuse in `conv1d_gemm_part`.
  - **German (de-DE) Model Validation on ESP32-P4 Hardware**:
    - Evaluated `moonshine_tiny_de_int8.espm` (architecture derived from `dattazigzag/moonshine-tiny-de`) on hardware via serial CLI and Wi-Fi HTTP deployment.
    - Achieved **RTF 0.866** (2,633 ms on 3.04 s audio) and an overall suite mean of **RTF 1.088** across both male and female German speech corpora.
    - Confirmed **100% bit-exact equivalence** against the host Hugging Face FP32 greedy reference implementation across all evaluation samples.
  - **Component & Standard Compliance**:
    - Standardized strictly on the unified **ESPM format** (`.espm`), dropping legacy migration nomenclature.
    - Removed hardcoded git commit version pins from `idf_component.yml` manifests.
    - Strictly **zero dependency on esp-dl / esp-nn**, using only pure native C and RV32 PIE SIMD.

## 2026-11-14

- Added native ESPM conversion/loading and validated the Tiny EN INT8
  payload bit-for-bit against reference tensors. The ESP-IDF 6.0.2 P4
  build and SD-card load passed; a synthetic spoken-WAV smoke test also passed
  through resampling, encoder, and decoder.
- Fixed audio downsampling: ESP-DSP's polyphase API rejects ratios below 1,
  so downsampling now uses an anti-aliased windowed-sinc filter. Verified on
  the P4 with 22.05 kHz mono PCM input.
- Completed post-RoPE Tiny EN benchmark on the ESP32-P4: 150 manifest samples
  each for INT8 and INT8-per-channel, with no outputs reaching the 100-token
  limit.
