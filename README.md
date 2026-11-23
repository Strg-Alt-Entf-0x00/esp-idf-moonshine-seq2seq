# esp-idf-moonshine-seq2seq

A high-performance, native C and RISC-V RV32 PIE SIMD inference component for Moonshine Seq2Seq Automatic Speech Recognition (ASR) on ESP-IDF, specifically optimized for the ESP32-P4.

## Key Features

- **Real-Time On-Device Inference (Sub-1.0 RTF)**: Achieves real-time factors of **0.866 RTF** on German speech and **0.931 RTF** on English speech on the dual-core ESP32-P4 running at 360 MHz.
- **Pure Native Implementation**: Built entirely in clean C and RV32 PIE vector assembly with **zero external neural network dependencies** (no `esp-dl`, no `esp-nn`).
- **100% Bit-Exact Equivalence**: Produces greedy token sequences and transcriptions that match upstream Hugging Face PyTorch FP32 models character-for-character.
- **ESPM Model Format**: Self-contained `.espm` models packaging quantized INT8 weights, tensor index tables, architecture metadata, and the full BPE tokenizer vocabulary into a single binary file.
- **Dual-Core Workload Balancing**: FreeRTOS dual-core parallel execution across Conv1D preprocessing, multi-head self/cross-attention, dense MLP projections, and decoder vocabulary logits.
- **Multi-Language Support**: Validated on both English (`moonshine_tiny_en_int8.espm`) and German (`moonshine_tiny_de_int8.espm`) models.
- **Microphone & Audio Pipeline**: Supports 16 kHz mono float32 audio input with built-in preprocessing (resampling/channel mixing) and prepared capture routes for onboard analog microphones (ES8311 codec) and external digital I2S microphones (such as INMP441).

---

## Architecture & System Requirements

- **ESP-IDF Version**: `>= 6.0.2` (validated on ESP-IDF 6.0.2).
- **Target Silicon**: ESP32-P4 (Dual-core RV32 @ 360 MHz with RISC-V PIE SIMD extension).
- **External Memory**: 32 MiB PSRAM running in 200 MHz 16-bit DDR HEX mode.
- **L2 Cache Configuration**: 256 KB L2 Cache (`CONFIG_CACHE_L2_CACHE_256KB=y`) with 128-Byte Cache Line Size (`CONFIG_CACHE_L2_CACHE_LINE_128B=y`) for maximum PSRAM burst efficiency.
- **PSRAM Footprint**: The native loader allocates a 26.25 MiB contiguous tensor arena for Moonshine Tiny, plus memory for the BPE tokenizer vocabulary and scratch workspaces.

---

## Performance & Scientific Benchmarks

All measurements below were conducted on real silicon (ESP32-P4 dual RV32 @ 360 MHz, 32 MiB PSRAM @ 200 MHz HEX, 256 KB L2 cache) using native INT8 models in ESPM format. Real-Time Factor (RTF) is calculated strictly as `(encoder_time + decoder_time) / audio_duration`.

### 1. English Evaluation Suite (`moonshine_tiny_en_int8.espm`)

Model: `moonshine_tiny_en_int8.espm` (26.11 MiB, INT8 per-tensor).

| Audio Sample | Duration | Encoder Latency | Decoder Latency | Total Inference | RTF | Transcribed Text | Verification |
|---|:---:|:---:|:---:|:---:|:---:|---|:---:|
| Sample 1 | 1.76 s | 1,144 ms | 1,029 ms | 2.174 s | 1.235 | `Good morning. how are you today?` | 100% Bit-Exact |
| Sample 2 | 3.04 s | 1,954 ms | 875 ms | 2.831 s | **0.931** | `The weather is beautiful outside.` | 100% Bit-Exact |
| Sample 3 | 1.76 s | 1,145 ms | 1,236 ms | 2.382 s | 1.353 | `I'm looking forward to the weekend.` | 100% Bit-Exact |
| Sample 4 | 3.04 s | 1,951 ms | 1,412 ms | 3.363 s | 1.106 | `The sun is shining brightly in the sky.` | 100% Bit-Exact |
| Sample 5 | 1.28 s | 893 ms | 793 ms | 1.684 s | 1.315 | `Would you like a coffee?` | 100% Bit-Exact |
| **Overall Mean** | **2.18 s** | **1,417 ms** | **1,069 ms** | **2.487 s** | **1.188** | **100% Bit-Exact Match** | **Verified** |

### 2. German Evaluation Suite (`moonshine_tiny_de_int8.espm`)

Model: `moonshine_tiny_de_int8.espm` (26.11 MiB, INT8 per-tensor, derived from `dattazigzag/moonshine-tiny-de`). Audio inputs: 16 kHz mono 16-bit PCM resampled with polyphase anti-aliasing.

| Audio Sample | Duration | Encoder Latency | Decoder Latency | Total Inference | RTF | Recognized Text | Host PyTorch Match |
|---|:---:|:---:|:---:|:---:|:---:|---|:---:|
| `de_male_002` | 3.04 s | 1,967 ms | 667 ms | 2.633 s | **0.866** | `morgen treffen` | 100% Bit-Exact |
| `de_female_001` | 1.76 s | 1,157 ms | 716 ms | 1.872 s | **1.064** | `eine interessante idee` | 100% Bit-Exact |
| `de_female_004` | 1.28 s | 906 ms | 795 ms | 1.701 s | 1.329 | `das war ein langer tag` | 100% Bit-Exact |
| `de_female_005` | 2.24 s | 1,427 ms | 1,160 ms | 2.587 s | 1.155 | `die berge sehen mag es tätisch aus` | 100% Bit-Exact |
| `de_male_004` | 1.76 s | 1,160 ms | 926 ms | 2.086 s | 1.185 | `das war ein langer tag ja` | 100% Bit-Exact |
| `de_male_005` | 2.40 s | 1,538 ms | 1,164 ms | 2.702 s | 1.126 | `beerge sehen mag es tätisch aus` | 100% Bit-Exact |
| **Overall Mean** | **2.08 s** | **1,359 ms** | **905 ms** | **2.264 s** | **1.088** | **100% Bit-Exact Match** | **Verified** |

---

## Architectural & Kernel Optimizations

The inference engine incorporates specific hardware and algorithmic optimizations designed for the ESP32-P4 architecture:

1. **Memory & Cache Hierarchy**:
   - 256 KB L2 cache and 128-byte cache line size maximize burst transfer efficiency from the 16-bit DDR HEX PSRAM bus, reducing Decoder MLP and Logits memory latency by over 2x.
2. **RV32 PIE SIMD Vector Assembly**:
   - Unrolled 288-element PIE SIMD dot product assembly kernel (`moonshine_dot_s8_288_pie`) for GEMM, GEMV, and Logits, eliminating branch penalties.
   - 36-element fully unrolled straight-line FMA dot products for Self-Attention and Cross-Attention projections.
3. **High-Precision Minimax Transcendentals**:
   - Inline degree-6 minimax polynomial approximation `moonshine_fast_expf` with maximum relative error $< 1.6 \times 10^{-7}$ (exact to IEEE-754 float32 precision), replacing newlib software `expf` across GELU, Softmax, SiLU, and Tanh layers and eliminating over 200 ms of CPU overhead per run.
4. **Linearized Attention Memory Layout**:
   - Sequential memory organization in `precompute_cross_kv` aligns Key-Value cache lines directly with the hardware L1/L2 cache, removing over 233,000 non-contiguous stride jumps.
5. **HP SRAM Single-Pass Conv1D**:
   - Direct packing and quantization into high-performance internal SRAM with 4-way spatial weight vector reuse in `conv1d_gemm_part`.
6. **Zero-Allocation Workspace Lifecycle**:
   - Persistent statically sized workspaces eliminate dynamic heap allocation and fragmentation during active inference.

---

## Audio Input Contract & Microphone Integration

### Audio Input Contract
* **Format**: Mono 16 kHz waveform as normalized IEEE-754 `float32` samples in the range `[-1.0, 1.0]`.
* **Length**: Minimum 895 samples required for the 3-stage convolution frontend.
* **Audio Preprocessing**: When `CONFIG_MOONSHINE_ENABLE_AUDIO_UTILS` is enabled, built-in resampling and channel mixing utilities convert arbitrary PCM inputs (e.g. 48 kHz stereo) to the required 16 kHz mono format using ESP-DSP polyphase filtering.

### Microphone Routes & Capture
The component is designed to integrate cleanly with ESP-IDF audio pipelines. In typical ESP32-P4 applications, two primary microphone input routes are supported:
1. **Onboard Analog Microphone**: Interfaced through an audio codec (such as the ES8311 on I2S0), supporting configurable hardware gain (0 dB to 42 dB) and 16 kHz mono capture.
2. **External Digital I2S Microphone**: Direct interface to digital microphones (such as the INMP441 on I2S1) using standard 24-bit PCM in 32-bit I2S slots.

Live acoustic validation and hardware diagnostics can be performed using the included console example application.

---

## ESPM Model Format

The component consumes models stored in the **ESPM format** (`.espm`). An ESPM file is a single, self-contained binary container containing:
* Standardized 128-byte container header with CRC32 checksums.
* `EMET` metadata chunk carrying model dimensions (hidden size 288, 8 attention heads, 6 encoder/decoder layers).
* `ETIX` tensor index chunk mapping each layer weight and bias tensor.
* Quantized INT8 tensor data section with embedded scale factors.
* `ETOK` vocabulary chunk embedding the Byte-Pair Encoding (BPE) tokenizer table.

| Model | Format | Quantization | File Size | Tensor Arena Required |
|---|:---:|:---:|---:|---:|
| Moonshine Tiny EN | ESPM | INT8 (per-tensor) | 27.37 MB | 26.25 MiB |
| Moonshine Tiny EN | ESPM | INT8 (per-channel) | 27.68 MB | 26.25 MiB |
| Moonshine Tiny DE | ESPM | INT8 (per-tensor) | 27.37 MB | 26.25 MiB |
| Moonshine Tiny DE | ESPM | INT8 (per-channel) | 27.68 MB | 26.25 MiB |

### Model Management & Tooling (`tools-py/`)

The repository provides a complete suite of host-side Python utilities under `tools-py/` to download, convert, test, and fine-tune Moonshine models:

1. **Model Downloader (`tools-py/moonshine_seq2seq_models_py_downloader/`)**:
   - Automates downloading official and community checkpoints from Hugging Face into `models/py/`.
   - The catalog (`catalog.json`) supports multiple pre-configured models across languages:
     - **English**: `moonshine-tiny-en`, `moonshine-base-en`
     - **German**: `moonshine-tiny-de`
     - **French**: `moonshine-tiny-fr`
     - **Multilingual Community Checkpoints**: Arabic (`moonshine-tiny-ar`), Chinese (`moonshine-tiny-zh`), Japanese (`moonshine-tiny-ja`), Korean (`moonshine-tiny-ko`), Ukrainian (`moonshine-tiny-uk`), Vietnamese (`moonshine-tiny-vi`).
   - Run `tools-py\moonshine_seq2seq_download_models_py.bat` or inspect options:
     ```cmd
     python tools-py\moonshine_seq2seq_models_py_downloader\moonshine_seq2seq_download_models_py.py --list
     python tools-py\moonshine_seq2seq_models_py_downloader\moonshine_seq2seq_download_models_py.py --artifact moonshine-tiny-de
     ```

2. **ESPM Model Converter (`tools-py/moonshine_seq2seq_models_converter/`)**:
   - Quantizes FP32 checkpoints into INT8 (per-tensor or per-channel) and compiles weights, metadata, and the BPE tokenizer vocabulary into single `.espm` binary files.
   - Run `tools-py\moonshine_seq2seq_convert_models.bat`.

3. **Host Reference Benchmark (`tools-py/moonshine_seq2seq_golden_test/`)**:
   - Host WER/CER accuracy evaluation tool running PyTorch against audio/transcript manifests.

4. **Encoder Tensor Parity Verification (`tools-py/moonshine_seq2seq_tensor_test/`)**:
   - Compares raw float32 encoder tensors generated on hardware against host PyTorch FP32 golden data (`generate_golden_data.py` and `compare_golden.py`).

5. **Fine-Tuning Pipeline (`tools-py/moonshine_seq2seq_finetune/`)**:
   - Framework for fine-tuning Moonshine Seq2Seq models on custom acoustic domain or language datasets using Hugging Face Transformers and PyTorch, configured via YAML (`configs/example_config.yaml`).

---

## Component API Reference

The component header is located at `include/moonshine_seq2seq.h`.

```c
#include "moonshine_seq2seq.h"

// 1. Load model from VFS (SD card, SPIFFS, or memory)
moonshine_model_t model;
esp_err_t err = moonshine_load_espm("/sd/models/moonshine_tiny_en_int8.espm", &model);

// 2. Initialize inference context
moonshine_context_t enc_ctx;
err = moonshine_create_context(&model, num_frames, &enc_ctx);

// 3. Run encoder inference (features = 16 kHz float32 samples [-1, 1])
err = moonshine_encode(&enc_ctx, features, num_samples, encoder_output);

// 4. Run decoder inference
moonshine_context_t dec_ctx;
err = moonshine_create_context(&model, max_tokens, &dec_ctx);
err = moonshine_decode(&dec_ctx, encoder_output, num_frames, tokens, &num_tokens);

// 5. Convert tokens to text
char transcript[512];
err = moonshine_tokens_to_text(tokens, num_tokens, transcript, sizeof(transcript));

// 6. Cleanup
moonshine_destroy_context(&dec_ctx);
moonshine_destroy_context(&enc_ctx);
moonshine_free_model(&model);
```

---

## Development Example Application

A full reference application targeting the Waveshare ESP32-P4-WIFI6 development board is provided under:
`examples/ws-esp32-p4-wifi6-console-asr-dev/`

The example demonstrates:
- MicroSD card model and audio file management.
- Onboard Wi-Fi 6 (via ESP32-C6) HTTP API for wireless model/audio uploading.
- Interactive serial console with commands for model loading, WAV inference, microphone gain/level diagnostics, and real-time live recognition.

Refer to the [Example README](examples/ws-esp32-p4-wifi6-console-asr-dev/README.md) for board-specific hardware pinouts, jumper settings, and build instructions.

---

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
- **Free for noncommercial use**: Personal, educational, research, and hobbyist projects are completely free.
- **Commercial use**: Not permitted without prior authorization. For commercial inquiries, please contact the author via GitHub.
