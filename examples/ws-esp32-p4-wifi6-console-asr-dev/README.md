# ESP32-P4 Wi-Fi 6 Console ASR Development Application

This is the full reference and development application for the native Moonshine Seq2Seq
component on the Waveshare ESP32-P4-WIFI6 board. It loads `.espm` model files and WAV
inputs from microSD, supports wireless HTTP file management over Wi-Fi 6, provides
microphone diagnostics (onboard ES8311 codec and external I2S INMP441), and exposes an
interactive serial console for benchmark and live speech recognition testing.
Comprehensive performance and accuracy benchmarks are documented in the repository-root
README.

## Board and audio routing

This example targets the Waveshare [ESP32-P4-WIFI6](https://docs.waveshare.com/ESP32-P4-WIFI6).
The vendor documentation lists 32 MB stacked PSRAM, 32 MB onboard Nor Flash,
an ESP32-C6 wireless coprocessor, a TF/microSD slot, an onboard SMD microphone,
and a speaker interface. Confirm the exact board/SKU and revision before
relying on a pin assignment or memory capacity.

### Microphone routes

The default capture source is the board's onboard analog microphone through
the ES8311 codec on I2S0. The codec is configured for its analog microphone
input (`digital_mic=false`) and sends stereo 16-bit audio to the P4; the
firmware selects one channel and exposes 16 kHz mono PCM16 to the ASR path.
This configuration makes the onboard route the default, but does not by itself
prove microphone sensitivity, noise, or recording quality on every board
revision. Use the level diagnostic and validate the actual hardware.

An external digital I2S microphone is also prepared as an optional input on
I2S1. The default pin assignment follows the existing example wiring and is
configurable; verify the board revision and wiring before connecting anything.

| Route | Default signals/configuration | Status |
|---|---|---|
| Onboard analog microphone | ES8311 over I2C SDA GPIO 7 / SCL GPIO 8; I2S0 MCLK 13, BCLK 12, WS 10, data-in 11, data-out 9 | Default source; a basic live level check returned audio, but controlled acoustic validation remains open |
| External digital I2S microphone | BCLK GPIO 20, WS GPIO 21, data-in GPIO 22; 32-bit I2S slots carrying signed 24-bit samples; left slot by default | Optional route, not validated without a connected microphone |

Use `idf.py menuconfig` > **Moonshine ASR audio** to select the boot-time
source, configure the external I2S pins and slot to match the microphone's
L/R wiring, and set the ES8311 PGA gain. The gain control supports 0 through
42 dB in 6 dB steps; the current default is 36 dB and should be treated as an
initial development setting, not a calibrated recommendation. Runtime changes
are available with `mic gain <dB>` and `mic select <onboard|i2s>`. Runtime
microphone route and gain settings are saved in NVS and override the build
defaults after they are changed.

After loading an ESPM model, use `listen start` to start capture and live ASR;
`listen stop` stops capture. `mic status` reports the selected route and
configuration. With listening stopped, `mic level` measures one second of
RMS, peak, and near-full-scale samples. These are digital dBFS diagnostics,
not calibrated acoustic SPL or a substitute for controlled microphone tests.
Select the optional input with `mic select i2s`; return to the board microphone
with `mic select onboard`. Change routes only while capture is stopped.

The WAV-based model inference tests do not test the live microphone route.
For hardware development, `record_mic` captures the selected microphone to a
16 kHz mono PCM16 WAV file without requiring a model, and `play_wav` plays that
format through the ES8311 speaker output. Connect a compatible speaker to the
board's speaker interface; the board specification does not imply a built-in
speaker transducer. These commands do not implement microphone-to-speaker live
loopback. Speaker volume defaults to 50% and is saved in NVS when changed.
The percentage controls linear signal amplitude (0 mutes, 100 is maximum);
actual loudness depends on the speaker and its amplifier. Manual playback was
audible at 100%, while acoustic calibration remains open. Example:

```text
mic status
mic level
speaker_vol 50
record_mic /sd/audio/onboard-test.wav 5
play_wav /sd/audio/onboard-test.wav
```

`record_mic` accepts durations from 1 to 60 seconds. `play_wav` intentionally
accepts only non-empty 16 kHz mono PCM16 WAV files; use `infer` for its separate
ASR input path, which also supports stereo downmix and resampling. Stop
`listen` before recording or playback. Recorded WAVs under `/sd/audio` can be
downloaded through the authenticated WLAN file-transfer service.

`sdkconfig.defaults` selects 32 MB flash, matching the vendor specification for
the base board. `partitions.csv` assigns 3 MiB to each app slot and does not
define a data filesystem in flash. The application mounts microSD at `/sd`;
models, audio, and benchmark outputs are stored there.

## Requirements and build

- ESP-IDF 6.0.2 or later is required by the component manifests. Only
  ESP-IDF 6.0.2 has been validated.
- The P4/C6 WLAN link uses ESP-Hosted over SDIO. Its C6 slave firmware must be
  compatible with the resolved `esp_hosted` / `esp_wifi_remote` versions;
  building the P4 application alone does not establish that compatibility.
- A FAT-formatted microSD/TF card and a serial/USB connection.
- `esp-idf-pulsevad` and `esp-idf-model-manager` are fetched from their public
  Git repositories; no sibling checkouts are required for those components.
- The example intentionally uses the Moonshine component from this repository
  through `main/idf_component.yml` so it tests the checked-out source. WAV
  parsing utilities (`audio_wav.h`, `audio_wav.c`) are embedded directly in
  `main/` for self-contained operation without external repository dependencies.

From the repository root, in an ESP-IDF terminal:

```powershell
cd examples\ws-esp32-p4-wifi6-console-asr-dev
idf.py build flash monitor
```

## Model files

From the repository root:

```cmd
tools-py\moonshine_seq2seq_download_models_py.bat
tools-py\moonshine_seq2seq_convert_models.bat
```

The converter writes native ESPM files to the repository's `models/espm/`
folder. Each `.espm` contains both encoder and decoder tensors, the tensor
index, and the tokenizer. Copy a single model file to `/models/` on the SD card.

The Tiny model files are ~27.37 MB (INT8 per-tensor) and ~27.68 MB (INT8-per-channel). Each file contains 68 encoder and 92 decoder tensors plus the 32,768-token BPE tokenizer vocabulary. On the ESP32-P4, the loader allocates a persistent 26.25 MiB tensor arena in PSRAM, verifies container CRCs, and loads the model in ~7.2 seconds. After `unload`, the arena is retained in PSRAM for zero-allocation warm reloads.

Transfer model files over WLAN using the onboard HTTP service as described below, or copy them directly to `/models/` on the microSD card.

### Audio Input Requirements

The WAV inference path accepts 16-bit PCM mono or stereo WAV files. When `CONFIG_MOONSHINE_ENABLE_AUDIO_UTILS` is enabled, the firmware automatically downmixes stereo to mono and resamples non-16 kHz audio using an anti-aliased windowed-sinc filter. Other bit depths and floating-point WAV files are rejected. For reference benchmarks, standard 16 kHz mono 16-bit PCM is recommended.

## Console commands

Load one model file (encoder, decoder, and vocabulary are in the same
container) and run inference:

```text
load /sd/models/<model>.espm
infer /sd/audio/test.wav
```

Use `ls` to browse the SD card (`ls` lists `/sd`; `ls /sd/models` lists the
model directory). Other useful commands include `help`, `models`, `mem`,
`unload`, `listen`, `mic`, `speaker_vol`, `record_mic`, and `play_wav`.
`listen` requires a loaded model and a working microphone route. `record_mic`
and `play_wav` are separate hardware diagnostics and do not require a model.

## WLAN file transfer

The P4 uses the onboard C6 as its Wi-Fi co-processor over SDIO. The hosted link
uses SDMMC slot 1 (the board's C6 SDIO pins); the microSD card remains on slot 0.
The configured C6 link is 4-bit SDIO (CMD 19, CLK 18, D0-D3 14-17, reset 54);
the SD card uses slot 0 (CMD 44, CLK 43, D0-D3 39-42). These are board-reserved
signals and must not be reassigned to external peripherals.
The first setup is interactive so the password is not part of a command line:

```text
wifi setup
wifi status
wifi token
```

SSID and password are stored in the `wifi_cfg` NVS namespace and loaded again
after reboot. This developer example does not enable NVS flash encryption;
credentials are therefore not protected against physical flash extraction.
The firmware suppresses password echo; disable local echo in the serial
terminal if it is enabled. The firmware does not log the password. Use
`wifi disconnect` to stop WLAN,
`wifi connect` to reconnect with saved credentials, and `wifi forget` to erase
the saved credentials. `wifi status` reports the station IP and transfer URL.

After DHCP succeeds, the firmware starts an authenticated HTTP file-transfer
service on port 8080. It transfers model/audio files and result files; it does
not stream live benchmark telemetry. A new random transfer token is generated
at each boot and is available only through the serial command `wifi token`; it
is required as an `Authorization: Bearer` header. HTTP is not TLS-encrypted,
so the token and file contents are visible to other parties able to observe
LAN traffic. The service accepts only new files in fixed directories; it does
not allow path traversal, overwriting existing files, or remote deletion:

| URL path | SD-card path | Accepted files |
|---|---|---|
| `/api/models/<filename>` | `/sd/models/` | `.espm` |
| `/api/audio/<filename>` | `/sd/audio/` | `.wav` |
| `/api/results/<filename>` | `/sd/results/` | `.csv`, `.json`, `.bin` |

Example from a trusted development computer (replace the IP, token, and local
file path):

```powershell
$deviceIp = "192.168.1.50" # replace with the address from wifi status
$token = "<value printed by wifi token>"
curl.exe -H "Authorization: Bearer $token" --upload-file .\model.espm `
  "http://${deviceIp}:8080/api/models/model.espm"
curl.exe -H "Authorization: Bearer $token" `
  "http://${deviceIp}:8080/api/results/run.csv" -o .\run.csv
```

Uploads are streamed to the SD card and limited to 64 MiB per file. Keep the
device on a trusted LAN; this is a development transfer service, not an
internet-facing server. Stop inference before transferring files or running a
benchmark. A successful P4 build does not prove that the installed C6 firmware
can establish the Hosted link; confirm the boot log and DHCP result on hardware.

## Benchmarks and tensor comparison

The device `benchmark` command iterates through encoder files in the model
directory and WAV files in the audio directory. Optional filename prefixes
isolate a chosen model variant and audio subset. Stop listening and run
`unload` first:

```text
benchmark /sd/models /sd/audio /sd/results/run.csv --model-prefix moonshine_tiny_en_int8ch --audio-prefix holdout-
```

The model prefix may be the full encoder filename; this distinguishes `int8`
from `int8ch`, whose names otherwise share a prefix. Audio prefixes are
matched against WAV basenames.

Benchmark WAV files must be 16 kHz mono 16-bit PCM and at least 895 samples.
The CSV records model-pair load time, tokenizer load time, free PSRAM after
each stage, WAV-load time, and encoder/decoder inference times. Serial logs
also report per-file SD read throughput and model parsing time. RTF is
`(encoder time + decoder time) / audio duration`; WAV loading, audio
preprocessing, context creation, and model loading are excluded. RTF values depend on the exact board, firmware, model, and audio. The completed
post-RoPE run and its measured device values are documented in the root README;
they are exploratory, not validated accuracy claims. A CSV created with an
older header is rejected; choose a new output path rather than appending rows
with a different schema.

The host `moonshine_seq2seq_golden_test` evaluates full-precision Hugging Face
checkpoints against user-supplied transcripts and is distinct from the
firmware benchmark. The `moonshine_seq2seq_tensor_test` compares encoder
hidden-state arrays. To write a raw float32 encoder output from the device,
load models and run:

```text
dump_golden /sd/audio/test.wav /sd/audio/device_encoder_out.bin
```

Compare that output with the host's `encoder_out.bin` using
`tools-py/moonshine_seq2seq_tensor_test/compare_golden.py`. This is an
encoder-tensor comparison, not transcript-level accuracy or full-model
equivalence. The default comparison tolerance is not validated for INT8
quantization; do not report a pass until an acceptance tolerance is justified.

The completed post-RoPE 150-sample Tiny EN P4 benchmark, including per-sample
encoder and decoder profiling for both quantizations, is documented in the
[repository-root README](../../README.md). The result is exploratory because
its synthetic reference transcripts have not been acoustically verified.

## Release status

The device example and model format remain experimental. Synthetic audio used
during quantization calibration is not an independent evaluation set. The
repeated load comparison above covers one P4 and one model per quantization;
it does not replace full-corpus inference and accuracy evaluation. Before
publishing accuracy claims, use an independent test set with verified
transcripts. Retain exact configurations and raw results, and report hardware,
model revision/quantization, sample count, text normalization, and timing
scope. These targeted results are not release-level accuracy evidence.
