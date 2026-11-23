#!/usr/bin/env python3
"""Export a complete Moonshine checkpoint directly to native ESPM."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    from transformers import AutoConfig, MoonshineForConditionalGeneration
except ImportError as error:
    print(f"[ERROR] Missing converter dependency: {error}", file=sys.stderr)
    raise SystemExit(1)

from espm_format import (
    MODEL_QUANT_INT8,
    MODEL_QUANT_INT8_PER_CH,
    PART_DECODER,
    PART_ENCODER,
    encode_tokenizer_json,
    write_espm,
)


def _model_name(directory: Path) -> str:
    name = directory.name
    for prefix in ("moonshine-seq2seq-py-", "moonshine-seq2seq-", "moonshine-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    result = "moonshine_" + re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    if not result or len(result.encode("utf-8")) >= 32:
        raise ValueError(f"model directory name cannot form an ESPM model name: {directory.name}")
    return result


def _load_vocabulary(model_dir: Path) -> tuple[bytes, int, str]:
    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.is_file():
        raw = tokenizer_path.read_bytes()
        vocabulary, count = encode_tokenizer_json(raw)
        return vocabulary, count, hashlib.sha256(raw).hexdigest()
    vocab_path = model_dir / "vocab.bin"
    if vocab_path.is_file():
        raw = vocab_path.read_bytes()
        return raw, 0, hashlib.sha256(raw).hexdigest()
    raise ValueError("model directory must contain tokenizer.json or vocab.bin")


def _quantize(array, quantize: str, is_weight: bool):
    if not np.isfinite(array).all():
        raise ValueError("model contains a non-finite tensor value")
    per_channel = quantize == "int8-per-channel" and is_weight and array.ndim >= 2
    if per_channel:
        channel_count = array.shape[0]
        scales = np.empty(channel_count, dtype="<f4")
        quantized = np.empty(array.shape, dtype=np.int8)
        for channel in range(channel_count):
            maximum = float(np.max(np.abs(array[channel])))
            scale = maximum / 127.0 if maximum > 0 else 1.0
            scales[channel] = scale
            quantized[channel] = np.clip(
                np.round(array[channel] / scale), -127, 127).astype(np.int8)
        return quantized.tobytes(order="C"), scales.tobytes(), 0, MODEL_QUANT_INT8_PER_CH

    maximum = float(np.max(np.abs(array)))
    scale = maximum / 127.0 if maximum > 0 else 1.0
    quantized = np.clip(np.round(array / scale), -127, 127).astype(np.int8)
    scales = np.asarray([scale], dtype="<f4")
    return quantized.tobytes(order="C"), scales.tobytes(), 0xFF, MODEL_QUANT_INT8


def export_model(model_dir: Path, output_dir: Path, quantize: str, device: str) -> Path:
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    vocabulary, vocabulary_count, tokenizer_sha256 = _load_vocabulary(model_dir)
    if vocabulary_count and vocabulary_count != int(config.vocab_size):
        raise ValueError(
            f"tokenizer vocabulary has {vocabulary_count} entries; "
            f"model config requires {config.vocab_size}")

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    torch_device = torch.device(device)
    model = MoonshineForConditionalGeneration.from_pretrained(
        str(model_dir), dtype=torch.float32, local_files_only=True)
    model.eval()
    model.to(torch_device)

    tensors = []
    for part, module in (
        (PART_ENCODER, model.get_encoder()),
        (PART_DECODER, model.get_decoder()),
    ):
        for name, tensor in module.state_dict().items():
            array = tensor.detach().to(device="cpu").numpy().astype(np.float32, copy=False)
            data, scales, axis, tensor_quant = _quantize(
                array, quantize, "weight" in name)
            tensors.append({
                "name": name,
                "part": part,
                "shape": array.shape,
                "quant_type": tensor_quant,
                "channel_axis": axis,
                "scales": scales,
                "data": data,
            })

    model_config = {
        "vocab_size": int(config.vocab_size),
        "hidden_size": int(config.hidden_size),
        "encoder_layers": int(config.encoder_num_hidden_layers),
        "decoder_layers": int(config.decoder_num_hidden_layers),
        "attention_heads": int(config.encoder_num_attention_heads),
        "ffn_size": int(config.intermediate_size),
    }
    manifest_path = model_dir / "artifact-manifest.json"
    source_metadata = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest.get("source", {})
        source_metadata = {
            "artifact_id": manifest.get("artifact_id"),
            "source_repo": source.get("repo_id"),
            "source_revision": source.get("revision"),
        }
    metadata = {
        "format": "ESPM",
        "format_version": 2,
        "model_id": _model_name(model_dir),
        "model_config": model_config,
        "quantization": quantize,
        "tokenizer_sha256": tokenizer_sha256,
        **{key: value for key, value in source_metadata.items() if value is not None},
    }
    model_name = _model_name(model_dir)
    quant_suffix = "int8ch" if quantize == "int8-per-channel" else "int8"
    registry_name = f"{model_name}_{quant_suffix}"
    output_path = output_dir / f"{registry_name}.espm"
    result = write_espm(
        output_path,
        model_name=registry_name,
        version_str=str(source_metadata.get("source_revision", "unversioned"))[:15],
        quant_type=MODEL_QUANT_INT8_PER_CH if quantize == "int8-per-channel" else MODEL_QUANT_INT8,
        vocab=vocabulary,
        model_config=model_config,
        tensors=tensors,
        metadata=metadata,
    )
    print(f"[OK] Wrote {output_path}")
    print(f"     {result['file_size']:,} bytes; {result['tensor_count']} tensors; "
          f"resident PSRAM estimate {result['psram_required_kb']} KiB")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert a Moonshine checkpoint to native ESPM")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quantize", required=True, choices=("int8", "int8-per-channel"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error(f"model directory does not exist: {args.model_dir}")
    try:
        export_model(args.model_dir, args.output, args.quantize, args.device)
    except (OSError, ValueError, KeyError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
