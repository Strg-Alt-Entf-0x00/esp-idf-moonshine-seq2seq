#!/usr/bin/env python3
"""Manifest-driven, reproducible evaluation of local Moonshine checkpoints."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BenchmarkError(ValueError):
    """Raised when a benchmark configuration or input dataset is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"Expected a JSON object in {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return value.strip("-._") or "benchmark"


def _resolve_path(root: Path, value: str, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{field} must be a non-empty path string")
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def initialize_config(config_path: Path, example_path: Path) -> bool:
    """Create the local editable config once; never replace user configuration."""
    config_path = config_path.resolve()
    example_path = example_path.resolve()
    example_bytes = example_path.read_bytes()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with config_path.open("xb") as destination:
            destination.write(example_bytes)
    except FileExistsError:
        return False
    return True


def load_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    config_bytes = config_path.read_bytes()
    if config.get("schema_version") != 1:
        raise BenchmarkError("Configuration schema_version must be 1")
    config_dir = config_path.parent
    root_value = config.get("project_root")
    root = _resolve_path(config_dir, root_value, "project_root") if root_value else PROJECT_ROOT
    output_root = _resolve_path(root, config.get("output_root", ".data/golden_test/runs"), "output_root")
    data_root = (root / ".data").resolve()
    if not output_root.is_relative_to(data_root):
        raise BenchmarkError(f"output_root must be inside the project .data directory: {output_root}")

    models = config.get("models")
    datasets = config.get("datasets")
    if not isinstance(models, list) or not models:
        raise BenchmarkError("Configuration must contain at least one model")
    if not isinstance(datasets, list) or not datasets:
        raise BenchmarkError("Configuration must contain at least one dataset")

    model_ids: set[str] = set()
    resolved_models = []
    for model in models:
        if not isinstance(model, dict):
            raise BenchmarkError("Each model entry must be an object")
        model_id = model.get("id")
        if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", model_id) or model_id in model_ids:
            raise BenchmarkError(f"Model IDs must be unique non-empty strings: {model_id!r}")
        model_ids.add(model_id)
        language = model.get("language")
        if not isinstance(language, str) or not language.strip():
            raise BenchmarkError(f"Model {model_id!r} must declare its language")
        path = _resolve_path(root, model.get("path"), f"models[{model_id}].path")
        if not path.is_dir():
            raise BenchmarkError(f"Model directory does not exist: {path}")
        if not (path / "config.json").is_file():
            raise BenchmarkError(f"Model directory has no config.json: {path}")
        if not any((path / filename).is_file() for filename in ("model.safetensors", "pytorch_model.bin")):
            if not list(path.glob("model-*.safetensors")) and not (path / "pytorch_model.bin.index.json").is_file():
                raise BenchmarkError(f"No supported model weights found in {path}")
        resolved_models.append({
            **model,
            "path": str(path),
            "language": language.split("-", 1)[0].casefold(),
        })

    dataset_ids: set[str] = set()
    resolved_datasets = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise BenchmarkError("Each dataset entry must be an object")
        dataset_id = dataset.get("id")
        if not isinstance(dataset_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", dataset_id) or dataset_id in dataset_ids:
            raise BenchmarkError(f"Dataset IDs must be unique non-empty strings: {dataset_id!r}")
        dataset_ids.add(dataset_id)
        language = dataset.get("language")
        if not isinstance(language, str) or not language.strip():
            raise BenchmarkError(f"Dataset {dataset_id!r} must declare its language")
        manifest_path = _resolve_path(root, dataset.get("manifest"), f"datasets[{dataset_id}].manifest")
        if not manifest_path.is_file():
            raise BenchmarkError(f"Dataset manifest does not exist: {manifest_path}")
        excluded_categories = dataset.get("exclude_categories", [])
        if not isinstance(excluded_categories, list) or not all(
            isinstance(category, str) and category.strip() for category in excluded_categories
        ):
            raise BenchmarkError(f"datasets[{dataset_id}].exclude_categories must be an array of strings")
        resolved_datasets.append({
            **dataset,
            "manifest_path": str(manifest_path),
            "language": language,
            "exclude_categories": [category.casefold() for category in excluded_categories],
        })

    max_samples = config.get("max_samples_per_dataset")
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise BenchmarkError("max_samples_per_dataset must be a positive integer or null")
    sample_seed = config.get("sample_seed", 0)
    if type(sample_seed) is not int:
        raise BenchmarkError("sample_seed must be an integer")
    generation = config.get("generation", {})
    if not isinstance(generation, dict):
        raise BenchmarkError("generation must be a JSON object")
    num_beams = generation.get("num_beams", 1)
    max_new_tokens = generation.get("max_new_tokens", 128)
    if type(num_beams) is not int or num_beams < 1:
        raise BenchmarkError("generation.num_beams must be a positive integer")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise BenchmarkError("generation.max_new_tokens must be a positive integer")
    device = config.get("device", "auto")
    if not isinstance(device, str) or not device.strip():
        raise BenchmarkError("device must be a non-empty string")
    if device not in ("auto", "cpu", "cuda", "mps") and not re.fullmatch(r"cuda:\d+", device):
        raise BenchmarkError("device must be auto, cpu, cuda, cuda:<index>, or mps")
    allow_unverified = config.get("allow_unverified_references", False)
    if not isinstance(allow_unverified, bool):
        raise BenchmarkError("allow_unverified_references must be true or false")
    run_label = config.get("run_label", "asr-benchmark")
    if not isinstance(run_label, str):
        raise BenchmarkError("run_label must be a string")

    return {
        **config,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_snapshot_bytes": config_bytes,
        "project_root_resolved": str(root),
        "output_root_resolved": str(output_root),
        "models_resolved": resolved_models,
        "datasets_resolved": resolved_datasets,
        "max_samples_per_dataset": max_samples,
        "sample_seed": sample_seed,
        "run_label": run_label,
        "generation": {
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "length_penalty": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "early_stopping": False,
        },
        "device": device,
    }


def load_manifest_dataset(
    dataset: dict[str, Any],
    max_samples: int | None,
    sample_seed: int = 0,
) -> list[dict[str, Any]]:
    manifest_path = Path(dataset["manifest_path"])
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"Invalid JSONL at {manifest_path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise BenchmarkError(f"Expected an object at {manifest_path}:{line_number}")
        audio_key = dataset.get("audio_key", "audio")
        text_key = dataset.get("text_key", "text")
        if not isinstance(audio_key, str) or not audio_key:
            raise BenchmarkError(f"audio_key must be a non-empty string in dataset {dataset['id']!r}")
        if not isinstance(text_key, str) or not text_key:
            raise BenchmarkError(f"text_key must be a non-empty string in dataset {dataset['id']!r}")
        audio_value = record.get(audio_key)
        text_value = record.get(text_key)
        source = record.get("source_text")
        category = source.get("category") if isinstance(source, dict) else None
        if not isinstance(category, str) or not category.strip():
            category = None
        if isinstance(category, str) and category.casefold() in dataset.get("exclude_categories", []):
            continue
        if not isinstance(audio_value, str) or not audio_value:
            raise BenchmarkError(f"Missing {audio_key!r} at {manifest_path}:{line_number}")
        if not isinstance(text_value, str) or not text_value.strip():
            raise BenchmarkError(f"Missing non-empty {text_key!r} at {manifest_path}:{line_number}")
        if not normalize_text(text_value):
            raise BenchmarkError(f"Reference contains no scorable words at {manifest_path}:{line_number}")
        audio_path = (manifest_path.parent / audio_value).resolve()
        if not audio_path.is_file():
            raise BenchmarkError(f"Audio file does not exist at {manifest_path}:{line_number}: {audio_path}")
        audio_metadata = record.get("audio_metadata")
        if not isinstance(audio_metadata, dict):
            audio_metadata = {}
        verified = audio_metadata.get(
            "transcription_verified",
            record.get("transcription_verified", False),
        )
        records.append({
            "sample_id": str(record.get("id", f"line-{line_number:06d}")),
            "audio_path": str(audio_path),
            "reference": text_value,
            "reference_verified": verified is True,
            "source": source,
            "category": category,
            "speaker": record.get("speaker"),
            "manifest_line": line_number,
        })
    if not records:
        raise BenchmarkError(f"No records found in {manifest_path}")
    sample_ids = [record["sample_id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise BenchmarkError(f"Sample IDs must be unique in manifest: {manifest_path}")
    if max_samples is not None and max_samples < len(records):
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            groups.setdefault(record["category"] or "uncategorized", []).append(record)
        for category_records in groups.values():
            category_records.sort(key=lambda record: hashlib.sha256(
                f"{sample_seed}:{record['sample_id']}".encode("utf-8")
            ).digest())
        categories = sorted(groups)
        rotation = int(hashlib.sha256(str(sample_seed).encode("ascii")).hexdigest()[:8], 16) % len(categories)
        categories = categories[rotation:] + categories[:rotation]
        selected_ids: set[str] = set()
        positions = dict.fromkeys(categories, 0)
        while len(selected_ids) < max_samples:
            found_record = False
            for category in categories:
                position = positions[category]
                if position < len(groups[category]):
                    selected_ids.add(groups[category][position]["sample_id"])
                    positions[category] = position + 1
                    found_record = True
                    if len(selected_ids) == max_samples:
                        break
            if not found_record:
                break
        records = [record for record in records if record["sample_id"] in selected_ids]
    return records


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).split())


def edit_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Compute deterministic Levenshtein insertion/deletion/substitution counts."""
    ref = reference.split()
    hyp = hypothesis.split()
    rows = len(ref) + 1
    cols = len(hyp) + 1
    distance = [[0] * cols for _ in range(rows)]
    operation = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        distance[i][0] = i
        operation[i][0] = "deletion"
    for j in range(1, cols):
        distance[0][j] = j
        operation[0][j] = "insertion"
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
                operation[i][j] = "hit"
                continue
            choices = (
                (distance[i - 1][j - 1] + 1, "substitution"),
                (distance[i - 1][j] + 1, "deletion"),
                (distance[i][j - 1] + 1, "insertion"),
            )
            distance[i][j], operation[i][j] = min(choices, key=lambda item: item[0])

    counts = Counter()
    i, j = len(ref), len(hyp)
    while i or j:
        action = operation[i][j]
        if action == "hit":
            i -= 1
            j -= 1
        elif action == "substitution":
            counts[action] += 1
            i -= 1
            j -= 1
        elif action == "deletion":
            counts[action] += 1
            i -= 1
        elif action == "insertion":
            counts[action] += 1
            j -= 1
        else:
            raise RuntimeError(f"Invalid alignment state at ({i}, {j})")
    return {
        "substitutions": counts["substitution"],
        "deletions": counts["deletion"],
        "insertions": counts["insertion"],
        "errors": distance[-1][-1],
        "reference_units": len(ref),
    }


def compute_metrics(references: list[str], hypotheses: list[str]) -> dict[str, Any]:
    if len(references) != len(hypotheses) or not references:
        raise BenchmarkError("Metrics require equal, non-empty reference and hypothesis lists")
    word_totals = Counter()
    char_totals = Counter()
    for reference, hypothesis in zip(references, hypotheses):
        ref_norm = normalize_text(reference)
        hyp_norm = normalize_text(hypothesis)
        word_totals.update(edit_counts(ref_norm, hyp_norm))
        char_counts = edit_counts(" ".join(ref_norm), " ".join(hyp_norm))
        char_totals.update(char_counts)
    word_denominator = word_totals["reference_units"]
    char_denominator = char_totals["reference_units"]
    return {
        "wer": word_totals["errors"] / word_denominator if word_denominator else None,
        "cer": char_totals["errors"] / char_denominator if char_denominator else None,
        "word_errors": word_totals["errors"],
        "reference_words": word_denominator,
        "substitutions": word_totals["substitutions"],
        "deletions": word_totals["deletions"],
        "insertions": word_totals["insertions"],
        "character_errors": char_totals["errors"],
        "reference_characters": char_denominator,
    }


def _load_audio(audio_path: Path, sample_rate: int) -> tuple[np.ndarray, float, str]:
    try:
        audio_bytes = audio_path.read_bytes()
        audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()
        audio, input_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except (OSError, RuntimeError, sf.LibsndfileError) as exc:
        raise BenchmarkError(f"Could not read audio {audio_path}: {exc}") from exc
    if audio.shape[0] == 0:
        raise BenchmarkError(f"Audio is empty: {audio_path}")
    if not np.isfinite(audio).all():
        raise BenchmarkError(f"Audio contains NaN or infinite values: {audio_path}")
    audio = audio.mean(axis=1, dtype=np.float32)
    duration = len(audio) / input_rate
    if input_rate != sample_rate:
        divisor = math.gcd(int(input_rate), sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, int(input_rate) // divisor).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak == 0:
        raise BenchmarkError(f"Audio is silent: {audio_path}")
    return np.ascontiguousarray(audio, dtype=np.float32), duration, audio_sha256


def _model_fingerprint(model_path: Path) -> dict[str, str]:
    important = (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    files = [model_path / name for name in important if (model_path / name).is_file()]
    files.extend(model_path.glob("model*.safetensors"))
    files.extend(model_path.glob("pytorch_model*.bin"))
    files = sorted({path for path in files if path.is_file()})
    if not files:
        raise BenchmarkError(f"Cannot fingerprint model files in {model_path}")
    return {str(path.relative_to(model_path)): _sha256_file(path) for path in files}


def _resolve_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise BenchmarkError(f"Requested device {requested!r}, but CUDA is unavailable")
    mps_backend = getattr(torch.backends, "mps", None)
    if requested == "mps" and (mps_backend is None or not mps_backend.is_available()):
        raise BenchmarkError("Requested device 'mps', but this PyTorch build has no MPS backend")
    return requested


def _load_model(model_config: dict[str, Any], device: str) -> tuple[Any, Any, float]:
    try:
        import torch
        from transformers import AutoProcessor, MoonshineForConditionalGeneration
    except ImportError as exc:
        raise BenchmarkError(
            "Benchmark inference dependencies are missing. Install "
            "tools-py/moonshine_seq2seq_golden_test/requirements.txt"
        ) from exc
    start = time.perf_counter()
    model_path = model_config["path"]
    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = MoonshineForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise BenchmarkError(f"Could not load Moonshine model {model_config['id']!r} from {model_path}: {exc}") from exc
    return model, processor, time.perf_counter() - start


def _synchronize(device: str, torch: Any) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _transcribe(
    audio: np.ndarray,
    processor: Any,
    model: Any,
    generation: dict[str, Any],
    device: str,
    torch: Any,
) -> tuple[str, float]:
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    input_values = inputs.input_values.to(device)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    generation_kwargs = {
        "max_new_tokens": generation["max_new_tokens"],
        "num_beams": generation["num_beams"],
        "do_sample": False,
        "length_penalty": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "early_stopping": False,
    }
    _synchronize(device, torch)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_values=input_values,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
    _synchronize(device, torch)
    elapsed = time.perf_counter() - start
    text = processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
    return text, elapsed


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summarize_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, bool], list[dict[str, Any]]] = {}
    for row in predictions:
        key = (row["model_id"], row["dataset_id"], row["dataset_language"], row["language_match"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (model_id, dataset_id, dataset_language, language_match), rows in groups.items():
        metrics = compute_metrics(
            [row["reference"] for row in rows],
            [row["transcription"] for row in rows],
        )
        audio_seconds = sum(row["audio_duration_sec"] for row in rows)
        inference_seconds = sum(row["inference_sec"] for row in rows)
        summaries.append({
            "model_id": model_id,
            "dataset_id": dataset_id,
            "dataset_language": dataset_language,
            "language_match": language_match,
            "samples": len(rows),
            "references_acoustically_verified": all(row["reference_verified"] for row in rows),
            "interpretation": "measured" if all(row["reference_verified"] for row in rows) else "exploratory_unverified_references",
            **metrics,
            "audio_duration_sec": audio_seconds,
            "inference_sec": inference_seconds,
            "real_time_factor": inference_seconds / audio_seconds if audio_seconds else None,
        })
    return summaries


def _summarize_categories(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, bool], list[dict[str, Any]]] = {}
    for row in predictions:
        key = (
            row["model_id"],
            row["dataset_id"],
            row.get("category") or "uncategorized",
            row["language_match"],
        )
        groups.setdefault(key, []).append(row)
    results = []
    for (model_id, dataset_id, category, language_match), rows in groups.items():
        metrics = compute_metrics(
            [row["reference"] for row in rows],
            [row["transcription"] for row in rows],
        )
        results.append({
            "model_id": model_id,
            "dataset_id": dataset_id,
            "category": category,
            "language_match": language_match,
            "samples": len(rows),
            "references_acoustically_verified": all(row["reference_verified"] for row in rows),
            **metrics,
        })
    return results


def _render_report(
    metadata: dict[str, Any],
    summaries: list[dict[str, Any]],
    category_summaries: list[dict[str, Any]],
) -> str:
    lines = [
        f"# ASR benchmark: {metadata['run_id']}",
        "",
        f"- Started: `{metadata['started_at']}`",
        f"- Status: `{metadata['status']}`",
        f"- Device: `{metadata.get('device', 'not resolved')}`",
        f"- Configuration SHA-256: `{metadata['config_sha256']}`",
        f"- Audio preprocessing: mono, 16 kHz, no loudness normalization",
        f"- Text normalization: Unicode NFKC, case-folding, punctuation removed",
        "",
        "> WER/CER are exploratory when references were not acoustically verified. "
        "Cross-language model/dataset results are controls, not fair in-language scores.",
        "",
        "| Model | Dataset | Language match | Samples | WER | CER | Substitutions | Deletions | Insertions | RTF | Reference status |",
        "|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        wer = "n/a" if row["wer"] is None else f"{row['wer'] * 100:.2f}%"
        cer = "n/a" if row["cer"] is None else f"{row['cer'] * 100:.2f}%"
        rtf = "n/a" if row["real_time_factor"] is None else f"{row['real_time_factor']:.3f}"
        language_match = "yes" if row["language_match"] else "no (control)"
        status = "verified" if row["references_acoustically_verified"] else "unverified"
        lines.append(
            f"| {row['model_id']} | {row['dataset_id']} | {language_match} | {row['samples']} "
            f"| {wer} | {cer} | {row['substitutions']} | {row['deletions']} | {row['insertions']} | {rtf} | {status} |"
        )
    lines.extend([
        "",
        "## Results by text category",
        "",
        "| Model | Dataset | Category | Language match | Samples | WER | CER | Reference status |",
        "|---|---|---|:---:|---:|---:|---:|---|",
    ])
    for row in category_summaries:
        wer = "n/a" if row["wer"] is None else f"{row['wer'] * 100:.2f}%"
        cer = "n/a" if row["cer"] is None else f"{row['cer'] * 100:.2f}%"
        language_match = "yes" if row["language_match"] else "no (control)"
        status = "verified" if row["references_acoustically_verified"] else "unverified"
        lines.append(
            f"| {row['model_id']} | {row['dataset_id']} | {row['category']} | "
            f"{language_match} | {row['samples']} | {wer} | {cer} | {status} |"
        )
    lines.extend(["", "## Reproducibility", "", f"- Run ID: `{metadata['run_id']}`"])
    for model in metadata["models"]:
        lines.append(f"- Model `{model['id']}` at `{model['path']}`")
        for filename, digest in model["sha256"].items():
            lines.append(f"  - `{filename}` SHA-256: `{digest}`")
    for dataset in metadata["datasets"]:
        lines.append(f"- Dataset `{dataset['id']}` manifest SHA-256: `{dataset['manifest_sha256']}`")
    lines.append("")
    return "\n".join(lines)


def dry_run(config: dict[str, Any]) -> list[dict[str, Any]]:
    planned = []
    for dataset in config["datasets_resolved"]:
        records = load_manifest_dataset(
            dataset,
            config["max_samples_per_dataset"],
            config.get("sample_seed", 0),
        )
        if not config.get("allow_unverified_references", False) and not all(
            record["reference_verified"] for record in records
        ):
            raise BenchmarkError(
                f"Dataset {dataset['id']!r} contains unverified reference text. "
                "Review it or set allow_unverified_references to true for an explicitly exploratory run."
            )
        planned.append({
            "dataset_id": dataset["id"],
            "language": dataset["language"],
            "samples": len(records),
            "verified_references": sum(record["reference_verified"] for record in records),
            "excluded_categories": dataset.get("exclude_categories", []),
        })
    return planned


def run_benchmark(config: dict[str, Any], run_label: str | None = None) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise BenchmarkError(
            "PyTorch is missing. Install tools-py/moonshine_seq2seq_golden_test/requirements.txt"
        ) from exc

    planned_datasets = dry_run(config)
    records_by_dataset = {
        dataset["id"]: load_manifest_dataset(
            dataset,
            config["max_samples_per_dataset"],
            config.get("sample_seed", 0),
        )
        for dataset in config["datasets_resolved"]
    }
    device = _resolve_device(config["device"], torch)
    started = datetime.now().astimezone()
    timestamp = started.strftime("%Y%m%dT%H%M%S%f%z")
    run_id = f"{timestamp}_{_safe_name(run_label or config.get('run_label', 'asr-benchmark'))}"
    output_root = Path(config["output_root_resolved"])
    run_dir = output_root / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise BenchmarkError(f"Refusing to overwrite existing benchmark run: {run_dir}") from exc

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at": started.isoformat(),
        "finished_at": None,
        "config_path": config["config_path"],
        "config_sha256": config["config_sha256"],
        "device": device,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_names": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else []
        ),
        "transformers_version": None,
        "runtime_versions": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "soundfile")
        },
        "sample_seed": config.get("sample_seed", 0),
        "generation": config["generation"],
        "datasets": [],
        "models": [],
        "planned_datasets": planned_datasets,
        "warnings": [],
    }
    if any(item["verified_references"] != item["samples"] for item in planned_datasets):
        metadata["warnings"].append(
            "At least one reference transcript is not acoustically verified; WER/CER are exploratory."
        )
    _write_json(run_dir / "run_metadata.json", metadata)
    predictions: list[dict[str, Any]] = []
    try:
        (run_dir / "config_snapshot.json").write_bytes(config["config_snapshot_bytes"])
        import transformers

        metadata["transformers_version"] = transformers.__version__
        metadata["benchmark_script_sha256"] = _sha256_file(Path(__file__))
        for dataset in config["datasets_resolved"]:
            manifest_path = Path(dataset["manifest_path"])
            metadata["datasets"].append({
                "id": dataset["id"],
                "language": dataset["language"],
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "excluded_categories": dataset.get("exclude_categories", []),
                "sample_count": next(
                    item["samples"] for item in planned_datasets if item["dataset_id"] == dataset["id"]
                ),
            })
        for model_config in config["models_resolved"]:
            model_path = Path(model_config["path"])
            model_entry = {
                "id": model_config["id"],
                "language": model_config["language"],
                "path": str(model_path),
                "sha256": _model_fingerprint(model_path),
            }
            metadata["models"].append(model_entry)
            _write_json(run_dir / "run_metadata.json", metadata)
            model, processor, model_load_seconds = _load_model(model_config, device)
            model_entry["load_seconds"] = model_load_seconds
            _write_json(run_dir / "run_metadata.json", metadata)
            for dataset in config["datasets_resolved"]:
                records = records_by_dataset[dataset["id"]]
                for index, record in enumerate(records, start=1):
                    audio_path = Path(record["audio_path"])
                    audio, duration, audio_sha256 = _load_audio(audio_path, 16000)
                    transcription, inference_seconds = _transcribe(
                        audio, processor, model, config["generation"], device, torch
                    )
                    row = {
                        "model_id": model_config["id"],
                        "model_language": model_config["language"],
                        "dataset_id": dataset["id"],
                        "dataset_language": dataset["language"],
                        "language_match": model_config["language"] == dataset["language"].split("-", 1)[0].casefold(),
                        "sample_id": record["sample_id"],
                        "manifest_line": record["manifest_line"],
                        "audio_path": str(audio_path),
                        "audio_sha256": audio_sha256,
                        "audio_duration_sec": duration,
                        "reference": record["reference"],
                        "reference_normalized": normalize_text(record["reference"]),
                        "reference_verified": record["reference_verified"],
                        "speaker": record["speaker"],
                        "source": record["source"],
                        "category": record["category"],
                        "transcription": transcription,
                        "transcription_normalized": normalize_text(transcription),
                        "inference_sec": inference_seconds,
                        "real_time_factor": inference_seconds / duration if duration else None,
                        "word_errors": edit_counts(
                            normalize_text(record["reference"]),
                            normalize_text(transcription),
                        ),
                        "character_errors": edit_counts(
                            " ".join(normalize_text(record["reference"])),
                            " ".join(normalize_text(transcription)),
                        ),
                    }
                    predictions.append(row)
                    with (run_dir / "predictions.jsonl").open("a", encoding="utf-8") as output:
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(
                        f"[{model_config['id']} | {dataset['id']}] "
                        f"{index}/{len(records)} {record['sample_id']}: {transcription}"
                    )
            del model, processor
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        summaries = _summarize_predictions(predictions)
        category_summaries = _summarize_categories(predictions)
        _write_json(run_dir / "summary.json", {
            "schema_version": 1,
            "run_id": run_id,
            "sample_count": len(predictions),
            "results": summaries,
            "category_results": category_summaries,
        })
        csv_path = run_dir / "predictions.csv"
        if predictions:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(predictions[0]))
                writer.writeheader()
                writer.writerows(predictions)
        metadata["status"] = "complete"
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        metadata["prediction_count"] = len(predictions)
        metadata["summary"] = "summary.json"
        metadata["predictions"] = ["predictions.jsonl", "predictions.csv"]
        _write_json(run_dir / "run_metadata.json", metadata)
        (run_dir / "report.md").write_text(
            _render_report(metadata, summaries, category_summaries),
            encoding="utf-8",
        )
        return run_dir
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        metadata["prediction_count"] = len(predictions)
        _write_json(run_dir / "run_metadata.json", metadata)
        raise


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare local Moonshine models against manifest reference transcripts."
    )
    default_config = PROJECT_ROOT / ".data" / "golden_test" / "benchmark.json"
    example_config = Path(__file__).with_name("benchmark.example.json")
    parser.add_argument(
        "--config",
        type=Path,
        help=f"Path to benchmark JSON configuration (default: {default_config})",
    )
    parser.add_argument("--limit", type=int, help="Override the maximum samples selected per dataset")
    parser.add_argument("--device", help="Override configured device (auto, cpu, cuda, cuda:0, ...)")
    parser.add_argument("--label", help="Short descriptive label included in the output run directory name")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the plan without inference")
    args = parser.parse_args(argv)
    try:
        config_path = args.config.resolve() if args.config else default_config
        if not config_path.is_file():
            if config_path == default_config.resolve():
                if initialize_config(config_path, example_config):
                    print(f"Created neutral local benchmark configuration: {config_path}")
                    print("Edit its model and manifest paths, then run the benchmark again.")
                    return 0
            raise BenchmarkError(f"Configuration file does not exist: {config_path}")
        config = load_config(config_path)
        if args.limit is not None:
            if args.limit < 1:
                raise BenchmarkError("--limit must be a positive integer")
            config["max_samples_per_dataset"] = args.limit
        if args.device:
            config["device"] = args.device
        plan = dry_run(config)
        for dataset in plan:
            print(
                f"Dataset {dataset['dataset_id']}: {dataset['samples']} samples, "
                f"language={dataset['language']}, verified references="
                f"{dataset['verified_references']}/{dataset['samples']}"
            )
        print(f"Models: {len(config['models_resolved'])}; device: {config['device']}")
        if not config.get("allow_unverified_references", False):
            print("All references are acoustically verified.")
        elif any(item["verified_references"] != item["samples"] for item in plan):
            print("Warning: unverified references allowed; results are exploratory, not ground-truth accuracy.")
        if args.dry_run:
            print("Dry run complete; no files were written.")
            return 0
        run_dir = run_benchmark(config, args.label)
        print(f"Benchmark complete: {run_dir}")
        return 0
    except (BenchmarkError, OSError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
