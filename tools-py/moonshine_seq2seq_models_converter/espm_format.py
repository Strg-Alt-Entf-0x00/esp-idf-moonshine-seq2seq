"""ESPM writer and strict host-side reader used by the converter tests."""

from __future__ import annotations

import math
import json
import operator
import struct
import zlib
from pathlib import Path
from typing import Any, Iterable

MAGIC = b"ESPM"
HEADER_SIZE = 128
SECTION_ENTRY_SIZE = 40
ALIGNMENT = 64
INDEX_MAGIC = b"ETIX"
MAX_VOCAB_SIZE = 32768
MAX_HIDDEN_SIZE = 512
MAX_LAYERS = 12

V3_FORMAT_VERSION = 3
V3_MAX_SECTIONS = 16
V3_INDEX_VERSION = 2
V3_INDEX_HEADER_SIZE = 36
V3_TENSOR_RECORD_SIZE = 160
V3_MAX_TENSORS = 512
V3_SECTION_OPTIONAL = 1
V3_METADATA_HEADER = struct.Struct("<4sHHII")
V3_METADATA_RECORD = struct.Struct("<HBBI")
V3_INDEX_HEADER = struct.Struct("<4sHHHHIIQQ")
V3_TENSOR = struct.Struct("<II4BI8IIB3sQIQ Q64s12s")
assert V3_METADATA_HEADER.size == 16
assert V3_METADATA_RECORD.size == 8
assert V3_INDEX_HEADER.size == V3_INDEX_HEADER_SIZE
assert V3_TENSOR.size == V3_TENSOR_RECORD_SIZE

MODEL_TYPE_ASR_SEQ2SEQ = 3
MODEL_QUANT_INT8 = 1
MODEL_QUANT_INT8_PER_CH = 2
MODEL_QUANT_MIXED = 5
SECTION_VOCAB = 1
SECTION_TENSOR_INDEX = 6
SECTION_TENSOR_DATA = 7
SECTION_METADATA = 8
PART_ENCODER = 1
PART_DECODER = 2
DTYPE_INT8 = 1

_HEADER = struct.Struct("<IHHHH32s16sIIIQIHHIII28s")
_SECTION = struct.Struct("<HHQQI16s")
assert _HEADER.size == HEADER_SIZE
assert _SECTION.size == SECTION_ENTRY_SIZE


def _align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _uint32(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an unsigned 32-bit integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{label} must be an unsigned 32-bit integer") from error
    if not 0 <= result <= 0xFFFFFFFF:
        raise ValueError(f"{label} must be an unsigned 32-bit integer")
    return result


def _bytes_like(value: Any, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"{label} must be bytes-like")
    return bytes(value)


def _decode_fixed_string(value: bytes, label: str) -> str:
    terminator = value.find(b"\0")
    if terminator <= 0 or any(value[terminator + 1:]):
        raise ValueError(f"invalid {label} string")
    try:
        return value[:terminator].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 in {label}") from error


def _count_vocabulary_tokens(vocabulary: bytes) -> int:
    position = 0
    count = 0
    while position < len(vocabulary):
        first = vocabulary[position]
        position += 1
        if first >= 128:
            if position >= len(vocabulary):
                raise ValueError("truncated vocabulary token length")
            high = vocabulary[position]
            position += 1
            if high > 127:
                raise ValueError("invalid vocabulary token length")
            length = high * 128 + (first & 0x7F)
            if length < 128:
                raise ValueError("non-canonical vocabulary token length")
        else:
            length = first
        if length > len(vocabulary) - position:
            raise ValueError("truncated vocabulary token")
        token = vocabulary[position:position + length]
        if b"\0" in token:
            raise ValueError("vocabulary token contains a NUL byte")
        try:
            token.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("vocabulary token is not valid UTF-8") from error
        position += length
        count += 1
        if count > MAX_VOCAB_SIZE:
            raise ValueError("vocabulary exceeds the runtime limit")
    if count == 0:
        raise ValueError("vocabulary cannot be empty")
    return count


def _reject_json_constant(value: str):
    raise ValueError(f"invalid JSON constant {value}")


def encode_tokenizer_json(tokenizer_json: bytes) -> tuple[bytes, int]:
    """Encode the tokenizer's ID-ordered vocabulary in the firmware wire format."""
    try:
        tokenizer = json.loads(tokenizer_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid tokenizer JSON: {error}") from error
    if not isinstance(tokenizer, dict):
        raise ValueError("tokenizer JSON must contain an object")
    model = tokenizer.get("model")
    vocabulary = model.get("vocab") if isinstance(model, dict) else None
    if not isinstance(vocabulary, dict):
        raise ValueError("tokenizer JSON has no model vocabulary")
    tokens_by_id = {}

    def add_token(token_id: int, token: str) -> None:
        if token_id in tokens_by_id and tokens_by_id[token_id] != token:
            raise ValueError(f"duplicate tokenizer ID {token_id}")
        tokens_by_id[token_id] = token

    for token, token_id in vocabulary.items():
        if (not isinstance(token, str) or not isinstance(token_id, int) or
                isinstance(token_id, bool)):
            raise ValueError("tokenizer vocabulary has an invalid token or ID")
        add_token(token_id, token)
    added_tokens = tokenizer.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise ValueError("tokenizer added_tokens must be a list")
    for token in added_tokens:
        if (not isinstance(token, dict) or not isinstance(token.get("id"), int) or
                isinstance(token.get("id"), bool) or
                not isinstance(token.get("content"), str)):
            raise ValueError("tokenizer added token has invalid content or ID")
        add_token(token["id"], token["content"])
    if not tokens_by_id or min(tokens_by_id) < 0:
        raise ValueError("tokenizer vocabulary is empty or has negative token IDs")
    max_id = max(tokens_by_id)
    if max_id >= 32768:
        raise ValueError("tokenizer vocabulary exceeds the runtime limit")

    encoded = bytearray()
    for token_id in range(max_id + 1):
        token = tokens_by_id.get(token_id, "")
        try:
            token_bytes = token.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(f"token {token_id} is not valid UTF-8") from error
        if b"\0" in token_bytes:
            raise ValueError(f"token {token_id} contains a NUL byte")
        length = len(token_bytes)
        if length >= 16384:
            raise ValueError(f"token {token_id} exceeds the vocabulary format limit")
        if length < 128:
            encoded.append(length)
        else:
            encoded.extend(((length & 0x7F) | 0x80, length >> 7))
        encoded.extend(token_bytes)
    return bytes(encoded), max_id + 1


def _fixed_string(value: str, capacity: int, label: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    if not encoded or len(encoded) >= capacity or b"\0" in encoded:
        raise ValueError(f"{label} must fit in {capacity - 1} UTF-8 bytes")
    return encoded + bytes(capacity - len(encoded))








def _encode_typed_metadata(records: dict[str, tuple[int, bytes, int]]) -> bytes:
    encoded = bytearray(V3_METADATA_HEADER.pack(b"EMET", 1, 16, len(records), 0))
    for key in sorted(records):
        key_bytes = key.encode("utf-8")
        value_type, value, flags = records[key]
        if not key_bytes or b"\0" in key_bytes or len(key_bytes) > 0xFFFF:
            raise ValueError(f"invalid metadata key {key!r}")
        encoded.extend(V3_METADATA_RECORD.pack(len(key_bytes), value_type, flags, len(value)))
        encoded.extend(key_bytes)
        encoded.extend(value)
    if len(encoded) > 64 * 1024:
        raise ValueError("typed metadata exceeds 64 KiB")
    return bytes(encoded)


def _typed_u32(value: int) -> tuple[int, bytes, int]:
    return 1, struct.pack("<I", _uint32(value, "metadata value")), 1


def _typed_f32(value: float) -> tuple[int, bytes, int]:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("metadata float must be finite")
    return 4, struct.pack("<f", value), 1


def _typed_utf8(value: str) -> tuple[int, bytes, int]:
    encoded = value.encode("utf-8")
    if b"\0" in encoded:
        raise ValueError("metadata strings cannot contain NUL bytes")
    return 6, encoded, 1


def _tensor_payloads_v3(tensors: Iterable[dict[str, Any]]):
    records: list[bytes] = []
    data = bytearray()
    seen: set[tuple[int, str]] = set()
    part_counts = {PART_ENCODER: 0, PART_DECODER: 0}
    quant_types: set[int] = set()
    total_elements = 0
    total_data_bytes = 0
    tensor_items = list(tensors)
    if not tensor_items or len(tensor_items) > V3_MAX_TENSORS:
        raise ValueError(f"tensor count must be between 1 and {V3_MAX_TENSORS}")

    for tensor in tensor_items:
        if not isinstance(tensor, dict):
            raise ValueError("tensor record must be an object")
        try:
            name = tensor["name"]
            part = _uint32(tensor["part"], "tensor scope")
            shape = tuple(_uint32(dimension, "tensor dimension")
                          for dimension in tensor["shape"])
            quant = _uint32(tensor["quant_type"], "tensor quantization")
            channel_axis = _uint32(tensor["channel_axis"], "channel axis")
            scales = _bytes_like(tensor["scales"], "tensor scales")
            values = _bytes_like(tensor["data"], "tensor data")
        except KeyError as error:
            raise ValueError(f"tensor record is missing {error.args[0]}") from error

        if part not in part_counts:
            raise ValueError(f"invalid tensor scope for {name}")
        if not 1 <= len(shape) <= 8 or any(dimension == 0 for dimension in shape):
            raise ValueError(f"invalid shape for {name}")
        element_count = math.prod(shape)
        if element_count > 0xFFFFFFFF or len(values) != element_count:
            raise ValueError(f"invalid INT8 payload size for {name}")
        if quant not in (MODEL_QUANT_INT8, MODEL_QUANT_INT8_PER_CH):
            raise ValueError(f"unsupported tensor quantization for {name}")
        if len(scales) == 0 or len(scales) % 4:
            raise ValueError(f"invalid scale bytes for {name}")
        scale_count = len(scales) // 4
        if quant == MODEL_QUANT_INT8:
            if channel_axis != 0xFF or scale_count != 1:
                raise ValueError(f"per-tensor quantization metadata is invalid for {name}")
        elif (channel_axis == 0xFF or channel_axis >= len(shape) or
              scale_count != shape[channel_axis]):
            raise ValueError(f"per-axis scales do not match shape for {name}")
        if any(not math.isfinite(scale) or scale <= 0
               for scale in struct.unpack(f"<{scale_count}f", scales)):
            raise ValueError(f"invalid quantization scale for {name}")

        name_bytes = _fixed_string(name, 64, "tensor name")
        identity = (part, name)
        if identity in seen:
            raise ValueError(f"duplicate tensor name {name}")
        seen.add(identity)
        quant_types.add(quant)
        data.extend(bytes((-len(data)) % 4))
        scale_offset = len(data)
        data.extend(scales)
        data.extend(bytes((-len(data)) % 16))
        tensor_offset = len(data)
        data.extend(values)

        dimensions = shape + (0,) * (8 - len(shape))
        name_hash = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
        records.append(V3_TENSOR.pack(
            name_hash,
            part - 1,
            1,
            len(shape),
            quant,
            0,
            element_count,
            *dimensions,
            scale_count,
            channel_axis,
            bytes(3),
            scale_offset,
            len(scales),
            tensor_offset,
            len(values),
            name_bytes,
            bytes(12),
        ))
        part_counts[part] += 1
        total_elements += element_count
        total_data_bytes += len(values)

    if not part_counts[PART_ENCODER] or not part_counts[PART_DECODER]:
        raise ValueError("ESPM ASR model requires encoder and decoder tensors")
    if total_elements > 0xFFFFFFFFFFFFFFFF or total_data_bytes > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("tensor index totals exceed the format limit")
    summary_quant = next(iter(quant_types)) if len(quant_types) == 1 else MODEL_QUANT_MIXED
    return records, data, part_counts, total_elements, total_data_bytes, summary_quant


def _build_espm_v3(
    output_path: Path,
    *,
    model_name: str,
    version_str: str,
    quant_type: int,
    vocab: bytes,
    model_config: dict[str, int],
    tensors: Iterable[dict[str, Any]],
    metadata: dict[str, Any] | None,
) -> dict[str, int]:
    quant_type = _uint32(quant_type, "container quantization")
    if quant_type not in (MODEL_QUANT_INT8, MODEL_QUANT_INT8_PER_CH, MODEL_QUANT_MIXED):
        raise ValueError("unsupported container quantization")
    vocab = _bytes_like(vocab, "vocabulary")
    vocabulary_size = _count_vocabulary_tokens(vocab)
    if not isinstance(model_config, dict):
        raise ValueError("model configuration must be an object")
    config_keys = (
        "vocab_size", "hidden_size", "encoder_layers", "decoder_layers",
        "attention_heads", "ffn_size",
    )
    try:
        config = {key: _uint32(model_config[key], f"model configuration {key}")
                  for key in config_keys}
    except KeyError as error:
        raise ValueError("model configuration is incomplete") from error
    if any(config[key] == 0 for key in config_keys):
        raise ValueError("model configuration values must be positive")
    if config["vocab_size"] != vocabulary_size or config["vocab_size"] > MAX_VOCAB_SIZE:
        raise ValueError("vocabulary token count does not match model configuration")
    if (config["hidden_size"] > MAX_HIDDEN_SIZE or
            config["encoder_layers"] > MAX_LAYERS or
            config["decoder_layers"] > MAX_LAYERS or
            config["hidden_size"] % config["attention_heads"] != 0):
        raise ValueError("model configuration exceeds runtime limits")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")

    (tensor_records, tensor_data, part_counts, total_elements,
     total_data_bytes, actual_quant) = _tensor_payloads_v3(tensors)
    if (quant_type == MODEL_QUANT_INT8 and actual_quant != MODEL_QUANT_INT8) or \
            (quant_type == MODEL_QUANT_INT8_PER_CH and
             actual_quant not in (MODEL_QUANT_INT8_PER_CH, MODEL_QUANT_MIXED)) or \
            (quant_type == MODEL_QUANT_MIXED and actual_quant != MODEL_QUANT_MIXED):
        raise ValueError(
            f"container quantization {quant_type} does not match tensor summary {actual_quant}")

    index = V3_INDEX_HEADER.pack(
        INDEX_MAGIC, V3_INDEX_VERSION, V3_INDEX_HEADER_SIZE, V3_TENSOR_RECORD_SIZE,
        0, len(tensor_records), 2, total_elements, total_data_bytes,
    ) + b"".join(tensor_records)
    metadata_records = {
        "general.architecture": _typed_utf8("moonshine.seq2seq"),
        "general.architecture_version": _typed_u32(1),
        "moonshine.vocab_size": _typed_u32(config["vocab_size"]),
        "moonshine.hidden_size": _typed_u32(config["hidden_size"]),
        "moonshine.encoder_layers": _typed_u32(config["encoder_layers"]),
        "moonshine.decoder_layers": _typed_u32(config["decoder_layers"]),
        "moonshine.attention_heads": _typed_u32(config["attention_heads"]),
        "moonshine.ffn_size": _typed_u32(config["ffn_size"]),
        "moonshine.encoder_tensor_count": _typed_u32(part_counts[PART_ENCODER]),
        "moonshine.decoder_tensor_count": _typed_u32(part_counts[PART_DECODER]),
        "moonshine.total_params": _typed_u32(total_elements),
    }
    typed_metadata = _encode_typed_metadata(metadata_records)
    sections = [
        (SECTION_VOCAB, 0, "vocab", bytes(vocab)),
        (SECTION_TENSOR_INDEX, 0, "tensor_index", index),
        (SECTION_TENSOR_DATA, 0, "tensor_data", bytes(tensor_data)),
        (SECTION_METADATA, 0, "metadata", typed_metadata),
    ]
    if metadata is not None:
        diagnostic_json = json.dumps(
            metadata, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")
        sections.append((5, V3_SECTION_OPTIONAL, "config_json", diagnostic_json))
    if len(sections) > V3_MAX_SECTIONS:
        raise ValueError("too many ESPM sections")

    table_end = HEADER_SIZE + len(sections) * SECTION_ENTRY_SIZE
    next_offset = _align(table_end)
    rows = []
    for section_type, section_flags, name, payload in sections:
        if not payload:
            raise ValueError(f"section {name} cannot be empty")
        rows.append((section_type, section_flags, name, next_offset, payload,
                     zlib.crc32(payload) & 0xFFFFFFFF))
        next_offset = _align(next_offset + len(payload))
    file_size = rows[-1][3] + len(rows[-1][4])
    table = b"".join(_SECTION.pack(
        section_type, section_flags, offset, len(payload), checksum,
        _fixed_string(name, 16, "section name"),
    ) for section_type, section_flags, name, offset, payload, checksum in rows)
    table_crc = zlib.crc32(table) & 0xFFFFFFFF
    psram_required_kb = (len(tensor_data) + 1023) // 1024
    header_values = (
        int.from_bytes(MAGIC, "little"), V3_FORMAT_VERSION, MODEL_TYPE_ASR_SEQ2SEQ,
        actual_quant, len(rows), _fixed_string(model_name, 32, "model name"),
        _fixed_string(version_str, 16, "model version"), 16000, psram_required_kb,
        0, file_size, HEADER_SIZE, HEADER_SIZE, SECTION_ENTRY_SIZE, ALIGNMENT,
        0, table_crc, bytes(28),
    )
    header_without_crc = _HEADER.pack(*header_values)
    header_crc = zlib.crc32(header_without_crc) & 0xFFFFFFFF
    header = _HEADER.pack(*header_values[:9], header_crc, *header_values[10:])
    output = bytearray(file_size)
    output[:HEADER_SIZE] = header
    output[HEADER_SIZE:HEADER_SIZE + len(table)] = table
    for _, _, _, offset, payload, _ in rows:
        output[offset:offset + len(payload)] = payload
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)
    return {
        "file_size": file_size,
        "psram_required_kb": psram_required_kb,
        "tensor_count": len(tensor_records),
        "tensor_data_size": len(tensor_data),
    }


def write_espm(
    output_path: Path,
    *,
    model_name: str,
    version_str: str,
    quant_type: int,
    vocab: bytes,
    model_config: dict[str, int],
    tensors: Iterable[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Write a version 3 Moonshine container with typed metadata and ETIX v2."""
    return _build_espm_v3(
        output_path,
        model_name=model_name,
        version_str=version_str,
        quant_type=quant_type,
        vocab=vocab,
        model_config=model_config,
        tensors=tensors,
        metadata=metadata,
    )


def _decode_typed_metadata(data: bytes) -> dict[str, tuple[int, Any, int]]:
    if len(data) < V3_METADATA_HEADER.size or len(data) > 64 * 1024:
        raise ValueError("invalid typed metadata size")
    magic, version, header_size, count, reserved = V3_METADATA_HEADER.unpack_from(data)
    if (magic != b"EMET" or version != 1 or header_size != 16 or
            count > 512 or reserved != 0):
        raise ValueError("unsupported typed metadata header")
    position = V3_METADATA_HEADER.size
    values = {}
    for _ in range(count):
        if len(data) - position < V3_METADATA_RECORD.size:
            raise ValueError("truncated metadata record")
        key_size, value_type, flags, value_size = V3_METADATA_RECORD.unpack_from(data, position)
        position += V3_METADATA_RECORD.size
        end = position + key_size + value_size
        if not key_size or flags & ~1 or end > len(data):
            raise ValueError("invalid metadata record size or flags")
        try:
            key = data[position:position + key_size].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("invalid metadata key UTF-8") from error
        if "\0" in key or key in values:
            raise ValueError("invalid or duplicate metadata key")
        position += key_size
        raw_value = data[position:position + value_size]
        position += value_size
        if value_type == 1 and value_size == 4:
            value = struct.unpack("<I", raw_value)[0]
        elif value_type == 2 and value_size == 8:
            value = struct.unpack("<Q", raw_value)[0]
        elif value_type == 3 and value_size == 4:
            value = struct.unpack("<i", raw_value)[0]
        elif value_type == 4 and value_size == 4:
            value = struct.unpack("<f", raw_value)[0]
            if not math.isfinite(value):
                raise ValueError("non-finite metadata float")
        elif value_type == 5:
            if value_size != 1 or raw_value[0] > 1:
                raise ValueError("invalid metadata boolean")
            value = bool(raw_value[0])
        elif value_type == 6:
            try:
                value = raw_value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("invalid metadata UTF-8") from error
            if "\0" in value:
                raise ValueError("metadata string contains NUL")
        elif value_type == 7:
            value = raw_value
        elif value_type in (1, 2, 3, 4):
            raise ValueError("invalid metadata value length")
        elif flags & 1:
            raise ValueError("unsupported required metadata type")
        else:
            value = raw_value
        values[key] = (value_type, value, flags)
    if position != len(data):
        raise ValueError("trailing bytes in typed metadata")
    return values


def _read_espm_v3(container: bytes) -> dict[str, Any]:
    if len(container) < HEADER_SIZE:
        raise ValueError("truncated ESPM header")
    fields = _HEADER.unpack_from(container)
    (magic, version, model_type, quant_type, section_count, model_name, version_str,
     sample_rate, psram_kb, header_crc, file_size, table_offset, header_size,
     entry_size, alignment, flags, table_crc, reserved) = fields
    if (magic != int.from_bytes(MAGIC, "little") or version != V3_FORMAT_VERSION or
            model_type != MODEL_TYPE_ASR_SEQ2SEQ or
            quant_type not in (MODEL_QUANT_INT8, MODEL_QUANT_INT8_PER_CH, MODEL_QUANT_MIXED) or
            sample_rate != 16000 or psram_kb == 0 or file_size != len(container) or
            table_offset != HEADER_SIZE or header_size != HEADER_SIZE or
            entry_size != SECTION_ENTRY_SIZE or alignment != ALIGNMENT or
            flags != 0 or reserved != bytes(28) or
            not 1 <= section_count <= V3_MAX_SECTIONS):
        raise ValueError("unsupported ESPM header fields")
    header_copy = bytearray(container[:HEADER_SIZE])
    struct.pack_into("<I", header_copy, 68, 0)
    if zlib.crc32(header_copy) & 0xFFFFFFFF != header_crc:
        raise ValueError("ESPM header CRC mismatch")
    model_name = _decode_fixed_string(model_name, "model name")
    version_str = _decode_fixed_string(version_str, "model version")
    table_end = table_offset + section_count * entry_size
    if table_end > len(container):
        raise ValueError("truncated ESPM section table")
    table = container[table_offset:table_end]
    if zlib.crc32(table) & 0xFFFFFFFF != table_crc:
        raise ValueError("ESPM section table CRC mismatch")

    sections = {}
    section_names = set()
    ranges = []
    for index in range(section_count):
        kind, section_flags, offset, size, checksum, raw_name = _SECTION.unpack_from(
            table, index * entry_size)
        name = _decode_fixed_string(raw_name, "section name")
        if ((section_flags & ~V3_SECTION_OPTIONAL) != 0 or not size or
                offset % ALIGNMENT or offset < _align(table_end) or
                offset > len(container) or size > len(container) - offset or
                kind in sections or name in section_names):
            raise ValueError("invalid ESPM section record")
        if kind in (SECTION_VOCAB, SECTION_TENSOR_INDEX,
                    SECTION_TENSOR_DATA, SECTION_METADATA) and section_flags:
            raise ValueError("required ESPM section is marked optional")
        if kind == 5 and not section_flags:
            raise ValueError("diagnostic JSON section must be optional")
        if kind not in (SECTION_VOCAB, 5, SECTION_TENSOR_INDEX,
                        SECTION_TENSOR_DATA, SECTION_METADATA):
            if not section_flags & V3_SECTION_OPTIONAL:
                raise ValueError("unknown required ESPM section")
        payload = container[offset:offset + size]
        if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
            raise ValueError("ESPM section CRC mismatch")
        sections[kind] = payload
        section_names.add(name)
        ranges.append((offset, offset + size))
    ranges.sort()
    if (any(ranges[i][0] < ranges[i - 1][1] for i in range(1, len(ranges))) or
            max(end for _, end in ranges) != len(container)):
        raise ValueError("overlapping or trailing ESPM section data")
    required_sections = {SECTION_VOCAB, SECTION_TENSOR_INDEX,
                         SECTION_TENSOR_DATA, SECTION_METADATA}
    if not required_sections.issubset(sections):
        raise ValueError("missing required Moonshine ESPM section")

    values = _decode_typed_metadata(sections[SECTION_METADATA])
    required_keys = {
        "general.architecture": (6, "moonshine.seq2seq"),
        "general.architecture_version": (1, 1),
        "moonshine.vocab_size": (1, None),
        "moonshine.hidden_size": (1, None),
        "moonshine.encoder_layers": (1, None),
        "moonshine.decoder_layers": (1, None),
        "moonshine.attention_heads": (1, None),
        "moonshine.ffn_size": (1, None),
        "moonshine.encoder_tensor_count": (1, None),
        "moonshine.decoder_tensor_count": (1, None),
        "moonshine.total_params": (1, None),
    }
    if any(entry[2] & 1 and key not in required_keys
           for key, entry in values.items()):
        raise ValueError("unknown required metadata key")
    for key, (expected_type, expected_value) in required_keys.items():
        if key not in values or values[key][0] != expected_type or not values[key][2] & 1:
            raise ValueError(f"missing or invalid required metadata {key}")
        if expected_value is not None and values[key][1] != expected_value:
            raise ValueError(f"unsupported metadata value for {key}")
    config = {
        key: values[f"moonshine.{key}"][1]
        for key in ("vocab_size", "hidden_size", "encoder_layers",
                    "decoder_layers", "attention_heads", "ffn_size")
    }

    vocab = sections[SECTION_VOCAB]
    if psram_kb * 1024 < len(sections[SECTION_TENSOR_DATA]):
        raise ValueError("ESPM PSRAM lower bound is smaller than tensor data")
    if (_count_vocabulary_tokens(vocab) != config["vocab_size"] or
            not 1 <= config["vocab_size"] <= MAX_VOCAB_SIZE or
            not 1 <= config["hidden_size"] <= MAX_HIDDEN_SIZE or
            not 1 <= config["encoder_layers"] <= MAX_LAYERS or
            not 1 <= config["decoder_layers"] <= MAX_LAYERS or
            not config["attention_heads"] or
            config["hidden_size"] % config["attention_heads"] or
            not config["ffn_size"]):
        raise ValueError("invalid ESPM model configuration or vocabulary")

    index = sections[SECTION_TENSOR_INDEX]
    if len(index) < V3_INDEX_HEADER_SIZE:
        raise ValueError("truncated ETIX v2 header")
    index_fields = V3_INDEX_HEADER.unpack_from(index)
    (index_magic, index_version, index_header_size, record_size, index_flags,
     tensor_count, scope_count, total_elements, total_data_bytes) = index_fields
    if (index_magic != INDEX_MAGIC or index_version != V3_INDEX_VERSION or
            index_header_size != V3_INDEX_HEADER_SIZE or
            record_size != V3_TENSOR_RECORD_SIZE or index_flags != 0 or
            tensor_count == 0 or tensor_count > V3_MAX_TENSORS or scope_count != 2 or
            len(index) != V3_INDEX_HEADER_SIZE + tensor_count * V3_TENSOR_RECORD_SIZE):
        raise ValueError("invalid ETIX v2 index header")

    tensor_data = sections[SECTION_TENSOR_DATA]
    tensors = []
    payload_ranges = []
    scope_counts = [0, 0]
    quant_types = set()
    observed_elements = 0
    observed_bytes = 0
    seen = set()
    for item in range(tensor_count):
        record = V3_TENSOR.unpack_from(
            index, V3_INDEX_HEADER_SIZE + item * V3_TENSOR_RECORD_SIZE)
        (name_hash, scope, dtype, rank, tensor_quant, layout, num_elements, *tail) = record
        shape = tuple(tail[:8])
        num_channels, channel_axis, reserved, scales_offset, scales_size, data_offset, \
            data_size, raw_name, reserved_v2 = tail[8:]
        name = _decode_fixed_string(raw_name, "tensor name")
        if (scope > 1 or dtype != 1 or tensor_quant not in (1, 2) or layout != 0 or
                not 1 <= rank <= 8 or any(dim <= 0 for dim in shape[:rank]) or
                any(shape[rank:]) or math.prod(shape[:rank]) != num_elements or
                reserved != bytes(3) or reserved_v2 != bytes(12) or
                zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF != name_hash):
            raise ValueError("invalid ETIX v2 tensor record")
        identity = (scope, name)
        if identity in seen:
            raise ValueError("duplicate tensor identity")
        seen.add(identity)
        if tensor_quant == MODEL_QUANT_INT8:
            if channel_axis != 0xFF or num_channels != 1:
                raise ValueError("invalid per-tensor quantization metadata")
        elif channel_axis >= rank or shape[channel_axis] != num_channels:
            raise ValueError("invalid per-axis quantization metadata")
        scales_end = scales_offset + scales_size
        data_end = data_offset + data_size
        if (scales_offset % 4 or data_offset % 16 or scales_size != num_channels * 4 or
                data_size != num_elements or scales_end > len(tensor_data) or
                data_end > len(tensor_data)):
            raise ValueError("tensor payload is outside tensor data section")
        scales = tensor_data[scales_offset:scales_end]
        values_payload = tensor_data[data_offset:data_end]
        if any(not math.isfinite(scale) or scale <= 0
               for scale in struct.unpack(f"<{num_channels}f", scales)):
            raise ValueError("invalid tensor quantization scale")
        payload_ranges.extend(((scales_offset, scales_end), (data_offset, data_end)))
        tensors.append({
            "name": name, "part": scope + 1, "dtype": dtype, "rank": rank,
            "quant_type": tensor_quant, "shape": shape[:rank],
            "num_elements": num_elements, "num_channels": num_channels,
            "channel_axis": channel_axis, "scales": scales, "data": values_payload,
        })
        scope_counts[scope] += 1
        quant_types.add(tensor_quant)
        observed_elements += num_elements
        observed_bytes += data_size
    payload_ranges.sort()
    if any(payload_ranges[i][0] < payload_ranges[i - 1][1]
           for i in range(1, len(payload_ranges))):
        raise ValueError("overlapping tensor payloads")
    summary_quant = next(iter(quant_types)) if len(quant_types) == 1 else MODEL_QUANT_MIXED
    if (observed_elements != total_elements or observed_bytes != total_data_bytes or
            scope_counts != [values["moonshine.encoder_tensor_count"][1],
                             values["moonshine.decoder_tensor_count"][1]] or
            values["moonshine.total_params"][1] != total_elements or
            summary_quant != quant_type):
        raise ValueError("ETIX v2 tensor index summary mismatch")

    diagnostic_metadata = None
    if 5 in sections:
        try:
            diagnostic_metadata = json.loads(
                sections[5], parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("invalid diagnostic config JSON") from error
        if not isinstance(diagnostic_metadata, dict):
            raise ValueError("diagnostic config JSON must contain an object")
    return {
        "model_type": model_type,
        "quant_type": quant_type,
        "model_name": model_name,
        "version": version_str,
        "sample_rate": sample_rate,
        "psram_required_kb": psram_kb,
        "vocab": vocab,
        "metadata": diagnostic_metadata,
        "typed_metadata": {key: value[1] for key, value in values.items()},
        "model_config": config,
        "tensors": tensors,
    }


def read_espm(container: bytes) -> dict[str, Any]:
    """Validate an ESPM v3 container."""
    if len(container) < HEADER_SIZE:
        raise ValueError("truncated ESPM header")
    version = struct.unpack_from("<H", container, 4)[0]
    if version != V3_FORMAT_VERSION:
        raise ValueError("unsupported ESPM format version")
    return _read_espm_v3(container)
