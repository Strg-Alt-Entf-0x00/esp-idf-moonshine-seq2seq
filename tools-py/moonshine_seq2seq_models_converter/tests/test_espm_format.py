import json
import struct
import tempfile
import unittest
import zlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moonshine_seq2seq_models_converter.espm_format import (
    MODEL_QUANT_INT8,
    MODEL_QUANT_INT8_PER_CH,
    PART_DECODER,
    PART_ENCODER,
    read_espm,
    encode_tokenizer_json,
    write_espm,
)


class ESPMFormatTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "vocab_size": 8,
            "hidden_size": 4,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "attention_heads": 2,
            "ffn_size": 8,
        }
        self.vocab = b"\x01a\x01b\x01c\x01d\x01e\x01f\x01g\x01h"
        self.tensors = [
            {
                "name": "conv.weight",
                "part": PART_ENCODER,
                "shape": (2, 2),
                "quant_type": MODEL_QUANT_INT8_PER_CH,
                "channel_axis": 0,
                "scales": struct.pack("<2f", 0.25, 0.5),
                "data": bytes((1, 2, 3, 4)),
            },
            {
                "name": "embed_tokens.weight",
                "part": PART_DECODER,
                "shape": (2, 2),
                "quant_type": MODEL_QUANT_INT8,
                "channel_axis": 0xFF,
                "scales": struct.pack("<f", 0.125),
                "data": bytes((5, 6, 7, 8)),
            },
        ]

    def _write(self, path, *, vocab=None, model_config=None, tensors=None,
               quant_type=5):
        return write_espm(
            path,
            model_name="moonshine-test",
            version_str="test-1",
            quant_type=quant_type,
            vocab=self.vocab if vocab is None else vocab,
            model_config=self.config if model_config is None else model_config,
            tensors=self.tensors if tensors is None else tensors,
            metadata={"source_revision": "example-revision"},
        )

    def _read_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.espm"
            self._write(path)
            return bytearray(path.read_bytes())

    @staticmethod
    def _update_checksums(container):
        table_offset = struct.unpack_from("<I", container, 80)[0]
        section_count = struct.unpack_from("<H", container, 10)[0]
        for item in range(section_count):
            entry = table_offset + item * 40
            _, _, offset, size, _, _ = struct.unpack_from("<HHQQI16s", container, entry)
            struct.pack_into(
                "<I", container, entry + 20,
                zlib.crc32(container[offset:offset + size]) & 0xFFFFFFFF)
        table_end = table_offset + section_count * 40
        struct.pack_into(
            "<I", container, 96,
            zlib.crc32(container[table_offset:table_end]) & 0xFFFFFFFF)
        struct.pack_into("<I", container, 68, 0)
        struct.pack_into("<I", container, 68, zlib.crc32(container[:128]) & 0xFFFFFFFF)

    def test_round_trip_preserves_native_tensor_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.espm"
            self._write(path)
            parsed = read_espm(path.read_bytes())

        self.assertEqual(parsed["model_name"], "moonshine-test")
        self.assertEqual(parsed["model_config"], self.config)
        self.assertEqual(parsed["metadata"]["source_revision"], "example-revision")
        self.assertEqual(
            [(tensor["name"], tensor["part"], tensor["shape"], tensor["data"])
             for tensor in parsed["tensors"]],
            [(tensor["name"], tensor["part"], tensor["shape"], tensor["data"])
             for tensor in self.tensors],
        )

    def test_writer_emits_v3_mixed_quantization_summary(self):
        container = self._read_bytes()
        self.assertEqual(struct.unpack_from("<H", container, 4)[0], 3)
        self.assertEqual(struct.unpack_from("<H", container, 8)[0], 5)

    def test_writer_and_reader_enforce_tensor_data_psram_lower_bound(self):
        large_tensor = {
            "name": "psram_test.weight",
            "part": PART_ENCODER,
            "shape": (32, 32),
            "quant_type": MODEL_QUANT_INT8,
            "channel_axis": 0xFF,
            "scales": struct.pack("<f", 0.125),
            "data": bytes(range(256)) * 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.espm"
            self._write(path, tensors=self.tensors + [large_tensor])
            invalid = bytearray(path.read_bytes())

        table_offset = struct.unpack_from("<I", invalid, 80)[0]
        section_count = struct.unpack_from("<H", invalid, 10)[0]
        tensor_data_size = None
        for index in range(section_count):
            entry = table_offset + index * 40
            kind, _, _, size, _, _ = struct.unpack_from("<HHQQI16s", invalid, entry)
            if kind == 7:
                tensor_data_size = size
                break
        self.assertIsNotNone(tensor_data_size)
        minimum_kb = (tensor_data_size + 1023) // 1024
        self.assertEqual(struct.unpack_from("<I", invalid, 64)[0], minimum_kb)

        struct.pack_into("<I", invalid, 64, minimum_kb - 1)
        self._update_checksums(invalid)
        with self.assertRaisesRegex(ValueError, "lower bound is smaller"):
            read_espm(bytes(invalid))

    def test_reader_enforces_section_optional_flags(self):
        for section_type, invalid_flags, expected_error in (
                (5, 0, "diagnostic JSON section must be optional"),
                (7, 1, "required ESPM section is marked optional")):
            invalid = self._read_bytes()
            table_offset = struct.unpack_from("<I", invalid, 80)[0]
            section_count = struct.unpack_from("<H", invalid, 10)[0]
            for index in range(section_count):
                entry = table_offset + index * 40
                kind = struct.unpack_from("<H", invalid, entry)[0]
                if kind == section_type:
                    struct.pack_into("<H", invalid, entry + 2, invalid_flags)
                    break
            else:
                self.fail(f"missing section type {section_type}")
            self._update_checksums(invalid)
            with self.assertRaisesRegex(ValueError, expected_error):
                read_espm(bytes(invalid))

    def test_reader_rejects_v2_container(self):
        container = self._read_bytes()
        struct.pack_into("<H", container, 4, 2)
        with self.assertRaisesRegex(ValueError, "unsupported ESPM format version"):
            read_espm(bytes(container))

    def test_reader_rejects_unknown_required_metadata(self):
        invalid = self._read_bytes()
        table_offset = struct.unpack_from("<I", invalid, 80)[0]
        section_count = struct.unpack_from("<H", invalid, 10)[0]
        metadata_offset = None
        for item in range(section_count):
            entry = table_offset + item * 40
            kind, _, offset, _, _, _ = struct.unpack_from("<HHQQI16s", invalid, entry)
            if kind == 8:
                metadata_offset = offset
                break
        self.assertIsNotNone(metadata_offset)
        key_size = struct.unpack_from("<H", invalid, metadata_offset + 16)[0]
        self.assertEqual(key_size, len(b"general.architecture"))
        start = metadata_offset + 24
        invalid[start:start + key_size] = b"unknown.architecture"
        self._update_checksums(invalid)
        with self.assertRaisesRegex(ValueError, "unknown required"):
            read_espm(bytes(invalid))

    def test_writer_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.espm"
            second = Path(directory) / "second.espm"
            self._write(first)
            self._write(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_rejects_corrupted_section_and_overlapping_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.espm"
            self._write(path)
            original = path.read_bytes()

        corrupted = bytearray(original)
        corrupted[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "section CRC"):
            read_espm(bytes(corrupted))

        overlapping = bytearray(original)
        table_offset = struct.unpack_from("<I", overlapping, 80)[0]
        first_section = struct.unpack_from("<HHQQI16s", overlapping, table_offset)
        second_entry = table_offset + 40
        struct.pack_into("<Q", overlapping, second_entry + 4, first_section[2])
        section_count = struct.unpack_from("<H", overlapping, 10)[0]
        self._update_checksums(overlapping)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            read_espm(bytes(overlapping))

    def test_rejects_invalid_tensor_shapes(self):
        bad_tensors = [dict(self.tensors[0], shape=(2, 3))]
        bad_tensors.append(self.tensors[1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.espm"
            with self.assertRaisesRegex(ValueError, "payload size"):
                write_espm(
                    path,
                    model_name="moonshine-test",
                    version_str="test-1",
                    quant_type=MODEL_QUANT_INT8_PER_CH,
                    vocab=self.vocab,
                    model_config=self.config,
                    tensors=bad_tensors,
                )

    def test_tokenizer_encoding_preserves_ids_and_fills_gaps(self):
        source = b'{"model":{"vocab":{"a":0,"c":2}},"added_tokens":[{"id":1,"content":"b"}]}'
        encoded, count = encode_tokenizer_json(source)
        self.assertEqual(count, 3)
        self.assertEqual(encoded, b"\x01a\x01b\x01c")

    def test_tokenizer_encoding_handles_long_tokens_and_rejects_bad_ids(self):
        long_token = "x" * 130
        source = json.dumps({"model": {"vocab": {long_token: 0}}}).encode()
        encoded, count = encode_tokenizer_json(source)
        self.assertEqual(count, 1)
        self.assertEqual(encoded[:2], b"\x82\x01")

        invalid = b'{"model":{"vocab":{"a":true}}}'
        with self.assertRaisesRegex(ValueError, "invalid token or ID"):
            encode_tokenizer_json(invalid)

    def test_tokenizer_rejects_conflicting_ids_and_embedded_nuls(self):
        duplicate = b'{"model":{"vocab":{"a":0,"b":0}}}'
        with self.assertRaisesRegex(ValueError, "duplicate tokenizer ID"):
            encode_tokenizer_json(duplicate)

        embedded_nul = b'{"model":{"vocab":{"a\\u0000b":0}}}'
        with self.assertRaisesRegex(ValueError, "NUL"):
            encode_tokenizer_json(embedded_nul)

    def test_writer_rejects_vocabulary_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.espm"
            with self.assertRaisesRegex(ValueError, "token count"):
                self._write(path, vocab=b"\x01a")

    def test_writer_rejects_per_tensor_quantization_with_channel_axis(self):
        bad_tensors = [self.tensors[0], dict(self.tensors[1], channel_axis=0)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.espm"
            with self.assertRaisesRegex(ValueError, "per-tensor quantization metadata"):
                self._write(path, tensors=bad_tensors)

    def test_reader_rejects_runtime_unsupported_header_fields(self):
        for offset, field_format, value in (
                (6, "<H", 99),
                (8, "<H", 3),
                (60, "<I", 8000)):
            with self.subTest(offset=offset):
                invalid = self._read_bytes()
                struct.pack_into(field_format, invalid, offset, value)
                self._update_checksums(invalid)
                with self.assertRaisesRegex(ValueError, "unsupported ESPM header fields"):
                    read_espm(bytes(invalid))

        invalid = self._read_bytes()
        invalid[100] = 1
        self._update_checksums(invalid)
        with self.assertRaisesRegex(ValueError, "unsupported ESPM header fields"):
            read_espm(bytes(invalid))

    def test_reader_rejects_tensor_total_mismatch(self):
        invalid = self._read_bytes()
        table_offset = struct.unpack_from("<I", invalid, 80)[0]
        section_count = struct.unpack_from("<H", invalid, 10)[0]
        index_offset = None
        for item in range(section_count):
            entry = table_offset + item * 40
            section_type, _, offset, _, _, _ = struct.unpack_from(
                "<HHQQI16s", invalid, entry)
            if section_type == 6:
                index_offset = offset
                struct.pack_into("<Q", invalid, offset + 20, 7)
                break
        self.assertIsNotNone(index_offset)
        self._update_checksums(invalid)
        with self.assertRaisesRegex(ValueError, "summary mismatch"):
            read_espm(bytes(invalid))

    def test_reader_rejects_inconsistent_tensor_quantization_metadata(self):
        for field_offset, value, error in (
                (10, MODEL_QUANT_INT8, "per-tensor quantization metadata"),
                (52, 0xFF, "per-axis quantization metadata")):
            with self.subTest(field_offset=field_offset):
                invalid = self._read_bytes()
                table_offset = struct.unpack_from("<I", invalid, 80)[0]
                section_count = struct.unpack_from("<H", invalid, 10)[0]
                for item in range(section_count):
                    entry = table_offset + item * 40
                    section_type, _, offset, _, _, _ = struct.unpack_from(
                        "<HHQQI16s", invalid, entry)
                    if section_type == 6:
                        invalid[offset + 36 + field_offset] = value
                        break
                self._update_checksums(invalid)
                with self.assertRaisesRegex(ValueError, error):
                    read_espm(bytes(invalid))

    def test_reader_rejects_noncanonical_or_invalid_vocabulary_tokens(self):
        for value, error in ((0x80, "non-canonical"),
                             (0xFF, "UTF-8"),
                             (0x00, "NUL")):
            with self.subTest(value=value):
                invalid = self._read_bytes()
                table_offset = struct.unpack_from("<I", invalid, 80)[0]
                section_count = struct.unpack_from("<H", invalid, 10)[0]
                for item in range(section_count):
                    entry = table_offset + item * 40
                    section_type, _, offset, _, _, _ = struct.unpack_from(
                        "<HHQQI16s", invalid, entry)
                    if section_type == 1:
                        if value == 0x80:
                            invalid[offset:offset + 2] = b"\x80\x00"
                        else:
                            invalid[offset + 1] = value
                        break
                self._update_checksums(invalid)
                with self.assertRaisesRegex(ValueError, error):
                    read_espm(bytes(invalid))

    def test_writer_rejects_non_positive_quantization_scales(self):
        bad_tensors = [dict(self.tensors[0], scales=struct.pack("<2f", 0.25, 0.0))]
        bad_tensors.append(self.tensors[1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.espm"
            with self.assertRaisesRegex(ValueError, "quantization scale"):
                write_espm(
                    path,
                    model_name="moonshine-test",
                    version_str="test-1",
                    quant_type=MODEL_QUANT_INT8_PER_CH,
                    vocab=self.vocab,
                    model_config=self.config,
                    tensors=bad_tensors,
                )

if __name__ == "__main__":
    unittest.main()
