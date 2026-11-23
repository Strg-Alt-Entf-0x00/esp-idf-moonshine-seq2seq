import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moonshine_seq2seq_golden_test.benchmark import (
    BenchmarkError,
    compute_metrics,
    dry_run,
    edit_counts,
    _load_audio,
    load_manifest_dataset,
    normalize_text,
    _summarize_categories,
    initialize_config,
    PROJECT_ROOT,
)
from moonshine_seq2seq_golden_test import benchmark as benchmark_module


class TextMetricTests(unittest.TestCase):
    def test_normalization_is_unicode_aware_and_ignores_punctuation(self):
        self.assertEqual(normalize_text("  FÜR, München!  "), "für münchen")

    def test_edit_counts_separate_substitutions_insertions_and_deletions(self):
        self.assertEqual(
            edit_counts("we test models", "we compare model"),
            {
                "substitutions": 2,
                "deletions": 0,
                "insertions": 0,
                "errors": 2,
                "reference_units": 3,
            },
        )

    def test_metrics_use_corpus_totals_not_average_sentence_rates(self):
        result = compute_metrics(["one two three", "four"], ["one two", "four five"])
        self.assertEqual(result["wer"], 0.5)
        self.assertEqual(result["deletions"], 1)
        self.assertEqual(result["insertions"], 1)

    def test_audio_is_resampled_and_hashes_the_exact_input_bytes(self):
        import hashlib

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audio.wav"
            sf.write(path, np.ones(4800, dtype=np.float32) * 0.1, 48000)
            samples, duration, audio_hash = _load_audio(path, 16000)
            self.assertEqual(len(samples), 1600)
            self.assertAlmostEqual(duration, 0.1, places=3)
            self.assertEqual(audio_hash, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_metrics_reject_empty_inputs(self):
        with self.assertRaises(BenchmarkError):
            compute_metrics([], [])

    def test_category_summary_groups_speech_types(self):
        rows = [
            {
                "model_id": "tiny-de",
                "dataset_id": "de-female",
                "category": "everyday",
                "language_match": True,
                "reference": "Hallo Welt",
                "transcription": "Hallo Welt",
                "reference_verified": False,
            },
            {
                "model_id": "tiny-de",
                "dataset_id": "de-female",
                "category": "basic",
                "language_match": True,
                "reference": "Guten Morgen",
                "transcription": "Guten Morgen",
                "reference_verified": False,
            },
        ]
        results = _summarize_categories(rows)
        self.assertEqual({row["category"] for row in results}, {"everyday", "basic"})

    def test_config_rejects_output_outside_dot_data(self):
        from moonshine_seq2seq_golden_test.benchmark import load_config

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "benchmark.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "project_root": ".",
                    "output_root": "../outside",
                    "models": [],
                    "datasets": [],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BenchmarkError, "inside the project .data"):
                load_config(root / "benchmark.json")

    def test_default_project_root_is_portable_and_does_not_depend_on_config_location(self):
        self.assertTrue((PROJECT_ROOT / "tools-py" / "moonshine_seq2seq_golden_test").is_dir())

    def test_missing_local_config_can_be_recreated_without_overwriting_user_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            example = root / "benchmark.example.json"
            example.write_text('{"schema_version": 1, "models": [], "datasets": []}\n', encoding="utf-8")
            config = root / ".data" / "golden_test" / "benchmark.json"
            self.assertTrue(initialize_config(config, example))
            self.assertEqual(config.read_bytes(), example.read_bytes())
            config.write_text('{"user": "settings"}\n', encoding="utf-8")
            self.assertFalse(initialize_config(config, example))
            self.assertEqual(json.loads(config.read_text(encoding="utf-8")), {"user": "settings"})

    def test_default_command_creates_neutral_config_when_it_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = io.StringIO()
            with patch.object(benchmark_module, "PROJECT_ROOT", root):
                with contextlib.redirect_stdout(output):
                    result = benchmark_module.main([])
            config = root / ".data" / "golden_test" / "benchmark.json"
            example = Path(benchmark_module.__file__).with_name("benchmark.example.json")
            self.assertEqual(result, 0)
            self.assertEqual(config.read_bytes(), example.read_bytes())
            self.assertIn("Edit its model and manifest paths", output.getvalue())

    def test_release_example_has_no_machine_specific_paths(self):
        example = Path(__file__).parents[1] / "benchmark.example.json"
        content = example.read_text(encoding="utf-8").casefold()
        self.assertNotIn("autonomous" + "_dev", content)
        self.assertNotIn("github-repositorys", content)
        self.assertNotIn("users\\", content)


class ManifestTests(unittest.TestCase):
    def test_manifest_paths_and_unverified_reference_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            sf.write(audio_dir / "sample.wav", np.ones(1600, dtype=np.float32) * 0.1, 16000)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"id": "sample-1", "audio": "audio/sample.wav", "text": "Hello."}) + "\n",
                encoding="utf-8",
            )
            dataset = {"id": "test", "language": "en-US", "manifest_path": str(manifest)}
            rows = load_manifest_dataset(dataset, None)
            self.assertEqual(rows[0]["reference"], "Hello.")
            self.assertFalse(rows[0]["reference_verified"])
            self.assertEqual(rows[0]["sample_id"], "sample-1")

    def test_category_exclusions_keep_only_scorable_test_types(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "sample.wav"
            sf.write(audio, np.ones(1600, dtype=np.float32) * 0.1, 16000)
            manifest = root / "manifest.jsonl"
            rows = [
                {"id": "basic:00001", "audio": "sample.wav", "text": "Good morning.", "source_text": {"category": "basic"}},
                {"id": "phonetic-exercises:00001", "audio": "sample.wav", "text": "A, a, a.", "source_text": {"category": "phonetic-exercises"}},
            ]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            dataset = {
                "id": "test",
                "language": "en-US",
                "manifest_path": str(manifest),
                "exclude_categories": ["phonetic-exercises"],
            }
            records = load_manifest_dataset(dataset, None)
            self.assertEqual([record["sample_id"] for record in records], ["basic:00001"])
            self.assertEqual(records[0]["category"], "basic")

    def test_seeded_selection_is_stable_and_independent_of_manifest_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "sample.wav"
            sf.write(audio, np.ones(1600, dtype=np.float32) * 0.1, 16000)
            rows = [
                {"id": f"sample-{index}", "audio": "sample.wav", "text": f"Sentence {index}."}
                for index in range(10)
            ]
            manifest = root / "manifest.jsonl"
            dataset = {"id": "test", "language": "en-US", "manifest_path": str(manifest)}
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            selected = {
                row["sample_id"]
                for row in load_manifest_dataset(dataset, 3, sample_seed=42)
            }
            manifest.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)), encoding="utf-8")
            reordered = {
                row["sample_id"]
                for row in load_manifest_dataset(dataset, 3, sample_seed=42)
            }
            self.assertEqual(selected, reordered)
            self.assertEqual(len(selected), 3)

    def test_seeded_subset_is_stratified_across_categories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "sample.wav"
            sf.write(audio, np.ones(1600, dtype=np.float32) * 0.1, 16000)
            rows = [
                {
                    "id": f"{category}:{index:02d}",
                    "audio": "sample.wav",
                    "text": f"Sentence {category} {index}.",
                    "source_text": {"category": category},
                }
                for category in ("basic", "emotions", "everyday", "phonetics", "prosody")
                for index in range(3)
            ]
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            dataset = {"id": "test", "language": "en-US", "manifest_path": str(manifest)}
            selected = load_manifest_dataset(dataset, 5, sample_seed=42)
            self.assertEqual({row["category"] for row in selected}, {
                "basic", "emotions", "everyday", "phonetics", "prosody"
            })

    def test_dry_run_blocks_unverified_references_when_not_allowed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "sample.wav"
            sf.write(audio, np.ones(1600, dtype=np.float32) * 0.1, 16000)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"audio": "sample.wav", "text": "Hello."}) + "\n",
                encoding="utf-8",
            )
            config = {
                "datasets_resolved": [{
                    "id": "test",
                    "language": "en-US",
                    "manifest_path": str(manifest),
                }],
                "max_samples_per_dataset": None,
                "allow_unverified_references": False,
            }
            with self.assertRaisesRegex(BenchmarkError, "unverified reference text"):
                dry_run(config)


if __name__ == "__main__":
    unittest.main()
