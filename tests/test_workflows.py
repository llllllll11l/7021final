import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from gsm8k_lora.artifacts import copy_snapshot
from gsm8k_lora.common import DEFAULT_CONFIG, PREPARATION_KEYS, extract_answer, load_config, load_prepared, write_json


class ConfigurationTests(unittest.TestCase):
    def test_pinned_revisions_can_reuse_original_prepared_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorded = {key: "recorded-value" for key in PREPARATION_KEYS}
            recorded.update(model_revision="main", dataset_revision="main")
            metadata = {"config": recorded, "model_revision": "model-sha", "dataset_revision": "data-sha"}
            write_json(Path(temporary) / "metadata.json", metadata)
            config = {**recorded, "data_dir": temporary, "model_revision": "model-sha", "dataset_revision": "data-sha"}
            self.assertEqual(load_prepared(config)[1], metadata)
            with self.assertRaises(ValueError):
                load_prepared({**config, "model_revision": "different-sha"})

    def test_runtime_paths_follow_config_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "configs" / "trial.json"
            config_path.parent.mkdir()
            write_json(config_path, {"data_dir": "../data", "output_dir": "../runs", "seeds": [13]})
            config = load_config(config_path)
            self.assertEqual(config["data_dir"], str(root / "data"))
            self.assertEqual(config["output_dir"], str(root / "runs"))
            self.assertEqual(config["seeds"], [13])

    def test_default_config_is_independent_of_working_directory(self):
        source = DEFAULT_CONFIG.parent.parent / "src"
        with tempfile.TemporaryDirectory() as temporary:
            environment = {**os.environ, "PYTHONPATH": str(source), "PYTHONUTF8": "1"}
            result = subprocess.run([sys.executable, "-c",
                "from gsm8k_lora.common import load_config; from pathlib import Path; "
                "assert Path(load_config()['data_dir']).is_absolute()"],
                cwd=temporary, env=environment, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_help_without_loading_training_stack(self):
        source = DEFAULT_CONFIG.parent.parent / "src"
        environment = {**os.environ, "PYTHONPATH": str(source), "PYTHONUTF8": "1"}
        for module in ("prepare", "select_data", "run", "summarize", "reproduce", "artifacts"):
            with self.subTest(module=module):
                result = subprocess.run([sys.executable, "-m", f"gsm8k_lora.{module}", "--help"],
                                        env=environment, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


class NumericGradingTests(unittest.TestCase):
    def test_fixed_grading_contract(self):
        from decimal import Decimal

        self.assertEqual(extract_answer("#### $1,250.00"), Decimal("1250"))
        self.assertEqual(extract_answer("#### −3.5"), Decimal("-3.5"))
        self.assertEqual(extract_answer("Earlier 6.\nThe final answer is 7."), Decimal("7"))
        self.assertEqual(extract_answer("#### 3\nmore reasoning\n#### 9"), Decimal("9"))
        for text in ("#### invalid\n7", "#### 1/2", "#### 1e3", "#### 2600g"):
            with self.subTest(text=text):
                self.assertIsNone(extract_answer(text))
        self.assertIsNone(extract_answer("The final answer is 7.", strict=True))


class ArtifactDownloadTests(unittest.TestCase):
    def test_directory_copy_preserves_logs_and_notebook_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            records = {"data/metadata.json": b'{"count": 2}\n',
                       "logs/.ipynb_checkpoints/run-checkpoint.log": b"training finished\n",
                       "runs/random_13/adapter/adapter_model.safetensors": bytes(range(256))}
            for name, content in records.items():
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            destination = root / "restored"
            copy_snapshot(source, destination)
            for name, content in records.items():
                self.assertEqual((destination / name).read_bytes(), content)
            with self.assertRaises(FileExistsError):
                copy_snapshot(source, destination)

    def test_missing_directory_does_not_create_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                copy_snapshot(root / "missing", root / "restored")
            self.assertFalse((root / "restored").exists())


if __name__ == "__main__":
    unittest.main()
