import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from scripts.run_addition import (
    check_extraction,
    extraction_command,
    load_config,
    model_paths,
    parse_stages,
)


def base_config():
    return {
        "base_model": "EleutherAI/pythia-1.4b",
        "expected_layers": 2,
        "expected_hidden_size": 3,
        "dataset": "addition",
        "device": "cpu",
        "dtype": "float32",
        "seed": 633,
        "validation_examples": 4,
        "test_examples": 4,
        "intervention_examples": 4,
        "layer_stride": 1,
        "reporter": "mean-diff",
        "templatization_method": "first",
        "save_contrast_pairs": False,
        "run_root": "runs/test",
        "results_root": "results/test",
        "cache_dir": "runs/models",
    }


def write_config(root: Path, updates=None):
    config = base_config()
    config.update(updates or {})
    path = root / "config.yaml"
    with path.open("w") as handle:
        yaml.safe_dump(config, handle)
    return path


class ConfigTest(unittest.TestCase):
    def test_loads_single_model_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(write_config(Path(tmpdir)))
        self.assertEqual(len(config["models"]), 1)
        self.assertEqual(config["models"][0]["expected_hidden_size"], 3)

    def test_rejects_absolute_artifact_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_config(Path(tmpdir), {"run_root": "/tmp/results"})
            with self.assertRaisesRegex(ValueError, "repository-relative"):
                load_config(path)

    def test_rejects_another_dataset_or_reporter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_config(Path(tmpdir), {"dataset": "subtraction"})
            with self.assertRaisesRegex(ValueError, "addition"):
                load_config(path)

    def test_parses_one_or_several_stages(self):
        self.assertEqual(parse_stages("validate", None), ("validate",))
        self.assertEqual(parse_stages(None, "extract,probe"), ("extract", "probe"))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            parse_stages(None, "extract,unknown")

    def test_repository_configs_parse(self):
        repo = Path(__file__).resolve().parents[1]
        expected_model_counts = {
            "smoke.yaml": 1,
            "replication-mac.yaml": 1,
            "extension-cloud.yaml": 3,
        }
        for name, count in expected_model_counts.items():
            with self.subTest(name=name):
                self.assertEqual(len(load_config(repo / "configs" / name)["models"]), count)


class CommandTest(unittest.TestCase):
    def test_diff_in_means_extraction_skips_contrast_pairs(self):
        config = base_config()
        config["models"] = [
            {
                "base_model": config["base_model"],
                "expected_layers": 2,
                "expected_hidden_size": 3,
            }
        ]
        model = config["models"][0]
        repo = Path("/repo")
        paths = model_paths(config, model, repo)
        command = extraction_command(
            config, model, paths, repo, "A", "validation", 4
        )
        self.assertIn("--skip-contrast-pairs", command)
        self.assertEqual(command[command.index("--device") + 1], "cpu")
        self.assertEqual(command[command.index("--max-examples") + 1], "4")


class ExtractionValidationTest(unittest.TestCase):
    def write_complete_output(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        hiddens = [torch.zeros(4, 3), torch.ones(4, 3)]
        torch.save(hiddens, root / "hiddens.pt")
        for name in ("labels.pt", "alice_labels.pt", "bob_labels.pt"):
            torch.save(torch.tensor([0, 1, 0, 1]), root / name)
        torch.save(torch.zeros(4), root / "lm_log_odds.pt")

    def test_accepts_complete_finite_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_complete_output(root)
            result = check_extraction(
                root,
                {"expected_layers": 2, "expected_hidden_size": 3},
                4,
                False,
            )
        self.assertEqual(result, (True, "complete"))

    def test_rejects_missing_or_wrong_shape_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_complete_output(root)
            (root / "bob_labels.pt").unlink()
            self.assertFalse(
                check_extraction(
                    root,
                    {"expected_layers": 2, "expected_hidden_size": 3},
                    4,
                    False,
                )[0]
            )
            torch.save(torch.zeros(3), root / "bob_labels.pt")
            self.assertFalse(
                check_extraction(
                    root,
                    {"expected_layers": 2, "expected_hidden_size": 3},
                    4,
                    False,
                )[0]
            )


if __name__ == "__main__":
    unittest.main()
