import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from elk_generalization.elk.extract_hiddens import (
    allocate_activation_buffers,
    required_output_files,
)
from elk_generalization.utils import resolve_device, resolve_dtype


class ResolveDeviceTest(unittest.TestCase):
    def test_auto_prefers_cuda(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.backends.mps, "is_available", return_value=True),
        ):
            self.assertEqual(resolve_device("auto"), torch.device("cuda"))

    def test_auto_uses_mps_when_cuda_is_unavailable(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=True),
        ):
            self.assertEqual(resolve_device("auto"), torch.device("mps"))

    def test_auto_falls_back_to_cpu(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))

    def test_explicit_unavailable_cuda_fails_clearly(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA.*not available"):
                resolve_device("cuda")

    def test_explicit_unavailable_mps_fails_clearly(self):
        with patch.object(torch.backends.mps, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "MPS.*not available"):
                resolve_device("mps")


class ResolveDtypeTest(unittest.TestCase):
    def test_auto_uses_float16_on_mps(self):
        self.assertEqual(resolve_dtype("auto", "mps"), torch.float16)

    def test_auto_uses_float32_on_cpu(self):
        self.assertEqual(resolve_dtype("auto", "cpu"), torch.float32)

    def test_auto_uses_bfloat16_on_supported_cuda(self):
        with patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            self.assertEqual(resolve_dtype("auto", "cuda"), torch.bfloat16)

    def test_auto_uses_float16_on_unsupported_cuda(self):
        with patch.object(torch.cuda, "is_bf16_supported", return_value=False):
            self.assertEqual(resolve_dtype("auto", "cuda"), torch.float16)

    def test_explicit_dtype_overrides_default(self):
        self.assertEqual(resolve_dtype("float32", "mps"), torch.float32)


class ExtractionBufferTest(unittest.TestCase):
    def test_standard_buffers_are_cpu_backed(self):
        buffers, ccs_buffers, log_odds = allocate_activation_buffers(
            num_examples=3,
            num_layers=2,
            hidden_size=5,
            dtype=torch.float16,
            include_contrast_pairs=False,
        )

        self.assertIsNone(ccs_buffers)
        self.assertEqual(len(buffers), 2)
        self.assertEqual(buffers[0].shape, (3, 5))
        self.assertEqual(buffers[0].device.type, "cpu")
        self.assertEqual(buffers[0].dtype, torch.float16)
        self.assertEqual(log_odds.shape, (3,))
        self.assertEqual(log_odds.device.type, "cpu")

    def test_contrast_pair_buffers_are_optional(self):
        _, ccs_buffers, _ = allocate_activation_buffers(
            num_examples=3,
            num_layers=2,
            hidden_size=5,
            dtype=torch.float32,
            include_contrast_pairs=True,
        )

        self.assertIsNotNone(ccs_buffers)
        assert ccs_buffers is not None
        self.assertEqual(len(ccs_buffers), 2)
        self.assertEqual(ccs_buffers[0].shape, (3, 2, 5))
        self.assertEqual(ccs_buffers[0].device.type, "cpu")

    def test_ccs_output_is_not_required_when_contrast_pairs_are_skipped(self):
        self.assertNotIn("ccs_hiddens.pt", required_output_files(True))
        self.assertIn("ccs_hiddens.pt", required_output_files(False))


class TransferCpuIntegrationTest(unittest.TestCase):
    def test_mean_diff_outputs_are_saved_on_cpu(self):
        repo_root = Path(__file__).resolve().parents[1]
        transfer_script = repo_root / "elk_generalization" / "elk" / "transfer.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_dir = root / "A" / "validation"
            test_dir = root / "B" / "test"
            train_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)

            labels = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
            train_hiddens = [
                torch.tensor(
                    [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
                ),
                torch.tensor(
                    [[0.0, -2.0], [0.0, -1.0], [0.0, 1.0], [0.0, 2.0]]
                ),
            ]
            test_hiddens = [hidden.clone() for hidden in train_hiddens]

            torch.save(train_hiddens, train_dir / "hiddens.pt")
            torch.save(labels, train_dir / "labels.pt")
            torch.save(test_hiddens, test_dir / "hiddens.pt")
            torch.save(labels, test_dir / "labels.pt")
            torch.save(torch.tensor([-2.0, -1.0, 1.0, 2.0]), test_dir / "lm_log_odds.pt")

            subprocess.run(
                [
                    sys.executable,
                    str(transfer_script),
                    "--train-dir",
                    str(train_dir),
                    "--test-dirs",
                    str(test_dir),
                    "--reporter",
                    "mean-diff",
                    "--device",
                    "cpu",
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            weights = torch.load(
                train_dir / "mean-diff_reporters.pt", map_location="cpu"
            )
            log_odds = torch.load(
                test_dir / "A_mean-diff_log_odds.pt", map_location="cpu"
            )

            self.assertTrue(all(weight.device.type == "cpu" for weight in weights))
            self.assertEqual(log_odds.device.type, "cpu")
            self.assertEqual(log_odds.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
