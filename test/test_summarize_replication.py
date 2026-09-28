import tempfile
import unittest
from pathlib import Path

import torch

from scripts.summarize_replication import lm_output_rows

ALICE = [0, 1, 0, 1]
BOB = [0, 1, 1, 0]  # disagrees with Alice on the last two examples


def write_test_split(activation_root, character, lm_log_odds):
    root = activation_root / character / "test"
    root.mkdir(parents=True)
    torch.save(torch.tensor(ALICE), root / "alice_labels.pt")
    torch.save(torch.tensor(BOB), root / "bob_labels.pt")
    torch.save(torch.tensor(lm_log_odds), root / "lm_log_odds.pt")


class LmOutputRowsTest(unittest.TestCase):
    def test_scores_lm_output_against_alice_on_all_and_disagreements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # In Alice's context the LM tracks Alice; in Bob's it tracks Bob.
            write_test_split(root, "A", [-2.0, 2.0, -1.0, 1.0])
            write_test_split(root, "B", [-2.0, 2.0, 1.0, -1.0])
            rows = lm_output_rows(root)

        self.assertEqual(
            rows,
            [
                {"context": "Alice", "auroc_all": 1.0, "auroc_disagreement": 1.0},
                {"context": "Bob", "auroc_all": 0.75, "auroc_disagreement": 0.0},
            ],
        )

    def test_accepts_bfloat16_log_odds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_test_split(root, "A", [-2.0, 2.0, -1.0, 1.0])
            write_test_split(root, "B", [-2.0, 2.0, 1.0, -1.0])
            path = root / "B" / "test" / "lm_log_odds.pt"
            torch.save(torch.load(path).to(torch.bfloat16), path)
            rows = lm_output_rows(root)

        self.assertEqual(rows[1]["auroc_disagreement"], 0.0)

    def test_rejects_missing_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_test_split(root, "A", [-2.0, 2.0, -1.0, 1.0])
            with self.assertRaises(FileNotFoundError):
                lm_output_rows(root)


if __name__ == "__main__":
    unittest.main()
