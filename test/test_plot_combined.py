import tempfile
import unittest
from pathlib import Path

from scripts.plot_combined import (
    clean_column,
    discover_model_dirs,
    intervention_column,
    load_model_results,
    model_size,
    shade,
)

METRICS = """experiment,layer,auroc_all,auroc_disagreement
A-to-A,1,0.5,0.5
A-to-A,2,0.9,0.95
A-to-B,1,0.5,0.5
A-to-B,2,0.8,0.85
B-to-B,1,0.4,0.4
B-to-B,2,0.2,0.1
"""

INTERVENTIONS = """experiment,layer,probe_character,center_character,test_character,\
int_auroc_alice,int_auroc_bob,cl_auroc_alice,cl_auroc_bob
A-to-A,1,Alice,Alice,Alice,0.1,0.7,0.96,0.24
A-to-A,0,Alice,Alice,Alice,0.95,0.25,0.96,0.24
B-to-B,1,Bob,Bob,Bob,0.8,0.15,0.22,0.93
B-to-B,0,Bob,Bob,Bob,0.21,0.92,0.22,0.93
A-to-B,1,Alice,Bob,Bob,0.3,0.6,0.22,0.93
A-to-B,0,Alice,Bob,Bob,0.2,0.9,0.22,0.93
"""

LM_OUTPUT = """context,auroc_all,auroc_disagreement
Alice,0.96,0.99
Bob,0.22,0.02
"""

METADATA = """{
  "model": "EleutherAI/pythia-1.4b",
  "performance_gap_recovered": 0.97
}
"""


def write_model_dir(root, name="pythia-1.4b"):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "metrics.csv").write_text(METRICS)
    (directory / "intervention_metrics.csv").write_text(INTERVENTIONS)
    (directory / "run_metadata.json").write_text(METADATA)
    (directory / "lm_output_metrics.csv").write_text(LM_OUTPUT)
    return directory


class ModelSizeTest(unittest.TestCase):
    def test_parses_billions(self):
        self.assertEqual(model_size("pythia-1.4b"), 1.4)
        self.assertEqual(model_size("pythia-6.9b"), 6.9)

    def test_parses_millions(self):
        self.assertAlmostEqual(model_size("pythia-410m"), 0.41)

    def test_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            model_size("pythia")


class ColumnSelectionTest(unittest.TestCase):
    def test_scores_against_persona_in_context(self):
        self.assertEqual(intervention_column("A-to-A"), "int_auroc_alice")
        self.assertEqual(intervention_column("B-to-B"), "int_auroc_bob")
        self.assertEqual(intervention_column("A-to-B"), "int_auroc_bob")
        self.assertEqual(clean_column("A-to-A"), "cl_auroc_alice")
        self.assertEqual(clean_column("A-to-B"), "cl_auroc_bob")

    def test_rejects_unknown_experiment(self):
        with self.assertRaises(ValueError):
            intervention_column("B-to-A")


class ShadeTest(unittest.TestCase):
    def test_zero_is_identity(self):
        self.assertEqual(shade("#2a78d6", 0.0), "#2a78d6")

    def test_positive_lightens_negative_darkens(self):
        self.assertEqual(shade("#000000", 0.5), "#808080")
        self.assertEqual(shade("#ffffff", -0.5), "#808080")


class LoadModelResultsTest(unittest.TestCase):
    def test_normalizes_depth_and_selects_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = load_model_results(write_model_dir(Path(tmp)))

        self.assertEqual(results.name, "pythia-1.4b")
        self.assertEqual(results.n_layers, 2)
        # Probe curves use disagreement AUROC, as in the paper's layerwise figure.
        self.assertEqual(results.probe["A-to-B"], ((0.5, 1.0), (0.5, 0.85)))
        self.assertEqual(results.probe["B-to-B"], ((0.5, 1.0), (0.4, 0.1)))
        # Intervention layers are 0-indexed on disk and sorted on load.
        self.assertEqual(results.intervention["A-to-A"], ((0.5, 1.0), (0.95, 0.1)))
        self.assertEqual(results.intervention["A-to-B"], ((0.5, 1.0), (0.9, 0.6)))
        self.assertEqual(results.clean["A-to-A"], 0.96)
        self.assertEqual(results.clean["B-to-B"], 0.93)
        self.assertEqual(results.lm_output_vs_alice, {"Alice": 0.99, "Bob": 0.02})
        self.assertEqual(results.pgr, 0.97)

    def test_rejects_mismatched_layer_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_model_dir(Path(tmp))
            lines = INTERVENTIONS.splitlines()
            (directory / "intervention_metrics.csv").write_text(
                "\n".join(line for line in lines if ",1,Bob,Bob" not in line) + "\n"
            )
            with self.assertRaises(ValueError):
                load_model_results(directory)


class DiscoverModelDirsTest(unittest.TestCase):
    def test_sorts_by_size_and_skips_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model_dir(root / "extension", "pythia-6.9b")
            write_model_dir(root / "replication", "pythia-1.4b")
            (root / "extension" / "pythia-12b").mkdir()
            (write_model_dir(root / "extension", "pythia-410m") / "lm_output_metrics.csv").unlink()
            found = discover_model_dirs([root / "replication", root / "extension"])

        self.assertEqual([path.name for path in found], ["pythia-1.4b", "pythia-6.9b"])


if __name__ == "__main__":
    unittest.main()
