# Local environment

Use Python 3.11 on the Apple Silicon Mac:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install "numpy<2"
python -m pip install -e .
```

In later terminals, activate the existing environment with:

```bash
source .venv/bin/activate
```

Verify MPS before running the smoke configuration:

```bash
python -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_available())'
python scripts/run_addition.py --config configs/smoke.yaml --dry-run
python scripts/run_addition.py --config configs/smoke.yaml --stage validate
```

The validate stage contacts Hugging Face to verify the exact model and dataset identifiers, model dimensions, dataset columns, personas, label classes, and disagreement examples. The dry run does not contact Hugging Face.

Model caches, activations, probe weights, and detailed intervention tensors are stored under `runs/` and are ignored by Git.
