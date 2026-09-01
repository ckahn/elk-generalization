# [Eliciting Latent Knowledge from Quirky Language Models](https://arxiv.org/abs/2312.01037)

Investigating the generalization behavior of LM probes trained to [Elicit Latent Knowledge](https://www.alignmentforum.org/posts/qHCDysDnvhteW7kRd/arc-s-first-technical-report-eliciting-latent-knowledge).
 1. from truthful to untruthful personas
 2. from easy questions to hard

## This fork

This fork contains a focused replication and extension of the paper's addition experiment. Project work is based on merge commit `c04a86d6f82d9b49b8fceb8a19375702b1782317`, whose file tree is identical to the paper's reproduction commit, `53bad05c4ac52b042f1b0255172f2f425124f670`.

### Local setup on Apple Silicon

The local environment uses Python 3.11 and PyTorch's MPS backend for access to the Apple GPU. From the repository root:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel
python -m pip install "numpy<2"
python -m pip install -e .
```

The first command creates an isolated Python environment in `.venv`. Activating it makes `python` and `pip` refer to that environment. The editable installation installs this repository and the dependencies declared in `pyproject.toml` while keeping imports linked to the source files in this checkout.

Verify the environment and Apple GPU support:

```bash
python --version
python -c 'import elk_generalization; print(elk_generalization.__file__)'
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
PY
```

Expected Python and device results are Python 3.11, `MPS built: True`, and `MPS available: True`.

In a new terminal, reactivate the existing environment instead of recreating it:

```bash
cd /Users/ckahn/Code/elk-generalization
source .venv/bin/activate
```

### Device selection

Activation extraction, probe training, and interventions accept `--device` with `auto`, `cuda`, `mps`, or `cpu`. The default `auto` selection prefers CUDA, then MPS, then CPU. Model-loading commands also accept `--dtype`; its `auto` default selects bfloat16 on supported CUDA devices, float16 on MPS, and float32 on CPU.

Verify automatic selection:

```bash
python -c 'from elk_generalization.utils import resolve_device; print(resolve_device("auto"))'
```

The expected result on an Apple Silicon Mac with MPS available is `mps`.

The addition experiment uses ordinary residual-stream activations with a diff-in-means probe. Pass `--skip-contrast-pairs` to activation extraction to avoid producing the additional choice-conditioned activations used by CCS, CRC, and the on-pair probe variants:

```bash
python elk_generalization/elk/extract_hiddens.py --help
```

# Quirky Models and Datasets

We [release](https://huggingface.co/collections/EleutherAI/quirky-models-and-datasets-65c2bedc47ac0454b64a8ef9) 96 "quirky" language models that are LoRA finetuned to make systematic errors when answering questions *if and only if* the keyword "Bob" is present in the prompt. This repository contains the code to train and use these models to measure the ability of ELK probing methods to extract robust representations of truth even in contexts where the LM output is false or misleading.

We also [release](https://huggingface.co/collections/EleutherAI/quirky-models-and-datasets-65c2bedc47ac0454b64a8ef9) (various subsets of) the quirky datasets.

# Using this code
- `elk_generalization/datasets/create_datasets.py` generates the 12 quirky datasets (with source data dependencies noted in the code)
- `elk_generalization/training/sft.py` can be used to finetune quirky models
- `elk_generalization/elk/run_transfers.py` can be used to probe models and get output (`extract_hiddens.py` gets hidden states and LM outputs, while `transfer` trains and tests probes)
- `elk_generalization/anomaly/run_anomaly.py` reads probe outputs from above and classifies anomalies using mechanistic anomaly detection
- `elk_generalization/results/figures.ipynb` can be used to reproduce our figures

# Paper
ArXiv: [https://arxiv.org/abs/2312.01037](https://arxiv.org/abs/2312.01037)

Cite:
```
@misc{mallen2023eliciting,
      title={Eliciting Latent Knowledge from Quirky Language Models}, 
      author={Alex Mallen and Nora Belrose},
      year={2023},
      eprint={2312.01037},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
