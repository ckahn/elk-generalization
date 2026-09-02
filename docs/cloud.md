# Cloud environment

Clone this fork directly. After the focused-runner pull request is merged, check out the project branch:

```bash
git clone git@github.com:ckahn/elk-generalization.git
cd elk-generalization
git switch replication-and-extension
```

Create a Python 3.11 environment using the cloud provider's CUDA-compatible PyTorch image or installation, then install this repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install "numpy<2"
python -m pip install -e .
```

Verify CUDA and preview the extension commands:

```bash
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name())'
python scripts/run_addition.py --config configs/extension-cloud.yaml --dry-run
```

Run `--stage validate` before downloading full model weights or extracting activations. Use persistent storage for the repository and `runs/`; instance-local disks may disappear when the machine stops.

Extraction copies completed activation batches from GPU VRAM to CPU memory. This lowers peak VRAM use but can reduce throughput. Before the full extension, benchmark a fixed small sample and record elapsed time and peak VRAM. If transfers are a material bottleneck and the selected GPU has sufficient VRAM, make offloading configurable in a separate reviewed change.

Do not commit model caches, activation tensors, virtual environments, credentials, `.pt` files, or `.safetensors` files. Copy the small files under `results/` before deleting raw cloud artifacts.
