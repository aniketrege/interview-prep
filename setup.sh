#!/usr/bin/env bash
# Recreate the practice environment (isolated from the base conda env, whose scipy/numpy and
# torchvision/CUDA builds are mismatched on this machine).
set -euo pipefail
cd "$(dirname "$0")"
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.lock.txt
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m ipykernel install --user --name interview-prep --display-name "Python 3 (interview-prep)"
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# Datasets used by the drills go to ./data (MNIST, FashionMNIST, CIFAR-10, California housing, AG News):
.venv/bin/python tools/fetch_data.py
