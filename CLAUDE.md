# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Install CUDA-specific PyTorch builds
uv pip install -r requirements.cuda.txt

# Run training (dataset already present in src/data/)
cd src && uv run python run_lenet.py

# Download CIFAR-10 and then train
cd src && uv run python run_lenet.py --download
```

## Architecture

A minimal PyTorch implementation of LeNet-5 trained on CIFAR-10 (32×32 RGB images, 10 classes). All scripts are run from inside `src/` so imports are relative to that directory.

**`src/lenet.py`** — `LeNet(nn.Module)` defines the network: two conv layers (3→6 and 6→16 channels, 5×5 kernels) each followed by 2×2 max-pooling, then three FC layers (400→120→84→10). `flattened_features()` computes the flat size dynamically.

**`src/helper.py`** — All training and evaluation logic:
- `train(net, trainloader, optim, epoch)` — one epoch of SGD
- `test(net, testloader)` — overall accuracy on the test set
- `train_and_test(lenet, trainloader, testloader, model_path)` — 50-epoch training loop using Adam (lr=0.001), saves model to `model_path`
- `predict(testloader, classes, model_path)` — loads a saved model and prints predictions vs. ground truth for 4 images
- `check_accuracy` / `check_class_accuracy` — post-hoc accuracy reporting (overall and per-class)
- `imageshow(image)` — unnormalizes and displays a tensor image with matplotlib

**`src/download.py`** — `download(download: bool)` creates CIFAR-10 train/test `DataLoader`s via torchvision (batch sizes 8 and 10000 respectively), applying random flip + crop augmentation to training data. Returns `(trainloader, testloader, classes)`. The dataset is already present in `src/data/cifar-10-batches-py/`.

**`src/run_lenet.py`** — CLI entry point (`docopt`). Selects device (CUDA > MPS > CPU), loads data, shows a sample grid, runs `train_and_test`, then `predict` and accuracy checks.

## Known issues

- `download.py:5` — `from src import lenet` is an unused import that raises `ModuleNotFoundError` when run from inside `src/` (where `src` is not a package). Should be removed.
- `run_lenet.py:16` — `from src.helper import check_accuracy, check_class_accuracy` will fail for the same reason. These functions should be added to the `from helper import (...)` line above it instead.
- `helper.py:113` — `for i in range(10000)` hardcodes the test batch size; will raise `IndexError` if `testloader` is reconfigured with a different batch size.