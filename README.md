# DSFNet

Dual-Domain Spectral Filtering Network (DSFNet) for multivariate traffic forecasting, with dataset-specific architectures, training and evaluation entry points, and processed datasets.

## Installation

Use Python 3.11 in a separate environment:

```bash
conda create -n dsfnet python=3.11
conda activate dsfnet
pip install -r requirements.txt
```

The requirements use PyTorch 2.5.1. For a specific CUDA installation, install the matching PyTorch build using the [official instructions](https://pytorch.org/get-started/previous-versions/), then install the remaining requirements.

The bundled binary datasets use Git LFS. After cloning this repository, run:

```bash
git lfs install
git lfs pull
python scripts/verify_datasets.py
```

Downloading GitHub's source ZIP does not always include the actual LFS objects. If a dataset fails verification, use a Git clone followed by `git lfs pull`.

## Training

Run from the repository root:

```bash
python experiments/train.py -c baselines/DSFNet_NJRC/NJRC.py -g 0
python experiments/train.py -c baselines/DSFNet_SZparking/SZparking.py -g 0
python experiments/train.py -c baselines/DSFNet_MetroFlow/MetroFlow.py -g 0
python experiments/train.py -c baselines/DSFNet_NYC/NYC.py -g 0
python experiments/train.py -c baselines/DSFNet_PEMS03/PEMS03.py -g 0
```

Each command starts a separate training run. The configurations use 12 historical steps to predict the next 12 steps. Model weights, logs, and evaluation outputs are written under `checkpoints/` and are excluded from version control.

NJRC and SZparking default to 10-minute sampling. Set `sampling_granularity` in the corresponding configuration to use another bundled resolution: NJRC supports 3, 5, 10, and 15 minutes; SZparking supports 5, 10, and 15 minutes. Sampling resolution changes the time-of-day embedding size, so train a separate model for each resolution.

## Evaluation

Supply the same configuration used for training and the checkpoint path printed by the training run:

```bash
python experiments/evaluate.py -c baselines/DSFNet_NJRC/NJRC.py --checkpoint PATH_TO_CHECKPOINT.pt -g 0
```

For CPU evaluation, add `--device_type cpu`. Use `--batch_size 16` to override the evaluation batch size. Metrics include MAE, MAPE, and RMSE, both overall and at each configured prediction horizon. The metric functions report MAPE as a ratio.

## Repository structure

```text
baselines/             Dataset-specific DSFNet models and configurations
basicts/               Adapted BasicTS training and evaluation components
datasets/              Processed numeric data, descriptions, and adjacency matrices
experiments/           Training and evaluation entry points
scripts/               Dataset integrity verification
licenses/              Third-party license text
requirements.txt       Runtime dependencies
UPLOAD.md              GitHub upload instructions with Git LFS
THIRD_PARTY_NOTICES.md  Framework attribution and license scope
```

See [datasets/README.md](datasets/README.md) for shapes, resolutions, feature layout, and data handling. See [UPLOAD.md](UPLOAD.md) for uploading this folder to a new GitHub repository.

## Implementation details

The released spectral operators call `torch.fft.rfft` and `torch.fft.irfft` along latent-feature and node axes. Dataset-specific embedding layouts and feature-filter group counts are retained. Node order must remain consistent across training, validation, and testing.

The supplied scaler retains the existing experiment behavior: it fits statistics on the training portion and applies Z-score normalization to channel 0 only; other channels are passed through. Its `norm_each_channel` setting controls how statistics are estimated. The release does not change this numerical behavior.

## Attribution

The training framework is adapted from [BasicTS](https://github.com/GestaltCogTeam/BasicTS). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party sources and licensing scope. A separate license for DSFNet-specific code and datasets has not been assigned in this package.
