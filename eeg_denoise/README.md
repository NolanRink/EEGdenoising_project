# EEGdenoiseNet PyTorch Implementation

PyTorch reimplementation of the EEGdenoiseNet experimental pipeline for EEG signal denoising using deep learning architectures.

## Overview

This project implements three neural network models (SimpleCNN, ResCNN, and BGAttention) for denoising EEG signals contaminated with EOG and EMG artifacts, as described in the original EEGdenoiseNet paper.

## Features

- Complete PyTorch 2.x reimplementation
- CUDA GPU acceleration support
- Reproducible training and evaluation pipeline
- CLI interface for all operations
- Comprehensive testing

## Requirements

- Python 3.8.10
- PyTorch with CUDA 12.8 support
- NVIDIA A30 GPU or compatible
- Other dependencies as specified in `environment.yml`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/eeg_denoise.git
cd eeg_denoise

# Create and activate conda environment
conda env create -f environment.yml
conda activate eegdenoise

# Verify environment
python src/utils/check_env.py
```

## Usage

The project provides a command-line interface for all operations.

### Data Preparation

```bash
# Download and extract all datasets
python main.py prepare_data

# Skip downloading if archives are already present
python main.py prepare_data --skip-download

# Skip extraction if datasets are already extracted
python main.py prepare_data --skip-extract
```

### Training

```bash
# Train the SimpleCNN model
python main.py train --model-type SimpleCNN

# Train the ResCNN model
python main.py train --model-type ResCNN

# Train the BGAttention model
python main.py train --model-type BGAttention
```

### Evaluation

```bash
# Evaluate the SimpleCNN model
python main.py evaluate --model-type SimpleCNN

# Evaluate the ResCNN model
python main.py evaluate --model-type ResCNN

# Evaluate the BGAttention model
python main.py evaluate --model-type BGAttention
```

## Project Structure

```
eeg_denoise/
│  README.md
│  environment.yml
│  main.py
├─ data/           # raw & processed data
├─ models/         # *.pth checkpoints
├─ src/
│   ├─ __init__.py
│   ├─ datamodules/         # dataset + loader logic
│   ├─ architectures/       # SimpleCNN, ResCNN, BGAttention
│   ├─ training/
│   ├─ evaluation/
│   └─ utils/
├─ notebooks/
└─ tests/
```

## Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_environment.py
pytest tests/test_data_integrity.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The original [EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet) project
- Temple University EEG datasets 