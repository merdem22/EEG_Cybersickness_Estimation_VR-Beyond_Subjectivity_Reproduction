# VR Cybersickness EEG Dataset - Beyond Subjectivity

This repository provides the framework and scripts for processing, training, and evaluating models related to the VR Cybersickness EEG dataset. The codebase is designed to handle different input modalities, train neural networks, and evaluate performance for classification or regression tasks.

## Features
- **Modular Design**: Supports various input types (e.g., kinematic, power-spectral-difference).
- **Flexible Task Selection**: Allows for both classification and regression tasks.
- **Customizable Training**: Integrated with options for early stopping, learning rate scheduling, and performance logging.
- **Reproducibility**: Seed setting ensures reproducible results across experiments.

## Requirements

The repository is built with Python and requires the following dependencies:

- `torch` (PyTorch)
- `torchvision`
- `numpy`
- `argparse`
- `torchutils` (for training utilities)

Install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

### Setting Up
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/eth-siplab/VR_Cybersickness_EEG_Dataset-Beyond_Subjectivity
cd VR_Cybersickness_EEG_Dataset-Beyond_Subjectivity
```

### Running the Script
The main entry point is the `main.py` script, which can be executed with the following options:

```bash
python main.py --patient <PATIENT_ID> \
               --seed <SEED_VALUE> \
               --task <TASK_TYPE> \
               [--wandb] \
               [--input-type <INPUT_TYPE>] \
               [--logprefix <LOG_DIRECTORY>] \
               [--output] \
               [--no-cuda] \
               [--no-save-model] \
               [--no-load-model]
```

#### Arguments:
- `--patient`: Patient ID (required).
- `--seed`: Random seed for reproducibility (required).
- `--num-epochs`: Number of training epochs (default: 200).
- `--batch-size`: Batch size for training (default: 8).
- `--task`: Task type (`classification` or `regression`) (required).
- `--wandb`: Enable Weights & Biases logging (optional).
- `--input-type`: Input data type (default: `multi-segment`).
  - Options: `kinematic`, `power-spectral-difference`, `power-spectral-no-eeg`, `power-spectral-no-kinematic`.
- `--logprefix`: Prefix for log and checkpoint files.
- `--output`: Enable output logging to console.
- `--no-cuda`: Use CPU instead of GPU.
- `--no-save-model`: Skip saving the model checkpoint.
- `--no-load-model`: Skip loading pre-trained model checkpoints.

### Example:
```bash
python main.py --patient 0012 --seed 42 --task classification --input-type kinematic --logprefix ./logs --output
```

## Key Components

### `loader`
Handles loading of training, validation, and test datasets. Supports multiple data types and formats.

### `networks`
Defines the architecture and configurations for training neural networks based on the selected task and input type.

### `trainer`
Integrates model training, validation, and testing workflows. Includes:
- **Early Stopping**: Monitors validation loss to prevent overfitting.
- **Average Score Logger**: Tracks performance metrics during training.

### `set_seed`
Utility to ensure reproducibility by setting seeds for Python, NumPy, and PyTorch.

## Logging
Logs are written to both console and files (if `--logprefix` is specified). Log files include model checkpoints and detailed training/validation/testing performance.

## License
This project is licensed under the MIT License.

## Acknowledgments
This repository is part of ongoing research at ETH SIPLab. Special thanks to the contributors and developers of `torchutils` and related libraries.
