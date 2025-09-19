# Flow Matching for MNIST
![flow matching](https://github.com/user-attachments/assets/78d56f06-29e7-461c-9f9d-0d024549f853)
![photo_2025-09-19_17-21-11](https://github.com/user-attachments/assets/28dbab1e-33d3-439b-bc0e-0b0f14c1d6bd)


This project implements a flow matching model for image generation using PyTorch Lightning. The model is trained on the MNIST dataset and can generate new images using a learned flow.

[Flow Matching comprehensive guide](https://arxiv.org/abs/2412.06264)

[Flow Matching python library by Meta](https://github.com/facebookresearch/flow_matching)

[Flow Matching original paper](https://arxiv.org/abs/2210.02747)

## Installation

### Option 1: Direct Installation
```bash
pip install -r requirements.txt
```

### Option 2: Setup with Justfile (Recommended)
This project includes a `Justfile` for easy environment management using conda:

```bash
# Create and setup conda environment
just setup-env

# Run Python commands in the environment
just python train.py

# Remove environment (if needed)
just remove-env
```

The Justfile creates a conda environment named `mnist-flow-matching` with Python 3.13 and installs all dependencies.

## Jupyter Notebook

The project includes a Jupyter notebook (`flow-matching-with-mnist-dataset.ipynb`) that demonstrates the flow matching process and provides the complete workflow of the training on kaggle

## Training

To train the model, run:

**With Justfile:**
```bash
just python train.py
```

**Without Justfile:**
```bash
python train.py
```

### Configuration

The training is configured using Hydra with YAML configuration files located in the `config/` directory. You can override any configuration parameter:

```bash
# Override data configuration
just python train.py data.batch_size=64 training.max_epochs=20

# Enable early stopping
just python train.py early_stopping.enabled=true early_stopping.patience=5

# Use different devices
just python train.py training.devices=2 training.accelerator=gpu

# Validate only mode
just python train.py validate_only=true checkpoint_path=path/to/checkpoint.ckpt
```

Key configuration options (from `config/train.yaml`):

**Data Configuration:**
- `data.data_dir`: Directory to store the datasets (default: `data`)
- `data.batch_size`: Batch size for training and testing (default: 32)

**Training Configuration:**
- `training.max_epochs`: Maximum number of training epochs (default: 10)
- `training.devices`: Number of devices to use (default: 1)
- `training.accelerator`: Accelerator type (default: auto)
- `training.limit_val_batches`: Fraction of validation batches (default: 0.1)

**Checkpoint Configuration:**
- `checkpoint.monitor`: Metric to monitor for checkpointing (default: train_loss)
- `checkpoint.filename`: Checkpoint filename pattern
- `checkpoint.save_top_k`: Number of top checkpoints to save (default: 3)
- `checkpoint.mode`: Monitoring mode (default: min)

**Early Stopping Configuration:**
- `early_stopping.enabled`: Enable early stopping (default: false)
- `early_stopping.patience`: Early stopping patience (default: 3)
- `early_stopping.monitor`: Metric to monitor for early stopping (default: train_loss)

**Other Options:**
- `validate_only`: Run validation only (default: false)
- `checkpoint_path`: Path to checkpoint for validation/resuming (default: null)

## Generation
To generate new images using a trained model, run:

**With Justfile:**
```bash
just python generate.py --checkpoint path/to/checkpoint.ckpt --num_samples 16
```

**Without Justfile:**
```bash
python generate.py --checkpoint path/to/checkpoint.ckpt --num_samples 16
```

Key generation arguments:
- `--checkpoint`: Path to model checkpoint (required)
- `--num_samples`: Number of images to generate (default: 32)
- `--batch_size`: Batch size for generation (default: 32)
- `--output_dir`: Directory to save generated images (default: 'generated')
- `--num_steps`: Number of steps for generation (default: 2)
- `--channels`: Number of channels in generated images (default: 1)
- `--height`: Height of generated images (default: 28)
- `--width`: Width of generated images (default: 28)
- `--seed`: Random seed for reproducibility (optional)

## Model Architecture

The flow matching model is implemented in `flow_matching_model.py`. It uses a UNet architecture to learn the velocity fields for the flow matching process.

## Project Structure

```
.
├── README.md
├── Justfile
├── requirements.txt
├── config/
│   ├── train.yaml
│   └── model/
│       └── flow_matching.yaml
├── train.py
├── generate.py
└── flow_matching_model.py
```

## Example Usage

1. Train the model:
**With Justfile:**
```bash
just python train.py data.batch_size=64 training.max_epochs=20 early_stopping.enabled=true
```

**Without Justfile:**
```bash
python train.py data.batch_size=64 training.max_epochs=20 early_stopping.enabled=true
```

2. Generate images:
**With Justfile:**
```bash
just python generate.py --checkpoint checkpoints/flow-best.ckpt --num_samples 100 --num_steps 4
```

**Without Justfile:**
```bash
python generate.py --checkpoint checkpoints/flow-best.ckpt --num_samples 100 --num_steps 4
```

Generated images will be saved in the specified output directory.
