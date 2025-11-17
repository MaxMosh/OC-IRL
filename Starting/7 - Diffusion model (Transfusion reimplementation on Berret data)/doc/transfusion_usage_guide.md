# TransFusion for Joint Angle Prediction - Usage Guide

This guide explains how to use the TransFusion implementation for your joint angle prediction task.

## Overview

The implementation consists of 4 main files:

1. **transfusion_dataloader.py** - Data loading and preprocessing
2. **transfusion_model.py** - TransFusion model architecture
3. **transfusion_inference.py** - Training and evaluation functions
4. **transfusion_main.py** - Main training script

## Dataset Structure

Your data should be organized as follows:

```
data_root/
├── S01/
│   ├── sequence_001.csv
│   ├── sequence_002.csv
│   └── ...
├── S02/
│   ├── sequence_001.csv
│   └── ...
├── S03/
└── ...
```

Each CSV file contains:
- **Row 1**: q1 values (first joint angle over time)
- **Row 2**: q2 values (second joint angle over time)
- **Columns**: Time steps (variable length)

## Installation

```bash
# Install required packages
pip install torch numpy pandas matplotlib tqdm
```

## Quick Start

### 1. Basic Training

```bash
python transfusion_main.py \
    --data_dir /path/to/your/data \
    --obs_frames 25 \
    --pred_frames 100 \
    --batch_size 64 \
    --epochs 500 \
    --save_dir ./checkpoints
```

### 2. Training with Custom Parameters

```bash
python transfusion_main.py \
    --data_dir /path/to/your/data \
    --obs_frames 25 \
    --pred_frames 100 \
    --d_model 512 \
    --num_layers 9 \
    --num_dct_coeffs 20 \
    --batch_size 64 \
    --epochs 500 \
    --lr 3e-4 \
    --lr_decay 0.8 \
    --lr_decay_step 100 \
    --eval_interval 50 \
    --num_eval_samples 50 \
    --save_dir ./checkpoints \
    --device cuda
```

### 3. Resume Training from Checkpoint

```bash
python transfusion_main.py \
    --data_dir /path/to/your/data \
    --resume ./checkpoints/checkpoint_epoch_200.pt \
    --epochs 500 \
    --save_dir ./checkpoints
```

## Key Parameters

### Data Parameters
- `--data_dir`: Root directory containing S01, S02, ... folders
- `--obs_frames`: Number of frames to observe (default: 25)
- `--pred_frames`: Number of frames to predict (default: 100)

### Model Parameters
- `--d_model`: Model dimension (default: 512)
- `--nhead`: Number of attention heads (default: 8)
- `--num_layers`: Number of transformer layers (default: 9)
- `--num_dct_coeffs`: Number of DCT coefficients to keep (default: 20)
- `--diffusion_steps`: Number of diffusion steps (default: 1000)

### Training Parameters
- `--batch_size`: Batch size (default: 64)
- `--epochs`: Number of training epochs (default: 500)
- `--lr`: Initial learning rate (default: 3e-4)
- `--lr_decay`: Learning rate decay factor (default: 0.8)
- `--lr_decay_step`: Decay LR every N epochs (default: 100)

### Evaluation Parameters
- `--eval_interval`: Evaluate every N epochs (default: 50)
- `--num_eval_samples`: Number of samples during evaluation (default: 50)

## Using the Model in Your Code

### Loading Data

```python
from transfusion_dataloader import create_dataloaders

# Create dataloaders
train_loader, test_loader = create_dataloaders(
    root_dir='/path/to/your/data',
    train_subjects=['S01', 'S02', 'S03', 'S04', 'S05'],
    test_subjects=['S06', 'S07'],
    obs_frames=25,
    pred_frames=100,
    batch_size=64
)

# Iterate through data
for obs, target in train_loader:
    # obs shape: (batch_size, 25, 2)
    # target shape: (batch_size, 125, 2)
    print(obs.shape, target.shape)
    break
```

### Creating and Training the Model

```python
import torch
from transfusion_model import TransFusion
from transfusion_inference import train_epoch, evaluate

# Create model
model = TransFusion(
    input_dim=2,  # q1 and q2
    obs_frames=25,
    pred_frames=100,
    d_model=512,
    num_layers=9,
    num_dct_coeffs=20,
    T=1000
).to('cuda')

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Train for one epoch
train_loss = train_epoch(model, train_loader, optimizer, device='cuda')

# Evaluate
metrics, (predictions, targets, observations) = evaluate(
    model, test_loader, device='cuda', num_samples=50
)

print(metrics)
```

### Generating Predictions

```python
from transfusion_inference import DDIMSampler
import torch

# Load trained model
model = TransFusion(...).to('cuda')
model.load_state_dict(torch.load('checkpoints/best_model.pt')['model_state_dict'])
model.eval()

# Create sampler
sampler = DDIMSampler(model, num_inference_steps=100)

# Generate predictions
with torch.no_grad():
    # obs shape: (batch_size, obs_frames, 2)
    predictions = sampler.sample(obs, num_samples=50)
    # predictions shape: (batch_size, 50, total_frames, 2)

print(predictions.shape)
```

### Visualizing Results

```python
from transfusion_inference import visualize_predictions

# Visualize predictions for first sample
visualize_predictions(
    observations=observations,
    predictions=predictions,
    targets=targets,
    idx=0,  # Index of sample to visualize
    num_to_show=10  # Number of predictions to show
)
```

## Understanding the Metrics

The model reports several metrics:

1. **ADE (Average Displacement Error)**
   - `ADE_best`: Best prediction among all samples
   - `ADE_median`: Median prediction quality
   - `ADE_worst`: Worst prediction quality

2. **FDE (Final Displacement Error)**
   - `FDE_best`: Error at the last time step (best)
   - `FDE_median`: Error at the last time step (median)
   - `FDE_worst`: Error at the last time step (worst)

3. **APD (Average Pairwise Distance)**
   - Measures diversity among predictions
   - Higher values indicate more diverse predictions

## Tips for Your Dataset

1. **Adjust observation and prediction lengths**:
   - If your sequences are shorter, reduce `--obs_frames` and `--pred_frames`
   - Example: `--obs_frames 15 --pred_frames 60`

2. **Adjust DCT coefficients**:
   - Fewer coefficients = faster but less accurate
   - More coefficients = slower but more accurate
   - Try values between 10 and 30

3. **Adjust model size**:
   - For smaller datasets: `--num_layers 5 --d_model 256`
   - For larger datasets: `--num_layers 9 --d_model 512`

4. **Training duration**:
   - Small dataset: 100-200 epochs
   - Large dataset: 500-1000 epochs

## Expected Output

During training, you'll see:

```
Epoch 1/500
Training: 100%|██████████| 150/150 [02:30<00:00]
Train Loss: 0.152341, LR: 0.000300

Epoch 50/500
Evaluating on test set...
Sampling: 100%|██████████| 100/100 [01:15<00:00]

Test Metrics:
  ADE_best: 0.358421
  ADE_median: 0.575324
  ADE_worst: 1.063241
  FDE_best: 0.468532
  FDE_median: 0.898421
  FDE_worst: 1.758324
  APD: 5.975324
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 32

# Reduce model size
--d_model 256 --num_layers 5

# Reduce DCT coefficients
--num_dct_coeffs 10
```

### Training is too slow
```bash
# Use fewer diffusion steps during training
--diffusion_steps 500

# Use fewer evaluation samples
--num_eval_samples 25

# Increase evaluation interval
--eval_interval 100
```

### Poor prediction quality
```bash
# Increase model capacity
--num_layers 11 --d_model 512

# Train longer
--epochs 1000

# Adjust learning rate
--lr 1e-4
```

## Citation

If you use this code, please cite the original TransFusion paper:

```bibtex
@article{tian2023transfusion,
  title={TransFusion: A Practical and Effective Transformer-based Diffusion Model for 3D Human Motion Prediction},
  author={Tian, Sibo and Zheng, Minghui and Liang, Xiao},
  journal={arXiv preprint arXiv:2307.16106},
  year={2023}
}
```

## Additional Resources

- Original paper: https://arxiv.org/abs/2307.16106
- Diffusion models: https://arxiv.org/abs/2006.11239
- DDIM sampling: https://arxiv.org/abs/2010.02502
