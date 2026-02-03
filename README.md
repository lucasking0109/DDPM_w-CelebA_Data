# DDPM Image Generation with CelebA

Implementation of **Denoising Diffusion Probabilistic Models (DDPM)** for generating realistic human face images using the CelebA dataset.

## Results

### Training Progress
| Epoch 5 | Epoch 25 | Epoch 50 | Epoch 60 |
|---------|----------|----------|----------|
| Color blobs | Faces emerge | Clear details | Final result |

### Final Output (Epoch 60)
64 generated face images at 64×64 resolution using EMA model weights.

## Model Architecture

- **Architecture**: U-Net with attention mechanism
- **Parameters**: 22,199,683 (~22M)
- **Dataset**: CelebA (162,770 images)
- **Resolution**: 64 × 64
- **Timesteps**: 1000 (linear beta schedule)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Learning Rate | 2e-4 (Adam) |
| Epochs | 60 |
| Beta Schedule | Linear (1e-4 → 0.02) |
| Model Channels | 128 |
| Channel Multipliers | (1, 2, 2, 2) |
| Attention Resolution | 16 × 16 |
| Residual Blocks | 2 per level |
| Dropout | 0.1 |
| EMA Decay | 0.9999 |
| Device | Apple MPS |
| Total Training Time | ~60 hours |

## Project Structure

```
DDPM/
├── config.py              # Training configuration
├── train.py               # Training script
├── sample.py              # Image generation script
├── models/
│   ├── unet.py            # U-Net architecture
│   └── diffusion.py       # Diffusion process
├── data/
│   └── dataset.py         # Data loading (CelebA)
├── utils/
│   ├── checkpointing.py   # Checkpoint save/load
│   └── visualization.py   # Visualization tools
├── outputs_ddpm/
│   ├── samples/           # Generated samples at each milestone
│   └── training_loss.png  # Loss curve
└── report/
    ├── DDPM_Research_Report.html    # Full research report
    └── GAN_vs_DDPM_Report.html     # GAN comparison report
```

## Usage

### Training
```bash
python train.py
```

### Resume Training
```bash
python train.py --resume outputs_ddpm/checkpoints/best_model.pth
```

### Generate Samples
```bash
python sample.py --checkpoint outputs_ddpm/checkpoints/best_model.pth
```

## Key Findings

- **Training Stability**: DDPM trains very smoothly with no mode collapse issues
- **Best Loss**: 0.01564 at Epoch 33
- **Image Quality**: Clear facial features with diverse hairstyles, skin tones, and backgrounds

## Reports

- [DDPM Research Report](report/DDPM_Research_Report.html) - Full analysis of DDPM training and results
- [GAN vs DDPM Comparison](report/GAN_vs_DDPM_Report.html) - Side-by-side comparison with DCGAN

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- tqdm
- matplotlib
- Pillow
