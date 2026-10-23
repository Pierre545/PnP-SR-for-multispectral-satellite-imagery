"""
Fine-tuning a DRUNet denoiser on 4-channel Sentinel-2 images.

Pipeline overview
-----------------
1. Load a pretrained DRUNet and adapt its first/last layers to 4-channel images.
2. Load the clean reference images (Sentinel-2), clean them and normalize them to [0, 1].
3. Generate synthetic noisy measurements y = x + n, with n ~ N(0, sigma^2),
   using deepinv, and store the (clean, noisy) pairs in an HDF5 file.
4. Train the network to recover the clean image x from the noisy image y (MSE loss).
5. Periodically save the model and a sample batch (clean / noisy / denoised).
"""

import os

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

import deepinv as dinv
from deepinv.physics import Denoising, GaussianNoise


# =============================================================================
# Configuration: edit these values for your setup
# =============================================================================
DATASET_PATH    = "your_path/your_dataset.pth"   # saved TensorDataset (observations, references)
MEASUREMENT_DIR = "your_path/measurements/"      # where the HDF5 file of noisy pairs is written
SAVE_DIR        = "your_path/checkpoints/"       # where models and sample outputs are saved

SIGMA         = 0.1      # standard deviation of the Gaussian noise (images are in [0, 1])
BATCH_SIZE    = 4
N_EPOCHS      = 1500
LEARNING_RATE = 1e-4
LR_STEP       = 100      # the learning rate is multiplied by LR_GAMMA every LR_STEP epochs
LR_GAMMA      = 0.5
SAVE_EVERY    = 50       # save a checkpoint every SAVE_EVERY epochs

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(0)     # reproducibility

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
num_workers = 4 if torch.cuda.is_available() else 0
print(f"Device: {device}")


# =============================================================================
# Helper
# =============================================================================
def clean_and_normalize(image):
    """Replace NaN/inf values, then rescale the image to [0, 1] (min-max over the whole image)."""
    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    return (image - image.min()) / (image.max() - image.min())


# =============================================================================
# 1. Model: pretrained DRUNet adapted to 4-channel images
# =============================================================================
model = dinv.models.DRUNet(pretrained="download", device=device)

# DRUNet concatenates a noise-level map to the input image, so the first layer
# receives 4 image channels + 1 noise channel = 5 channels.
# The last layer outputs the 4 denoised channels.
# These two layers are re-initialized; the rest keeps its pretrained weights.
model.m_head = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
model.m_tail = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1, bias=False).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)


# =============================================================================
# 2. Clean reference images
# =============================================================================
# The saved dataset contains (observations, references). Only the references
# (index 1, Sentinel-2) are used here: the noisy inputs are generated synthetically.
dataset = torch.load(DATASET_PATH, weights_only=False)
references = dataset[:][1]

y_clean = torch.stack([clean_and_normalize(img) for img in references])
clean_dataset = TensorDataset(y_clean)


# =============================================================================
# 3. Synthetic noisy measurements
# =============================================================================
# Forward model: y = x + n, with n ~ N(0, SIGMA^2)
physics = Denoising(GaussianNoise(sigma=SIGMA)).to(device)

# deepinv applies the physics to every clean image and saves the (x, y) pairs to HDF5.
hdf5_path = dinv.datasets.generate_dataset(
    train_dataset=clean_dataset,
    test_dataset=None,
    physics=physics,
    device=device,
    save_dir=MEASUREMENT_DIR,
    num_workers=num_workers,
)

train_dataset = dinv.datasets.HDF5Dataset(path=hdf5_path, train=True)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True
)


# =============================================================================
# 4. Training
# =============================================================================
mse_loss = dinv.loss.SupLoss(metric=dinv.metric.MSE())    # training loss
psnr_metric = dinv.loss.SupLoss(metric=dinv.metric.PSNR()) # monitoring only

sigma = torch.tensor(SIGMA, device=device)  # noise level given to DRUNet as input
model.train()

for epoch in range(N_EPOCHS):
    epoch_mse, epoch_psnr = 0.0, 0.0

    for x_true, noisy in train_dataloader:      # x_true: clean image, noisy: x_true + noise
        x_true, noisy = x_true.to(device), noisy.to(device)

        x_denoised = model(noisy, sigma)
        loss = mse_loss(x_denoised, x_true).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_mse += loss.item()
        epoch_psnr += psnr_metric(x_denoised, x_true).mean().item()

    scheduler.step()

    n_batches = len(train_dataloader)
    print(f"Epoch {epoch}/{N_EPOCHS} | MSE {epoch_mse / n_batches:.2e} | "
          f"PSNR {epoch_psnr / n_batches:.2f} dB | lr {scheduler.get_last_lr()[0]:.2e}")

    # =========================================================================
    # 5. Checkpoints: model + last batch of the epoch (for visual inspection)
    # =========================================================================
    if epoch % SAVE_EVERY == 0:
        torch.save(x_true, os.path.join(SAVE_DIR, f"xtrue_epoch{epoch}.pt"))
        torch.save(noisy, os.path.join(SAVE_DIR, f"noisy_epoch{epoch}.pt"))
        torch.save(x_denoised.detach(), os.path.join(SAVE_DIR, f"denoised_epoch{epoch}.pt"))
        torch.save(model, os.path.join(SAVE_DIR, f"drunet_epoch{epoch}.pt"))
