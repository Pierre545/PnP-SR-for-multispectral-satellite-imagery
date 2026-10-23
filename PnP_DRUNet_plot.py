"""
Plug-and-Play (PnP) reconstruction of 4-channel Sentinel-2 images with a fine-tuned DRUNet.

Problem
-------
We observe y = A x, where A blurs and then downsamples the image x.
We want to recover x from y.

PnP proximal gradient descent
-----------------------------
Starting from x_0 = A^T y, repeat:
    1. gradient step on the data fidelity:  z_k     = x_k - gamma * A^T (A x_k - y)
    2. denoising step (replaces the prox):  x_{k+1} = D(z_k, sigma)   with sigma = gamma * lambda
The denoiser D acts as the image prior (regularizer).

For each test image, the script prints the PSNR/SSIM and runtime, then plots
the reconstruction, the reference, and an NDWI water mask computed from the reconstruction.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap

import deepinv as dinv
from deepinv.loss.metric import SSIM


# =============================================================================
# Configuration: edit these values for your setup
# =============================================================================
DATASET_PATH    = "your_path/your_dataset.pth"     # saved TensorDataset (observations, references)
CHECKPOINT_PATH = "your_path/your_checkpoint.pth"  # fine-tuned DRUNet (see denoiser_training.py)

IMG_SIZE    = [4, 300, 300]  # (channels, height, width)
BLUR_SIGMA  = 1.5            # std of the Gaussian blur kernel (pixels)
DOWN_FACTOR = 3              # downsampling factor

N_ITER = 250    # number of PnP iterations
GAMMA  = 10     # step size of the gradient step
LAMBDA = 0.01   # regularization weight; the denoiser receives sigma = GAMMA * LAMBDA
# GAMMA * LAMBDA = 0.1 matches the noise level used to train the denoiser (sigma = 0.1).

NDWI_THRESHOLD = 0.01        # pixels with NDWI above this value are classified as water

dinv.utils.disable_tex()
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


# =============================================================================
# Helpers
# =============================================================================
def clean_and_normalize(image):
    """Replace NaN/inf values, then rescale the image to [0, 1] (min-max over the whole image)."""
    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    return (image - image.min()) / (image.max() - image.min())


@torch.no_grad()
def operator_norm_sq(physics, x_shape, device, n_iter=100, tol=1e-6):
    """Estimate ||A||^2 (largest eigenvalue of A^T A) by power iteration.

    ||A||^2 is the Lipschitz constant L of the data-fidelity gradient;
    gradient descent requires the step size GAMMA < 2 / L.
    """
    v = torch.randn(x_shape, device=device)
    v = v / v.norm()
    lam_old = 0.0
    for _ in range(n_iter):
        AtAv = physics.A_adjoint(physics.A(v))
        lam = torch.dot(v.flatten(), AtAv.flatten()).item()
        v = AtAv / AtAv.norm()
        if abs(lam - lam_old) <= tol * abs(lam):
            break
        lam_old = lam
    return lam


# =============================================================================
# 1. Forward operator A = downsampling ∘ blur
# =============================================================================
blur = dinv.physics.Blur(
    dinv.physics.blur.gaussian_blur(sigma=BLUR_SIGMA, angle=0.0),
    device=device, padding="reflect",
)
downsampling = dinv.physics.Downsampling(
    img_size=IMG_SIZE, factor=DOWN_FACTOR, device=device, padding="reflect",
)
physics = downsampling * blur   # blur first, then downsample

L = operator_norm_sq(physics, (1, *IMG_SIZE), device)
print(f"||A||^2 ~ {L:.4f}  ->  step size must stay below {2 / L:.3f}")
if GAMMA >= 2 / L:
    print(f"Warning: GAMMA = {GAMMA} exceeds 2/L; the gradient step may diverge.")

data_fidelity = dinv.optim.L2()   # f(x) = 1/2 ||A x - y||^2, grad = A^T (A x - y)


# =============================================================================
# 2. Denoiser: DRUNet adapted to 4 channels, loaded from the fine-tuned checkpoint
# =============================================================================
# Same architecture as in training: 4 image channels + 1 noise-level map as input,
# 4 channels as output.
denoiser = dinv.models.DRUNet(pretrained=None, device=device)
denoiser.m_head = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
denoiser.m_tail = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1, bias=False).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
denoiser.load_state_dict(checkpoint["model_state_dict"])
denoiser.eval()


# =============================================================================
# 3. Test data: clean reference images
# =============================================================================
# The saved dataset contains (observations, references). Only the references are
# used: the observations are simulated below with y = A x.
dataset = torch.load(DATASET_PATH, weights_only=False)
x_test = [clean_and_normalize(img) for img in dataset[:][1]]


# =============================================================================
# 4. PnP reconstruction of each test image
# =============================================================================
psnr_metric = dinv.metric.PSNR()
ssim_metric = SSIM()
water_cmap = ListedColormap(["#c17767", "#add8e6"])   # land = brown, water = light blue
runtimes = []

for idx, x_true in enumerate(x_test):
    x_true = x_true.unsqueeze(0).to(device)   # add batch dimension: [1, C, H, W]
    y = physics(x_true)                       # synthetic low-resolution observation

    xk = physics.A_adjoint(y)                 # initialization: back-projection A^T y
    psnr_history = []

    start = time.time()
    with torch.no_grad():
        for _ in range(N_ITER):
            xk = xk - GAMMA * data_fidelity.grad(xk, y, physics)   # data-fidelity step
            xk = denoiser(xk, GAMMA * LAMBDA)                      # prior (denoising) step
            psnr_history.append(psnr_metric(x_true, xk).item())
    runtimes.append(time.time() - start)

    ssim = ssim_metric(x_true, xk).item()
    print(f"Image {idx}: PSNR {psnr_history[-1]:.4f} dB | SSIM {ssim:.4f} | "
          f"time {runtimes[-1]:.2f} s (mean {np.mean(runtimes):.2f} s)")

    # -------------------------------------------------------------------------
    # Display: channels [2, 1, 0] form an RGB composite.
    # A square root (gamma correction) brightens the images for display only.
    # -------------------------------------------------------------------------
    dinv.utils.plot([(xk ** 0.5)[0, [2, 1, 0]]], titles=["Reconstruction"])
    dinv.utils.plot([(x_true ** 0.5)[0, [2, 1, 0]]], titles=["Reference"])

    # -------------------------------------------------------------------------
    # Water mask from the reconstruction:
    # NDWI = (Green - NIR) / (Green + NIR); channel 1 = green, channel 3 = NIR.
    # -------------------------------------------------------------------------
    green, nir = xk[0, 1], xk[0, 3]
    ndwi = (green - nir) / (green + nir + 1e-8)   # +eps avoids division by zero
    dinv.utils.plot([ndwi.unsqueeze(0) > NDWI_THRESHOLD], cmap=water_cmap, titles=["Water mask"])
