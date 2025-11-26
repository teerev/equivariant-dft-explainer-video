import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
BATCH_SIZE = 128
EPOCHS = 2          # MNIST is trivial, this is enough
LR = 1e-3
OUT_DIR = "mnist_viz"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Dataset & Dataloader
# ----------------------------
transform = transforms.ToTensor()

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# ----------------------------
# Simple CNN for MNIST
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x28x28 -> 16x24x24
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        # 16x12x12 -> 32x8x8
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # conv1 feature maps
        x = F.max_pool2d(x, 2)         # 16x12x12
        x = F.relu(self.conv2(x))      # conv2 feature maps
        x = F.max_pool2d(x, 2)         # 32x4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# ----------------------------
# Training loop
# ----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS}, "
          f"loss = {total_loss/len(train_loader.dataset):.4f}")

# ----------------------------
# Plotting helpers
# ----------------------------
def plot_kernels(weight_tensor, save_path, title="Conv filters", max_filters=6):
    """
    weight_tensor: (out_channels, in_channels, kH, kW)
    max_filters: save only the first N filters
    Produces grayscale filters on black background with red outline and white text.
    """
    w = weight_tensor.detach().cpu().clone()
    out_ch = min(w.shape[0], max_filters)

    # Only grayscale channel
    w = w[:out_ch, 0, :, :]

    # Normalize each filter to [0,1]
    w_min = w.view(out_ch, -1).min(dim=1)[0].view(-1, 1, 1)
    w_max = w.view(out_ch, -1).max(dim=1)[0].view(-1, 1, 1)
    w = (w - w_min) / (w_max - w_min + 1e-8)

    cols = out_ch
    rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, 2),
                             facecolor='black')
    axes = np.atleast_1d(axes)

    for i in range(out_ch):
        ax = axes[i]
        ax.imshow(w[i], cmap="gray", vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f"f{i}", color='white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("black")

        # Red outline around each kernel
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2.0)

    plt.suptitle(title, color='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor="black")
    plt.close(fig)


def plot_feature_maps(feature_maps, save_path, title="Feature maps", max_maps=6):
    """
    feature_maps: (1, C, H, W)
    max_maps: save only the first N feature maps
    Grayscale on black background, white text, no outline.
    """
    fmap = feature_maps.detach().cpu().squeeze(0)  # (C, H, W)
    C = min(fmap.shape[0], max_maps)
    fmap = fmap[:C]

    # Normalize each map to [0,1]
    fm_min = fmap.view(C, -1).min(dim=1)[0].view(-1, 1, 1)
    fm_max = fmap.view(C, -1).max(dim=1)[0].view(-1, 1, 1)
    fmap = (fmap - fm_min) / (fm_max - fm_min + 1e-8)

    cols = C
    rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, 2),
                             facecolor='black')
    axes = np.atleast_1d(axes)

    for i in range(C):
        ax = axes[i]
        ax.imshow(fmap[i], cmap="gray", vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f"c{i}", color='white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("black")

    plt.suptitle(title, color='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor="black")
    plt.close(fig)


def save_input_image(img_tensor, save_path, upscale=3):
    """
    img_tensor: (1, 1, 28, 28) or (1, 28, 28)
    Saves MNIST digit on a black background.
    """
    img = img_tensor.detach().cpu().squeeze().numpy()  # (28, 28)

    fig = plt.figure(figsize=(2 * upscale, 2 * upscale), facecolor='black')
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title("Input image", color='white')
    ax.axis('off')
    ax.set_facecolor('black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='black')
    plt.close(fig)

# ----------------------------
# Save conv1 filters
# ----------------------------
print("Saving conv1 filters...")
plot_kernels(
    model.conv1.weight,
    save_path=os.path.join(OUT_DIR, "conv1_filters.png"),
    title="Conv1 Filters",
    max_filters=6
)

# ----------------------------
# Capture feature maps for a single test image
# ----------------------------
model.eval()
with torch.no_grad():
    sample_img, sample_label = next(iter(test_loader))
    sample_img = sample_img.to(device)

    # Save the input image itself
    save_input_image(
        sample_img,
        save_path=os.path.join(OUT_DIR, "input_digit.png"),
        upscale=3
    )
    print("Sample label:", sample_label.item())

    # Manually forward through layers to grab maps
    x = F.relu(model.conv1(sample_img))
    conv1_maps = x.clone()
    x = F.max_pool2d(x, 2)
    x = F.relu(model.conv2(x))
    conv2_maps = x.clone()

print("Saving feature maps...")
plot_feature_maps(
    conv1_maps,
    save_path=os.path.join(OUT_DIR, "conv1_feature_maps.png"),
    title="Conv1 Feature Maps",
    max_maps=6
)
plot_feature_maps(
    conv2_maps,
    save_path=os.path.join(OUT_DIR, "conv2_feature_maps.png"),
    title="Conv2 Feature Maps",
    max_maps=6
)

print("Done. Images saved in:", OUT_DIR)
