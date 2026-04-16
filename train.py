"""
train.py

Balanced training setup
"""

import torch
import cv2
from torch.utils.data import DataLoader
from dataset import BSDDataset
from model import TNRD
from utils import psnr


train_path = "data/train"
test_path = "data/test"

epochs = 10
batch_size = 4

# CHANGE: increased LR
# WHY: previous 1e-4 too slow → no learning
lr = 3e-4


train_dataset = BSDDataset(train_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("Number of training images:", len(train_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

model = TNRD(T=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


print("Training...\n")

for epoch in range(epochs):
    total_loss = 0

    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()

        output, _ = model(noisy)

        # CHANGE: removed *10 scaling
        # WHY: was distorting gradients
        loss = torch.mean((output - clean) ** 2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.6f}")


# ================= TEST =================
print("\nTesting...\n")

test_dataset = BSDDataset(test_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    for i, (noisy, clean) in enumerate(test_loader):
        noisy, clean = noisy.to(device), clean.to(device)

        output, _ = model(noisy)

        print("PSNR:", psnr(output, clean).item())

        cv2.imwrite("noisy.png", noisy.squeeze().cpu().numpy() * 255)
        cv2.imwrite("denoised.png", output.squeeze().cpu().numpy() * 255)
        cv2.imwrite("clean.png", clean.squeeze().cpu().numpy() * 255)

        print("Saved output images.")
        break