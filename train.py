"""
train.py

Convergence-based training:
- Trains until loss plateaus near zero (not a fixed epoch count)
- LR scheduler reduces LR on plateau automatically
- Gradient clipping for stability
- Best model checkpointing
"""

import torch
import cv2
from torch.utils.data import DataLoader
from dataset import BSDDataset
from model import TNRD
from utils import psnr


# ─── Config ───────────────────────────────────────────────────────────────────
train_path = "data/train"
test_path  = "data/test"

batch_size = 4
lr_initial = 3e-4          # starting LR

# Convergence criteria
LOSS_TARGET   = 1e-4       # stop if avg loss drops below this
PATIENCE      = 15         # stop if no improvement for this many epochs
MAX_EPOCHS    = 500        # hard ceiling so it never runs forever
MIN_LR        = 1e-7       # stop if LR decays this low (model has converged as far as it can)

# Gradient clipping
GRAD_CLIP     = 1.0
# ──────────────────────────────────────────────────────────────────────────────


train_dataset = BSDDataset(train_path)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f"Training images : {len(train_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device          : {device}\n")

model     = TNRD(T=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_initial)

# Reduce LR by 0.5 if loss doesn't improve for 5 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode      = "min",
    factor    = 0.5,
    patience  = 5,
    min_lr    = MIN_LR,
    verbose   = True,
)


# ─── Training loop ────────────────────────────────────────────────────────────
print("Training (runs until convergence)...\n")

best_loss       = float("inf")
epochs_no_improv = 0
epoch           = 0

while True:
    epoch += 1
    model.train()
    total_loss = 0.0

    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()

        output, _ = model(noisy)
        loss = torch.mean((output - clean) ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

    avg_loss   = total_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

    # Step scheduler
    scheduler.step(avg_loss)

    # Save best checkpoint
    if avg_loss < best_loss:
        best_loss = avg_loss
        epochs_no_improv = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"           ↳ New best model saved (loss={best_loss:.6f})")
    else:
        epochs_no_improv += 1

    # ── Stop conditions ──────────────────────────────────────────────────────
    if avg_loss <= LOSS_TARGET:
        print(f"\n✓ Converged: loss {avg_loss:.6f} ≤ target {LOSS_TARGET}")
        break

    if epochs_no_improv >= PATIENCE:
        print(f"\n✓ Early stop: no improvement for {PATIENCE} epochs")
        break

    if current_lr <= MIN_LR:
        print(f"\n✓ Early stop: LR decayed to minimum ({MIN_LR})")
        break

    if epoch >= MAX_EPOCHS:
        print(f"\n✓ Reached max epoch cap ({MAX_EPOCHS})")
        break


# ─── Load best weights for testing ───────────────────────────────────────────
print("\nLoading best checkpoint for evaluation...")
model.load_state_dict(torch.load("best_model.pth", map_location=device))

# ─── Test ─────────────────────────────────────────────────────────────────────
print("\nTesting...\n")

test_dataset = BSDDataset(test_path)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
with torch.no_grad():
    for i, (noisy, clean) in enumerate(test_loader):
        noisy, clean = noisy.to(device), clean.to(device)

        output, _ = model(noisy)

        score = psnr(output, clean).item()
        print(f"PSNR: {score:.2f} dB")

        cv2.imwrite("noisy.png",    (noisy.squeeze().cpu().numpy()  * 255).astype("uint8"))
        cv2.imwrite("denoised.png", (output.squeeze().cpu().numpy() * 255).astype("uint8"))
        cv2.imwrite("clean.png",    (clean.squeeze().cpu().numpy()  * 255).astype("uint8"))

        print("Saved: noisy.png, denoised.png, clean.png")
        break