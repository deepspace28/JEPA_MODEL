import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset.video_stream_dataset import LocalVideoDataset
from models.encoder import ViTEncoder
from models.predictor import TransformerPredictor

# Config
CLIP_LEN = 16
IMG_SIZE = 224
BATCH_SIZE = 1
LR = 1e-4
EMA_MOMENTUM = 0.99
PRED_STEPS = [1, 2, 4]
MAX_STEPS = 500

dataset = LocalVideoDataset(root="data/clips", clip_len=CLIP_LEN, size=IMG_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

enc = ViTEncoder(image_size=IMG_SIZE).to(device)
pred = TransformerPredictor(dim=768).to(device)
ema_enc = ViTEncoder(image_size=IMG_SIZE).to(device)
ema_enc.load_state_dict(enc.state_dict())

opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=LR)
scaler = GradScaler("cuda")

for step, clip in enumerate(loader):
    clip = clip.to(device)

    with autocast("cuda"):
        z = enc(clip)
        with torch.no_grad():
            zt = ema_enc(clip)

        total_loss = 0.0
        for k in PRED_STEPS:
            z_input = z[:, :-k]
            z_target = zt[:, k:]

            z_roll = z_input
            for _ in range(k):
                z_roll = pred(z_roll)

            total_loss = total_loss + torch.nn.functional.l1_loss(z_roll, z_target)

        loss = total_loss / len(PRED_STEPS)

    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    with torch.no_grad():
        for p, q in zip(enc.parameters(), ema_enc.parameters()):
            q.data = EMA_MOMENTUM * q.data + (1 - EMA_MOMENTUM) * p.data

    if step % 25 == 0:
        print(f"Step {step:04d} | Loss: {loss.item():.5f}")

    if step >= MAX_STEPS:
        break

import os
os.makedirs("checkpoints", exist_ok=True)

torch.save(enc.state_dict(), "checkpoints/enc.pt")
torch.save(pred.state_dict(), "checkpoints/pred.pt")

print("Saved checkpoints to ./checkpoints/")

