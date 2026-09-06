from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from jepa_model.config import load_config
from jepa_model.data import LocalVideoDataset
from jepa_model.models import TransformerPredictor, ViTEncoder
from jepa_model.utils import choose_device, save_json, set_seed


def train(config_path: str = "configs/base.yaml") -> dict:
    cfg = load_config(config_path).raw
    set_seed(cfg["seed"])
    device = choose_device(cfg["device"])

    train_ds = LocalVideoDataset(
        root=cfg["paths"]["train_root"],
        clip_len=16,
        size=cfg["model"]["image_size"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
    )

    enc = ViTEncoder(image_size=cfg["model"]["image_size"]).to(device)
    pred = TransformerPredictor(dim=cfg["model"]["dim"]).to(device)
    ema_enc = ViTEncoder(image_size=cfg["model"]["image_size"]).to(device)
    ema_enc.load_state_dict(enc.state_dict())

    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=cfg["training"]["lr"])
    scaler = GradScaler(device if device == "cuda" else "cpu", enabled=cfg["training"]["amp"] and device == "cuda")

    pred_steps = cfg["training"]["pred_steps"]
    max_steps = cfg["training"]["max_steps"]
    losses = []

    pbar = tqdm(enumerate(train_loader), total=min(len(train_loader), max_steps + 1), desc="stage1")
    for step, clip in pbar:
        if step > max_steps:
            break
        clip = clip.to(device)
        amp_enabled = cfg["training"]["amp"] and device == "cuda"
        with autocast(device_type="cuda", enabled=amp_enabled):
            z = enc(clip)
            with torch.no_grad():
                zt = ema_enc(clip)
            total_loss = 0.0
            for k in pred_steps:
                z_input = z[:, :-k]
                z_target = zt[:, k:]
                z_roll = z_input
                for _ in range(k):
                    z_roll = pred(z_roll)
                total_loss = total_loss + F.l1_loss(z_roll, z_target)
            loss = total_loss / len(pred_steps)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            m = cfg["training"]["ema_momentum"]
            for p, q in zip(enc.parameters(), ema_enc.parameters()):
                q.data = m * q.data + (1.0 - m) * p.data

        losses.append(float(loss.item()))
        if step % cfg["training"]["log_interval"] == 0:
            pbar.set_postfix(loss=float(loss.item()))

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(enc.state_dict(), ckpt_dir / "enc.pt")
    torch.save(pred.state_dict(), ckpt_dir / "pred.pt")

    metrics = {
        "seed": cfg["seed"],
        "device": device,
        "steps": len(losses),
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
    }
    save_json(ckpt_dir / "train_metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    m = train()
    print(m)
