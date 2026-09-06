from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jepa_model.config import load_config
from jepa_model.data import LocalVideoDataset
from jepa_model.models import TransformerPredictor, ViTEncoder
from jepa_model.utils import choose_device, save_json, set_seed


def evaluate(config_path: str = "configs/base.yaml") -> dict:
    cfg = load_config(config_path).raw
    set_seed(cfg["seed"])
    device = choose_device(cfg["device"])

    val_ds = LocalVideoDataset(cfg["paths"]["val_root"], clip_len=16, size=cfg["model"]["image_size"])
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"])

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    enc = ViTEncoder(image_size=cfg["model"]["image_size"]).to(device)
    pred = TransformerPredictor(dim=cfg["model"]["dim"]).to(device)
    enc.load_state_dict(torch.load(ckpt_dir / "enc.pt", map_location=device))
    pred.load_state_dict(torch.load(ckpt_dir / "pred.pt", map_location=device))
    enc.eval()
    pred.eval()

    losses = []
    with torch.no_grad():
        for clip in val_loader:
            clip = clip.to(device)
            z = enc(clip)
            zt = z.clone()
            total = 0.0
            for k in cfg["training"]["pred_steps"]:
                z_input = z[:, :-k]
                z_target = zt[:, k:]
                z_roll = z_input
                for _ in range(k):
                    z_roll = pred(z_roll)
                total = total + F.l1_loss(z_roll, z_target)
            losses.append(float((total / len(cfg["training"]["pred_steps"])).item()))

    metrics = {
        "seed": cfg["seed"],
        "eval_batches": len(losses),
        "eval_latent_l1": sum(losses) / len(losses) if losses else None,
    }
    save_json(ckpt_dir / "eval_metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    m = evaluate()
    print(m)
