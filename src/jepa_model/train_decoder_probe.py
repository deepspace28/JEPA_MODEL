from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from jepa_model.config import load_config
from jepa_model.data import LocalVideoDataset
from jepa_model.models import PatchDecoder, ViTEncoder
from jepa_model.utils import choose_device, save_json, set_seed


def train_decoder_probe(config_path: str = "configs/base.yaml") -> dict:
    cfg = load_config(config_path).raw
    set_seed(cfg["seed"])
    device = choose_device(cfg["device"])

    ds = LocalVideoDataset(cfg["paths"]["train_root"], clip_len=16, size=cfg["model"]["image_size"])
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"])

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    enc = ViTEncoder(image_size=cfg["model"]["image_size"]).to(device)
    enc.load_state_dict(torch.load(ckpt_dir / "enc.pt", map_location=device))
    enc.eval()

    dec = PatchDecoder(dim=cfg["model"]["dim"], patch_size=cfg["model"]["patch_size"]).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)

    losses = []
    for step, clip in tqdm(enumerate(loader), total=min(len(loader), 1001), desc="decoder_probe"):
        if step > 1000:
            break
        clip = clip.to(device)
        with torch.no_grad():
            z = enc(clip)

        patch_imgs = dec(z)
        b, t, p, c, h, w = patch_imgs.shape
        grid = int(p ** 0.5)
        recon = patch_imgs.view(b, t, grid, grid, c, h, w).permute(0, 1, 4, 2, 5, 3, 6).reshape(b, t, c, grid * h, grid * w)

        loss = F.mse_loss(recon, clip)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    torch.save(dec.state_dict(), ckpt_dir / "dec.pt")
    metrics = {
        "seed": cfg["seed"],
        "steps": len(losses),
        "final_mse": losses[-1] if losses else None,
        "mean_mse": sum(losses) / len(losses) if losses else None,
    }
    save_json(ckpt_dir / "decoder_metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    m = train_decoder_probe()
    print(m)
