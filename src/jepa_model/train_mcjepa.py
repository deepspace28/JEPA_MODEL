from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from jepa_model.config import load_config
from jepa_model.data import LocalVideoDataset
from jepa_model.losses import photometric_loss, smoothness_loss, tokens_to_grid, warp_image_with_flow
from jepa_model.models import FlowHead, TransformerPredictor, ViTEncoder
from jepa_model.utils import choose_device, save_json, set_seed
from jepa_model.visualization import save_training_curves, save_triplet


def train_mc_jepa(config_path: str = "configs/mc_jepa.yaml") -> dict:
    cfg = load_config(config_path).raw
    set_seed(cfg["seed"])
    device = choose_device(cfg["device"])

    clip_len = cfg["training"]["clip_len"]
    if clip_len < 2:
        raise ValueError("clip_len must be >= 2 for motion training")

    train_ds = LocalVideoDataset(
        root=cfg["paths"]["train_root"],
        clip_len=clip_len,
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
    flow_head = FlowHead(dim=cfg["model"]["dim"], hidden_dim=cfg["model"]["flow_hidden_dim"]).to(device)

    ema_enc = ViTEncoder(image_size=cfg["model"]["image_size"]).to(device)
    ema_enc.load_state_dict(enc.state_dict())

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(pred.parameters()) + list(flow_head.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = GradScaler("cuda", enabled=cfg["training"]["amp"] and device == "cuda")

    pred_steps = cfg["training"]["pred_steps"]
    max_steps = cfg["training"]["max_steps"]
    lambda_content = cfg["training"]["lambda_content"]
    lambda_photo = cfg["training"]["lambda_photo"]
    lambda_smooth = cfg["training"]["lambda_smooth"]

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(cfg["paths"].get("artifacts_dir", "artifacts/mc_jepa"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    vis_every = int(cfg.get("visualization", {}).get("save_every", 100))
    max_images = int(cfg.get("visualization", {}).get("max_images", 8))

    content_losses, photo_losses, smooth_losses, total_losses = [], [], [], []
    saved_images = 0

    pbar = tqdm(enumerate(train_loader), total=min(len(train_loader), max_steps + 1), desc="mc-jepa")
    for step, clip in pbar:
        if step > max_steps:
            break
        clip = clip.to(device)
        frame_t = clip[:, 0]
        frame_t1 = clip[:, 1]

        amp_enabled = cfg["training"]["amp"] and device == "cuda"
        with autocast(device_type="cuda", enabled=amp_enabled):
            z = enc(clip)
            with torch.no_grad():
                zt = ema_enc(clip)

            total_content = 0.0
            for k in pred_steps:
                z_input = z[:, :-k]
                z_target = zt[:, k:]
                z_roll = z_input
                for _ in range(k):
                    z_roll = pred(z_roll)
                total_content = total_content + F.l1_loss(z_roll, z_target)
            content_loss = total_content / len(pred_steps)

            tokens_t = z[:, 0]
            feat_grid_t = tokens_to_grid(tokens_t)
            lowres_flow = flow_head(feat_grid_t)
            flow = F.interpolate(lowres_flow, size=(frame_t.shape[-2], frame_t.shape[-1]), mode="bilinear", align_corners=False)

            photo = photometric_loss(frame_t, frame_t1, flow)
            smooth = smoothness_loss(flow)

            total = lambda_content * content_loss + lambda_photo * photo + lambda_smooth * smooth

        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            m = cfg["training"]["ema_momentum"]
            for p, q in zip(enc.parameters(), ema_enc.parameters()):
                q.data = m * q.data + (1.0 - m) * p.data

        content_losses.append(float(content_loss.item()))
        photo_losses.append(float(photo.item()))
        smooth_losses.append(float(smooth.item()))
        total_losses.append(float(total.item()))

        if step % cfg["training"]["log_interval"] == 0:
            pbar.set_postfix(total=float(total.item()), content=float(content_loss.item()), photo=float(photo.item()))

        if step % vis_every == 0 and saved_images < max_images:
            with torch.no_grad():
                warped = warp_image_with_flow(frame_t[:1], flow[:1])
            save_triplet(
                artifacts_dir / f"sample_step_{step:06d}.png",
                frame_t[0],
                frame_t1[0],
                warped[0],
                flow[0],
            )
            saved_images += 1

    torch.save(enc.state_dict(), ckpt_dir / "enc_mcjepa.pt")
    torch.save(pred.state_dict(), ckpt_dir / "pred_mcjepa.pt")
    torch.save(flow_head.state_dict(), ckpt_dir / "flow_head_mcjepa.pt")

    save_training_curves(
        artifacts_dir / "mcjepa_losses.png",
        {
            "total": total_losses,
            "content": content_losses,
            "photometric": photo_losses,
            "smoothness": smooth_losses,
        },
    )

    metrics = {
        "seed": cfg["seed"],
        "device": device,
        "steps": len(total_losses),
        "final_total": total_losses[-1] if total_losses else None,
        "final_content": content_losses[-1] if content_losses else None,
        "final_photo": photo_losses[-1] if photo_losses else None,
        "final_smooth": smooth_losses[-1] if smooth_losses else None,
        "mean_total": sum(total_losses) / len(total_losses) if total_losses else None,
        "artifacts_dir": str(artifacts_dir),
    }
    save_json(ckpt_dir / "mcjepa_train_metrics.json", metrics)
    save_json(artifacts_dir / "mcjepa_train_metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    print(train_mc_jepa())
