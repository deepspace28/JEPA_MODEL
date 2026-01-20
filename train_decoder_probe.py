import torch
from torch.utils.data import DataLoader
from dataset.video_stream_dataset import LocalVideoDataset
from models.encoder import ViTEncoder
from models.decoder import PatchDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"

enc = ViTEncoder(image_size=224).to(device)
enc.load_state_dict(torch.load("checkpoints/enc.pt"))  
enc.eval()

dec = PatchDecoder(dim=768).to(device)
opt = torch.optim.Adam(dec.parameters(), lr=1e-3)

dataset = LocalVideoDataset("data/clips", clip_len=16, size=224)
loader = DataLoader(dataset, batch_size=1)

for step, clip in enumerate(loader):
    clip = clip.to(device)

    with torch.no_grad():
        z = enc(clip)

    patch_imgs = dec(z)

    B, T, P, C, H, W = patch_imgs.shape
    grid = int(P ** 0.5)
    patch_imgs = patch_imgs.view(B, T, grid, grid, C, H, W)
    recon = patch_imgs.permute(0,1,4,2,5,3,6).reshape(B, T, C, grid*H, grid*W)

    loss = torch.nn.functional.mse_loss(recon, clip)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"Step {step} | Loss: {loss.item():.4f}")
    if step > 1000:
        break

torch.save(dec.state_dict(), "checkpoints/dec.pt")

