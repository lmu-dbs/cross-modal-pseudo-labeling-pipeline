#!/usr/bin/env python3
import argparse, os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

class ImgFolderRecursive(Dataset):
    def __init__(self, root, size=512):
        self.root = Path(root)
        self.size = int(size)
        self.imgs = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in [".png",".jpg",".jpeg"]])
        if len(self.imgs) == 0:
            raise ValueError(f"Found 0 images under {root}")
        self.tf = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        x = Image.open(self.imgs[i]).convert("RGB")
        return self.tf(x)

def load_sd_into_model(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    model.load_state_dict(sd, strict=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--ckpt_in", required=True)
    ap.add_argument("--ckpt_out", required=True)
    ap.add_argument("--num_classes", type=int, default=19)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    ds = ImgFolderRecursive(args.imgs, size=args.size)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = models.segmentation.deeplabv3_resnet50(num_classes=args.num_classes, weights=None).to(device)
    load_sd_into_model(model, args.ckpt_in)

    # --- AdaBN: update BN running stats only ---
    model.train()  # important: enables BN running_mean/var updates
    # freeze affine params too (optional; safest)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)

    seen = 0
    with torch.no_grad():
        for x in dl:
            x = x.to(device, non_blocking=True)
            _ = model(x)["out"]  # forward updates BN stats
            seen += x.size(0)
            if args.max_images > 0 and seen >= args.max_images:
                break

    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.ckpt_out)
    print(f"[AdaBN] Updated BN stats using {min(seen, len(ds))} images")
    print(f"[Saved] {args.ckpt_out}")

if __name__ == "__main__":
    main()
