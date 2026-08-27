import os, argparse, json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as I
from torchvision.models.segmentation import (
    deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50
)
from pathlib import Path


IGNORE = 255
try:
    from PIL import Image as PILImage
    RESIZE_NEAREST = PILImage.Resampling.NEAREST
except Exception:
    RESIZE_NEAREST = Image.NEAREST



class SegFolder(Dataset):
    def __init__(self, imgs_dir, lbls_dir, size):
        self.imgs_dir = imgs_dir
        self.lbls_dir = lbls_dir
        self.size = size

        from pathlib import Path
        img_paths = sorted(
            str(p) for p in Path(imgs_dir).rglob("*.png")
            if not p.name.startswith(".")
        )
        lbl_paths = sorted(
    str(p) for p in Path(lbls_dir).rglob("*_gtFine_labelTrainIds.png")
    if not p.name.startswith(".")
)


        assert len(img_paths) > 0, f"No PNGs found under {imgs_dir}"
        assert len(lbl_paths) > 0, f"No PNGs found under {lbls_dir}"

        def key_fn(p: str) -> str:
            """Normalize name so foggy image + gtFine label share key."""
            name = Path(p).stem
            # strip foggy suffixes
            for fog in [
                "_leftImg8bit_foggy_beta_0.005",
                "_leftImg8bit_foggy_beta_0.01",
                "_leftImg8bit_foggy_beta_0.02",
                "_leftImg8bit_foggy_beta_0_005",
                "_leftImg8bit_foggy_beta_0_01",
                "_leftImg8bit_foggy_beta_0_02",
            ]:
                if name.endswith(fog):
                    name = name[: -len(fog)]
                    break
            # strip gtFine suffix
            if name.endswith("_gtFine_labelTrainIds"):
                name = name[: -len("_gtFine_labelTrainIds")]
            return name

        img_map = {key_fn(p): p for p in img_paths}
        lbl_map = {key_fn(p): p for p in lbl_paths}

        common_keys = sorted(set(img_map.keys()) & set(lbl_map.keys()))

        self.pairs = [(img_map[k], lbl_map[k]) for k in common_keys]

        assert len(self.pairs) > 0, (
            f"No image/label pairs found in {imgs_dir} and {lbls_dir}. "
            f"Found {len(img_paths)} imgs, {len(lbl_paths)} labels, "
            f"but 0 matching normalized keys. Example img key: "
            f"{key_fn(img_paths[0])}, lbl key: {key_fn(lbl_paths[0])}"
        )

        print(f"SegFolder: using {len(self.pairs)} img/label pairs from:")
        print(f"  imgs_dir = {imgs_dir}")
        print(f"  lbls_dir = {lbls_dir}")

        self.tf = T.Compose([
            T.Resize((size,size), interpolation=I.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],
                        [0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, lbl_path = self.pairs[i]
        x = Image.open(img_path).convert("RGB")
        y = Image.open(lbl_path).resize((self.size,self.size),
                                        resample=RESIZE_NEAREST)
        y = torch.from_numpy(np.array(y, dtype=np.int64))
        return self.tf(x), y




def build_model(arch, num_classes):
    if arch=='deeplabv3_resnet50':  return deeplabv3_resnet50(weights=None, num_classes=num_classes)
    if arch=='deeplabv3_resnet101': return deeplabv3_resnet101(weights=None, num_classes=num_classes)
    if arch=='fcn_resnet50':        return fcn_resnet50(weights=None, num_classes=num_classes)
    raise ValueError(f"Unknown arch: {arch}")

def _unwrap_state_dict(sd):
    if isinstance(sd, dict):
        if 'state_dict' in sd and isinstance(sd['state_dict'], dict): sd = sd['state_dict']
        elif 'model' in sd and isinstance(sd['model'], dict):         sd = sd['model']
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k,v in sd.items():
        nk = k
        if nk.startswith('module.'): nk = nk[7:]
        if nk.startswith('model.'):  nk = nk[6:]
        new_sd[nk] = v
    return new_sd

@torch.no_grad()
@torch.no_grad()
def evaluate(model, dl, num_classes, device):
    inter = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    pred_area = torch.zeros(num_classes, device=device)
    gt_area   = torch.zeros(num_classes, device=device)

    correct = 0
    total = 0

    model.eval()
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        valid = (y != IGNORE)
        if valid.sum() == 0:
            continue

        # logits + flip-TTA (optional; you can disable if slow)
        logits = model(x)["out"]
        xf = torch.flip(x, dims=[3])
        logits_f = model(xf)["out"]
        logits_f = torch.flip(logits_f, dims=[3])
        logits = (logits + logits_f) / 2

        pred = logits.argmax(1)

        correct += (pred[valid] == y[valid]).sum().item()
        total += valid.sum().item()

        for c in range(num_classes):
            p = (pred == c) & valid
            g = (y == c) & valid

            inter[c] += (p & g).sum()
            union[c] += (p | g).sum()
            pred_area[c] += p.sum()
            gt_area[c] += g.sum()

    iou = inter / (union + 1e-10)

    present = (gt_area > 0)  # present in GT (more standard)
    miou_all = float(iou.mean().cpu())
    miou_present = float(iou[present].mean().cpu()) if present.any() else 0.0

    dice = (2 * inter) / (pred_area + gt_area + 1e-10)
    mdice_present = float(dice[present].mean().cpu()) if present.any() else 0.0

    pixacc = correct / max(total, 1)

    return {
        "miou_all": miou_all,
        "miou_present": miou_present,
        "mdice_present": mdice_present,
        "pixel_accuracy": pixacc,
        "class_iou": [float(v) for v in iou.detach().cpu().tolist()],
        "class_dice": [float(v) for v in dice.detach().cpu().tolist()],
        "present_classes": [bool(v) for v in present.detach().cpu().tolist()],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imgs', required=True)
    ap.add_argument('--lbls', required=True)
    ap.add_argument('--arch', required=True)
    ap.add_argument('--num_classes', type=int, required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out_json', default='')
    args = ap.parse_args()

    ds = SegFolder(args.imgs, args.lbls, args.size)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = build_model(args.arch, args.num_classes).to(dev)

    sd = torch.load(args.ckpt, map_location=dev)
    sd = _unwrap_state_dict(sd)
    try:   model.load_state_dict(sd, strict=True)
    except: model.load_state_dict(sd, strict=False)
    print(f"Loaded: {args.ckpt}")

    metrics = evaluate(model, dl, args.num_classes, dev)
    print(
    f"mIoU(all): {metrics['miou_all']:.4f} | "
    f"mIoU(present): {metrics['miou_present']:.4f} | "
    f"Dice(present): {metrics['mdice_present']:.4f} | "
    f"PixelAcc: {metrics['pixel_accuracy']:.4f}"
)
    print("class_iou:", [round(x, 4) for x in metrics["class_iou"]])

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        json.dump(metrics, open(args.out_json, "w"), indent=2)

if __name__ == '__main__':
    main()
