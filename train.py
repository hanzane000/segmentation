import os
import argparse
from datetime import datetime

import torch
from torch.optim import Adam

from data.data import build_dataloaders
from losses.segmentation_loss import BCEDiceLoss
from utils.metrics import dice_score, iou_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net baseline for binary segmentation")

    # Paths
    parser.add_argument("--data_root", default="dataset")
    parser.add_argument("--img_dir", default="")
    parser.add_argument("--mask_dir", default="")
    parser.add_argument("--save_dir", default="./checkpoints/baseline")
    parser.add_argument("--best_model_name", default="best_model.pth")
    parser.add_argument("--last_model_name", default="last_model.pth")
    parser.add_argument("--logs_root", default="./logs")
    parser.add_argument("--model_name", default="unet", choices=["unet", "unet_A", "unet_B"])

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    # Data
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fg_crop_prob", type=float, default=0.7)
    parser.add_argument("--val_resize_short", type=int, default=512)

    # Model
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--bilinear", type=int, default=1)
    parser.add_argument("--base_c", type=int, default=32)

    # Loss
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)

    return parser.parse_args()


def build_model(model_name, in_channels, num_classes, bilinear, base_c=32):
    if model_name == "unet":
        from models.unet import UNet
    elif model_name == "unet_A":
        from models.unet_A import UNet
    elif model_name == "unet_B":
        from models.unet_B import UNet
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        bilinear=bool(bilinear),
        base_c=base_c,
    )
    return model


def save_args_as_yaml(args, yaml_path, extra=None):
    data = vars(args).copy()
    if extra is not None:
        data.update(extra)

    with open(yaml_path, "w", encoding="utf-8") as f:
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, str):
                f.write(f'{key}: "{value}"\n')
            else:
                f.write(f"{key}: {value}\n")


# =========================
# Train / Val
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)

            total_loss += loss.item()
            total_dice += dice_score(logits, masks)
            total_iou += iou_score(logits, masks)

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)
    return avg_loss, avg_dice, avg_iou


def main(args):
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(args.logs_root, run_time)
    os.makedirs(run_log_dir, exist_ok=True)

    log_txt_path = os.path.join(run_log_dir, "train_log.txt")
    yaml_path = os.path.join(run_log_dir, "config.yaml")

    img_dir = args.img_dir if args.img_dir else os.path.join(args.data_root, "images")
    mask_dir = args.mask_dir if args.mask_dir else os.path.join(args.data_root, "masks")
    best_ckpt_path = os.path.join(args.save_dir, args.best_model_name)
    last_ckpt_path = os.path.join(args.save_dir, args.last_model_name)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    save_args_as_yaml(
        args=args,
        yaml_path=yaml_path,
        extra={
            "run_time": run_time,
            "img_dir_resolved": img_dir,
            "mask_dir_resolved": mask_dir,
            "best_ckpt_path": best_ckpt_path,
            "last_ckpt_path": last_ckpt_path,
            "device_resolved": str(device),
        },
    )

    def log(message):
        print(message)
        with open(log_txt_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    log(f"Run time: {run_time}")
    log(f"Log dir: {run_log_dir}")
    log(f"Device: {device}")

    train_loader, val_loader, train_stems, val_stems = build_dataloaders(
        img_dir=img_dir,
        mask_dir=mask_dir,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        fg_crop_prob=args.fg_crop_prob,
        val_resize_short=args.val_resize_short,
    )

    log(f"Train samples: {len(train_stems)}")
    log(f"Val samples: {len(val_stems)}")

    model = build_model(
        model_name=args.model_name,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        bilinear=bool(args.bilinear),
        base_c=args.base_c,
    ).to(device)
    criterion = BCEDiceLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion, device)

        log(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val dice: {val_dice:.4f} | "
            f"val iou: {val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_ckpt_path)

        torch.save(model.state_dict(), last_ckpt_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
