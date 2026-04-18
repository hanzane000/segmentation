import os
import argparse
from datetime import datetime

import torch
from torch.optim import Adam

from data.data_C import build_dataloaders
from models.unet_C import UNet
from losses.segmentation_loss import MultiBranchRefineLoss
from utils.metrics import dice_score, iou_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet_C with multi-branch refinement loss")

    parser.add_argument("--data_root", default="dataset")
    parser.add_argument("--img_dir", default="")
    parser.add_argument("--mask_dir", default="")
    parser.add_argument("--save_dir", default="./checkpoints/C")
    parser.add_argument("--logs_root", default="./logs")
    parser.add_argument("--best_model_name", default="best_model.pth")
    parser.add_argument("--last_model_name", default="last_model.pth")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fg_crop_prob", type=float, default=0.7)
    parser.add_argument("--val_resize_short", type=int, default=512)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--bilinear", type=int, default=1)
    parser.add_argument("--base_c", type=int, default=32)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--lambda_coarse", type=float, default=0.5)
    parser.add_argument("--lambda_final", type=float, default=1.0)
    parser.add_argument("--lambda_boundary", type=float, default=0.3)
    parser.add_argument("--lambda_uncertainty", type=float, default=0.2)

    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)

    return parser.parse_args()


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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    coarse_loss_sum = 0.0
    final_loss_sum = 0.0
    boundary_loss_sum = 0.0
    uncertainty_loss_sum = 0.0

    for images, seg_masks, boundary_masks, uncertainty_masks, stems in loader:
        images = images.to(device, non_blocking=True)
        seg_masks = seg_masks.to(device, non_blocking=True)
        boundary_masks = boundary_masks.to(device, non_blocking=True)
        uncertainty_masks = uncertainty_masks.to(device, non_blocking=True)

        outputs = model(images)
        total, loss_dict = criterion(
            coarse_logits=outputs["coarse_logits"],
            final_logits=outputs["final_logits"],
            boundary_logits=outputs["boundary_logits"],
            uncertainty_map=outputs["uncertainty_map"],
            seg_targets=seg_masks,
            boundary_targets=boundary_masks,
            uncertainty_targets=uncertainty_masks,
        )

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        total_loss += total.item()
        coarse_loss_sum += loss_dict["coarse_loss"]
        final_loss_sum += loss_dict["final_loss"]
        boundary_loss_sum += loss_dict["boundary_loss"]
        uncertainty_loss_sum += loss_dict["uncertainty_loss"]

    num_batches = len(loader)
    stats = {
        "total_loss": total_loss / num_batches,
        "coarse_loss": coarse_loss_sum / num_batches,
        "final_loss": final_loss_sum / num_batches,
        "boundary_loss": boundary_loss_sum / num_batches,
        "uncertainty_loss": uncertainty_loss_sum / num_batches,
    }
    return stats


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    coarse_loss_sum = 0.0
    final_loss_sum = 0.0
    boundary_loss_sum = 0.0
    uncertainty_loss_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0

    with torch.no_grad():
        for images, seg_masks, boundary_masks, uncertainty_masks, stems in loader:
            images = images.to(device, non_blocking=True)
            seg_masks = seg_masks.to(device, non_blocking=True)
            boundary_masks = boundary_masks.to(device, non_blocking=True)
            uncertainty_masks = uncertainty_masks.to(device, non_blocking=True)

            outputs = model(images)
            total, loss_dict = criterion(
                coarse_logits=outputs["coarse_logits"],
                final_logits=outputs["final_logits"],
                boundary_logits=outputs["boundary_logits"],
                uncertainty_map=outputs["uncertainty_map"],
                seg_targets=seg_masks,
                boundary_targets=boundary_masks,
                uncertainty_targets=uncertainty_masks,
            )

            final_logits = outputs["final_logits"]

            total_loss += total.item()
            coarse_loss_sum += loss_dict["coarse_loss"]
            final_loss_sum += loss_dict["final_loss"]
            boundary_loss_sum += loss_dict["boundary_loss"]
            uncertainty_loss_sum += loss_dict["uncertainty_loss"]
            dice_sum += dice_score(final_logits, seg_masks)
            iou_sum += iou_score(final_logits, seg_masks)

    num_batches = len(loader)
    stats = {
        "total_loss": total_loss / num_batches,
        "coarse_loss": coarse_loss_sum / num_batches,
        "final_loss": final_loss_sum / num_batches,
        "boundary_loss": boundary_loss_sum / num_batches,
        "uncertainty_loss": uncertainty_loss_sum / num_batches,
        "dice": dice_sum / num_batches,
        "iou": iou_sum / num_batches,
    }
    return stats


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

    model = UNet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        bilinear=bool(args.bilinear),
        base_c=args.base_c,
    ).to(device)

    criterion = MultiBranchRefineLoss(
        lambda_coarse=args.lambda_coarse,
        lambda_final=args.lambda_final,
        lambda_boundary=args.lambda_boundary,
        lambda_uncertainty=args.lambda_uncertainty,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
    )
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = validate_one_epoch(model, val_loader, criterion, device)

        log(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train total loss: {train_stats['total_loss']:.4f} | "
            f"val total loss: {val_stats['total_loss']:.4f} | "
            f"val dice: {val_stats['dice']:.4f} | "
            f"val iou: {val_stats['iou']:.4f} | "
            f"coarse loss: {val_stats['coarse_loss']:.4f} | "
            f"final loss: {val_stats['final_loss']:.4f} | "
            f"boundary loss: {val_stats['boundary_loss']:.4f} | "
            f"uncertainty loss: {val_stats['uncertainty_loss']:.4f}"
        )

        if val_stats["dice"] > best_dice:
            best_dice = val_stats["dice"]
            torch.save(model.state_dict(), best_ckpt_path)

        torch.save(model.state_dict(), last_ckpt_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
