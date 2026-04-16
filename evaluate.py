import os
import argparse

import cv2
import numpy as np
import torch

from data.data import build_dataloaders
from models.unet import UNet
from utils.metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score,
    pixel_accuracy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate binary segmentation model")

    # Paths
    parser.add_argument("--data_root", default="dataset")
    parser.add_argument("--img_dir", default="")
    parser.add_argument("--mask_dir", default="")
    parser.add_argument("--weights", default="checkpoints/best_model.pth")
    parser.add_argument("--save_dir", default="predictions")

    # Data loader
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fg_crop_prob", type=float, default=0.7)
    parser.add_argument("--val_resize_short", type=int, default=512)

    # Model
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--bilinear", type=int, default=1)

    # Eval
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)

    return parser.parse_args()


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def save_predictions(logits, stems, save_dir, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.detach().cpu().numpy()

    for i, stem in enumerate(stems):
        mask = preds[i, 0]
        mask = (mask * 255).astype(np.uint8)
        out_path = os.path.join(save_dir, f"{stem}.png")
        cv2.imwrite(out_path, mask)


def evaluate(model, val_loader, device, save_dir, threshold=0.5):
    model.eval()

    dice_sum = 0.0
    iou_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    pixel_acc_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks, stems in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            batch_size = images.size(0)

            dice_sum += dice_score(logits, masks, threshold=threshold) * batch_size
            iou_sum += iou_score(logits, masks, threshold=threshold) * batch_size
            precision_sum += precision_score(logits, masks, threshold=threshold) * batch_size
            recall_sum += recall_score(logits, masks, threshold=threshold) * batch_size
            pixel_acc_sum += pixel_accuracy(logits, masks, threshold=threshold) * batch_size
            total_samples += batch_size

            save_predictions(logits, stems, save_dir, threshold=threshold)

    metrics = {
        "dice": dice_sum / total_samples,
        "iou": iou_sum / total_samples,
        "precision": precision_sum / total_samples,
        "recall": recall_sum / total_samples,
        "pixel_accuracy": pixel_acc_sum / total_samples,
    }
    return metrics


def main():
    args = parse_args()

    img_dir = args.img_dir if args.img_dir else os.path.join(args.data_root, "images")
    mask_dir = args.mask_dir if args.mask_dir else os.path.join(args.data_root, "masks")

    os.makedirs(args.save_dir, exist_ok=True)

    device = get_device(args.device)
    print(f"Device: {device}")

    _, val_loader, _, val_stems = build_dataloaders(
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

    print(f"Val samples: {len(val_stems)}")

    model = UNet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        bilinear=bool(args.bilinear),
    ).to(device)

    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)

    metrics = evaluate(
        model=model,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_dir,
        threshold=args.threshold,
    )

    print(f"dice: {metrics['dice']:.4f}")
    print(f"iou: {metrics['iou']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"pixel accuracy: {metrics['pixel_accuracy']:.4f}")


if __name__ == "__main__":
    main()
