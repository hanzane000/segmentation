import os
import argparse
from datetime import datetime

import cv2
import numpy as np
import torch

from data.data_C import build_dataloaders
from models.unet_C import UNet
from utils.metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score,
    pixel_accuracy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multi-branch refinement segmentation model")

    parser.add_argument("--data_root", default="dataset")
    parser.add_argument("--img_dir", default="")
    parser.add_argument("--mask_dir", default="")
    parser.add_argument("--weights", default="checkpoints/C/best_model.pth")
    parser.add_argument("--save_dir", default="predictions/C")
    parser.add_argument("--visualize_dir", default="")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fg_crop_prob", type=float, default=0.7)
    parser.add_argument("--val_resize_short", type=int, default=512)

    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--bilinear", type=int, default=1)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_aux", type=int, default=1)

    return parser.parse_args()


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_visualize_image(stem, visualize_dir, h, w):
    for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        path = os.path.join(visualize_dir, stem + ext)
        if os.path.exists(path):
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
                return image
    return np.zeros((h, w, 3), dtype=np.uint8)


def colorize_mask(mask_uint8):
    return np.stack([mask_uint8, mask_uint8, mask_uint8], axis=-1)


def overlay_mask(image_rgb, mask_uint8, alpha=0.5, color=(255, 0, 0)):
    overlay = image_rgb.copy()
    binary = mask_uint8 > 0
    overlay[binary] = color
    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, overlay, alpha, 0)
    return blended


def save_predictions(outputs, gt_masks, stems, save_dir, visualize_dir, threshold=0.5, save_aux=True):
    final_logits = outputs["final_logits"]
    coarse_logits = outputs["coarse_logits"]
    boundary_logits = outputs["boundary_logits"]
    uncertainty_map = outputs["uncertainty_map"]

    final_prob = torch.sigmoid(final_logits)
    coarse_prob = torch.sigmoid(coarse_logits)
    boundary_prob = torch.sigmoid(boundary_logits)

    final_pred = (final_prob > threshold).float()
    coarse_pred = (coarse_prob > threshold).float()

    final_pred = final_pred.detach().cpu().numpy()
    coarse_pred = coarse_pred.detach().cpu().numpy()
    boundary_prob = boundary_prob.detach().cpu().numpy()
    uncertainty_map = uncertainty_map.detach().cpu().numpy()
    gt_masks = gt_masks.detach().cpu().numpy()

    for i, stem in enumerate(stems):
        final_mask = (final_pred[i, 0] * 255).astype(np.uint8)
        coarse_mask = (coarse_pred[i, 0] * 255).astype(np.uint8)
        boundary_map = (boundary_prob[i, 0] * 255).astype(np.uint8)
        uncertainty_vis = (uncertainty_map[i, 0] * 255).astype(np.uint8)
        gt_mask = (gt_masks[i, 0] * 255).astype(np.uint8)

        h, w = gt_mask.shape
        vis_image = load_visualize_image(stem, visualize_dir, h, w)
        final_overlay = overlay_mask(vis_image, final_mask, alpha=0.5, color=(255, 0, 0))

        final_path = os.path.join(save_dir, f"{stem}_final.png")
        cv2.imwrite(final_path, final_mask)

        if save_aux:
            coarse_path = os.path.join(save_dir, f"{stem}_coarse.png")
            boundary_path = os.path.join(save_dir, f"{stem}_boundary.png")
            uncertainty_path = os.path.join(save_dir, f"{stem}_uncertainty.png")
            cv2.imwrite(coarse_path, coarse_mask)
            cv2.imwrite(boundary_path, boundary_map)
            cv2.imwrite(uncertainty_path, uncertainty_vis)

        gt_rgb = colorize_mask(gt_mask)
        final_rgb = colorize_mask(final_mask)
        top_row = np.concatenate([gt_rgb, vis_image], axis=1)
        bottom_row = np.concatenate([final_rgb, final_overlay], axis=1)
        grid = np.concatenate([top_row, bottom_row], axis=0)
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        grid_path = os.path.join(save_dir, f"{stem}_grid.png")
        cv2.imwrite(grid_path, grid_bgr)


def evaluate(model, val_loader, device, save_dir, visualize_dir, threshold=0.5, save_aux=True):
    model.eval()

    dice_sum = 0.0
    iou_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    pixel_acc_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, seg_masks, boundary_masks, uncertainty_masks, stems in val_loader:
            images = images.to(device, non_blocking=True)
            seg_masks = seg_masks.to(device, non_blocking=True)

            outputs = model(images)
            final_logits = outputs["final_logits"]
            batch_size = images.size(0)

            dice_sum += dice_score(final_logits, seg_masks, threshold=threshold) * batch_size
            iou_sum += iou_score(final_logits, seg_masks, threshold=threshold) * batch_size
            precision_sum += precision_score(final_logits, seg_masks, threshold=threshold) * batch_size
            recall_sum += recall_score(final_logits, seg_masks, threshold=threshold) * batch_size
            pixel_acc_sum += pixel_accuracy(final_logits, seg_masks, threshold=threshold) * batch_size
            total_samples += batch_size

            save_predictions(
                outputs=outputs,
                gt_masks=seg_masks,
                stems=stems,
                save_dir=save_dir,
                visualize_dir=visualize_dir,
                threshold=threshold,
                save_aux=save_aux,
            )

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
    visualize_dir = args.visualize_dir if args.visualize_dir else os.path.join(args.data_root, "visualize")

    os.makedirs(args.save_dir, exist_ok=True)
    parent_dir = os.path.dirname(args.save_dir)
    if parent_dir == "":
        parent_dir = "."
    result_txt_path = os.path.join(parent_dir, f"{os.path.basename(args.save_dir)}.txt")

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

    checkpoint = torch.load(args.weights, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    metrics = evaluate(
        model=model,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_dir,
        visualize_dir=visualize_dir,
        threshold=args.threshold,
        save_aux=bool(args.save_aux),
    )

    print(f"dice: {metrics['dice']:.4f}")
    print(f"iou: {metrics['iou']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"pixel accuracy: {metrics['pixel_accuracy']:.4f}")

    with open(result_txt_path, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"weights: {args.weights}\n")
        f.write(f"save_dir: {args.save_dir}\n")
        f.write(f"threshold: {args.threshold}\n")
        f.write(f"val_samples: {len(val_stems)}\n")
        f.write(f"dice: {metrics['dice']:.6f}\n")
        f.write(f"iou: {metrics['iou']:.6f}\n")
        f.write(f"precision: {metrics['precision']:.6f}\n")
        f.write(f"recall: {metrics['recall']:.6f}\n")
        f.write(f"pixel_accuracy: {metrics['pixel_accuracy']:.6f}\n")


if __name__ == "__main__":
    main()
