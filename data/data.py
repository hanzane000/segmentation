"""
数据加载和处理模块
用于图像分割任务的数据增强、裁剪和数据加载
"""

import os
import random
from typing import List, Tuple, Optional

import cv2
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader


def split_dataset(stems: List[str], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    stems = stems.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)
    val_count = int(round(len(stems) * val_ratio))
    return stems[val_count:], stems[:val_count]


def load_image_mask(img_dir: str, mask_dir: str, stem: str) -> Tuple[np.ndarray, np.ndarray]:
    img_path = os.path.join(img_dir, stem + ".jpg")
    mask_path = os.path.join(mask_dir, stem + ".png")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图像文件不存在: {img_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask 文件不存在: {mask_path}")

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像文件: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取 Mask 文件: {mask_path}")

    return image, mask


def pad_if_needed(image: np.ndarray, mask: np.ndarray, crop_size: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)

    if pad_h == 0 and pad_w == 0:
        return image, mask

    image = cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    mask = cv2.copyMakeBorder(
        mask, 0, pad_h, 0, pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    return image, mask


def random_crop(image: np.ndarray, mask: np.ndarray, crop_size: int) -> Tuple[np.ndarray, np.ndarray]:
    image, mask = pad_if_needed(image, mask, crop_size)
    h, w = image.shape[:2]
    y1 = random.randint(0, h - crop_size)
    x1 = random.randint(0, w - crop_size)

    image_crop = image[y1:y1 + crop_size, x1:x1 + crop_size]
    mask_crop = mask[y1:y1 + crop_size, x1:x1 + crop_size]
    return image_crop, mask_crop


def foreground_aware_crop(image: np.ndarray, mask: np.ndarray, crop_size: int) -> Tuple[np.ndarray, np.ndarray]:
    image, mask = pad_if_needed(image, mask, crop_size)
    h, w = image.shape[:2]

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return random_crop(image, mask, crop_size)

    idx = random.randint(0, len(xs) - 1)
    jitter = crop_size // 8

    x1 = max(
        0,
        min(xs[idx] - crop_size // 2 + random.randint(-jitter, jitter), w - crop_size)
    )
    y1 = max(
        0,
        min(ys[idx] - crop_size // 2 + random.randint(-jitter, jitter), h - crop_size)
    )

    image_crop = image[y1:y1 + crop_size, x1:x1 + crop_size]
    mask_crop = mask[y1:y1 + crop_size, x1:x1 + crop_size]
    return image_crop, mask_crop


def resize_short_side(image: np.ndarray, mask: np.ndarray, target_short: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    scale = target_short / min(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return image_resized, mask_resized


def resize_to_fixed_size(image: np.ndarray, mask: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    将图像和 mask 直接缩放到固定尺寸。
    用于验证阶段，替代原来的 center crop，保证 DataLoader 可正常 batch。
    """
    image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return image_resized, mask_resized


def build_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(
            std_range=(0.05, 0.15),
            mean_range=(0.0, 0.0),
            p=0.3
        ),
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2
        ),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-8, 8),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.5
        ),
    ])


def build_val_transform():
    return A.Compose([])


class TunnelDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        file_list: List[str],
        crop_size: int,
        mode: str = "train",
        train_transform: Optional[A.Compose] = None,
        val_transform: Optional[A.Compose] = None,
        fg_crop_prob: float = 0.7,
        val_resize_short: Optional[int] = None,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.crop_size = crop_size
        self.mode = mode

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.fg_crop_prob = fg_crop_prob

        # 为兼容原有接口，保留 val_resize_short 参数
        # 现在它表示验证阶段最终输出尺寸；若不指定则默认使用 crop_size
        self.val_output_size = val_resize_short or crop_size

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        stem = self.file_list[idx]
        image, mask = load_image_mask(self.img_dir, self.mask_dir, stem)

        if self.mode == "train":
            if (mask > 0).any() and random.random() < self.fg_crop_prob:
                image, mask = foreground_aware_crop(image, mask, self.crop_size)
            else:
                image, mask = random_crop(image, mask, self.crop_size)

            if self.train_transform is not None:
                result = self.train_transform(image=image, mask=mask)
                image, mask = result["image"], result["mask"]

        else:
            # 验证阶段去掉 center crop，直接缩放到固定尺寸
            image, mask = resize_to_fixed_size(image, mask, self.val_output_size)

            if self.val_transform is not None:
                result = self.val_transform(image=image, mask=mask)
                image, mask = result["image"], result["mask"]

        # 归一化
        image = image.astype(np.float32) / 255.0

        # mask 二值化：前景只要大于 0 就视为有效区域
        mask = (mask > 0).astype(np.float32)

        # 显式转为 torch.Tensor
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask_tensor = torch.from_numpy(np.expand_dims(mask, axis=0)).float()

        return image_tensor, mask_tensor, stem


def build_dataloaders(
    img_dir: str,
    mask_dir: str,
    crop_size: int,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    fg_crop_prob: float,
    val_resize_short: Optional[int] = None,
):
    stems = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(img_dir)
        if f.lower().endswith(".jpg")
    ])

    train_stems, val_stems = split_dataset(
        stems=stems,
        val_ratio=val_ratio,
        seed=seed
    )

    train_dataset = TunnelDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        file_list=train_stems,
        crop_size=crop_size,
        mode="train",
        train_transform=build_train_transform(),
        val_transform=None,
        fg_crop_prob=fg_crop_prob,
        val_resize_short=val_resize_short,
    )

    val_dataset = TunnelDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        file_list=val_stems,
        crop_size=crop_size,
        mode="val",
        train_transform=None,
        val_transform=build_val_transform(),
        fg_crop_prob=fg_crop_prob,
        val_resize_short=val_resize_short,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_stems, val_stems