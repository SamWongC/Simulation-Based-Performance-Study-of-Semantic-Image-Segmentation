"""PyTorch dataset, transforms and class weights for the extracted CARLA image/label pairs."""
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
 
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

AUG_VARIANTS = ("photo", "geo", "full")

def get_transforms(height: int, width: int,
                   augment: Union[bool, str] = False) -> A.Compose:
    """Build the albumentations pipeline shared by all splits.

    Resizes images to (height, width) with INTER_AREA, which suits downscaling, and
    applies ImageNet normalisation. `augment` selects a training-time variant; each
    variant adds a horizontal flip:

    - False: raw baseline (resize + normalise)
    - "photo": strong darkening and colour jitter, targeting weather/lighting
      (Block B); darkened noon approximates night without street lights, headlights
      or night sky
    - "geo": shift/scale/rotate with 255-filled mask borders, targeting camera
      geometry (Block D); a 2-D warp contains no real parallax
    - "full": photo and geo combined
    """
    variant = None if augment is False else augment
    if variant is not None and variant not in AUG_VARIANTS:
        raise ValueError(f"augment must be False or one of {AUG_VARIANTS}, "
                         f"got {augment!r}")

    ops = [A.Resize(height, width, interpolation=cv2.INTER_AREA)]
    if variant is not None:
        ops.append(A.HorizontalFlip(p=0.5))
    if variant in ("geo", "full"):
        ops.append(A.ShiftScaleRotate(
            shift_limit_x=0.05, shift_limit_y=0.15,
            scale_limit=(-0.35, 0.5), rotate_limit=3,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255,
            p=0.9))
    if variant in ("photo", "full"):
        ops += [
            A.ColorJitter(brightness=(0.2, 1.2), contrast=(0.6, 1.2),
                          saturation=0.2, hue=0.05, p=0.8),
            A.GaussNoise(var_limit=(5, 40), p=0.3),
        ]
    ops += [
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]
    return A.Compose(ops)


def compute_class_weights(label_dir: Union[str, Path], num_classes: int,
                          c: float = 1.02) -> torch.Tensor:
    """ENet-style log-frequency class weights for CrossEntropyLoss.

    w_c = 1 / ln(c + f_c), where f_c is the pixel frequency of class c over all
    label PNGs in `label_dir`.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for p in sorted(Path(label_dir).glob("*.png")):
        a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if a is None:
            raise RuntimeError(f"Could not read label: {p}")
        counts += np.bincount(a.ravel(), minlength=256)[:num_classes]
    freq = counts / counts.sum()
    weights = 1.0 / np.log(c + freq)
    return torch.tensor(weights, dtype=torch.float32)


class SegmentationDataset(Dataset):
    """Image/label PNG pairs from a folder containing images/ and labels/.

    data_extraction.py and prepare_datasets.py write these folders. Augmentation applies
    to split == "train"; other splits get resize and normalisation. Items are
    (float32 image (3, H, W), int64 label (H, W)).
    """
    def __init__(self,
                 root: Union[str, Path],
                 split: str = "train",
                 height: int=256,
                 width: int=512,
                 augment: Union[bool, str] = False,
                 ignore_index=255):
        
        self.root = Path(root)
        self.split = split
        self.ignore_index = ignore_index

        self.img_dir = self.root / "images"
        self.label_dir = self.root / "labels"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"images/ not found under {self.root}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"labels/ not found under {self.root}")
        
        self.img_paths   = sorted(self.img_dir.glob("*.png"))
        self.label_paths = sorted(self.label_dir.glob("*.png"))

        if len(self.img_paths) != len(self.label_paths):
            raise RuntimeError(
                f"Mismatch: {len(self.img_paths)} images vs "
                f"{len(self.label_paths)} labels in {self.root}"
            )
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No PNG files found in {self.img_dir}")
        use_aug = augment if split == "train" else False
        self.transforms = get_transforms(height, width, augment=use_aug)

    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = cv2.imread(str(self.img_paths[idx]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {self.img_paths[idx]}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        # Load label (stored as uint8 class IDs)
        label = cv2.imread(str(self.label_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise RuntimeError(f"Could not read label: {self.label_paths[idx]}")
 
        # Apply transforms 
        transformed = self.transforms(image=img, mask=label)
        image_t = transformed["image"]                          # float32 (C, H, W)
        label_t = transformed["mask"].long()                    # int64  (H, W)
 
        return image_t, label_t
    
    def __repr__(self) -> str:
        return (f"SegmentationDataset(root={self.root}, split={self.split}, "
                f"n={len(self)})")


def build_dataloaders(
    train_dataset: SegmentationDataset,
    val_datasets:  Dict[str, SegmentationDataset],
    batch_size:    int = 8,
    num_workers:   int = 4,
    pin_memory:    bool = True,
) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """Build a shuffled training DataLoader that drops the incomplete last batch, and one
    validation DataLoader (batch size 1) per entry of `val_datasets`."""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
 
    val_loaders = {
        name: DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for name, ds in val_datasets.items()
    }
 
    return train_loader, val_loaders
 
        