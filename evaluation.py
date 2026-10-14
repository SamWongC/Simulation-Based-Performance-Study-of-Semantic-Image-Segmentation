"""mIoU and per-class IoU for a trained model. Run it as a script to evaluate one
checkpoint on one dataset folder."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from dataset import SegmentationDataset
from model import LightUNet
from carla_labels import REDUCED_CLASSES

def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-class intersection and union pixel counts for one batch, skipping ignore pixels.

    The caller accumulates the counts over a whole domain and divides once.
    """
    pred = pred.argmax(dim=1)  # (B, H, W)
    valid = target != ignore_index  # drop void pixels from every class's counts

    intersection = np.zeros(num_classes, dtype=np.int64)
    union        = np.zeros(num_classes, dtype=np.int64)

    for cls in range(num_classes):
        pred_mask   = (pred == cls) & valid
        target_mask = (target == cls) & valid

        inter = (pred_mask & target_mask).sum().item()
        # union via inclusion-exclusion: |A| + |B| - |A∩B|
        uni   = pred_mask.sum().item() + target_mask.sum().item() - inter

        intersection[cls] = inter
        union[cls]        = uni

    return intersection, union


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, 
                   dataloader: DataLoader, 
                   num_classes: int = 8, 
                   device: str = "cuda") -> Dict[str, float]:
    """Evaluate `model` on a whole dataloader.

    Sums intersections and unions over all batches before dividing; a class with an
    empty union gets NaN and drops out of the mean. Returns
    {"mIoU": float, "per_class_iou": {class name: float}}.
    """
    
    model.eval()

    # Accumulate counts over the whole dataset, then divide once.
    total_inter = np.zeros(num_classes, dtype=np.int64)
    total_union = np.zeros(num_classes, dtype=np.int64)

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)                    # (B, num_classes, H, W)
        inter, union = compute_iou(outputs, labels, num_classes)
        total_inter += inter
        total_union += union

    # A class with union == 0 appears in neither the labels nor the predictions of
    # the domain. NaN keeps it out of nanmean; a 0 would drag the mean down.
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_iou = np.where(total_union > 0, total_inter / total_union, np.nan)
    mIoU = float(np.nanmean(per_class_iou))        # overall mIoU over present classes

    results = {
        "mIoU": mIoU,
        "per_class_iou": {REDUCED_CLASSES[i]: float(per_class_iou[i]) for i in range(num_classes)}
    }

    return results


def main():
    """Evaluate one checkpoint on one dataset folder; print the results and save them
    to <data>/evaluation_results.txt."""
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data", type=str, required=True, help="Path to validation dataset folder")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Load model
    model = LightUNet(num_classes=8).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    print(f"✅ Model loaded from {args.model}")
    
    # Load validation dataset
    val_dataset = SegmentationDataset(root=args.data, split="val")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Evaluating on dataset: {args.data} ({len(val_dataset)} samples)")
    
    # evaluation
    results = evaluate_model(model, val_loader, num_classes=8, device=device)
    
    # print
    print("\n" + "="*30)
    print("Evaluation Results")
    print("="*30)
    print(f"Overall mIoU: {results['mIoU']:.4f}\n")
    print("Per-class IoU:")
    for cls_name, iou in results['per_class_iou'].items():
        print(f"  {cls_name:12s}: {iou:.4f}")
    print("="*30)

    # save
    save_path = Path(args.data) / "evaluation_results.txt"
    with open(save_path, "w") as f:
        f.write(f"mIoU: {results['mIoU']:.4f}\n")
        for cls_name, iou in results['per_class_iou'].items():
            f.write(f"{cls_name}: {iou:.4f}\n")
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()