"""Train LightUNet from the command line.

The defaults reproduce the report's recipe (50 epochs, batch 8, Adam 1e-3,
log-frequency class weights), and the script keeps the checkpoint with the best mIoU
on the held-out validation split. Train the report's augmentation variants with, e.g.::

    python train.py --data data/train --augment photo --save-dir checkpoints_photo
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

from dataset import SegmentationDataset, build_dataloaders, compute_class_weights
from model import LightUNet
from evaluation import evaluate_model

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Run one training epoch and return the mean batch loss."""
    model.train()
    total_loss = 0.0  # summed over the epoch, averaged on return
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device) # move the batch to the device
        
        optimizer.zero_grad()
        outputs = model(images) # (B, num_classes, H, W) forward
        
        loss = criterion(outputs, labels)
        loss.backward() # backpropagate
        optimizer.step() # update the weights
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader) # mean batch loss


def main():
    """Train, selecting on validation mIoU; writes best_model.pth and last_model.pth to --save-dir."""
    parser = argparse.ArgumentParser(description="Train semantic segmentation model")
    parser.add_argument("--data", type=str, required=True, 
                        help="Path to training dataset (contains images/ and labels/)")
    parser.add_argument("--val-data", type=str, nargs="+", default=[],
                        help="Extra sets to monitor each epoch (NOT used for selection)")
    parser.add_argument("--id-val", type=str, default="data/val",
                        help="In-distribution validation split used for checkpoint selection")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--augment", type=str, default=None,
                        choices=("photo", "geo", "full"),
                        help="Training-time augmentation variant (default: off, raw baseline)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint to resume training")
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # dataset
    train_dataset = SegmentationDataset(root=args.data, split="train", augment=args.augment) # load training datasets
    
    val_datasets = {} # optional monitoring sets (--val-data)
    for path in args.val_data:
        name = Path(path).name
        val_datasets[name] = SegmentationDataset(root=path, split="val")
    
    train_loader, val_loaders = build_dataloaders(
        train_dataset=train_dataset,
        val_datasets=val_datasets,
        batch_size=args.batch_size,
        num_workers=0,       
        pin_memory=False,
    ) # loaders for training and the monitoring sets

    # In-distribution validation split.
    id_val_dataset = SegmentationDataset(root=args.id_val, split="val")
    id_val_loader  = DataLoader(id_val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0, pin_memory=False)

    print(f"Train samples: {len(train_dataset)}  |  in-dist val: {len(id_val_dataset)}")
    print(f"Monitor sets: {list(val_datasets.keys())}")

    # Model
    model = LightUNet(num_classes=args.num_classes).to(device) 
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Resumed from {args.resume}")

    # Loss & Optimizer
    class_weights = compute_class_weights(Path(args.data) / "labels", args.num_classes).to(device)
    print("Class weights:", [round(w, 2) for w in class_weights.tolist()])
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # Adam

    best_mIoU = 0.0
    best_epoch = 0

    # Training Loop
    for epoch in range(1, args.epochs + 1): # Default 50
        print(f"\nEpoch {epoch}/{args.epochs}") # epoch header
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Checkpoint selection on the held-out in-distribution val split.
        id_val = evaluate_model(model, id_val_loader, num_classes=args.num_classes, device=device)
        print(f"  in-dist val mIoU: {id_val['mIoU']:.4f}")

        # Monitoring on the extra sets; checkpoint selection ignores them.
        for name, loader in val_loaders.items():
            r = evaluate_model(model, loader, num_classes=args.num_classes, device=device)
            print(f"  [monitor] {name:12s} mIoU: {r['mIoU']:.4f}")

        if id_val["mIoU"] > best_mIoU:
            best_mIoU = id_val["mIoU"]
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mIoU': best_mIoU,
            }, save_dir / "best_model.pth")
            print(f"✅ New best model saved! val mIoU = {best_mIoU:.4f}")

    # Save the final epoch, so a run without validation data still leaves a checkpoint.
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mIoU': best_mIoU,
    }, save_dir / "last_model.pth")

    print(f"\nTraining finished! Best mIoU: {best_mIoU:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()