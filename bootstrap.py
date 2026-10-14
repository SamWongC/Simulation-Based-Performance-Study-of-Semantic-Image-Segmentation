# Block-bootstrap confidence intervals for the accumulated-count mIoU.
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from carla_labels import NUM_CLASSES

DOMAINS = [f"V{i}" for i in range(1, 17)]
BLOCKS = [("A in-dist", 0, 3), ("B weather", 3, 7), ("C map", 7, 10),
          ("D camera", 10, 13), ("E combined", 13, 16)]

# Reference scheme of the quantitative table; Block A rows have none.
# Delta intervals resample a domain and its reference with separate draws: they
# are different recordings with independent noise.
REFS = {**{d: ["V1"] for d in ("V4", "V5", "V6", "V7", "V11", "V12", "V13")},
        **{d: ["V1", "V2", "V3"] for d in ("V8", "V9", "V10")},
        "V14": ["V9"], "V15": ["V10"], "V16": ["V8"]}

# The two contrasts we read as ties in the report's limits paragraph.
CONTRASTS = (("V9", "V10"), ("V4", "V6"))


@torch.no_grad()
def per_frame_counts(model: torch.nn.Module, root: Union[str, Path],
                     device: torch.device, batch_size: int = 8
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame per-class intersection/union pixel counts, frames in
    chronological order (sorted filenames), as two (N, 8) int64 arrays."""
    ds = SegmentationDataset(root, split="val")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    inters, unions = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        pred  = model(images).argmax(1)
        valid = labels != 255
        tgt   = torch.where(valid, labels, 0)
        idx   = tgt * NUM_CLASSES + pred          # confusion-matrix bin per pixel
        for b in range(images.shape[0]):
            cm = torch.bincount(idx[b][valid[b]].flatten(),
                                minlength=NUM_CLASSES ** 2
                                ).reshape(NUM_CLASSES, NUM_CLASSES)
            inter = torch.diag(cm)
            union = cm.sum(0) + cm.sum(1) - inter
            inters.append(inter.cpu().numpy())
            unions.append(union.cpu().numpy())
    return np.array(inters, dtype=np.int64), np.array(unions, dtype=np.int64)


def miou(inter_sum: np.ndarray, union_sum: np.ndarray) -> float:
    # Same convention as evaluate_model: nanmean skips classes absent from the
    # whole sample (union == 0).
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union_sum > 0, inter_sum / union_sum, np.nan)
    return float(np.nanmean(iou))


def bootstrap_mious(inter: np.ndarray, union: np.ndarray,
                    rng: np.random.Generator,
                    block: int = 30, reps: int = 1000) -> np.ndarray:
    """`reps` moving-block-bootstrap replicates of the accumulated mIoU."""
    n         = len(inter)
    n_blocks  = int(np.ceil(n / block))
    max_start = max(n - block, 0)
    out = np.empty(reps)
    for r in range(reps):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        take   = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[r] = miou(inter[take].sum(0), union[take].sum(0))
    return out


def ci(samples: np.ndarray) -> Tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return lo, hi


def run_bootstrap(model: torch.nn.Module, device: torch.device,
                  data_root: str = "data/extracted",
                  domains: List[str] = DOMAINS,
                  block: int = 30, reps: int = 1000, seed: int = 0
                  ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    point, boot = {}, {}
    for dom in domains:
        inter, union = per_frame_counts(model, Path(data_root) / dom, device)
        point[dom] = miou(inter.sum(0), union.sum(0))
        boot[dom]  = bootstrap_mious(inter, union, rng, block, reps)
        print(f"{dom:<4s} evaluated: {len(inter)} frames, mIoU {point[dom]:.4f}")
    return point, boot


def print_summary(point: Dict[str, float], boot: Dict[str, np.ndarray],
                  block: int = 30, reps: int = 1000, seed: int = 0) -> None:
    print(f"\nMoving-block bootstrap: {block}-frame blocks, {reps} replicates, "
          f"seed {seed}")
    print(f"{'dom':<5s}{'mIoU':>8s}{'95% CI':>19s}{'Δ vs ref':>11s}"
          f"{'95% CI(Δ)':>21s}")
    print("-" * 64)
    half_widths = []
    for dom in point:
        lo, hi = ci(boot[dom])
        half_widths.append((hi - lo) / 2)
        line = f"{dom:<5s}{point[dom]:>8.4f}   [{lo:.4f}, {hi:.4f}]"
        if dom in REFS:
            ref_reps  = np.mean([boot[r] for r in REFS[dom]], axis=0)
            ref_point = np.mean([point[r] for r in REFS[dom]])
            dlo, dhi  = ci(boot[dom] - ref_reps)
            line += (f"{point[dom] - ref_point:>+11.4f}   "
                     f"[{dlo:+.4f}, {dhi:+.4f}]")
        print(line)
    print(f"\nMedian CI half-width across domains: "
          f"±{np.median(half_widths):.4f} (max ±{np.max(half_widths):.4f})")

    print("\nPairwise contrasts from the analysis text (independent "
          "resampling):")
    for a, b in CONTRASTS:
        dlo, dhi = ci(boot[a] - boot[b])
        verdict = "excludes 0" if dlo > 0 or dhi < 0 else "includes 0"
        print(f"  {a} − {b}: {point[a] - point[b]:+.4f}  "
              f"[{dlo:+.4f}, {dhi:+.4f}]  → {verdict}")


def plot_intervals(point: Dict[str, float], boot: Dict[str, np.ndarray],
                   domains: List[str] = DOMAINS) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(domains), dtype=float)
    for _, s, e in BLOCKS[1:]:
        x[s:] += 0.8

    fig, ax = plt.subplots(figsize=(12, 4))
    y    = np.array([point[d] for d in domains])
    yerr = np.array([[point[d] - ci(boot[d])[0] for d in domains],
                     [ci(boot[d])[1] - point[d] for d in domains]])
    ax.errorbar(x, y, yerr=yerr, fmt="o", color="#2a78d6", markersize=7,
                capsize=3, lw=1.5, zorder=3)
    for label, s, e in BLOCKS:
        ax.text((x[s] + x[e - 1]) / 2, 0.88, label, ha="center", va="bottom",
                fontsize=10, color="#555555", fontweight="bold")
        if s > 0:
            ax.axvline((x[s] + x[s - 1]) / 2, color="#cccccc", lw=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 0.95)
    ax.set_title("Baseline mIoU with 95% block-bootstrap intervals")
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.show()
