"""Merge the extracted training recordings T1-T3 into data/train and data/val.

The last 15% of each town (contiguous in time) becomes the validation split for
checkpoint selection. Run it from the topic directory after extracting T1-T3 with
data_extraction.py; it replaces existing data/train and data/val.
"""
import shutil
from pathlib import Path

TRAIN_SOURCES = [
    Path("data/extracted/T1"),
    Path("data/extracted/T2"),
    Path("data/extracted/T3"),
]
TRAIN_OUT = Path("data/train")
VAL_OUT   = Path("data/val")
VAL_FRACTION = 0.15   # last 15% of each town (contiguous in time) = validation


def _reset(out: Path):
    """Delete `out` if it exists and recreate empty images/ and labels/ folders."""
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)


def _copy_pair(img: Path, lbl: Path, out: Path, idx: int):
    """Copy one image/label pair into `out` under the running index `idx`."""
    fname = f"{idx:06d}.png"
    shutil.copy(img, out / "images" / fname)
    shutil.copy(lbl, out / "labels" / fname)


def split_and_merge(sources, train_out, val_out, val_fraction):
    """Split each source into a leading training part and a trailing `val_fraction`
    validation part, and merge all sources with filenames unique across them."""
    _reset(train_out)
    _reset(val_out)

    train_idx = val_idx = 0
    for src in sources:
        imgs   = sorted((src / "images").glob("*.png"))
        labels = sorted((src / "labels").glob("*.png"))
        assert len(imgs) == len(labels), f"Mismatch in {src}"

        n = len(imgs)
        n_val   = int(round(n * val_fraction))
        n_train = n - n_val
        for i, (img, lbl) in enumerate(zip(imgs, labels)):
            if i < n_train:                      # train
                _copy_pair(img, lbl, train_out, train_idx); train_idx += 1
            else:                                # val
                _copy_pair(img, lbl, val_out, val_idx); val_idx += 1
        print(f"  {src.name}: {n} frames -> {n_train} train, {n_val} val")

    print(f"Train total: {train_idx} frames -> {train_out}")
    print(f"Val   total: {val_idx} frames -> {val_out}")


if __name__ == "__main__":
    missing = [s for s in TRAIN_SOURCES if not s.exists()]
    if missing:
        print("Missing training sources (extract the MCAP files first):")
        for m in missing:
            print(f"  {m}")
    else:
        print("Splitting + merging training data")
        split_and_merge(TRAIN_SOURCES, TRAIN_OUT, VAL_OUT, VAL_FRACTION)
