# Simulation-Based Performance Study of Semantic Image Segmentation

### :arrow_forward: [Task Description](task.ipynb)

## Execution Environment

This notebook is developed for the ACDC Research Project Docker container (Jupyter, PyTorch, and Python 3 pre-installed). The offline pipeline (dataset generation, training, evaluation, all sections up to and excluding "ROS 2 Integration") needs no ROS installation and runs in any Python 3 environment with the packages listed in `report.ipynb`'s Dependencies cell. The ROS 2 sections require the project's ROS 2 Jazzy container.

## Downloads

Every code cell below the "Synthetic Dataset Generation" section in `report.ipynb` reads from a local `data/` directory (~18 GB: `data/train`, `data/val`, `data/extracted/V1`-`V16`) and from four checkpoint directories (~835 MB total: `checkpoints/`, `checkpoints_photo/`, `checkpoints_geo/`, `checkpoints_full/`). Both are excluded from the Git repository via `.gitignore` because of their size, per the template's guidance for data over 50 MB.

**For readers with access to the RWTH ACDC-RP filesystem**, the fastest route is to symlink the already-processed data and checkpoints directly:

```bash
SRC=/work/acdc-research-project-ss26/acdc-rp-buriachenko/acdc-research-project-ss26/topics/14-Simulation-Based-Performance-Study-of-Semantic-Image-Segmentation
for d in data checkpoints checkpoints_photo checkpoints_geo checkpoints_full; do
    ln -s "$SRC/$d" "<your-topic-directory>/$d"
done
```

Verify with `du -sh data` (expect ~18 GB). All four checkpoint directories are needed: `checkpoints/` holds the baseline, the other three the augmentation variants compared in the Evaluation section.

The checkpoint directories also hold cached evaluation results (`eval_16_domains.json`, `bootstrap_cache.npz`, `darkening_sweep.json`), so the evaluation cells finish in seconds instead of about 50 minutes. A cache is only used while it is newer than its checkpoint. Delete it to force a recomputation.

**To regenerate `data/` from scratch**, the raw CARLAB recordings (`.mcap`, one per simulation run) live under `/work/carlab/users/acdc-rp-buriachenko/data/<job-id>/<job-id>_0.mcap` on the same filesystem. The table below maps each job ID to the domain it produced (`veh_occupancy=0.4`, `veh_num=15`, `spawn_point=1` for training and Blocks B/D/E, `spawn_point=70` for Block A):

| Job ID | Domain(s) |
|---|---|
| `2026-06-09_10-49-12_0/1/2` | T1 (Town10HD_Opt), T2 (Town01), T3 (Town05) — training, 30 min each |
| `2026-06-09_13-06-17_0/1/2` | V1, V2, V3 — Block A |
| `2026-06-10_08-53-40_0/1/3`, `2026-06-10_09-34-11_0` | V4, V5, V7, V6 — Block B |
| `2026-06-10_09-58-03_0/1/2` | V8, V9, V10 — Block C |
| `2026-06-10_13-42-00_0`, `2026-06-10_13-58-00_0`, `2026-06-10_14-21-40_0` | V11, V12, V13 — Block D (with `sensors-V11-low-mount.json`, `sensors-V12-pitch-down.json`, `sensors-V13-wide-fov.json` respectively) |
| `2026-06-10_15-06-53_0`, `2026-06-10_15-30-15_0`, `2026-06-10_17-20-26_0` | V14, V15, V16 — Block E |

Extract each with `data_extraction.py --input <path>.mcap --output data/extracted/<ID> --every-nth 10` (Experiments, Section 2), then merge the three training recordings with `prepare_datasets.py`. Trained checkpoints are not regenerated this way and must come from the symlink above or from the training cells in Experiments, Section 3 (the baseline, then the three augmentation variants via `train.py`).
