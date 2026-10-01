# BIND

**Binding 3D Robot Actions to 2D Image Features.**

BIND is a visuomotor policy that predicts a robot's future end-effector waypoints
as a probability distribution over a **discretized 3D world voxel grid**, scored
directly from DINOv3 image features across multiple camera views. The argmax voxel
per future timestep is the predicted target; gripper and rotation are read from the
same voxel. No language, no diffusion — a single image→volume forward pass.

This repo is a **minimal, self-contained reference** to train and evaluate BIND on
one LIBERO task (`libero_spatial` → *"pick up the black bowl from table center and
place it on the plate"*), with the same visualizations and closed-loop eval we use
internally.

> **TL;DR (train, no simulator needed):**
> ```bash
> conda env create -f environment.yml && conda activate bind
> export DINOV3_WEIGHTS=/path/to/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
> bash scripts/download_data.sh           # ~0.4 GB packaged dataset -> ./data
> python train.py --cache_root ./data --task_ids 2
> ```

## Two environments

The training stack (modern PyTorch) and the LIBERO simulator (older `numpy`/`robosuite`
pins) **cannot coexist in one env**, so there are two:

| env | file | used for |
|---|---|---|
| `bind` | `environment.yml` / `requirements.txt` | **training** (+ the packaged dataset) — no simulator |
| `bind-sim` | `environment-sim.yml` / `requirements-sim.txt` | `prerender.py` + `eval.py` (LIBERO sim) |

```bash
conda env create -f environment.yml        # training env 'bind'
conda env create -f environment-sim.yml     # simulator env 'bind-sim' (only if you prerender/eval)
```
Reference setup: Python 3.10, PyTorch 2.6.0 (CUDA 12.4), numpy 2.2.6 (train) / 1.26.4 (sim).
If `import torch` fails with `libcudnn.so.9 not found`: `pip install --force-reinstall nvidia-cudnn-cu12`.

**DINOv3 backbone.** Download the ViT-S/16+ checkpoint from Meta's official release
(<https://github.com/facebookresearch/dinov3>, license-gated) and point an env var at it:
```bash
export DINOV3_WEIGHTS=/path/to/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
export DINOV3_REPO=/path/to/dinov3   # optional local clone; else pulled from torch hub
```

**LIBERO** (only for `prerender.py` / `eval.py`) — install editable so its task files register:
```bash
conda activate bind-sim
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..
```

## Get the data

**Option A — packaged dataset (recommended).** The 2-view cache (scene+wrist frames +
camera calibration + EE trajectory, 49 demos, JPG, ~0.4 GB). No simulator required:
```bash
bash scripts/download_data.sh            # -> ./data/libero_spatial/task_2/
```

**Option B — regenerate from official LIBERO demos** (needs the `bind-sim` env + LIBERO):
```bash
python prerender.py --benchmark libero_spatial --task_ids 2 --out_root ./data
```

## Train

Training needs only the `bind` env + the data above (no simulator):
```bash
python train.py --cache_root ./data --task_ids 2
```
Defaults reproduce the reference run: scene+wrist, `n_window=8`, `img_size=448`,
lr `5e-5`, batch 16, 30 epochs. Checkpoints land in `./checkpoints/bind_bowl/latest.pth`.

**Visualization.** By default, panels dump as PNGs to `./viz_out/` every 500 steps —
for both a training batch and a held-out **val** batch:
- `heatmap_grid` — per-waypoint volume confidence, stacked `t=0..T-1`, scene row over wrist row.
- `kp_gt_vs_pred` — GT (green) vs argmax-predicted (red) waypoints on scene + wrist.
- `z_bins` — the per-view height-bin marginal.

Pass `--wandb` to log these (and loss / `val/xyz_err_mm`) to Weights & Biases instead.

## Evaluate (closed-loop)

Needs the `bind-sim` env (LIBERO):
```bash
python eval.py --checkpoint ./checkpoints/bind_bowl/latest.pth \
    --benchmark libero_spatial --task_id 2 --teleport --n_episodes 6 \
    --save_video_dir ./out --viz_rollout_dir ./out/panels
```
Runs a teleport-servo rollout in the LIBERO sim, prints the success rate, and (with the
flags above) saves per-episode rollout mp4s (`--save_video_dir`) and per-inference panels
with the 8-waypoint keypoints + per-timestep heatmaps (`--viz_rollout_dir`).

## How it works (one paragraph)

Each camera view is encoded by a shared DINOv3 backbone → a per-pixel feature map.
A world-space voxel grid (N views × Z height bins × P×P) is projected into every view
and each voxel gathers its multi-view features; a small per-voxel MLP fuses them into a
scalar score. For each of the `T` future waypoints a query vector scores the whole grid,
giving `volume_logits (B, T, N·Z, P, P)`. The training target is the voxel nearest the
ground-truth EE position (cross-entropy); gripper and rotation are classified from the
chosen voxel's feature. At inference the per-waypoint argmax voxel gives the 3D target
the controller servos to.

## Credits

- **DINOv3** — Meta AI (<https://github.com/facebookresearch/dinov3>).
- **LIBERO** — <https://github.com/Lifelong-Robot-Learning/LIBERO>.
- If you use this code, please cite the BIND paper.

MIT licensed.
