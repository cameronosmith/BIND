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

<p align="center"><em>per-waypoint volume heatmaps (scene / wrist) + GT-vs-pred keypoints + closed-loop rollout</em></p>

## Environment

Reference setup (what this was built and tested on):

| | |
|---|---|
| Python | 3.10 |
| PyTorch | 2.6.0 (CUDA 12.4) · torchvision 0.21.0 |
| numpy / opencv / scipy | 2.2.6 / 4.13.0.92 / 1.15.3 |
| simulator (prerender + eval) | robosuite 1.4.0 · LIBERO 0.1.1 (pulls a compatible MuJoCo) |

**Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate bind
```

**Or pip** (into a Python 3.10 venv):

```bash
pip install -r requirements.txt
```

Both pin PyTorch 2.6.0 for CUDA 12.4 via the public PyTorch index; adjust the
`--extra-index-url` (e.g. `cu121`, `cpu`) if your CUDA differs. Training needs only
the core deps; `robosuite` + `LIBERO` are required for `prerender.py` and `eval.py`.

**LIBERO** (demos + task `.bddl` files) — install from git:

```bash
pip install "git+https://github.com/Lifelong-Robot-Learning/LIBERO.git"
```

**DINOv3 backbone.** The model loads DINOv3 ViT-S/16+ via `torch.hub`. Download the
checkpoint from Meta's official release (<https://github.com/facebookresearch/dinov3>,
license-gated) and point an env var at it:

```bash
export DINOV3_WEIGHTS=/path/to/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
# optional: a local clone of the dinov3 repo (otherwise pulled from the hub)
export DINOV3_REPO=/path/to/dinov3
```

## Get the data

The 2-view cache (per-demo scene+wrist frames + camera calibration + EE trajectory)
is built from the official LIBERO demos:

```bash
python prerender.py --benchmark libero_spatial --task_ids 2 --out_root ./data
```

This writes `./data/libero_spatial/task_2/demo_*/` with the frames and `.npy`
calibration/trajectory files that the dataloader reads. (`task_id 2` is the bowl task;
use `--task_ids all` for the whole suite.)

## Train

Training only needs the cache above (no simulator):

```bash
python train.py --cache_root ./data --task_ids 2
```

Defaults reproduce the reference run: scene+wrist, `n_window=8`, `img_size=448`,
lr `5e-5`, batch 16, 30 epochs. Checkpoints land in `./checkpoints/bind_bowl/latest.pth`.

**Visualization.** By default, panels are dumped as PNGs to `./viz_out/` every 500
steps — for both a training batch and a held-out **val** batch:
- `heatmap_grid` — the per-waypoint volume confidence, stacked `t=0..T-1`, scene row over wrist row.
- `kp_gt_vs_pred` — GT (green) vs argmax-predicted (red) waypoints projected on scene + wrist.
- `z_bins` — the per-view height-bin marginal.

Pass `--wandb` to log these (and loss / `val/xyz_err_mm`) to Weights & Biases instead.

## Evaluate (closed-loop)

```bash
python eval.py --checkpoint ./checkpoints/bind_bowl/latest.pth --task_id 2 --teleport --out_dir ./out
```

Runs a teleport-servo rollout in the LIBERO sim and saves per-episode videos with the
predicted keypoints overlaid on scene + wrist, plus the success rate.

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
