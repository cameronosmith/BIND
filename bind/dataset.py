"""Two-view libero dataset reader.

Reads the dual-camera cache produced by prerender_dataset_2view.py. Per-sample yields:
  rgb_bev, rgb_wrist : (3, H, W) ImageNet-normalised
  trajectory_2d_bev  : (T, 2) in 448-image pixel coords
  trajectory_3d, _gripper, _quat, _euler  : (T, ...)
  start_pix_bev, _wrist                   : (2,)
  wrist_world_to_cam                      : (T, 4, 4) per-step
  bev_K_norm                              : (3, 3)
  bev_world_to_cam                        : (4, 4)
  wrist_K_norm                            : (3, 3)
"""
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as ScipyR

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DEFAULT_CACHE = "./data"
CACHE_RENDER_SIZE = 448   # resolution the prerender cache was rendered at


def _derive_extrinsic_from_projection(K_norm, world_to_cam_full, image_size):
    """Robosuite's `get_camera_transform_matrix` returns `K_extended @ world_to_cam_extrinsic`,
    so to recover the pure extrinsic (cam→world) we need to undo the K multiplication.
    K_norm:           (3, 3) normalised
    world_to_cam_full: (4, 4) = K @ world_to_cam_extrinsic  (K on rows 0..2)
    Returns extrinsic (4, 4) cam→world.
    """
    K = K_norm.copy()
    K[0] *= float(image_size); K[1] *= float(image_size)
    K_ext = np.eye(4, dtype=np.float64); K_ext[:3, :3] = K.astype(np.float64)
    w2c_ext = np.linalg.inv(K_ext) @ world_to_cam_full.astype(np.float64)        # (4, 4) world→cam pure
    extrinsic = np.linalg.inv(w2c_ext)                                            # cam→world
    return extrinsic.astype(np.float32)


def _load_rgb(p, image_size):
    bgr = cv2.imread(str(p))
    if bgr is None:
        # truncated/corrupt/mid-write frame -> black fallback (rare; avoids crashing training)
        print(f"[data] WARN: unreadable frame {p}; using black fallback", flush=True)
        bgr = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if rgb.shape[0] != image_size or rgb.shape[1] != image_size:
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return rgb


class CachedTrajectory2ViewDataset(Dataset):
    def __init__(self, cache_root=DEFAULT_CACHE,
                 benchmark_name="libero_spatial",
                 task_ids=(0,),
                 image_size=448, n_window=8, frame_stride=3,
                 max_demos=0, upright=False, lang_cache_path=""):
        self.image_size   = image_size
        self.n_window     = n_window
        self.frame_stride = frame_stride
        self.benchmark_name = benchmark_name
        # Optional CLIP-text embedding cache: {"<bm>/task_<t>": (512,)}. When set, each
        # sample returns a "lang_emb" so the model can FiLM-condition on the instruction.
        self.lang_cache = dict(np.load(lang_cache_path)) if lang_cache_path else None
        # upright=True feeds the DINO backbone the natural (upright) agentview instead of the
        # legacy flipud "training convention". Geometrically handled by flipping the image AND
        # the intrinsics' v-axis (fy -> -fy, cy -> (S-1)-cy), keeping projection consistent.
        self.upright      = upright

        bench_root = Path(cache_root) / benchmark_name
        if not bench_root.exists():
            raise FileNotFoundError(f"Cache not found: {bench_root}")
        self.demos   = []
        self.samples = []

        for tid in sorted(task_ids):
            task_dir = bench_root / f"task_{tid}"
            if not task_dir.exists():
                continue
            task_demo_count = 0
            def _frames(d):  # accept the packaged .jpg asset or prerendered .png
                fs = sorted(d.glob("*.jpg"))
                return fs if fs else sorted(d.glob("*.png"))
            for demo_dir in sorted(task_dir.glob("demo_*")):
                if max_demos > 0 and task_demo_count >= max_demos:
                    break
                if not (demo_dir / "actions.npy").exists():
                    continue   # skip partially-rendered demos (prerender writes actions.npy LAST)
                bev_frames   = _frames(demo_dir / "frames_bev")
                wrist_frames = _frames(demo_dir / "frames_wrist")
                if not bev_frames or len(bev_frames) != len(wrist_frames):
                    continue
                T = len(bev_frames)
                demo = {
                    "bev_frames":     bev_frames,
                    "wrist_frames":   wrist_frames,
                    "eef_pos":        np.load(demo_dir / "eef_pos.npy"),
                    "eef_quat":       np.load(demo_dir / "eef_quat.npy"),
                    "gripper":        np.load(demo_dir / "gripper.npy"),
                    "pix_uv_bev":     np.load(demo_dir / "pix_uv_bev.npy"),
                    "pix_uv_wrist":   np.load(demo_dir / "pix_uv_wrist.npy"),
                    "bev_K_norm":     np.load(demo_dir / "bev_K_norm.npy"),
                    "bev_extrinsic":  np.load(demo_dir / "bev_extrinsic.npy"),
                    "bev_world_to_cam": np.load(demo_dir / "bev_world_to_cam.npy"),
                    "wrist_K_norm":   np.load(demo_dir / "wrist_K_norm.npy"),
                    "wrist_world_to_cam": np.load(demo_dir / "wrist_world_to_cam.npy"),
                    "base_z":         float(np.load(demo_dir / "base_z.npy")),
                    "T":              T,
                }
                # bev_extrinsic is already cam→world (saved from get_camera_extrinsic_matrix)
                demo["bev_extrinsic"] = demo["bev_extrinsic"].astype(np.float32)
                # We didn't save wrist_extrinsic explicitly — derive from wrist_world_to_cam.
                # (wrist_world_to_cam = K @ world_to_cam_extrinsic per frame, so undo K then invert.)
                wrist_ext = np.zeros_like(demo["wrist_world_to_cam"])
                for t in range(T):
                    wrist_ext[t] = _derive_extrinsic_from_projection(
                        demo["wrist_K_norm"], demo["wrist_world_to_cam"][t], image_size
                    )
                demo["wrist_extrinsic"] = wrist_ext.astype(np.float32)
                demo["task_id"] = tid
                self.demos.append(demo)
                for t in range(T):
                    self.samples.append((len(self.demos) - 1, t))
                task_demo_count += 1
        print(f"CachedTrajectory2ViewDataset: {len(self.demos)} demos, {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        demo_idx, start_t = self.samples[idx]
        demo = self.demos[demo_idx]
        T = demo["T"]

        # Sample T_window future frames (libero convention: start_t is current, future is [start_t..start_t + n*stride])
        traj_2d_bev, traj_3d, traj_grip, traj_quat = [], [], [], []
        rgb_bev_t0 = rgb_wrist_t0 = None
        wrist_extrinsic_t0 = None

        for k in range(self.n_window):
            t = min(start_t + k * self.frame_stride, T - 1)
            if k == 0:
                rgb_bev_t0   = _load_rgb(demo["bev_frames"][t],   self.image_size)
                rgb_wrist_t0 = _load_rgb(demo["wrist_frames"][t], self.image_size)
                wrist_extrinsic_t0 = demo["wrist_extrinsic"][t]
            traj_3d.append(demo["eef_pos"][t].astype(np.float64))
            traj_quat.append(demo["eef_quat"][t].astype(np.float64))
            traj_grip.append(float(np.clip(demo["gripper"][t], -1.0, 1.0)))
            traj_2d_bev.append(demo["pix_uv_bev"][t].astype(np.float32))

        traj_2d_bev   = np.stack(traj_2d_bev,   axis=0)
        # Cache is rendered at 448; frames get resized to image_size in _load_rgb, and K_norm is
        # resolution-free (normalized), so we only need to scale the pixel coords to image_size.
        if self.image_size != CACHE_RENDER_SIZE:
            traj_2d_bev = traj_2d_bev * (self.image_size / float(CACHE_RENDER_SIZE))
        traj_3d       = np.stack(traj_3d,       axis=0).astype(np.float32)
        traj_grip     = np.array(traj_grip,             dtype=np.float32)
        traj_quat     = np.stack(traj_quat,     axis=0).astype(np.float32)
        traj_euler    = np.stack([ScipyR.from_quat(q).as_euler('xyz') for q in traj_quat],
                                  axis=0).astype(np.float32)

        # Normalise RGB (ImageNet) — match training convention
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)
        rgb_bev_t   = torch.from_numpy(rgb_bev_t0.transpose(2, 0, 1).astype(np.float32))
        rgb_wrist_t = torch.from_numpy(rgb_wrist_t0.transpose(2, 0, 1).astype(np.float32))
        rgb_bev_t   = (rgb_bev_t   - torch.from_numpy(mean)) / torch.from_numpy(std)
        rgb_wrist_t = (rgb_wrist_t - torch.from_numpy(mean)) / torch.from_numpy(std)

        bev_K_out   = demo["bev_K_norm"].copy()
        wrist_K_out = demo["wrist_K_norm"].copy()
        if self.upright:
            S = self.image_size
            rgb_bev_t   = torch.flip(rgb_bev_t,   dims=[1])          # flip image rows (H)
            rgb_wrist_t = torch.flip(rgb_wrist_t, dims=[1])
            traj_2d_bev = traj_2d_bev.copy()
            traj_2d_bev[:, 1] = (S - 1) - traj_2d_bev[:, 1]          # flip v of start/GT pixels
            for K in (bev_K_out, wrist_K_out):                       # flip intrinsics' v-axis
                K[1, 1] = -K[1, 1]                                   # fy -> -fy (normalized)
                K[1, 2] = (S - 1) / S - K[1, 2]                      # cy -> (S-1)-cy (normalized)

        return {
            "rgb_bev":             rgb_bev_t,
            "rgb_wrist":           rgb_wrist_t,
            "trajectory_2d_bev":   torch.from_numpy(traj_2d_bev),
            "trajectory_3d":       torch.from_numpy(traj_3d),
            "trajectory_gripper":  torch.from_numpy(traj_grip),
            "trajectory_quat":     torch.from_numpy(traj_quat),
            "trajectory_euler":    torch.from_numpy(traj_euler),
            "wrist_extrinsic":     torch.from_numpy(wrist_extrinsic_t0),
            "bev_K_norm":          torch.from_numpy(bev_K_out),
            "bev_extrinsic":       torch.from_numpy(demo["bev_extrinsic"]),
            "wrist_K_norm":        torch.from_numpy(wrist_K_out),
            **({"lang_emb": torch.from_numpy(
                    self.lang_cache[f"{self.benchmark_name}/task_{demo['task_id']}"]).float()}
               if self.lang_cache is not None else {}),
            "demo_idx":            torch.tensor(demo_idx, dtype=torch.long),
            "start_t":             torch.tensor(start_t,  dtype=torch.long),
        }
