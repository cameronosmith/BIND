"""Train BIND (DinoVolumeSceneV4, scene+wrist per-voxel fusion) on a LIBERO task.

Data contract (from the 2-view cache built by prerender.py):
  rgb        = [rgb_bev (scene), rgb_wrist]
  K_in[k]    = unnormalize(*_K_norm)      (K_norm rows 0,1 are /image_size)
  T_w2c[k]   = inverse(*_extrinsic)       (cache stores cam->world; model wants world->cam)
  start_pix  = trajectory_2d_bev[:, 0]    (EE pixel in the scene view)
  target_xyz = trajectory_3d              (world EE XYZ; volume loss finds the nearest voxel)
  target_grip/rot = discretized bins      (gripper linear; rotation via a 1D PCA axis / clusters)

Loss = bind.losses.multiview_losses (volume CE to nearest voxel + grip CE + rot CE).

Visualization: by default dumps PNG panels (per-waypoint heatmap grid + GT-vs-pred
keypoints, for train and held-out val) to --viz_dir. Pass --wandb to log to Weights & Biases.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from bind.dataset import CachedTrajectory2ViewDataset
from bind.model import DinoVolumeSceneV4
from bind.losses import multiview_losses
from bind import viz as vizmod

HERE = Path(__file__).resolve().parent


def discretize(values, lo, hi, n_bins):
    norm = (values - lo) / max(hi - lo, 1e-8)
    return (norm.clamp(0, 1) * (n_bins - 1)).long().clamp(0, n_bins - 1)


def compute_dataset_stats(ds):
    all_z, all_g = [], []
    for demo in ds.demos:
        all_z.append(demo["eef_pos"][:, 2]); all_g.append(demo["gripper"])
    z = np.concatenate(all_z); g = np.concatenate(all_g)
    z_lo, z_hi = float(z.min()), float(z.max()); z_pad = (z_hi - z_lo) * 0.05
    g_lo, g_hi = float(g.min()), float(g.max()); g_pad = (g_hi - g_lo) * 0.05
    return {"min_height": z_lo - z_pad, "max_height": z_hi + z_pad,
            "min_grip": g_lo - g_pad, "max_grip": g_hi + g_pad}


def unnormalize_K(K_norm, S):
    K = K_norm.clone(); K[:, 0, :] *= S; K[:, 1, :] *= S
    return K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", type=str, default="./data")
    p.add_argument("--benchmark", type=str, default="libero_spatial")
    p.add_argument("--task_ids", type=str, default="2", help="comma-separated; 2 = from_table_center bowl")
    p.add_argument("--max_demos", type=int, default=0)
    p.add_argument("--n_window", type=int, default=8)
    p.add_argument("--frame_stride", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--n_height_bins", type=int, default=32)
    p.add_argument("--pred_size", type=int, default=56)
    p.add_argument("--n_gripper_bins", type=int, default=32)
    p.add_argument("--n_rot_bins", type=int, default=32)
    p.add_argument("--cross_view_layers", type=int, default=0,
                   help="0 = per-voxel MLP fusion; 4-5 = + cross-view attention")
    p.add_argument("--upright", type=int, default=1,
                   help="1 = feed DINO the upright agentview (matches DINO pretraining)")
    p.add_argument("--gripper_loss_weight", type=float, default=0.5)
    p.add_argument("--rotation_loss_weight", type=float, default=0.5)
    p.add_argument("--log_scalars_every", type=int, default=10)
    p.add_argument("--vis_every_steps", type=int, default=500, help=">0 logs viz panels every N steps")
    p.add_argument("--vis_n", type=int, default=2)
    p.add_argument("--viz_dir", type=str, default="./viz_out", help="dump viz PNGs here (ignored if --wandb)")
    p.add_argument("--save_every_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0, help=">0 stops early (smoke test)")
    p.add_argument("--rot_pca_path", type=str, default=str(HERE / "assets" / "rotation_pca_basis.npz"))
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--run_name", type=str, default="bind_bowl")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases instead of local PNGs")
    p.add_argument("--wandb_project", type=str, default="bind")
    args = p.parse_args()
    S = args.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    os.makedirs(args.viz_dir, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def log_scalars(d, step):
        if use_wandb:
            import wandb; wandb.log(d, step=step)

    def log_panels(panels, step):
        """panels: {name: HxWx3 BGR ndarray}."""
        if use_wandb:
            import wandb
            wandb.log({k: wandb.Image(v[:, :, ::-1]) for k, v in panels.items()}, step=step)
        else:
            import cv2
            for name, arr in panels.items():
                cv2.imwrite(os.path.join(args.viz_dir, name.replace("/", "_") + f"_s{step}.png"), arr)

    task_ids = [int(t) for t in args.task_ids.split(",") if t.strip()]
    print(f"Loading 2-view cache: {args.cache_root}/{args.benchmark} task_ids={task_ids}")
    full = CachedTrajectory2ViewDataset(
        cache_root=args.cache_root, benchmark_name=args.benchmark, task_ids=task_ids,
        image_size=S, n_window=args.n_window, frame_stride=args.frame_stride,
        max_demos=args.max_demos, upright=bool(args.upright))
    stats = compute_dataset_stats(full)
    print(f"  height range: [{stats['min_height']:.3f}, {stats['max_height']:.3f}]")

    pca = np.load(args.rot_pca_path)
    IS_CLUSTER = "centroids_quat" in pca
    if IS_CLUSTER:
        rot_centroids = torch.tensor(pca["centroids_quat"], dtype=torch.float32, device=device)
        rot_mean = rot_axis = None; rot_pca_min = rot_pca_max = 0.0
    else:
        rot_mean = torch.tensor(pca["mean"], dtype=torch.float32, device=device)
        rot_axis = torch.tensor(pca["principal_axis"], dtype=torch.float32, device=device)
        rot_pca_min = float(pca["pca_min"]); rot_pca_max = float(pca["pca_max"])

    n = len(full); n_val = max(1, int(n * 0.05))
    train_ds, val_ds = random_split(full, [n - n_val, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False,
                            persistent_workers=args.num_workers > 0)
    val_iter = iter(val_loader)
    print(f"  train={len(train_ds)} val={len(val_ds)}")

    print("Building DinoVolumeSceneV4 (views=[scene, wrist])...")
    model = DinoVolumeSceneV4(
        views=["scene", "wrist"],
        n_window=args.n_window, n_height_bins=args.n_height_bins, pred_size=args.pred_size,
        n_gripper_bins=args.n_gripper_bins, n_rot_clusters=args.n_rot_bins,
        z_lo=stats["min_height"], z_hi=stats["max_height"],
        img_size=S, past_n=0, cross_view_layers=args.cross_view_layers,
        cls_fusion="concat").to(device)
    n_t = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"  trainable: {n_t:,}  cross_view_layers={args.cross_view_layers}")

    if args.resume_from:
        sd = torch.load(args.resume_from, map_location=device, weights_only=False)["model_state_dict"]
        cur = model.state_dict()
        loaded = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
        model.load_state_dict(loaded, strict=False)
        print(f"  resumed: {len(loaded)} matching keys")

    opt = optim.AdamW(filter(lambda q: q.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    def make_inputs(batch):
        rgb = [batch["rgb_bev"].to(device), batch["rgb_wrist"].to(device)]
        K_in = [unnormalize_K(batch["bev_K_norm"].to(device), S),
                unnormalize_K(batch["wrist_K_norm"].to(device), S)]
        T_w2c = [torch.inverse(batch["bev_extrinsic"].to(device)),
                 torch.inverse(batch["wrist_extrinsic"].to(device))]
        start_pix = batch["trajectory_2d_bev"].to(device)[:, 0, :]
        return rgb, K_in, T_w2c, start_pix

    def make_targets(batch):
        grip_v = batch["trajectory_gripper"].to(device)
        gg = discretize(grip_v, stats["min_grip"], stats["max_grip"], args.n_gripper_bins)
        if IS_CLUSTER:
            tq = batch["trajectory_quat"].to(device); tq = tq / tq.norm(dim=-1, keepdim=True)
            gr = torch.abs(torch.einsum("btq,kq->btk", tq, rot_centroids)).argmax(-1)
        else:
            proj = (batch["trajectory_euler"].to(device) - rot_mean) @ rot_axis
            gr = discretize(proj, rot_pca_min, rot_pca_max, args.n_rot_bins)
        return gg, gr

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        last = None
        for batch in pbar:
            rgb, K_in, T_w2c, start_pix = make_inputs(batch)
            traj3d = batch["trajectory_3d"].to(device)
            B, T, _ = batch["trajectory_2d_bev"].shape
            out = model(rgb, start_pix, K_in, T_w2c, target_xyz=None)
            gg, gr = make_targets(batch)
            losses = multiview_losses(out, None, None, gg, gr, traj3d)
            total = (losses["loss/volume"]
                     + args.gripper_loss_weight * losses["loss/grip"]
                     + args.rotation_loss_weight * losses["loss/rot"])
            opt.zero_grad(); total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            last = losses
            pbar.set_postfix(v=f"{losses['loss/volume'].item():.2f}",
                             g=f"{losses['loss/grip'].item():.2f}",
                             r=f"{losses['loss/rot'].item():.2f}")
            global_step += 1

            if global_step % args.log_scalars_every == 0:
                with torch.no_grad():
                    vl = out["volume_logits"].reshape(B, T, -1)
                    vp = out["voxel_positions"].reshape(B, -1, 3)
                    pred_xyz = torch.gather(vp, 1, vl.argmax(-1).unsqueeze(-1).expand(-1, -1, 3))
                    xyz_err = (pred_xyz - traj3d).norm(dim=-1).mean().item() * 1000.0
                    grip_acc = (out["grip_logits"].argmax(-1) == gg).float().mean().item()
                    rot_acc = (out["rot_logits"].argmax(-1) == gr).float().mean().item()
                log_scalars({"train/volume_loss": losses["loss/volume"].item(),
                             "train/gripper_loss": losses["loss/grip"].item(),
                             "train/rotation_loss": losses["loss/rot"].item(),
                             "train/total": total.item(), "train/xyz_err_mm": xyz_err,
                             "train/grip_acc": grip_acc, "train/rot_acc": rot_acc, "epoch": epoch}, global_step)
                if global_step <= 20 or global_step % (args.log_scalars_every * 10) == 0:
                    print(f"  step {global_step}: vol={losses['loss/volume'].item():.3f} "
                          f"xyz_err={xyz_err:.1f}mm grip_acc={grip_acc:.2f} rot_acc={rot_acc:.2f}", flush=True)

            if args.vis_every_steps > 0 and global_step % args.vis_every_steps == 0:
                _emit_viz(model, batch, out, stats, args, S, device, make_inputs, make_targets,
                          val_iter, val_loader, multiview_losses, log_panels, log_scalars, global_step)

            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"[max_steps={args.max_steps}] stopping early."); break
        if last is not None:
            print(f"Epoch {epoch}: vol={last['loss/volume'].item():.3f} "
                  f"grip={last['loss/grip'].item():.3f} rot={last['loss/rot'].item():.3f}")
        if (epoch + 1) % max(1, args.save_every_epochs) == 0 or epoch + 1 == args.epochs:
            torch.save({"epoch": epoch, "global_step": global_step,
                        "model_state_dict": model.state_dict(), "args": vars(args),
                        "model_config": {"views": ["scene", "wrist"], "n_window": args.n_window,
                                         "n_height_bins": args.n_height_bins, "pred_size": args.pred_size,
                                         "n_gripper_bins": args.n_gripper_bins, "n_rot_clusters": args.n_rot_bins,
                                         "cross_view_layers": args.cross_view_layers,
                                         "z_lo": stats["min_height"], "z_hi": stats["max_height"], "img_size": S},
                        "min_height": stats["min_height"], "max_height": stats["max_height"],
                        "min_grip": stats["min_grip"], "max_grip": stats["max_grip"],
                        "rot_pca_mean": (np.asarray(pca["mean"]) if not IS_CLUSTER else np.zeros(3)),
                        "rot_pca_axis": (np.asarray(pca["principal_axis"]) if not IS_CLUSTER else np.zeros(3)),
                        "rot_pca_min": rot_pca_min, "rot_pca_max": rot_pca_max,
                        "rot_centroids_quat": (np.asarray(pca["centroids_quat"]) if IS_CLUSTER else None),
                        "image_size": S}, ckpt_dir / "latest.pth")
            print(f"  saved {ckpt_dir / 'latest.pth'}")
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    if use_wandb:
        import wandb; wandb.finish()
    print(f"Done. {ckpt_dir}")


def _emit_viz(model, batch, out, stats, args, S, device, make_inputs, make_targets,
              val_iter, val_loader, loss_fn, log_panels, log_scalars, step):
    """Train + held-out-val viz panels (heatmap grid + GT-vs-pred keypoints)."""
    def panels_for(batch_, out_, prefix):
        rgb = [batch_["rgb_bev"].to(device), batch_["rgb_wrist"].to(device)]
        bgr_scene = [vizmod.denorm_bgr(rgb[0][i]) for i in range(min(args.vis_n, rgb[0].shape[0]))]
        bgr_wrist = [vizmod.denorm_bgr(rgb[1][i]) for i in range(min(args.vis_n, rgb[1].shape[0]))]
        traj3d = batch_["trajectory_3d"].to(device)
        gz = discretize(traj3d[..., 2], stats["min_height"], stats["max_height"], args.n_height_bins)
        _, K_in, T_w2c, _ = make_inputs(batch_)
        P = vizmod.heatmap_grid(out_, bgr_scene, bgr_wrist, gz, S, args.vis_n,
                                flip_heat=not bool(args.upright), prefix=prefix)
        P.update(vizmod.kp_gt_vs_pred(out_, traj3d, K_in[0], T_w2c[0], K_in[1], T_w2c[1],
                                      bgr_scene, bgr_wrist, args.vis_n, prefix=prefix))
        return P
    try:
        log_panels(panels_for(batch, out, "viz"), step)
    except Exception as e:
        print("  [viz] failed:", repr(e), flush=True)
    try:
        try:
            vb = next(val_iter)
        except StopIteration:
            vb = next(iter(val_loader))
        rgb_v, K_in_v, T_w2c_v, start_pix_v = make_inputs(vb)
        model.eval()
        with torch.no_grad():
            out_v = model(rgb_v, start_pix_v, K_in_v, T_w2c_v, target_xyz=None)
        model.train()
        log_panels(panels_for(vb, out_v, "val"), step)
        with torch.no_grad():
            traj3d_v = vb["trajectory_3d"].to(device)
            Bv, Tv, _ = vb["trajectory_2d_bev"].shape
            vl = out_v["volume_logits"].reshape(Bv, Tv, -1)
            vp = out_v["voxel_positions"].reshape(Bv, -1, 3)
            pred = torch.gather(vp, 1, vl.argmax(-1).unsqueeze(-1).expand(-1, -1, 3))
            val_err = (pred - traj3d_v).norm(dim=-1).mean().item() * 1000.0
        log_scalars({"val/xyz_err_mm": val_err}, step)
        print(f"  [val] xyz_err={val_err:.1f}mm", flush=True)
    except Exception as e:
        model.train(); print("  [val viz] failed:", repr(e), flush=True)


if __name__ == "__main__":
    main()
