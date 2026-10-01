"""Visualization panels: per-waypoint volume heatmap grid + GT-vs-pred keypoints.

All functions return ``{name: HxWx3 BGR uint8}`` dicts so the caller can either
log them to wandb (``wandb.Image(arr[:, :, ::-1])``) or write PNGs with cv2.
"""
import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denorm_bgr(rgb_t):
    """ImageNet-normalized CHW tensor -> HxWx3 BGR uint8 (as-fed orientation)."""
    x = rgb_t.detach().cpu().numpy().transpose(1, 2, 0)
    x = np.clip(x * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
    return (x * 255).astype(np.uint8)[:, :, ::-1].copy()


def heatmap_grid(out, bgr_scene, bgr_wrist, gz, img_size, vis_n, flip_heat=True, prefix="viz"):
    """Stacked per-waypoint volume heatmaps (scene row over wrist row), t=0..T-1.

    ``out["volume_logits"]`` is (B, T, NZ, P, P); ``out["views"]`` the view names.
    For each waypoint we softmax its (Z,P,P) block, marginalize over Z, and overlay
    the (P,P) confidence on the image thumbnail.
    """
    vol = out["volume_logits"].detach().float()
    views = list(out["views"])
    B, T, NZ, P, _ = vol.shape
    N = len(views); Z = NZ // N
    TH = max(96, img_size // 2)
    panels = {}
    for b in range(min(vis_n, B)):
        view_rows, zbar_strips = [], []
        for k, v in enumerate(views):
            src = bgr_scene[b] if v == "scene" else (bgr_wrist[b] if bgr_wrist is not None else None)
            if src is None:
                continue
            bg_th = cv2.resize(src, (TH, TH), interpolation=cv2.INTER_AREA).astype(np.float32)
            thumbs = []
            for t in range(T):
                block = vol[b, t, k * Z:(k + 1) * Z]
                p = torch.softmax(block.reshape(-1), 0).reshape(Z, P, P)
                hm = p.sum(0).cpu().numpy()
                if flip_heat:
                    hm = np.flipud(hm)
                hm = (hm / max(hm.max(), 1e-9)) ** 0.5
                hm_big = cv2.resize(hm.astype(np.float32), (TH, TH), interpolation=cv2.INTER_LINEAR)
                hm_color = cv2.applyColorMap((hm_big * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
                alpha = np.clip(hm_big[..., None] * 1.3, 0, 1) * 0.8
                blend = (bg_th * (1 - alpha) + hm_color * alpha).astype(np.uint8)
                cv2.putText(blend, f"t{t}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                thumbs.append(blend)
            bar = np.full((22, TH * T, 3), 30, np.uint8)
            cv2.putText(bar, f"{v} heatmaps t=0..{T-1}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
            view_rows.append(np.vstack([bar, np.hstack(thumbs)]))
            block0 = vol[b, 0, k * Z:(k + 1) * Z]
            p0 = torch.softmax(block0.reshape(-1), 0).reshape(Z, P, P)
            zmarg = p0.sum((1, 2)).cpu().numpy(); zmarg /= max(zmarg.max(), 1e-9)
            zh, zw = 150, TH * T; bw = max(zw // Z, 1)
            zimg = np.full((zh, zw, 3), 30, np.uint8)
            for zi in range(Z):
                h = int(zmarg[zi] * (zh - 28))
                cv2.rectangle(zimg, (zi * bw + 1, zh - h), ((zi + 1) * bw - 1, zh), (90, 210, 130), -1)
            label = f"{v} z-bins"
            if v == "scene":
                gt_z = int(gz[b, 0].item())
                if 0 <= gt_z < Z:
                    cv2.line(zimg, (gt_z * bw + bw // 2, 0), (gt_z * bw + bw // 2, zh), (60, 60, 255), 2)
                label += " (red=GT)"
            else:
                label += " (own z-range)"
            bar2 = np.full((22, zw, 3), 30, np.uint8)
            cv2.putText(bar2, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
            zbar_strips.append(np.vstack([bar2, zimg]))
        if view_rows:
            panels[f"{prefix}/heatmap_grid_{b}"] = np.vstack(view_rows)
        if zbar_strips:
            panels[f"{prefix}/z_bins_{b}"] = np.vstack(zbar_strips)
    return panels


def _project(xyz, K_pix, T_w2c):
    """xyz (T,3) world -> (T,2) pixel. K_pix (3,3) pixel intrinsics, T_w2c (4,4) world->cam."""
    ph = np.concatenate([xyz, np.ones((len(xyz), 1))], 1)
    cam = (T_w2c @ ph.T).T[:, :3]
    uv = (K_pix @ cam.T).T
    return uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)


def _draw_track(img, uv, color, r=5):
    prev = None
    S0, S1 = img.shape[1], img.shape[0]
    for u, v in uv:
        u, v = int(round(u)), int(round(v))
        if 0 <= u < S0 and 0 <= v < S1:
            if prev is not None:
                cv2.line(img, prev, (u, v), color, 2, cv2.LINE_AA)
            cv2.circle(img, (u, v), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (u, v), r, (255, 255, 255), 1, cv2.LINE_AA)
            prev = (u, v)
        else:
            prev = None


def kp_gt_vs_pred(out, target_xyz, K_scene, T_scene, K_wrist, T_wrist,
                  bgr_scene, bgr_wrist, vis_n, prefix="viz"):
    """GT (green) vs argmax-predicted (red) waypoint keypoints on scene + wrist.

    K_* are pixel intrinsics (3,3), T_* are world->cam (4,4), per batch item.
    Images are in as-fed (upright) orientation; projection lands directly on them.
    """
    vol = out["volume_logits"]
    vp = out["voxel_positions"]
    B, T = vol.shape[0], vol.shape[1]
    pred_idx = vol.reshape(B, T, -1).argmax(-1)
    pred_xyz = torch.gather(vp.reshape(B, -1, 3), 1,
                            pred_idx.unsqueeze(-1).expand(-1, -1, 3)).detach().cpu().numpy()
    gt = target_xyz.detach().cpu().numpy()
    panels = {}
    GREEN, RED = (0, 200, 0), (0, 0, 255)
    for b in range(min(vis_n, B)):
        sc = bgr_scene[b].copy(); wr = bgr_wrist[b].copy()
        for img, K, Tw in [(sc, K_scene[b], T_scene[b]), (wr, K_wrist[b], T_wrist[b])]:
            Kp = K.detach().cpu().numpy(); Tp = Tw.detach().cpu().numpy()
            _draw_track(img, _project(gt[b], Kp, Tp), GREEN)
            _draw_track(img, _project(pred_xyz[b], Kp, Tp), RED)
        cv2.putText(sc, "scene GT=green pred=red", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(wr, "wrist", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        panels[f"{prefix}/kp_gt_vs_pred_{b}"] = np.hstack([sc, wr])
    return panels
