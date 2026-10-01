"""Closed-loop LIBERO eval for BIND (DinoVolumeSceneV4).

Runs a teleport-servo rollout: at each replan the model predicts the future
end-effector waypoints (argmax over the volume) + gripper + rotation, and the
arm position-servos to them. Saves per-episode rollout videos with the predicted
keypoints overlaid on scene + wrist, and reports the success rate.

Requires LIBERO installed (pip) and DINOV3_WEIGHTS set (see bind/model.py).
"""
import os, sys, argparse, json, time
import numpy as np
import torch
import cv2
import h5py
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as ScipyR

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from libero.libero import benchmark as bm, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_camera_transform_matrix,
    project_points_from_world_to_camera,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bind.model import DinoVolumeSceneV4

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_obs(rgb_obs, image_size):
    """HxWx3 uint8 LIBERO obs -> (1,3,H,W) ImageNet-normalized tensor.

    LIBERO obs images are already upright; flipud matches the training-image
    convention (flipud(obs) -> training image)."""
    img = rgb_obs.astype(np.float32) / 255.0
    img = np.flipud(img).copy()
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)


def eef_to_start_kp(eef_pos, world_to_camera, image_size):
    """Project current EEF world position -> (u, v) pixel (training convention)."""
    pix_rc = project_points_from_world_to_camera(
        points=eef_pos.reshape(1, 3).astype(np.float64),
        world_to_camera_transform=world_to_camera,
        camera_height=image_size, camera_width=image_size,
    )[0]
    return torch.tensor([float(pix_rc[1]), float(pix_rc[0])], dtype=torch.float32)

LIBERO_IMG = 448
OSC_POS_SCALE = 0.05
FORCE_GAIN = float(os.environ.get("FORCE_GAIN", "1.0"))  # >1: more action per position error; corrects ~8x under-scaling that starved contact-rich (drawer) tasks. Safe for free-space (large errors already saturate to 1.0).
OSC_ROT_SCALE = 0.5

# Task 0 object qpos offsets (state has +1 prefix)
TASK0_PICK_PLACE_QPOS = [9, 37]   # bowl, plate
TASK0_DISTRACTOR_QPOS = [16, 23, 30]
DISTRACTOR_FAR = np.array([10.0, 10.0, 0.9])


def shift_init_state(state, dx, dy):
    s = state.copy()
    for qp in TASK0_PICK_PLACE_QPOS:
        si = qp + 1
        s[si] += dx
        s[si + 1] += dy
    return s


def hide_distractors_in_state(state):
    s = state.copy()
    for qp in TASK0_DISTRACTOR_QPOS:
        si = qp + 1
        s[si:si + 3] = DISTRACTOR_FAR
    return s


def reposition_camera(sim, camera_name, theta_deg, phi_deg):
    """Spherical-cap camera reposition (matches generate_ood_viewpoint.py's grid)."""
    from scipy.spatial.transform import Rotation as ScipyR
    cam_id = sim.model.camera_name2id(camera_name)
    default_pos = sim.data.cam_xpos[cam_id].copy()
    cam_xmat = sim.data.cam_xmat[cam_id].reshape(3, 3)
    forward = -cam_xmat[:, 2]
    TABLE_Z = 0.90
    t_hit = (TABLE_Z - default_pos[2]) / (forward[2] + 1e-8)
    look_at = default_pos + t_hit * forward
    radius = np.linalg.norm(default_pos - look_at)
    default_dir = (default_pos - look_at) / radius
    up = np.array([0, 0, 1.0])
    if abs(np.dot(default_dir, up)) > 0.99:
        up = np.array([1, 0, 0.0])
    right = np.cross(default_dir, up); right /= np.linalg.norm(right)
    true_up = np.cross(right, default_dir)
    theta = np.radians(theta_deg); phi = np.radians(phi_deg)
    offset = (np.sin(theta) * np.cos(phi) * right +
              np.sin(theta) * np.sin(phi) * true_up +
              np.cos(theta) * default_dir)
    new_pos = look_at + radius * offset
    fwd = look_at - new_pos
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    cam_z = -fwd
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(fwd, up_hint)) > 0.99:
        up_hint = np.array([0.0, 1.0, 0.0])
    cam_x = np.cross(up_hint, cam_z); cam_x /= (np.linalg.norm(cam_x) + 1e-12)
    cam_y = np.cross(cam_z, cam_x)
    R = np.stack([cam_x, cam_y, cam_z], axis=-1)
    q = ScipyR.from_matrix(R).as_quat()
    new_quat = np.array([q[3], q[0], q[1], q[2]])
    sim.model.cam_pos[cam_id] = new_pos
    sim.model.cam_quat[cam_id] = new_quat
    sim.forward()


def apply_clean_scene(sim):
    """Hide furniture + distractors to match OOD training data."""
    for fname in ["wooden_cabinet_1_main", "flat_stove_1_main"]:
        try:
            bid = sim.model.body_name2id(fname)
            sim.model.body_pos[bid] = np.array([0, 0, -5.0])
        except Exception:
            pass
    sim.forward()
    distractor_bodies = set()
    for dn in ["akita_black_bowl_2_main", "cookies_1_main", "glazed_rim_porcelain_ramekin_1_main"]:
        try:
            distractor_bodies.add(sim.model.body_name2id(dn))
        except Exception:
            pass
    for gid in range(sim.model.ngeom):
        if sim.model.geom_bodyid[gid] in distractor_bodies:
            sim.model.geom_rgba[gid][3] = 0.0


def decode_actions_v4(out_dict, current_eef_pos, current_eef_quat,
                       min_g, max_g, rot_pca_mean, rot_pca_axis, rot_pca_min, rot_pca_max,
                       rot_centroids=None, max_delta=0.05):
    """DinoVolumeSceneV4 decode: predicted world XYZ = voxel_positions at volume argmax.
    No pinhole ray-cast / height-bin math needed (voxel_positions already carry world XYZ)."""
    vol   = out_dict["volume_logits"][0]          # (T, NZ, P, P)
    vpos  = out_dict["voxel_positions"][0]        # (NZ, P, P, 3)
    grip  = out_dict["grip_logits"][0]            # (T, n_grip)   (v4 key names)
    rot   = out_dict["rot_logits"][0]             # (T, n_rot)
    T = vol.shape[0]
    flat = vol.reshape(T, -1).argmax(dim=-1)      # (T,) index into NZ*P*P
    vpos_flat = vpos.reshape(-1, 3)               # (NZ*P*P, 3)
    pred_xyz = vpos_flat[flat].cpu().numpy().astype(np.float64)   # (T, 3)
    n_grip = grip.shape[-1]; n_rot = rot.shape[-1]
    grip_argmax = grip.argmax(dim=-1).cpu().numpy()
    rot_argmax  = rot.argmax(dim=-1).cpu().numpy()

    R_current = ScipyR.from_quat(current_eef_quat)
    actions = []; pred_3d_list = []; pred_quat_list = []
    ref_pos = current_eef_pos.copy()
    for t in range(T):
        p3d = pred_xyz[t]
        pred_3d_list.append(p3d)
        delta_pos = p3d - ref_pos
        norm = np.linalg.norm(delta_pos)
        if norm > max_delta:
            delta_pos = delta_pos / norm * max_delta
        delta_norm = np.clip(delta_pos / OSC_POS_SCALE, -1.0, 1.0)

        if rot_centroids is not None:
            R_pred = ScipyR.from_quat(rot_centroids[int(rot_argmax[t])])
        else:
            pca_val = rot_pca_min + (rot_argmax[t] + 0.5) / n_rot * (rot_pca_max - rot_pca_min)
            euler_pred = rot_pca_mean + pca_val * rot_pca_axis
            R_pred = ScipyR.from_euler('xyz', euler_pred)
        if os.environ.get("DEBUG_ROT"):
            import numpy as _np
            print("ROT t=%d pred_eul=%s cur_eul=%s" % (t, _np.round(R_pred.as_euler("xyz", degrees=True), 1).tolist(), _np.round(R_current.as_euler("xyz", degrees=True), 1).tolist()), flush=True)
        R_delta = R_pred * R_current.inv()
        delta_rot_norm = np.clip(R_delta.as_rotvec() / OSC_ROT_SCALE, -1.0, 1.0)

        grip_continuous = (grip_argmax[t] / max(n_grip - 1, 1)) * (max_g - min_g) + min_g
        gripper_cmd = 1.0 if grip_continuous > 0.0 else -1.0
        if os.environ.get("DEBUG_POS"):
            import numpy as _np
            print("POS t=%d pred3d=%s cur3d=%s grip=%.1f" % (t, _np.round(p3d, 3).tolist(), _np.round(ref_pos, 3).tolist(), float(gripper_cmd)), flush=True)

        action = np.zeros(7, dtype=np.float32)
        action[:3]  = delta_norm
        action[3:6] = delta_rot_norm
        action[6]   = gripper_cmd
        actions.append(action)
        pred_quat_list.append(R_pred.as_quat())     # absolute target orientation for optional rot-servo
    return actions, pred_3d_list, pred_quat_list


def _proj_w2p(pts_w, K, T_w2c):
    """world (N,3) -> pixel (N,2) via K(3,3) @ T_w2c(4,4)."""
    ph = np.concatenate([np.asarray(pts_w, float), np.ones((len(pts_w), 1))], 1)   # (N,4)
    cam = (T_w2c @ ph.T).T[:, :3]                                                   # (N,3)
    uv = (K @ cam.T).T                                                             # (N,3)
    return uv[:, :2] / uv[:, 2:3]


_VIZ = {"on": False}

def _draw_kps(img, uv, eef_uv):
    import cv2
    H, W = img.shape[0], img.shape[1]
    prev = None; n = len(uv)
    for i, (x, y) in enumerate(uv):
        xi, yi = int(round(x)), int(round(y))
        col = (int(255*(1-i/max(n-1,1))), 60, int(255*i/max(n-1,1)))
        if 0 <= xi < W and 0 <= yi < H:
            if prev is not None:
                cv2.line(img, prev, (xi, yi), (210,210,210), 1, cv2.LINE_AA)
            cv2.circle(img, (xi, yi), 5, col, -1, cv2.LINE_AA); prev=(xi,yi)
        else:
            prev = None
    ex, ey = int(round(eef_uv[0])), int(round(eef_uv[1]))
    if 0 <= ex < W and 0 <= ey < H:
        cv2.circle(img, (ex,ey), 7, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(img, (ex,ey), 8, (0,0,0), 2, cv2.LINE_AA)

def _overlay_frame(env, obs):
    import cv2
    S=_VIZ["S"]; up=_VIZ["up"]; pts=np.array(_VIZ["pred"], dtype=np.float64)
    Ks=_VIZ["Ks"]; Tsi=_VIZ["Tsi"]
    scene=np.ascontiguousarray(np.asarray(obs["agentview_image"])[:,:,::-1]).copy()
    wrist=np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"])[:,:,::-1]).copy()
    if scene.shape[0]!=S: scene=cv2.resize(scene,(S,S))
    if wrist.shape[0]!=S: wrist=cv2.resize(wrist,(S,S))
    uv_s=_proj_w2p(pts, Ks, Tsi)
    w_ext=get_camera_extrinsic_matrix(env.sim, "robot0_eye_in_hand").astype(np.float32)
    w_K=get_camera_intrinsic_matrix(env.sim, "robot0_eye_in_hand", S, S).astype(np.float32)
    if up:
        w_K=w_K.copy(); w_K[1,1]=-w_K[1,1]; w_K[1,2]=(S-1)-w_K[1,2]
    Twi=np.linalg.inv(w_ext)
    uv_w=_proj_w2p(pts, w_K, Twi)
    eef=np.array(obs["robot0_eef_pos"], dtype=np.float64)[None]
    _draw_kps(scene, uv_s, _proj_w2p(eef, Ks, Tsi)[0])
    _draw_kps(wrist, uv_w, _proj_w2p(eef, w_K, Twi)[0])
    cv2.putText(scene,"SCENE  pred kpts",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(wrist,"WRIST",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    return np.hstack([scene, wrist])



def _viz_view_row(frame_rgb, pred_3d_list, vol_block, K, T_w2c, image_size, label):
    """One view's row: (left) 8 predicted waypoints projected onto the frame as a numbered
    trajectory; (right) 2x4 grid of the 8 per-timestep heatmaps (this view's volume marginal)."""
    S = image_size
    bg = cv2.resize(np.ascontiguousarray(frame_rgb[:, :, ::-1]), (S, S))            # RGB->BGR
    uv = _proj_w2p(np.array(pred_3d_list), K, T_w2c)                                # (8,2)
    traj = bg.copy(); prev = None
    for t in range(len(uv)):
        u, v = int(round(uv[t, 0])), int(round(uv[t, 1]))
        col = (int(255 * (1 - t / max(len(uv) - 1, 1))), 60, int(255 * t / max(len(uv) - 1, 1)))
        if 0 <= u < S and 0 <= v < S:
            if prev is not None: cv2.line(traj, prev, (u, v), (210, 210, 210), 1, cv2.LINE_AA)
            cv2.circle(traj, (u, v), 6, col, -1, cv2.LINE_AA)
            cv2.putText(traj, str(t), (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            prev = (u, v)
    cv2.putText(traj, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    Tn, Zb, P, _ = vol_block.shape
    th = S // 2; thumbs = []; bg_th = cv2.resize(bg, (th, th)).astype(np.float32)
    for t in range(Tn):
        p = torch.softmax(vol_block[t].reshape(-1), 0).reshape(Zb, P, P)
        hm = p.sum(0).cpu().numpy(); hm = (hm / max(hm.max(), 1e-9)) ** 0.35
        hm_big = cv2.resize(hm.astype(np.float32), (th, th))
        hmc = cv2.applyColorMap((hm_big * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
        al = np.clip(hm_big[..., None] * 1.4, 0, 1) * 0.88
        blend = (bg_th * (1 - al) + hmc * al).astype(np.uint8)
        cv2.putText(blend, f"t{t}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(blend)
    grid = np.vstack([np.hstack(thumbs[i * 4:(i + 1) * 4]) for i in range(2)])      # (2th, 4th)
    trajr = cv2.resize(traj, (grid.shape[0], grid.shape[0]))
    return np.hstack([trajr, grid])


def viz_inference_panel(scene_frame, wrist_frame, pred_3d_list, out, K_scene, T_scene, K_wrist, T_wrist, image_size, tag=""):
    """Two-row panel: SCENE view (traj + 8 scene-block heatmaps) on top, WRIST view below."""
    vol = out["volume_logits"][0].detach().float()                                 # (T,NZ,P,P)
    T_, NZ, P, _ = vol.shape; Z = NZ // 2                                          # [0:Z]=scene, [Z:2Z]=wrist
    scene_row = _viz_view_row(scene_frame, pred_3d_list, vol[:, 0:Z], K_scene, T_scene, image_size, f"SCENE  {tag}")
    wrist_row = _viz_view_row(wrist_frame, pred_3d_list, vol[:, Z:2 * Z], K_wrist, T_wrist, image_size, "WRIST")
    return np.vstack([scene_row, wrist_row])


def servo_to_pos(env, target_pos, gripper_cmd, max_servo=25, threshold=0.005, frames=None,
                 target_quat=None, rot_threshold=0.08, grasp_settle=0):
    """Closed-loop servo to a 3D target position (and optionally target orientation) with given gripper.

    If `target_quat` (xyzw) is given, each micro-step also drives action[3:6] toward the target
    orientation (delta rotvec), so the predicted wrist rotation is actually applied — otherwise the
    EEF orientation is held constant (the teleport default).
    Returns (obs, n_steps, done). If `frames` is a list, appends the agentview render at each micro-step.
    """
    obs = None
    n_steps = 0
    done = False
    R_tgt = ScipyR.from_quat(target_quat) if target_quat is not None else None
    for _ in range(max_servo):
        cur_obs = env.env._get_observations()
        cur_pos = np.array(cur_obs["robot0_eef_pos"], dtype=np.float64)
        delta = target_pos - cur_pos
        dist = np.linalg.norm(delta)
        rot_action = np.zeros(3, dtype=np.float32); rot_dist = 0.0
        if R_tgt is not None:
            R_cur = ScipyR.from_quat(np.array(cur_obs["robot0_eef_quat"], dtype=np.float64))
            rotvec = (R_tgt * R_cur.inv()).as_rotvec()
            rot_dist = float(np.linalg.norm(rotvec))
            rot_action = np.clip(rotvec / OSC_ROT_SCALE, -1.0, 1.0)
        if dist < threshold and rot_dist < rot_threshold:
            obs = cur_obs
            break
        delta_clipped = np.clip(FORCE_GAIN * delta / OSC_POS_SCALE, -1.0, 1.0)
        action = np.zeros(7, dtype=np.float32)
        action[:3] = delta_clipped
        action[3:6] = rot_action
        action[6] = gripper_cmd
        obs, _, done, _ = env.step(action)
        if frames is not None and "agentview_image" in obs:
            frames.append(_overlay_frame(env, obs) if _VIZ.get("on") else np.asarray(obs["agentview_image"]).copy())  # already upright (human-viewable)
        n_steps += 1
        if done:
            break
    # grasp-settle: hold the closed gripper for K steps at the reached target so the physics secures the grasp (mirrors sim hold_and_ramp / cube-sim GRIP_TIGHTEN)
    if grasp_settle > 0 and gripper_cmd > 0:
        for _ in range(grasp_settle):
            act = np.zeros(7, dtype=np.float32); act[6] = gripper_cmd
            obs, _, done, _ = env.step(act)
            if frames is not None and "agentview_image" in obs:
                frames.append(_overlay_frame(env, obs) if _VIZ.get("on") else np.asarray(obs["agentview_image"]).copy())
            n_steps += 1
            if done: break
    if obs is None:
        obs = env.env._get_observations()
    return obs, n_steps, done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--benchmark",  type=str, default="libero_spatial")
    p.add_argument("--task_id",    type=int, default=0)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--max_steps",  type=int, default=600)
    p.add_argument("--max_plans",  type=int, default=0, help="if >0, budget the episode by NUMBER OF MODEL PREDICTIONS (inference calls) rather than env micro-steps; teleport-servo micro-steps in between waypoints do NOT count. Recommended for teleport eval on long-horizon tasks.")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--shift_dx",   type=float, default=0.0)
    p.add_argument("--shift_dy",   type=float, default=0.0)
    p.add_argument("--clean_scene", action="store_true")
    p.add_argument("--zero_rotation", action="store_true")
    p.add_argument("--teleport",     action="store_true")
    p.add_argument("--cam_theta",    type=float, default=0.0, help="BEV camera polar angle (deg)")
    p.add_argument("--cam_phi",      type=float, default=0.0, help="BEV camera azimuth (deg)")
    p.add_argument("--out_json",   type=str, default="")
    p.add_argument("--upright", type=int, default=1, help="must match training: 1 = upright DINO input (flip img + K v-axis)")
    p.add_argument("--lang_cache", type=str, default="", help="CLIP-text cache npz; required if the ckpt was trained with lang_dim>0")
    p.add_argument("--viz_rollout_dir", type=str, default="", help="if set, save a per-inference panel (8-waypoint trajectory + 8 per-timestep heatmaps)")
    p.add_argument("--replan_every", type=int, default=8, help="re-plan after executing this many of the 8 predicted waypoints (<8 = more closed-loop)")
    p.add_argument("--z_drop", type=float, default=0.0, help="subtract this many METERS from predicted waypoint height (world +z is up) before servo+viz — grasp-height hack (cube-sim used ~0.015)")
    p.add_argument("--grasp_settle", type=int, default=0, help="hold the closed gripper for K env-steps at a reached grasp waypoint to secure the grasp")
    p.add_argument("--servo_rot", type=int, default=0, help="teleport mode: if 1, also servo the EEF toward the model's PREDICTED orientation each waypoint (default 0 = hold orientation constant, which is why wrist rotation looks fixed in rollouts)")
    p.add_argument("--temporal_ensemble", type=int, default=0, help="ACT-style temporal ensembling: infer every executed step, servo to an exp-weighted blend of recent chunks' predictions for the current step (momentum carries through a single-inference collapse)")
    p.add_argument("--te_m", type=float, default=0.01, help="temporal-ensemble weight decay exp(-te_m*age); small=near-uniform (more momentum from older chunks)")
    p.add_argument("--save_video_dir", type=str, default="", help="if set, write per-episode agentview rollout mp4s here")
    p.add_argument("--save_video_max", type=int, default=3, help="max episodes to record")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    min_h, max_h = float(ckpt["min_height"]), float(ckpt["max_height"])
    min_g, max_g = float(ckpt["min_grip"]),   float(ckpt["max_grip"])
    rot_pca_mean = np.asarray(ckpt["rot_pca_mean"], dtype=np.float64)
    rot_pca_axis = np.asarray(ckpt["rot_pca_axis"], dtype=np.float64)
    rot_pca_min  = float(ckpt["rot_pca_min"]); rot_pca_max = float(ckpt["rot_pca_max"])
    _rc = ckpt.get("rot_centroids_quat", None)
    rot_centroids = np.asarray(_rc, dtype=np.float64) if _rc is not None else None
    mc = ckpt["model_config"]
    n_rot_bins   = int(mc["n_rot_clusters"])
    n_window     = int(ckpt.get("n_window", 8))
    image_size   = int(ckpt.get("image_size", LIBERO_IMG))
    bev_K_norm   = np.asarray(ckpt["bev_K_norm"], dtype=np.float32)
    bev_extrinsic = np.asarray(ckpt["bev_extrinsic"], dtype=np.float32)
    print(f"Loaded ckpt: epoch={ckpt['epoch']}, n_window={n_window}")
    print(f"  shift_dx={args.shift_dx:+.3f} shift_dy={args.shift_dy:+.3f}  clean={args.clean_scene}  zero_rot={args.zero_rotation}  teleport={args.teleport}")

    print(f"  model: DinoVolumeSceneV4 views={mc['views']} cross_view_layers={mc['cross_view_layers']}")
    model = DinoVolumeSceneV4(
        views=mc["views"], n_window=n_window, n_height_bins=mc["n_height_bins"],
        pred_size=mc["pred_size"], n_gripper_bins=mc["n_gripper_bins"],
        n_rot_clusters=mc["n_rot_clusters"], z_lo=mc["z_lo"], z_hi=mc["z_hi"], xyz_pe_dim=mc.get("xyz_pe_dim", 0),
        img_size=image_size, past_n=0, cross_view_layers=mc["cross_view_layers"],
        lang_dim=mc.get("lang_dim", 0), cls_fusion="concat").to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Language conditioning: look up this task's cached CLIP-text embedding once.
    lang_emb = None
    if mc.get("lang_dim", 0) > 0:
        assert args.lang_cache, "ckpt trained with lang_dim>0 but --lang_cache not provided"
        _c = dict(np.load(args.lang_cache))
        lang_emb = torch.from_numpy(_c[f"{args.benchmark}/task_{args.task_id}"]).float().unsqueeze(0).to(device)
        print(f"  lang: FiLM-conditioned on task {args.task_id} instruction (CLIP {lang_emb.shape})")

    # bev_xyz_table is rebuilt per-episode after the env (potentially with shifted camera) is set up
    bev_xyz_table = None

    bench = bm.get_benchmark_dict()[args.benchmark]()
    task  = bench.get_task(args.task_id)
    print(f"Task: [{args.benchmark}] {task.name}")
    demo_path = os.path.join(get_libero_path("datasets"), bench.get_task_demonstration(args.task_id))
    with h5py.File(demo_path, "r") as f:
        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
        init_states = [f[f"data/{k}/states"][0] for k in demo_keys]
    n_episodes = min(args.n_episodes, len(init_states))
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    # ignore_done + huge horizon: the teleport-servo burns many env.step micro-steps per waypoint,
    # which would otherwise exhaust robosuite's internal horizon and raise "executing action in
    # terminated episode". We control termination ourselves via --max_plans / --max_steps and detect
    # success via _check_success(); the env should never auto-terminate.
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=image_size, camera_widths=image_size,
        camera_names=["agentview", "robot0_eye_in_hand"],
        ignore_done=True, horizon=100000,
    )
    env.seed(args.seed); env.reset()
    if args.clean_scene:
        apply_clean_scene(env.env.sim)
        print("✓ Clean scene applied")

    K_bev_pixel = bev_K_norm.copy().astype(np.float64)
    K_bev_pixel[0] *= image_size; K_bev_pixel[1] *= image_size
    camera_pose_bev = bev_extrinsic.astype(np.float64)

    successes, step_counts, plan_counts = [], [], []
    for ep_idx in tqdm(range(n_episodes), desc="Episodes"):
        env.reset()
        if args.clean_scene:
            apply_clean_scene(env.env.sim)
        init_state = init_states[ep_idx].copy()
        if args.shift_dx != 0 or args.shift_dy != 0:
            init_state = shift_init_state(init_state, args.shift_dx, args.shift_dy)
            init_state = hide_distractors_in_state(init_state)
        obs = env.set_init_state(init_state)
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

        if args.cam_theta != 0.0 or args.cam_phi != 0.0:
            reposition_camera(env.env.sim, "agentview", args.cam_theta, args.cam_phi)
            obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

        # Per-episode BEV cam params (agentview may have been repositioned). v4 decodes XYZ
        # from voxel_positions, so no bev_xyz_table / pinhole K needed here.
        cur_bev_K = get_camera_intrinsic_matrix(env.env.sim, "agentview", image_size, image_size).astype(np.float32)  # pixel
        cur_bev_ext = get_camera_extrinsic_matrix(env.env.sim, "agentview").astype(np.float32)  # cam->world

        done = False; success = False
        step_idx = 0
        ep_frames = [] if (args.save_video_dir and ep_idx < args.save_video_max) else None
        # (initial frame skipped: overlay needs a prediction first)
        viz_panels = [] if (args.viz_rollout_dir and ep_idx < args.save_video_max) else None
        infer_idx = 0

        # Budget mode: by model predictions (--max_plans) or by env micro-steps (--max_steps).
        # In plan mode the teleport-servo micro-steps in between waypoints do NOT count toward the
        # budget; a high env-step safety cap only guards against a servo that never converges.
        use_plan_budget = args.max_plans > 0
        STEP_SAFETY = 20000
        n_plans = 0
        te_chunks = []; exec_step = 0   # temporal-ensemble buffer
        _ep_start = time.time()
        EP_WALL_LIMIT = float(os.environ.get('EP_WALL_LIMIT', '0'))
        def _budget_left():
            if EP_WALL_LIMIT > 0 and (time.time() - _ep_start) > EP_WALL_LIMIT:
                return False
            if step_idx >= STEP_SAFETY:
                return False
            return (n_plans < args.max_plans) if use_plan_budget else (step_idx < args.max_steps)

        while _budget_left() and not done:
            current_eef_pos  = np.array(obs["robot0_eef_pos"],  dtype=np.float64)
            current_eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float64)
            rgb_bev_obs   = obs["agentview_image"]
            rgb_wrist_obs = obs["robot0_eye_in_hand_image"]
            img_bev   = preprocess_obs(rgb_bev_obs,   image_size).to(device)
            img_wrist = preprocess_obs(rgb_wrist_obs, image_size).to(device)

            world_to_cam_bev = get_camera_transform_matrix(env.sim, "agentview", image_size, image_size)
            start_pix = eef_to_start_kp(current_eef_pos, world_to_cam_bev, image_size).to(device)
            if start_pix.dim() == 1: start_pix = start_pix.unsqueeze(0)

            wrist_ext_np = get_camera_extrinsic_matrix(env.sim, "robot0_eye_in_hand").astype(np.float32)  # cam->world
            wrist_K_np   = get_camera_intrinsic_matrix(env.sim, "robot0_eye_in_hand", image_size, image_size).astype(np.float32)  # pixel

            bev_K_use, wrist_K_use = cur_bev_K.copy(), wrist_K_np.copy()
            if args.upright:
                # match training: upright DINO input = flip images + intrinsics' v-axis (fy->-fy, cy->(S-1)-cy)
                img_bev   = torch.flip(img_bev,   dims=[2])
                img_wrist = torch.flip(img_wrist, dims=[2])
                start_pix = start_pix.clone(); start_pix[:, 1] = (image_size - 1) - start_pix[:, 1]
                for K in (bev_K_use, wrist_K_use):
                    K[1, 1] = -K[1, 1]; K[1, 2] = (image_size - 1) - K[1, 2]

            def _t(m): return torch.tensor(np.asarray(m), dtype=torch.float32, device=device).unsqueeze(0)
            rgb   = [img_bev, img_wrist]
            K_in  = [_t(bev_K_use), _t(wrist_K_use)]                                    # pixel intrinsics
            T_w2c = [_t(np.linalg.inv(cur_bev_ext)), _t(np.linalg.inv(wrist_ext_np))]   # world->cam

            with torch.no_grad():
                out = model(rgb, start_pix, K_in, T_w2c, lang_emb=lang_emb)
            n_plans += 1

            window_actions, pred_3d_list, pred_quat_list = decode_actions_v4(
                out, current_eef_pos, current_eef_quat,
                min_g, max_g, rot_pca_mean, rot_pca_axis, rot_pca_min, rot_pca_max,
                rot_centroids=rot_centroids,
            )
            if os.environ.get("DEBUG_FREEZE"):
                _dmm=[float(np.linalg.norm(np.asarray(pp,dtype=np.float64)-current_eef_pos))*1000 for pp in pred_3d_list]
                print(f"  infer {n_plans}: waypoint dist-from-gripper(mm)="+str([round(x) for x in _dmm]), flush=True)

            if args.z_drop:
                pred_3d_list = [np.asarray(pp, dtype=np.float64).copy() for pp in pred_3d_list]
                for _pp in pred_3d_list: _pp[2] -= args.z_drop
            _VIZ.clear(); _VIZ.update({"on": True, "pred": [np.asarray(p, dtype=np.float64) for p in pred_3d_list],
                       "Ks": bev_K_use, "Tsi": np.linalg.inv(cur_bev_ext), "S": image_size, "up": bool(args.upright)})
            if viz_panels is not None:
                panel = viz_inference_panel(
                    np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                    pred_3d_list, out,
                    bev_K_use, np.linalg.inv(cur_bev_ext),
                    wrist_K_use, np.linalg.inv(wrist_ext_np), image_size,
                    tag=f"inference {infer_idx}")
                viz_panels.append(panel); infer_idx += 1

            if args.temporal_ensemble:
                _grips=[float(a[6]) for a in window_actions]
                te_chunks.append({"s":exec_step,"pos":[np.asarray(p_,dtype=np.float64) for p_ in pred_3d_list],"quat":[np.asarray(q_) for q_ in pred_quat_list],"grip":_grips})
                te_chunks=[c for c in te_chunks if c["s"]+len(c["pos"])>exec_step]
                _newest=te_chunks[-1]["s"]; _ps=[];_qs=[];_gs=[];_ws=[]
                for c in te_chunks:
                    off=exec_step-c["s"]
                    if 0<=off<len(c["pos"]):
                        _ws.append(float(np.exp(-args.te_m*(_newest-c["s"])))); _ps.append(c["pos"][off]); _qs.append(c["quat"][off]); _gs.append(c["grip"][off])
                _ws=np.asarray(_ws); _ws=_ws/_ws.sum()
                ens_target=(_ws[:,None]*np.asarray(_ps)).sum(0)
                ens_grip=1.0 if float((_ws*np.asarray(_gs)).sum())>0.0 else -1.0
                ens_quat=None
                if args.servo_rot:
                    try: ens_quat=ScipyR.from_quat(np.asarray(_qs)).mean(weights=_ws).as_quat()
                    except Exception: ens_quat=_qs[-1]
                obs,n_servo,ep_done=servo_to_pos(env,ens_target.astype(np.float64),ens_grip,max_servo=25,frames=ep_frames,target_quat=ens_quat,grasp_settle=args.grasp_settle)
                step_idx+=n_servo; exec_step+=1
                if hasattr(env.env,"_check_success") and env.env._check_success(): success=True; done=True
                elif ep_done: done=True
                continue
            for t, action in enumerate(window_actions):
                if t >= args.replan_every:
                    break                                 # re-plan (re-infer) after replan_every waypoints
                if args.zero_rotation:
                    action[3:6] = 0.0
                if args.teleport:
                    # Servo to predicted 3D, then apply gripper
                    target = pred_3d_list[t].astype(np.float64)
                    gripper_cmd = float(action[6])
                    tgt_quat = pred_quat_list[t] if args.servo_rot else None
                    obs, n_servo, ep_done = servo_to_pos(env, target, gripper_cmd, max_servo=25, frames=ep_frames, target_quat=tgt_quat, grasp_settle=args.grasp_settle)
                    step_idx += n_servo
                    # SUCCESS is task completion (_check_success), NOT raw env-done (which fires at the
                    # horizon). ep_done only forces a stop (shouldn't happen with ignore_done=True).
                    if hasattr(env.env, '_check_success') and env.env._check_success():
                        success = True; done = True; break
                    if ep_done:
                        done = True; break
                else:
                    obs, _, done, _ = env.step(action)
                    step_idx += 1
                    if hasattr(env.env, '_check_success') and env.env._check_success():
                        success = True; done = True; break
                    if done: break
                # mid-chunk exit only for env-step budget mode (or the safety cap)
                if step_idx >= STEP_SAFETY or (not use_plan_budget and step_idx >= args.max_steps):
                    break
            if step_idx >= STEP_SAFETY or (not use_plan_budget and step_idx >= args.max_steps):
                break

        successes.append(int(success))
        if os.environ.get('STREAM_EVAL', '1') == '1':
            print('[EP %d] success=%s  running=%d/%d' % (ep_idx, success, sum(successes), len(successes)), flush=True)
            if args.out_json:
                Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
                json.dump({'n_done': len(successes), 'n_episodes': n_episodes,
                           'successes': successes, 'success_rate': float(np.mean(successes))},
                          open(args.out_json + '.partial', 'w'))
        step_counts.append(step_idx)
        plan_counts.append(n_plans)

        if viz_panels:
            Path(args.viz_rollout_dir).mkdir(parents=True, exist_ok=True)
            tag = "success" if success else "fail"
            h, w = viz_panels[0].shape[:2]
            vp = Path(args.viz_rollout_dir) / f"{args.benchmark}_task{args.task_id}_ep{ep_idx}_{tag}_viz.mp4"
            vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 2, (w, h))  # 2 fps: one frame per inference
            for pan in viz_panels:
                vw.write(cv2.resize(pan, (w, h)))
            vw.release()
            print(f"  wrote viz rollout: {vp} ({len(viz_panels)} inferences, {tag})", flush=True)

        if args.save_video_dir and ep_frames:
            Path(args.save_video_dir).mkdir(parents=True, exist_ok=True)
            tag = "success" if success else "fail"
            vp = Path(args.save_video_dir) / f"task{args.task_id}_ep{ep_idx}_{tag}.mp4"
            h, w = ep_frames[0].shape[:2]
            vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            for fr in ep_frames:
                vw.write(np.ascontiguousarray(fr))
            vw.release()
            print(f"  wrote rollout: {vp} ({len(ep_frames)} frames, {tag})")

    sr = float(np.mean(successes))
    print(f"\nSuccess Rate: {sr:.1%}  ({sum(successes)}/{n_episodes})  avg_steps={float(np.mean(step_counts)):.1f}  avg_plans={float(np.mean(plan_counts)):.1f}  (budget={'plans='+str(args.max_plans) if args.max_plans>0 else 'steps='+str(args.max_steps)})")

    if args.out_json:
        out = {
            "checkpoint": args.checkpoint, "shift_dx": args.shift_dx, "shift_dy": args.shift_dy,
            "n_episodes": n_episodes, "successes": successes, "step_counts": step_counts,
            "plan_counts": plan_counts, "max_plans": args.max_plans,
            "success_rate": sr, "clean_scene": args.clean_scene,
            "zero_rotation": args.zero_rotation, "teleport": args.teleport,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
