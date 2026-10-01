"""Pre-render libero demos with BOTH agentview AND robot0_eye_in_hand simultaneously.

Per demo, dumps:
  frames_bev/   — agentview JPGs (flipped to training convention)
  frames_wrist/ — robot0_eye_in_hand JPGs (flipped)
  eef_pos.npy, eef_quat.npy, gripper.npy, actions.npy   — robot state
  pix_uv_bev.npy   (T, 2) — EEF projected into agentview
  pix_uv_wrist.npy (T, 2) — EEF projected into wrist view (varies per frame)
  bev_K_norm.npy   (3, 3)        — agentview intrinsics, normalised
  bev_extrinsic.npy (4, 4)       — agentview camera→world (static)
  bev_world_to_cam.npy (4, 4)    — agentview world→camera (static)
  wrist_K_norm.npy (3, 3)        — wrist intrinsics, normalised
  wrist_world_to_cam.npy (T, 4, 4) — wrist world→camera per frame (varies)
  base_z.npy
"""
import argparse, os
from pathlib import Path
import numpy as np
import h5py
import cv2

from libero.libero import benchmark as bm_lib, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_camera_transform_matrix,
    project_points_from_world_to_camera,
)


def render_demo(env, states, actions, image_size, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "frames_bev").mkdir(parents=True, exist_ok=True)
    (out_dir / "frames_wrist").mkdir(parents=True, exist_ok=True)

    T = states.shape[0]
    eef_pos_list   = np.zeros((T, 3),   dtype=np.float32)
    eef_quat_list  = np.zeros((T, 4),   dtype=np.float32)
    gripper_list   = np.zeros((T,),     dtype=np.float32)
    pix_uv_bev     = np.zeros((T, 2),   dtype=np.float32)
    pix_uv_wrist   = np.zeros((T, 2),   dtype=np.float32)
    wrist_wtc_seq  = np.zeros((T, 4, 4), dtype=np.float32)

    bev_K_norm     = None
    bev_extrinsic  = None
    bev_wtc        = None
    wrist_K_norm   = None
    base_z         = None

    for t in range(T):
        obs = env.set_init_state(states[t])
        env.env.sim.forward()

        # ---- RGB save: both cameras ----
        for cam_name, sub in [("agentview", "frames_bev"), ("robot0_eye_in_hand", "frames_wrist")]:
            key = f"{cam_name}_image" if cam_name != "robot0_eye_in_hand" else "robot0_eye_in_hand_image"
            rgb = np.asarray(obs[key]).copy()
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            rgb = np.ascontiguousarray(np.flipud(rgb))
            if rgb.shape[0] != image_size or rgb.shape[1] != image_size:
                rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out_dir / sub / f"{t:06d}.jpg"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ---- EEF state ----
        eef_pos  = np.asarray(obs["robot0_eef_pos"],  dtype=np.float64)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
        eef_pos_list[t]  = eef_pos.astype(np.float32)
        eef_quat_list[t] = eef_quat.astype(np.float32)
        gripper_list[t]  = float(np.clip(actions[min(t, len(actions) - 1), 6], -1.0, 1.0))

        # ---- Camera transforms ----
        h, w = image_size, image_size
        bev_wtc_t   = get_camera_transform_matrix(env.env.sim, "agentview", h, w)
        wrist_wtc_t = get_camera_transform_matrix(env.env.sim, "robot0_eye_in_hand", h, w)

        # Project EEF into each camera
        for wtc, target in [(bev_wtc_t, pix_uv_bev), (wrist_wtc_t, pix_uv_wrist)]:
            pix_rc = project_points_from_world_to_camera(
                points=eef_pos.reshape(1, 3),
                world_to_camera_transform=wtc,
                camera_height=h, camera_width=w,
            )[0]
            v_raw, u_raw = float(pix_rc[0]), float(pix_rc[1])
            target[t] = [np.clip(u_raw, 0, w - 1), np.clip(v_raw, 0, h - 1)]

        wrist_wtc_seq[t] = wrist_wtc_t.astype(np.float32)

        if bev_K_norm is None:
            bev_K = get_camera_intrinsic_matrix(env.env.sim, "agentview", h, w).astype(np.float32)
            bev_K_norm = bev_K.copy()
            bev_K_norm[0] /= float(w); bev_K_norm[1] /= float(h)
            bev_extrinsic = get_camera_extrinsic_matrix(env.env.sim, "agentview").astype(np.float32)
            bev_wtc = bev_wtc_t.astype(np.float32)
            wrist_K = get_camera_intrinsic_matrix(env.env.sim, "robot0_eye_in_hand", h, w).astype(np.float32)
            wrist_K_norm = wrist_K.copy()
            wrist_K_norm[0] /= float(w); wrist_K_norm[1] /= float(h)
            base_body_id = env.env.sim.model.body_name2id("robot0_base")
            base_z = float(env.env.sim.data.xpos[base_body_id][2]) if base_body_id >= 0 else 0.0

    np.save(out_dir / "eef_pos.npy",          eef_pos_list)
    np.save(out_dir / "eef_quat.npy",         eef_quat_list)
    np.save(out_dir / "gripper.npy",          gripper_list)
    np.save(out_dir / "pix_uv_bev.npy",       pix_uv_bev)
    np.save(out_dir / "pix_uv_wrist.npy",     pix_uv_wrist)
    np.save(out_dir / "bev_K_norm.npy",       bev_K_norm)
    np.save(out_dir / "bev_extrinsic.npy",    bev_extrinsic)
    np.save(out_dir / "bev_world_to_cam.npy", bev_wtc)
    np.save(out_dir / "wrist_K_norm.npy",     wrist_K_norm)
    np.save(out_dir / "wrist_world_to_cam.npy", wrist_wtc_seq)
    np.save(out_dir / "base_z.npy",           np.float32(base_z))
    np.save(out_dir / "actions.npy",          actions.astype(np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark",  type=str, default="libero_spatial")
    p.add_argument("--out_root",   type=str, default="./data")
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--task_ids",   type=str, default="0")
    p.add_argument("--max_demos",  type=int, default=None)
    args = p.parse_args()

    bench    = bm_lib.get_benchmark_dict()[args.benchmark]()
    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip() != "all"] if args.task_ids != "all" \
               else list(range(bench.get_num_tasks()))

    for task_id in task_ids:
        task = bench.get_task(task_id)
        demo_path = os.path.join(get_libero_path("datasets"), bench.get_task_demonstration(task_id))
        print(f"\nTask {task_id}: {task.name}")

        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=args.image_size, camera_widths=args.image_size,
            camera_names=["agentview", "robot0_eye_in_hand"],
        )

        with h5py.File(demo_path, "r") as f:
            demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
            if args.max_demos:
                demo_keys = demo_keys[:args.max_demos]
            for demo_key in demo_keys:
                states  = f[f"data/{demo_key}/states"][:]
                actions = f[f"data/{demo_key}/actions"][:]
                out_dir = Path(args.out_root) / args.benchmark / f"task_{task_id}" / demo_key
                if (out_dir / "actions.npy").exists():
                    print(f"  skip {demo_key} (already exists)")
                    continue
                env.reset()
                render_demo(env, states, actions, args.image_size, out_dir)
                print(f"  done {demo_key}: T={states.shape[0]}")
        env.close()


if __name__ == "__main__":
    main()
