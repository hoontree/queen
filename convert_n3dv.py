"""
Convert N3DV dataset (camXX.mp4 + poses_bounds.npy) to the expected directory structure:
  [scene_name]/
    camXX/
      images/
        0000.png
        0001.png
        ...
    poses_bounds.npy
    points3D_downsample2.ply  (auto-generated from poses_bounds.npy)

Usage:
  python convert_n3dv.py --scene_path data/n3dv/flame_steak [--max_frames N] [--n_pts 100000] [--workers N]
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser("N3DV to queen format converter")
parser.add_argument("--scene_path", "-s", required=True, type=str)
parser.add_argument("--max_frames", default=None, type=int)
parser.add_argument("--n_pts", default=100_000, type=int)
parser.add_argument("--workers", default=None, type=int,
                    help="Number of parallel workers (default: number of cameras)")
args = parser.parse_args()

scene_path = Path(args.scene_path)
assert scene_path.exists(), f"Scene path not found: {scene_path}"

mp4_files = sorted(scene_path.glob("cam*.mp4"))
assert len(mp4_files) > 0, f"No cam*.mp4 files found in {scene_path}"


def extract_frames(mp4_path: Path):
    cam_name = mp4_path.stem
    out_dir = mp4_path.parent / cam_name / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = args.max_frames if args.max_frames is not None else total_frames

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(out_dir / f"{frame_idx:04d}.png"), frame)
        frame_idx += 1

    cap.release()
    return cam_name, frame_idx


print(f"Found {len(mp4_files)} camera videos in {scene_path}")

n_workers = min(args.workers or len(mp4_files), cpu_count())
print(f"Extracting frames with {n_workers} workers...")

with Pool(n_workers) as pool:
    for cam_name, n_frames in pool.imap_unordered(extract_frames, mp4_files):
        print(f"  {cam_name}: {n_frames} frames written")

# --- Generate PLY from poses_bounds.npy ---
ply_dst = scene_path / "points3D_downsample2.ply"
if ply_dst.exists():
    print(f"PLY already exists, skipping: {ply_dst}")
else:
    poses_path = scene_path / "poses_bounds.npy"
    assert poses_path.exists(), f"poses_bounds.npy not found: {poses_path}"

    poses_arr = np.load(poses_path)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    near_fars = poses_arr[:, -2:]

    cam_centers = poses[:, :3, 3]
    scene_center = cam_centers.mean(axis=0)
    scene_radius = np.linalg.norm(cam_centers - scene_center, axis=1).max()
    near = near_fars[:, 0].min()
    far = near_fars[:, 1].max()

    rng = np.random.default_rng(0)
    pts = rng.standard_normal((args.n_pts, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    radii = rng.uniform(near, far, size=(args.n_pts, 1))
    pts = scene_center + pts * radii * (scene_radius / far)

    colors = (rng.random((args.n_pts, 3)) * 255).astype(np.uint8)
    normals = np.zeros((args.n_pts, 3), dtype=np.float32)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elements = np.empty(args.n_pts, dtype=dtype)
    elements['x'], elements['y'], elements['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    elements['nx'], elements['ny'], elements['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
    elements['red'], elements['green'], elements['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]

    PlyData([PlyElement.describe(elements, 'vertex')]).write(str(ply_dst))
    print(f"Generated PLY with {args.n_pts} random points: {ply_dst}")

print("Done.")
