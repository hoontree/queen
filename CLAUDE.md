# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Conda environment: `queen` (Python 3.11, CUDA 11.8+)

**All commands must be run with `mamba run -n queen`:**
```bash
mamba run -n queen python train.py ...
```

## Key Commands

**Training:**
```bash
mamba run -n queen python train.py --config configs/dynerf.yaml -s data/n3dv/flame_steak -m ./output/flame_steak
```

**Rendering:**
```bash
mamba run -n queen python render.py -s <scene_path> -m <model_path>
mamba run -n queen python render_fvv.py --config configs/dynerf.yaml -s <scene_path> -m <model_path>
mamba run -n queen python render_fvv_compressed.py --config configs/dynerf.yaml -s <scene_path> -m <model_path>
```

**Metrics:**
```bash
mamba run -n queen python metrics.py -m <model_path>
mamba run -n queen python metrics_video.py -m <model_path>
```

**N3DV data preparation** (MP4 → per-camera image folders):
```bash
mamba run -n queen python convert_n3dv.py --scene_path data/n3dv/flame_steak
```
This extracts frames from `camXX.mp4` into `camXX/images/XXXX.png` and auto-generates `points3D_downsample2.ply` from `poses_bounds.npy` (random point initialization — sufficient for Gaussian densification to correct).

## Architecture Overview

QUEEN is a dynamic 3D Gaussian Splatting framework. It trains an **initial frame** with full densification (500 epochs), then trains **residual frames** sequentially (10 epochs each) using optical flow-guided updates. Gaussians are compressed via learned scalar quantization with latent decoders per attribute.

### Dataset Detection (`scene/__init__.py`)

Dataset type is auto-detected by file presence in `source_path`:
- `sparse/` → Colmap
- `poses_bounds.npy` → DyNeRF / N3DV (`readDynerfInfo`)
- `train_meta.json` → Panoptic
- `models.json` → Google Immersive
- `transforms_train.json` → Blender/NeRF Synthetic

### Key Modules

- **`train.py`** — main training loop; handles initial frame + residual frames separately; includes per-component latency profiling
- **`scene/gaussian_model.py`** — `GaussianModel` with per-attribute latent decoders (`DecoderSQ`, `DecoderIdentity`, etc.); `create_from_pcd` initializes from PLY, `create_from_gaussians` initializes residual frames from previous frame
- **`scene/dataset_readers.py`** — `readDynerfInfo` reads `poses_bounds.npy` + `camXX/images/`; `readImmersiveSceneInfo` for Google Immersive
- **`scene/decoders.py`** — quantization decoders (`sq`, `sq_res`, `none`) applied per Gaussian attribute
- **`utils/loader_utils.py`** — `MultiViewVideoDataset` loads per-camera image sequences from `camXX/images/` directories
- **`gaussian_renderer/`** — differentiable rasterizer wrapper

### Config Structure (`configs/dynerf.yaml`)

Four param groups map to argument classes in `arguments/__init__.py`:
- `model_params` → `ModelParams` (flow loss, depth init, MiDaS depth, gating)
- `quantize_params` → `QuantizeParams` (per-attribute quantization type and latent dims)
- `opt_params_initial` → `OptimizationParamsInitial` (first-frame training)
- `opt_params_rest` → `OptimizationParamsRest` (residual frame training)

### Data Format (N3DV / DyNeRF)

```
data/n3dv/<scene>/
  camXX/images/0000.png ... 0299.png   # one folder per camera
  poses_bounds.npy                      # camera poses + near/far (LLFF format)
  points3D_downsample2.ply             # point cloud for Gaussian init
```

`readDynerfInfo` reads image size from `cam00/images/0000.png`, loads all camera poses from `poses_bounds.npy`, and dispatches to `points3D_downsample2.ply` only when `"dynerf"` is in the path string — otherwise looks for `colmap/dense/workspace/fused.ply`.

### MiDaS Depth

Depth supervision uses MiDaS (`dpt_beit_large_512.pt`, must be downloaded to `MiDaS/weights/`). Controlled by `depth_init`, `lambda_depthssim`, `depth_from_iter`/`depth_until_iter` in config.

**Streaming Training (MP4 → frame-by-frame via OpenCV):**
```bash
mamba run -n queen python train_streaming.py \
    --config configs/n3dv.yaml \
    --video_dir data/n3dv/flame_steak \
    -s data/n3dv/flame_steak \
    -m ./output/streaming/flame_steak
```
- `--video_dir`: `cam*.mp4` 파일들이 있는 디렉토리 (카메라당 1개)
- `-s`: scene geometry 정보(`poses_bounds.npy`, `points3D_downsample2.ply`)가 있는 디렉토리 (N3DV의 경우 `--video_dir`과 동일)
- `lambda_flow`는 자동으로 `0.0`으로 강제됨 (다음 프레임이 없으므로)
- `adaptive_iters`는 자동으로 `False`로 강제됨 (`frame_diff.json` 불필요)

## Component Latency Profiling (`train.py`)

Per-iteration latency is measured for six pipeline components using wall-clock time (with optional CUDA sync via `dataset.timed`):

| Component | What is measured | Script |
|---|---|---|
| `gaussian_selection` | pixel mask (`pix_thresh_vals`) selection per iteration | both |
| `render_forward` | `render_mask()` call including LatentDecoder (decode is internal) | both |
| `motion_estimation` | flow loss computation (`render_mask_shift` or `flow_warp`) | both |
| `loss_backward` | `loss.backward()` | both |
| `optimizer_step` | `optimizer.step()` + gate optimizer step | both |
| `densify_prune` | densification stats, `densify_and_prune`/`densify_dynamic`, `influence_prune` | both |
| `video_decode` | OpenCV frame decode (all cameras combined) — **per frame, not per iteration** | `train_streaming.py` only |

> **Note:** `decode` (LatentDecoder) runs inside `render_forward` inside the CUDA rasterizer and cannot be timed separately. It is reported as an alias of `render_forward` in all outputs.

**Outputs:**
- `<model_path>/timing_metrics.json` — `total_sec`, `num_measurements`, `avg_ms` per component
- wandb metrics `frame/latency/<component>_ms` (per frame, step = `frame_idx`)
- wandb summary `average/latency/<component>_ms` (whole-run average)
- Console summary printed at end of training

**Enabling GPU-accurate timing:**
Set `dataset.timed = True` (or `timed: true` in config) to insert `torch.cuda.synchronize()` before each timestamp, preventing GPU/CPU overlap from under-reporting latency.
