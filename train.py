#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import os
import sys
import torch
import socket
from random import randint, Random
from utils.loss_utils import l1_loss, ssim, l2_loss, tv_loss, lp_loss, DepthRelLoss, mse_loss
from gaussian_renderer import render, network_gui, render_mask, render_mask_shift
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import cv2
import copy
import uuid
import json
import time
import yaml
import hashlib
import functools
import torchvision
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import torchvision.transforms.functional as F
from utils.image_utils import psnr, save_image, value2color
from scene.cameras import SequentialCamera, camName_from_Path, imageName_from_Path
from argparse import ArgumentParser, Namespace
from utils.general_utils import DecayScheduler, kthvalue
from utils.graphics_utils import adjust_depths
from utils.image_utils import resize_image, downsample_image, blur_image, get_mask, write_depth, coords_grid, flow_warp, coords_grid_proj, get_depth, resize_dims
from utils.loader_utils import MultiViewVideoDataset
from utils.loader_utils import SequentialMultiviewSampler, MultiViewVideoDataset
from arguments import ModelParams, PipelineParams, OptimizationParams, QuantizeParams, OptimizationParamsInitial, OptimizationParamsRest
from scene.utils import get_depth_model, get_depth_poses
from torchmetrics.functional.regression import pearson_corrcoef
from MiDaS.run import process
from scene.decoders import LatentDecoder, LatentDecoderRes, Gate
from generate_video_all import symlink

# Disable tqdm to make pdb easier to use
# Set to False to disable progress bars for debugging
enable_tqdm = True
enable_debug = False

EPS = 1.0e-7
try:
    from torch.utils.tensorboard import SummaryWriter
    if not ('SLURM_PROCID' in os.environ and os.environ['SLURM_PROCID']!='0'):
        TENSORBOARD_FOUND = True
    else:
        TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    if not ('SLURM_PROCID' in os.environ and os.environ['SLURM_PROCID']!='0'):
        WANDB_FOUND = True
    else:
        WANDB_FOUND = False
except ImportError:
    WANDB_FOUND = False


def _sync_cuda(enabled):
    """torch.cuda.synchronize() guarded by a flag and CUDA availability."""
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def next_batch_with_latency(loader, sync_cuda=False):
    """Pull one batch from a DataLoader and move it to CUDA, returning the wall
    time spent. With num_workers=0 (timed mode) this includes the actual RGB
    decode (PIL.Image.open + ToTensor) since it runs synchronously in-process.
    Returns (images_cuda, paths, elapsed_seconds)."""
    _sync_cuda(sync_cuda)
    t0 = time.perf_counter()
    images, paths = next(loader)
    images = images.cuda()
    _sync_cuda(sync_cuda)
    return images, paths, time.perf_counter() - t0


def _latency_stats(values):
    """count/mean/std/min/max for a list of per-frame latencies (seconds)."""
    n = len(values)
    if n == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {
        "count": n,
        "mean": round(mean, 6),
        "std": round(var ** 0.5, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, qp:QuantizeParams, testing_iterations: list,
             saving_iterations: list, checkpoint_iterations, checkpoint: str, debug_from, args):
    """Main training function for QUEEN compressed Gaussian splatting."""
    wandb_enabled = WANDB_FOUND and dataset.use_wandb
    tb_writer = prepare_output_and_logger(args)
    generator = Random(dataset.seed)
    qp.seed = dataset.seed

    qp.use_shift = [bool(el) for el in qp.use_shift]

    # Create dataset and loader for training and testing at each time instance
    train_image_dataset = MultiViewVideoDataset(dataset.source_path, split='train', test_indices=dataset.test_indices,
                                                max_frames=dataset.max_frames, start_idx=dataset.start_idx, img_format=dataset.img_fmt)
    test_image_dataset = MultiViewVideoDataset(dataset.source_path, split='test', test_indices=dataset.test_indices, 
                                               max_frames=dataset.max_frames, start_idx=dataset.start_idx, 
                                               img_format=dataset.img_fmt)

    train_sampler = SequentialMultiviewSampler(train_image_dataset)
    if test_image_dataset.n_cams > 0:
        test_sampler = SequentialMultiviewSampler(test_image_dataset)

    # Under timed mode use num_workers=0 so PNG decode runs synchronously in-process
    # and is captured inside the decode-timing region (background workers would hide it).
    loader_workers = 0 if dataset.timed else 4
    train_loader = iter(torch.utils.data.DataLoader(train_image_dataset, batch_size=train_image_dataset.n_cams,
                                                    sampler=train_sampler, num_workers=loader_workers))
    if test_image_dataset.n_cams > 0:
        test_loader = iter(torch.utils.data.DataLoader(test_image_dataset, batch_size=test_image_dataset.n_cams,
                                                        sampler=test_sampler, num_workers=loader_workers))
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    print(f"training(): dataset.white_background set to {dataset.white_background}")
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # Initial set of images to initialize camera and camera parameters
    # Image dimensions should remain constant throughout the video
    print(f"training(): loading data for the first frame...")
    tic = time.time()
    # Decode time (frame 1): make the first frame's training-view RGBs available on GPU.
    # Test-view decode is evaluation overhead and is excluded from the decode bucket.
    train_images, train_paths, frame1_decode_time = next_batch_with_latency(train_loader, sync_cuda=dataset.timed)

    if test_image_dataset.n_cams > 0:
        test_images, test_paths, _ = next_batch_with_latency(test_loader, sync_cuda=dataset.timed)
        test_image_data = {'image':test_images,'path':test_paths,'frame_idx':0}
    else:
        print('No test cameras found, disabling testing.')
        test_images, test_paths = None, None
        test_image_data = {'image':None,'path':None,'frame_idx':0}

    train_image_data = {'image':train_images,'path':train_paths,'frame_idx':0}

    print(f"training(): data loaded in {float(time.time() - tic):.2f} sec")

    # Resume frame 1 from a prebuilt .ply (skips the static 3DGS training below).
    if checkpoint is not None:
        if not (isinstance(checkpoint, str) and checkpoint.endswith(".ply")):
            raise ValueError(f"--start_checkpoint must be a .ply file, got: {checkpoint}")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"--start_checkpoint not found: {checkpoint}")
        print(f"training(): resuming frame 1 from checkpoint ply: {checkpoint}")

    # Static 3DGS initialization (frame 1, one-time). Reported in the dedicated
    # initialization table and excluded from steady-state per-frame latency.
    # When resuming from a checkpoint, this region instead measures the ply-load time.
    _sync_cuda(dataset.timed)
    static_init_start = time.time()
    # Create the gaussian model and scene, initialized with frame 1 images from dataset
    gaussians = GaussianModel(dataset.sh_degree, qp, dataset, use_xyz_legacy=args.use_xyz_legacy)

    max_frames = args.max_frames
    scene = Scene(
        dataset,
        gaussians,
        train_image_data= train_image_data,
        test_image_data=test_image_data,
        N_video_views=max_frames
    )
    if checkpoint is not None:
        # Overwrite the create_from_pcd init with the prebuilt frame-1 Gaussian.
        gaussians.load_ply(checkpoint)
    # Setup training arguments (rebuilds the optimizer over the loaded params)
    gaussians.training_setup(opt)
    _sync_cuda(dataset.timed)
    static_init_time = time.time() - static_init_start

    # Spiral cameras
    video_cameras = scene.getVideoCameras()

    # Metadata used by various components
    train_cameras = scene.getTrainCameras()
    n_frames, n_cams = train_image_dataset.n_frames, train_image_dataset.n_cams
    print(f"training(): running with {n_frames} frames from {n_cams} cameras")
    opt.iterations = opt.epochs*n_cams
    print(f"training(): opt.iterations set to {opt.iterations}")
    _,H,W = train_cameras[0].original_image.shape

    cur_frame_views = train_image_data['image']
    prev_frame_views = cur_frame_views

    # Vary number of iterations based on frame difference in json file
    if dataset.adaptive_iters and n_frames>1:
        frame_diff = json.load(open(os.path.join(dataset.source_path,'frame_diff.json'),'r'))['l2']
        frame_diff = np.array(frame_diff[:n_frames-1])
        epochs_rest = opt.opt_rest['epochs_rest']
        mult = np.clip(frame_diff/frame_diff.mean(),1/4,4) # between 0.25 to 4
        mult = mult/mult.mean()
        frame_epochs = np.ceil((mult*epochs_rest)).astype(np.int32)
        frame_iters = np.concatenate((np.array([opt.iterations]),frame_epochs*n_cams))
    else:
        frame_iters = np.array([opt.iterations]+[opt.opt_rest['epochs_rest']*n_cams]*(n_frames-1))
    # Depth supervision/init is frame-1-training-only; skip it when resuming from a checkpoint.
    if (opt.lambda_depth>0.0 or dataset.depth_init) and checkpoint is None:

        ## MiDas model for monocular depth estimation
        depth_model, transform, net_w, net_h = get_depth_model(dataset)
        for camera in train_cameras:
            gt_image = camera.original_image.permute(1,2,0).detach().cpu().numpy()
            image = transform({"image": gt_image})["image"]
            with torch.no_grad():
                prediction = process(torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                                     depth_model, 'dpt_beit_large_512', image, (net_w, net_h), 
                                     gt_image.shape[1::-1],
                                     False, False)
                camera.gt_depth = torch.tensor(prediction).cuda()

        # Add points to gaussian model using the monocular depth
        if dataset.depth_init:
            gaussians.create_from_depth_immersive(cameras=train_cameras, spatial_lr_scale=gaussians.spatial_lr_scale, downsample_scale=1,
                                        alpha_thresh=dataset.depth_thresh, renderFunc = functools.partial(render_mask, 
                                                                                        pipe=pipe, 
                                                                                        bg_color=bg, 
                                                                                        image_shape=camera.original_image.shape, 
                                                                                        color_mask=None, 
                                                                                        render_depth=True))
            
        # Loss function for relative depth
        depth_loss_fn = DepthRelLoss(camera.original_image.shape[1], camera.original_image.shape[2],
                                     pix_diff=dataset.depth_pix_range, num_comp=dataset.depth_num_comp, 
                                     tolerance=dataset.depth_tolerance)

    # Progressive training scheduler - OBSOLETE: Remove in future cleanup
    resize_scale_sched = DecayScheduler(
                                        total_steps=int(opt.resize_period*(opt.iterations+1)),
                                        decay_name='cosine',
                                        start=opt.resize_scale,
                                        end=1.0,
                                        )

    start_frame_idx = 1
    training_metrics = []
    net_elapsed_time = 0.0
    net_iter_time = 0.0

    # Coarse 3-bucket latency reported per frame (seconds): Decode, Update,
    # Render (held-out latency render of G_t into one held-out view), plus E2E.
    LATENCY_FIELDS = ["Decode time", "Update time", "Render time (heldout)", "E2E time"]

    # Coarse 3-bucket latency: Decode / Update / Render(+E2E), with CUDA syncs ONLY at
    # frame-level bucket boundaries (no per-stage/per-iteration syncs). The Update bucket
    # is a single wall-clock span over the reconstruction (frame setup + training loop);
    # decode (binding), the held-out render, and validation/spiral renders are kept out of
    # it via pause()/resume() or by subtracting the synced eval-render time.
    class SpanTimer:
        """Single wall-clock accumulator with optional CUDA sync at span boundaries."""
        def __init__(self, sync):
            self.sync = sync
            self.total = 0.0
            self._t0 = None
        def reset(self):
            self.total = 0.0
            self._t0 = None
        def resume(self):
            if self.sync:
                torch.cuda.synchronize()
            self._t0 = time.time()
        def pause(self):
            if self._t0 is None:
                return
            if self.sync:
                torch.cuda.synchronize()
            self.total += time.time() - self._t0
            self._t0 = None

    update_sw = SpanTimer(sync=dataset.timed)

    training_start = time.time()

    # Define video-wide metrics for wandb logging
    if wandb_enabled:
        wandb.define_metric("frame_idx")
        wandb.define_metric("frame/num_iterations", step_metric="frame_idx")
        wandb.define_metric("frame/test/loss_viewpoint/psnr", step_metric="frame_idx")
        wandb.define_metric("frame/test/loss_viewpoint/loss", step_metric="frame_idx")
        wandb.define_metric("frame/val/loss_viewpoint/psnr", step_metric="frame_idx")
        wandb.define_metric("frame/val/loss_viewpoint/loss", step_metric="frame_idx")
        wandb.define_metric("frame/size", step_metric="frame_idx")
        wandb.define_metric("frame/num_points", step_metric="frame_idx")
        wandb.define_metric("frame/update_points", step_metric="frame_idx")
        wandb.define_metric("frame/iter_time", step_metric="frame_idx")
        wandb.define_metric("frame/iter_time_io", step_metric="frame_idx")
        wandb.define_metric("frame/rendering_time", step_metric="frame_idx")
        wandb.define_metric("frame/rendering_frames", step_metric="frame_idx")
        wandb.define_metric("frame/rendering_fps", step_metric="frame_idx")
        wandb.define_metric("frame/elapsed", step_metric="frame_idx")
        for _lat in ["decode", "update", "render", "e2e"]:
            wandb.define_metric(f"frame/latency/{_lat}", step_metric="frame_idx")

    if opt.lambda_flow > 0.0:
        grid = coords_grid(1,H,W, device='cuda')

    # Train-decode latency of the current frame. Frame 1 is decoded at the top; each
    # later frame's train data is prefetched during the previous iteration, so its
    # measured decode cost is carried over here to the frame it belongs to.
    pending_train_decode_time = frame1_decode_time

    if enable_tqdm:
        progress_bar_frame = tqdm(range(1, n_frames+1), desc="Training progress")
        progress_bar_frame.update(start_frame_idx-1)
    else:
        progress_bar_frame = None
        frame_counter = 0

    # start frame index loop
    for frame_idx in range(start_frame_idx, n_frames+1):

        # Frame-wise metrics for wandb logging
        if wandb_enabled and frame_idx <= 2:
            frame_str = f"{str(frame_idx).zfill(4)}"
            iter_metric = "iter_"+frame_str
            frame_str = "frame_"+frame_str
            wandb.define_metric(iter_metric)
            wandb.define_metric(frame_str+"/test/loss_viewpoint/best_psnr", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/test/loss_viewpoint/psnr", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/test/loss_viewpoint/l1_loss", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/val/loss_viewpoint/psnr", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/val/loss_viewpoint/l1_loss", step_metric=iter_metric)

            wandb.define_metric(frame_str+"/train_loss_patches/l1_loss", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/train_loss_patches/total_loss", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/num_points", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/update_points", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/elapsed", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/size", step_metric=iter_metric)
            

        first_iter = 1
        scene.model_path = os.path.join(args.model_path,'frames',str(dataset.start_idx + frame_idx).zfill(4))

        os.makedirs(scene.model_path,exist_ok=True)

        ema_loss_for_log, cur_size, best_psnr = 0.0, 0.0, 0.0
        metrics = {'val':{'psnr':0.0, 'loss':0.0}, 'test':{'psnr':0.0, 'loss': 0.0}}
        camera_idx_stack = []
        report = None

        if dataset.timed:
            torch.cuda.synchronize()
        frame_start_io = time.time()
        frame_time_io = 0.0
        update_sw.reset()
        eval_render_time = 0.0   # validation/spiral renders inside the loop (excluded from Update)
        render_time = 0.0        # held-out render bucket
        rendering_frames = 0

        # Decode latency of the current frame's training views (carried over from the
        # previous iteration's prefetch; for frame 1 set from the top-level decode).
        decode_time = pending_train_decode_time

        try:
            # Pre-load data for next frame (this measures the NEXT frame's train decode)
            next_train_images, next_train_paths, pending_train_decode_time = next_batch_with_latency(
                train_loader, sync_cuda=dataset.timed)
            next_frame_views = next_train_images

            orig_size = cur_frame_views.shape[-2:]
            rescaled_size = resize_dims(orig_size, dataset.flow_scale)

        except StopIteration:
            assert frame_idx == n_frames
            pending_train_decode_time = 0.0
            opt.lambda_flow = 0.0

        # Begin the Update span (frame setup + training loop). resume() does the only
        # frame-boundary CUDA sync; the held-out render and eval renders are kept out below.
        update_sw.resume()
        frame_start = time.time()
        frame_time = 0.0

        # Update a bunch of variables and models for each new frame
        if frame_idx > 1:

            # Initialize gate probabilities based on gradient differences or frame differences
            if dataset.update_mask == "viewspace_diff":
                # Compute viewspace gradient differences for gate initialization
                grad_diff = torch.zeros(gaussians.get_xyz.shape[0],1).to(gaussians._xyz)
                denom = torch.zeros(gaussians.get_xyz.shape[0],1).to(gaussians._xyz)
                gaussians.optimizer.zero_grad(set_to_none=True)
                for cam_idx, camera in enumerate(train_cameras):
                    render_pkg = render_mask(camera, gaussians, pipe, bg, image_shape=camera.original_image.shape)
                    camera.prev_rendered = render_pkg["render"].detach()
                    image, viewspace_point_tensor = render_pkg["render"], render_pkg["viewspace_points"]
                    visibility_filter = render_pkg["visibility_filter"]
                    cur_gt_image = cur_frame_views[cam_idx]
                    prev_gt_image = prev_frame_views[cam_idx]
                    if dataset.update_loss == "mae":
                        Ll1 = mse_loss(image, cur_gt_image)
                        Ll1_prev = mse_loss(image, prev_gt_image)
                    elif dataset.update_loss == "mse":
                        Ll1 = mse_loss(image, cur_gt_image)
                        Ll1_prev = mse_loss(image, prev_gt_image)
                    elif dataset.update_loss == "ssim":
                        Ll1 = 1.0-ssim(image, cur_gt_image)
                        Ll1_prev = 1.0-ssim(image, prev_gt_image)
                    elif dataset.update_loss == "mae_orig":
                        Ll1 = l1_loss(image, cur_gt_image)
                        Ll1_prev = l1_loss(image, prev_gt_image)
                    cur_loss = Ll1-Ll1_prev
                    cur_loss.backward()
                    cur_grad = viewspace_point_tensor.grad[visibility_filter,:2].clone()
                    with torch.no_grad():
                        viewspace_point_tensor.grad *= 0
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    grad_diff[visibility_filter] += torch.norm(cur_grad,dim=-1,keepdim=True)
                    denom[visibility_filter] += 1

                grad_diff[grad_diff.isnan()] = 0.0

                with torch.no_grad():
                    if dataset.adaptive_render and dataset.adaptive_update_period>0.0:
                        for camera in train_cameras:
                            grad_mask = (grad_diff.flatten()>dataset.pixel_update_thresh)
                            render_pkg = render_mask(camera, scene.gaussians, pipe, bg, 
                                                    gaussian_mask=grad_mask)
                            alphamask = (render_pkg["alpha"]>0.5).float()
                            camera.orig_mask = alphamask
                            mask_down = torch.nn.functional.max_pool2d(alphamask.unsqueeze(0), (dataset.dilate_size,dataset.dilate_size))
                            mask_dilate = torch.nn.functional.interpolate(mask_down, size=(alphamask.shape[-2],alphamask.shape[-1]))
                            camera.mask = (mask_dilate.squeeze(0).squeeze(0)>0).float()

                gaussian_mask = grad_diff>dataset.gaussian_update_thresh

            with torch.no_grad():

                # Load optimizer hyperparams (initial or rest) based on frame index
                opt.set_params(frame_idx)
                opt.iterations = frame_iters[frame_idx-1]
                opt.epochs = (opt.iterations//n_cams)
                gaussians.frame_idx = frame_idx
                # Create decoder and latents for quantized residuals if first time
                # Else reset latent values to 0
                gaussians.update_residuals()
                # Redefine the optimizer and other tracked variables for the gaussian model
                gaussians.training_setup(opt)
                # Load the current test data (Preloaded data for next frame is only for training)
                train_images, train_paths = cur_train_images, cur_train_paths
                update_sw.pause()  # exclude test-data load from the Update span
                frame_time += time.time() - frame_start
                if test_image_dataset.n_cams > 0:
                    test_data = next(test_loader)
                    test_images, test_paths = test_data[0].cuda(), test_data[1]
                else:
                    if frame_idx == start_frame_idx:
                        print('No test cameras found, disabling testing.')
                    test_images, test_paths = None, None
                update_sw.resume()
                frame_start = time.time()
                train_image_data = {'image':train_images,'path':train_paths}
                test_image_data = {'image':test_images,'path':test_paths}

                # Update the images and paths for all cameras in the scene with new frame index.
                # Binding the decoded tensors into camera objects makes the frame available to
                # the model, so its cost is folded into decode_time (excluded from Update).
                update_sw.pause()
                _bind_t0 = time.time()
                scene.updateCameraImages(args, train_image_data, test_image_data, frame_idx, resolution_scales=[1.0])
                _sync_cuda(dataset.timed)
                decode_time += time.time() - _bind_t0
                update_sw.resume()
                train_cameras = scene.getTrainCameras()

                # If using a frame difference or 2d flow mask for gate initialization and adaptive masked training
                if dataset.update_mask =="diff":
                    flow_norm = torch.norm((prev_frame_views-cur_frame_views),dim=1,keepdim=True)/np.sqrt(3) # normalize across rgb

                    # Mask if using fixed threshold
                    flow_mask = flow_norm>dataset.pixel_update_thresh

                if dataset.update_mask =="diff":
                    if dataset.adaptive_render and dataset.adaptive_update_period>0.0:
                        bg = torch.rand((3), device="cuda") if opt.random_background else background
                        gaussian_mask = torch.zeros_like(gaussians.mask_xyz)
                        # Freeze mask by back projecting pixel mask
                        net_influence = None
                        for idx,camera in enumerate(train_cameras):
                            render_pkg = render_mask(camera, gaussians, pipe, bg, image_shape=camera.original_image.shape, 
                                                    pixel_mask=flow_mask[idx].float(), render_depth=False)
                            influence = render_pkg["influence"]
                            if net_influence is None:
                                net_influence = influence
                            else:
                                net_influence += influence
                            gaussian_mask = torch.logical_or(gaussian_mask,influence[...,None]>0)


                        # Pixel mask by rerendering gaussian mask 
                        # (otherwise directly use the 2d mask as
                        for idx,camera in enumerate(train_cameras):
                            alphamask = flow_mask[idx].float()
                            camera.orig_mask = alphamask
                            mask_down = torch.nn.functional.max_pool2d(alphamask.unsqueeze(0), (dataset.dilate_size,dataset.dilate_size))
                            mask_dilate = torch.nn.functional.interpolate(mask_down, size=(alphamask.shape[-2],alphamask.shape[-1]))
                            camera.mask = (mask_dilate.squeeze(0).squeeze(0)>0).float()

                    if (dataset.gaussian_update_thresh != dataset.pixel_update_thresh) or \
                        not (dataset.adaptive_render and dataset.adaptive_update_period>0.0):
                        flow_mask = flow_norm>dataset.gaussian_update_thresh
                        # Rerun backprojection if we want to use a different threshold for our gaussian mask
                        gaussian_mask = torch.zeros_like(gaussians.mask_xyz)
                        # Freeze mask by back projecting pixel mask
                        net_influence = None
                        for idx,camera in enumerate(train_cameras):
                            render_pkg = render_mask(camera, gaussians, pipe, bg, image_shape=camera.original_image.shape, 
                                                    pixel_mask=flow_mask[idx].float(), render_depth=False)
                            influence = render_pkg["influence"]
                            gaussian_mask = torch.logical_or(gaussian_mask,influence[...,None]>0)
                            if net_influence is None:
                                net_influence = influence
                            else:
                                net_influence += influence
                
                gaussians.update_masks(dataset, None if dataset.update_mask == "none" else gaussian_mask)
                gaussians.freeze_atts(dataset)

                if dataset.adaptive_render and dataset.adaptive_update_period>0.0:
                    adaptive_update_epochs = np.ceil(opt.epochs*dataset.adaptive_update_period).astype(np.int32)
                    pix_thresh_vals = torch.ones(adaptive_update_epochs*n_cams)*dataset.pixel_update_thresh

                    if opt.iterations>pix_thresh_vals.shape[0]:
                        addn_pix_vals = torch.zeros(opt.iterations-pix_thresh_vals.shape[0]).to(pix_thresh_vals)
                        pix_thresh_vals = torch.cat((pix_thresh_vals,addn_pix_vals),dim=0)
                    assert pix_thresh_vals.shape[0] == opt.iterations
                else:
                    pix_thresh_vals = None

                # Initialize gate probabilities based on computed differences
                if any([gating!="none" for gating in qp.gate_params]):
                    if dataset.update_mask == "viewspace_diff":
                        # Use gradient differences for gate initialization
                        init_probs = grad_diff/(grad_diff+grad_diff.median())
                        gaussians.init_probs = init_probs.flatten()
                    elif dataset.update_mask == "diff":
                        # Use frame differences for gate initialization  
                        init_probs = net_influence/(net_influence+net_influence.mean())
                        gaussians.init_probs = init_probs.flatten()
                    else:
                        gaussians.init_probs = None
                    if gaussians.gate_atts is None:
                        gaussians.gate_atts = Gate(gaussians._xyz.shape[0], 
                                                  gamma=dataset.gate_gamma,
                                                  eta=dataset.gate_eta,
                                                  lr = dataset.gate_lr, 
                                                  temp=dataset.gate_temp,
                                                  lambda_l2=dataset.gate_lambda_l2, 
                                                  lambda_l0=dataset.gate_lambda_l0, 
                                                  init_probs=gaussians.init_probs)
                        gaussians.gate_atts.train()
                    else:
                        gaussians.gate_atts.reset_params(init_probs=gaussians.init_probs)
                        gaussians.gate_atts.train()

                if dataset.flow_update and opt.lambda_flow>0.0:
                    gaussians.update_points_flow()
                prev_frame_views = cur_frame_views

        # When resuming frame 1 from a checkpoint, skip the static training loop entirely.
        load_checkpoint = (frame_idx == 1 and checkpoint is not None)

        if enable_tqdm and frame_idx == 1 and not load_checkpoint:
            progress_bar_iter = tqdm(range(first_iter, opt.iterations+1),
                                     desc="Frame iteration progress")
        else:
            progress_bar_iter = None

        if dataset.timed:
            torch.cuda.synchronize()
        frame_time += time.time()- frame_start
        frame_start = time.time()
        frame_time_io += time.time() - frame_start_io
        frame_start_io = time.time()

        # Frame 1 loaded from checkpoint: run a single eval pass (no optimization) so the
        # frame-1 PSNR/metrics are reported. The eval render counts as evaluation overhead,
        # not as update latency.
        if load_checkpoint:
            net_elapsed_time = time.time() - training_start
            cur_size = gaussians.size()/8/(10**6)
            _dummy = torch.zeros(1, device="cuda")
            report = training_report(tb_writer, wandb_enabled, dataset, frame_idx, opt.iterations,
                                     _dummy, _dummy, l1_loss, cur_size, frame_time, True, scene,
                                     render_mask, (pipe, background), prev_report=None,
                                     report_alpha=True, max_iterations=opt.iterations)
            if report:
                eval_render_time += report.get("_render_time", 0.0)
                rendering_frames += report.get("_render_count", 0)
                report_configs = ['test', 'val'] if 'test' in report.keys() else ['val']
                for config_name in report_configs:
                    metrics[config_name]['psnr'] = report[config_name]['psnr']
                    metrics[config_name]['loss'] = report[config_name]['l1']
                best_psnr = max(best_psnr, metrics['test']['psnr'])

        # Start training iteration loop for current frame (skipped when loading from checkpoint)
        loop_iters = 0 if load_checkpoint else opt.iterations
        for iteration in range(first_iter, loop_iters + 1):

            if enable_debug:
                print(f"DEBUG: started iteration {iteration}")

            iter_start = time.time()

            # Handle quantization and freezing of latent parameters
            if frame_idx>1:
                for i, att_name in enumerate(gaussians.param_names):
                    decoder = gaussians.latent_decoders[att_name]
                    # Switch from Identity Decoder to quantized encoding at specified iteration
                    if iteration == np.ceil(qp.quant_after[i]*opt.iterations) and type(decoder) == LatentDecoderRes:
                        decoder.identity = False
                        latent = gaussians._latents[att_name].data
                        if "f_" in att_name:
                            latent = latent.reshape(latent.shape[0],-1)
                        quant_latents = decoder.invert(latent)
                        new_lr = opt.latents_lr_scaling[i]*gaussians.orig_lr[att_name]
                        optimizable_tensors = gaussians.replace_tensor_to_optimizer(quant_latents, att_name,
                                                                                    lr=new_lr)
                        gaussians._latents[att_name] = optimizable_tensors[att_name]

                    # Handle parameter freezing schedule
                    assert qp.freeze_before[i]<= qp.freeze_after[i]
                    freeze_before_iter = np.ceil(qp.freeze_before[i]*opt.iterations)
                    freeze_after_iter = np.ceil(qp.freeze_after[i]*opt.iterations)
                    frz = gaussians.get_frz
                    if iteration==first_iter and iteration<freeze_before_iter:
                        gaussians.get_masks[att_name] *= False
                    elif iteration == freeze_before_iter:
                        gaussians.get_masks[att_name] += True
                        if frz[att_name] == "st":
                            # NOTE: might fail with densification
                            gaussians.get_masks[att_name] *= gaussian_mask

                    if iteration==(freeze_after_iter+1):
                        gaussians.get_masks[att_name] *= False
            
            gaussians.update_learning_rate(iteration, qp)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not camera_idx_stack:
                camera_idx_stack = list(range(n_cams))
            cam_idx = camera_idx_stack.pop(generator.randint(0, len(camera_idx_stack)-1))
            viewpoint_cam: SequentialCamera = train_cameras[cam_idx]

            # Render

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            # Loss
            gt_image = viewpoint_cam.original_image
            if opt.transform == "resize":
                gt_image = resize_image(gt_image, resize_scale_sched(iteration))
            elif "blur" in opt.transform and resize_scale_sched(iteration)!=1.0:
                if (iteration-1) % 100 == 0:
                    transform = blur_image(resize_scale_sched(iteration), opt.transform)
                gt_image = transform(gt_image)
            elif opt.transform == "downsample":
                gt_image = downsample_image(gt_image, resize_scale_sched(iteration))

            # GT mask. Depending on the data, can be float value or binarized. In range [0, 1].
            gt_mask = viewpoint_cam.original_alpha_mask  # (1, H, W)
            if opt.lambda_alpha > 0:
                # If enabled alpha loss, we require the data provide gt_mask
                if gt_mask is None:
                    raise RuntimeError(f"Alpha loss enabled, however no `gt_mask` is provided.")

                if opt.transform == "resize":
                    raise NotImplementedError(f"not yet tested")
                    gt_mask = resize_image(gt_mask, resize_scale_sched(iteration))
                elif "blur" in opt.transform and resize_scale_sched(iteration)!=1.0:
                    raise NotImplementedError(f"not yet tested")
                    if (iteration-1) % 100 == 0:
                        transform = blur_image(resize_scale_sched(iteration), opt.transform)
                    gt_mask = transform(gt_mask)
                elif opt.transform == "downsample":
                    gt_mask = downsample_image(gt_mask, resize_scale_sched(iteration))

            color_rw_mask = None

            # Initialize pixel_mask to None by default
            pixel_mask = None
            if frame_idx>1 and pix_thresh_vals is not None:
                if pix_thresh_vals[iteration-1]>0:
                    pixel_mask = viewpoint_cam.mask
            
            # render
            render_pkg = render_mask(viewpoint_cam, gaussians, pipe, bg, image_shape=gt_image.shape,
                                     color_mask=color_rw_mask, render_depth=opt.lambda_depth>0.0,
                                     backward_alpha=opt.lambda_alpha>0.0,
                                     render_flow=opt.lambda_flow>0.0 and iteration > (opt.flow_from_iter*opt.iterations),
                                     pixel_mask=pixel_mask,
                                     update_mask=None)

            image, viewspace_point_tensor = render_pkg["render"], render_pkg["viewspace_points"]
            visibility_filter, radii = render_pkg["visibility_filter"], render_pkg["radii"]

            # Compute main reconstruction losses
            loss, Ll1 = torch.Tensor([0.0]).to(image.device), torch.Tensor([0.0]).to(image.device)
            if iteration>opt.color_from_iter:
                if pixel_mask is not None:
                    # Apply pixel mask for selective training
                    Ll1 = l1_loss(image*pixel_mask.unsqueeze(0), gt_image*pixel_mask.unsqueeze(0))
                    Lssim = ssim(image*pixel_mask.unsqueeze(0), gt_image*pixel_mask.unsqueeze(0))
                else:
                    Ll1 = l1_loss(image, gt_image)
                    Lssim = ssim(image, gt_image)

                loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim)
                
            # alpha mask
            if opt.lambda_alpha > 0.0:
                pred_alpha = render_pkg["alpha"]  # (1, H, W)
                # Note: if need to apply selective training, use `pixel_mask`. See examples in the photometric loss above.
                loss_alpha = l1_loss(pred_alpha, gt_mask)  # L1 loss for now. Can do BCE if gt_mask is binarized.
                loss += opt.lambda_alpha * loss_alpha
            
            # Add regularization losses
            if opt.weight_decay>0.0:
                loss += opt.weight_decay * gaussians.std_reg()

            if gaussians.gate_atts is not None and gaussians.gate_atts.training:
                loss += gaussians.gate_atts.reg_loss(gaussians._ungated_xyz_res)

            if opt.lambda_posres>0.0:
                residual = gaussians.get_xyz-prev_xyz.detach()
                loss += opt.lambda_posres*torch.abs(residual).mean()
            if iteration > opt.alpha_from_iter and opt.lambda_alpha>0.0:
                loss += opt.lambda_alpha * l2_loss(render_pkg["alpha"],1.0)

            # Temporal flow consistency loss
            if opt.lambda_flow>0.0 and iteration > (opt.flow_from_iter*opt.iterations):
                if dataset.flow_loss_type == "render":
                    # Direct rendering approach for flow loss
                    render_pkg_flow = render_mask_shift(viewpoint_cam, gaussians, pipe, bg, image_shape=gt_image.shape)
                    next_image = render_pkg_flow["render"]
                    next_gt_image = next_frame_views[cam_idx]
                    if pixel_mask is not None:
                        next_Ll1 = l1_loss(next_image*pixel_mask.unsqueeze(0), next_gt_image*pixel_mask.unsqueeze(0))
                        next_Lssim = ssim(next_image*pixel_mask.unsqueeze(0), next_gt_image*pixel_mask.unsqueeze(0))
                    else:
                        next_Ll1 = l1_loss(next_image, next_gt_image)
                        next_Lssim = ssim(next_image, next_gt_image)
                    flow_loss = (1.0 - opt.lambda_fdssim) * next_Ll1 + opt.lambda_fdssim * (1.0 - next_Lssim)
                    loss += opt.lambda_flow * flow_loss

                elif dataset.flow_loss_type == "warp":
                    # Optical flow warping approach
                    rendered_flow = torch.clamp(render_pkg["flow"],-50,50) 
                    if dataset.use_gt_flow:
                        tgt_flow = cur_frame_views[cam_idx]
                    else:
                        tgt_flow = image.unsqueeze(0).detach()
                    warped = flow_warp(next_frame_views[cam_idx:cam_idx+1], rendered_flow.unsqueeze(0), grid)
                    if pixel_mask is not None:
                        flow_loss = (1-opt.lambda_dssim)*l1_loss(warped*pixel_mask.unsqueeze(0), 
                                                                 tgt_flow*pixel_mask.unsqueeze(0))+\
                                    opt.lambda_dssim*(1.0-ssim(warped*pixel_mask.unsqueeze(0), 
                                                                   tgt_flow*pixel_mask.unsqueeze(0)))
                    else:
                        flow_loss = (1-opt.lambda_dssim)*l1_loss(warped, tgt_flow)+\
                                    opt.lambda_dssim*(1.0-ssim(warped, tgt_flow))
                    loss += opt.lambda_flow * flow_loss  + opt.lambda_tv * tv_loss(rendered_flow)

            # Depth supervision loss (first frame only)
            if opt.lambda_depth>0.0 and iteration>opt.depth_from_iter and iteration<=opt.depth_until_iter and frame_idx == 1:
                pred_depth = render_pkg["depth"] 
                gt_depth = viewpoint_cam.gt_depth
                depth_loss = (1.0 - opt.lambda_depthssim) * depth_loss_fn(pred_depth, gt_depth)+ opt.lambda_depthssim * (1.0 - ssim(pred_depth.unsqueeze(0), gt_depth.unsqueeze(0)))
                loss += opt.lambda_depth * depth_loss + opt.lambda_tv * tv_loss(pred_depth)
                if iteration % dataset.depth_pair_interval == 0:
                    depth_loss_fn.resample_pairs()

            # Temporal consistency loss
            if opt.lambda_consistency>0.0:
                prev_image = viewpoint_cam.prev_rendered
                cur_image = render_pkg["render"]
                gt_diff = viewpoint_cam.image_diff
                # High consistency loss for low varying regions
                gt_diff = 1/(gt_diff+gt_diff.mean()) 
                # Normalize
                gt_diff = gt_diff/gt_diff.mean()
                consistency_loss = 1- l1_loss(prev_image*gt_diff, cur_image*gt_diff)
                loss += opt.lambda_consistency*consistency_loss
            loss.backward()
            if enable_debug:
                print(f'DEBUG ({iteration}): backpropagated')

            with torch.no_grad():
                frame_time += time.time() - iter_start
                frame_time_io += time.time() - iter_start
                net_elapsed_time = time.time() - training_start
                # Log and save
                if dataset.test_interval>0:
                    is_test = (iteration % dataset.test_interval == 0) and frame_idx == 1
                else:
                    is_test = (iteration in testing_iterations) and frame_idx == 1
                if iteration == opt.iterations:
                    is_test = True
                    
                report = training_report(tb_writer, wandb_enabled, dataset, frame_idx, iteration, Ll1, loss, 
                                         l1_loss, cur_size, frame_time, is_test, scene, 
                                         render_mask, (pipe, background), prev_report=report, report_alpha=True, max_iterations=opt.iterations)
                if report:
                    eval_render_time += report.get("_render_time", 0.0)
                    rendering_frames += report.get("_render_count", 0)
                if enable_debug:
                    print(f'DEBUG ({iteration}): training_report done')

                if report:
                    if 'test' in report.keys():
                        report_configs = ['test','val']
                    else:
                        report_configs = ['val']
                    for config_name in report_configs:
                        metrics[config_name]['psnr'] = report[config_name]['psnr']
                        metrics[config_name]['loss'] = report[config_name]['l1']
                    if metrics['test']['psnr'] > best_psnr:
                        best_psnr = metrics['test']['psnr']
                        if wandb_enabled and frame_idx<=2:
                            wandb.log({frame_str+"/test/loss_viewpoint/best_psnr": best_psnr,
                                       iter_metric:iteration})
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

                if (iteration) % dataset.log_interval == 0 or iteration == opt.iterations:
                    cur_size = gaussians.size()/8/(10**6)
                    log_dict = {
                                "Loss": f"{ema_loss_for_log:.{5}f}",
                                "Num points": f"{gaussians._xyz.shape[0]}",
                                "Update points": f"{torch.count_nonzero(gaussians.mask_xyz)}" \
                                                if frame_idx>1 else f"{gaussians._xyz.shape[0]}",
                                "Size (MB)": f"{cur_size:.{2}f}",
                                "PSNR (Test)": f"{metrics['test']['psnr']:.{2}f}",
                                "PSNR (Val)": f"{metrics['val']['psnr']:.{2}f}",
                                }
                    if progress_bar_iter:
                        progress_bar_iter.set_postfix(log_dict)
                        progress_bar_iter.update(dataset.log_interval)
                if iteration == opt.iterations and progress_bar_iter:
                    progress_bar_iter.close()
                # Note: PLY saving moved after iteration loop to match PKL timing (after densification/pruning)


                iter_start = time.time()

                if iteration <=opt.prune_until_iter:
                    gaussians.add_influence_stats(render_pkg["influence"])

                if iteration>opt.prune_from_iter and iteration<=opt.prune_until_iter and iteration % opt.prune_interval == 0:
                    out = gaussians.infl_accum/gaussians.infl_denom
                    out[out.isnan()] = 0.0

                # Gaussian Densification
                if iteration <= (np.ceil(opt.densify_until_epoch*n_cams*opt.iterations)) and iteration>(opt.calc_dense_stats*n_cams):
                    # Track max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], 
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # Wait for stats accumulation (at least 2 training epochs) before densifying
                    densify_from_epoch = max(opt.calc_dense_stats+2, opt.densify_from_epoch)

                    if iteration > (densify_from_epoch*n_cams) and iteration % (opt.densification_interval*n_cams) == 0:
                        size_threshold = opt.size_threshold if iteration > (opt.opacity_reset_interval*n_cams) else None
                        if frame_idx == 1:
                            # Standard densification for first frame
                            gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold)
                        else:
                            # Dynamic densification for subsequent frames
                            gaussians.densify_dynamic(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, opt.size_threshold)
                    
                    # Periodic opacity reset
                    if iteration % (opt.opacity_reset_interval*n_cams) == 0 or \
                        (dataset.white_background and iteration == (densify_from_epoch*n_cams)):
                        gaussians.reset_opacity()

                    if enable_debug:
                        print(f'DEBUG ({iteration}): densification done')

                # Pruning
                if iteration>opt.prune_from_iter and iteration<=opt.prune_until_iter and iteration % opt.prune_interval == 0:
                    gaussians.influence_prune(opt.prune_threshold)

                    if enable_debug:
                        print(f'DEBUG ({iteration}): pruning done')

                frame_time += time.time()-iter_start
                frame_time_io += time.time()-iter_start
                with torch.no_grad():
                    if (opt.iterations - iteration) < (2*n_cams): # Save most recent render for final epochs
                        viewpoint_cam.prev_rendered = render_pkg["render"].detach()

                    if (opt.iterations - iteration)<(n_cams) and cam_idx == 0 and (dataset.log_images or dataset.log_compressed or dataset.log_ply):
                        
                        if dataset.log_images:
                            save_image(gt_image,os.path.join(scene.model_path, "gt.png"))

                        if dataset.log_ply:
                            scene.save(iteration, save_point_cloud=True)
                        
                        if dataset.log_compressed:
                            if frame_idx == 1:
                                scene.save(frame_idx, save_point_cloud=True)                                

                        if frame_idx>1 and (dataset.adaptive_render and dataset.adaptive_update_period>0.0) and dataset.update_mask!="none":
                            torchvision.utils.save_image(train_cameras[cam_idx].mask.unsqueeze(0)*gt_image,
                                                         os.path.join(scene.model_path, "mask.png"))
                            torchvision.utils.save_image(train_cameras[cam_idx].orig_mask.unsqueeze(0)*gt_image,
                                                         os.path.join(scene.model_path, "orig_mask.png"))
                        video_camera = video_cameras[frame_idx-1]
                        _sync_cuda(dataset.timed)
                        _er0 = time.time()
                        spiral_img = render(video_camera, gaussians, pipe, background)["render"]
                        _sync_cuda(dataset.timed)
                        eval_render_time += time.time() - _er0
                        rendering_frames += 1
                        if frame_idx == 1:
                            os.makedirs(os.path.join(dataset.model_path,"spiral"), exist_ok=True)
                        save_image(torch.clip(spiral_img, 0.0, 1.0),os.path.join(dataset.model_path, "spiral", f"{str(dataset.start_idx + frame_idx).zfill(4)}.png"))

                        if frame_idx == 1:
                            with torch.no_grad():
                                _sync_cuda(dataset.timed)
                                _er0 = time.time()
                                render_pkg = render_mask(viewpoint_cam, gaussians, pipe, bg, image_shape=gt_image.shape,
                                                         color_mask=color_rw_mask, render_depth=True)
                                _sync_cuda(dataset.timed)
                                eval_render_time += time.time() - _er0
                                rendering_frames += 1
                                pred_depth = render_pkg["depth"]
                                render_depth = pred_depth.detach().cpu().numpy()

                            if opt.lambda_depth>0.0 or dataset.depth_init:
                                gt_depth = viewpoint_cam.gt_depth
                                gt_depth = gt_depth.detach().cpu().numpy()
                                gt_depth = (gt_depth-gt_depth.min())/(gt_depth.max()-gt_depth.min())
                                render_depth = (render_depth-render_depth.min())/(render_depth.max()-render_depth.min())
                                depth_ssim = ssim(torch.tensor(render_depth).cuda().unsqueeze(0), torch.tensor(gt_depth).cuda().unsqueeze(0)).item()
                                depth_psnr = psnr(torch.tensor(gt_depth).cuda().unsqueeze(0), torch.tensor(render_depth).cuda().unsqueeze(0)).item()
                                if wandb_enabled:
                                    wandb.run.summary["depth_SSIM"] = depth_ssim
                                    wandb.run.summary["depth_PSNR"] = depth_psnr
                                depth_err = np.abs(render_depth-gt_depth)
                                depth_err = torch.abs(render_pkg["depth"]-viewpoint_cam.gt_depth).detach().cpu().numpy()
                                torchvision.utils.save_image(torch.tensor(depth_err).unsqueeze(0),os.path.join(dataset.model_path,'err_depth_gray.png'))

                # Optimizer step
                iter_start = time.time()
                if iteration <= opt.iterations:
                    # gaussians.update_grads()
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if gaussians.gate_atts is not None and gaussians.gate_atts.training:
                        gaussians.gate_atts.step()
                        gaussians.gate_atts.clamp_params()
                frame_time += time.time()-iter_start
                frame_time_io += time.time()-iter_start
                if enable_debug:
                    print(f'DEBUG ({iteration}): Optimizer step done')
        # end training loop for this frame

        # Close the Update span (reconstruction is done) before the held-out render.
        # Update bucket = reconstruction wall-clock minus the validation/spiral renders that
        # ran inside the loop. Decode binding and the held-out render are already excluded
        # (binding via pause/resume, held-out render measured separately below).
        update_sw.pause()
        update_time = max(update_sw.total - eval_render_time, 0.0)

        # Render latency: rasterize the updated G_t into ONE held-out test view.
        # PSNR/SSIM/LPIPS are intentionally excluded (evaluation overhead, not latency).
        if test_image_dataset.n_cams > 0:
            heldout_cam = scene.getTestCameras()[0]
            with torch.no_grad():
                _sync_cuda(dataset.timed)
                _hr0 = time.time()
                _ = render_mask(heldout_cam, gaussians, pipe, background,
                                image_shape=heldout_cam.original_image.shape)["render"]
                _sync_cuda(dataset.timed)
                render_time = time.time() - _hr0
        elif frame_idx == start_frame_idx:
            print('No test cameras found; held-out render time will be 0.0 (requires a held-out view).')

        if dataset.timed:
            torch.cuda.synchronize()
        frame_start = time.time()

        # Save PLY/PKL files after iteration loop completes (after all densification/pruning)
        # This ensures PLY and PKL files represent the same Gaussian state
        if -1 in saving_iterations:
            print("\n[ITER {}] Saving Gaussians".format(opt.iterations))
            if args.save_format == "ply":
                scene.save(opt.iterations)
                symlink(os.path.join("..", "cfg_args"),
                        os.path.abspath(os.path.join(scene.model_path, "cfg_args")))
            elif args.save_format == "pkl":
                # PKL files handled by dataset.log_compressed below
                pass
            else:
                raise ValueError(f"Invalid save format {args.save_format}")

        if dataset.log_compressed:
            if frame_idx == 1:
                scene.save(frame_idx, save_point_cloud=True)
            else:
                gaussians.gate_atts.eval()
                scene.save_compressed(-1, qp)
                gaussians.gate_atts.train()


        # Update previous frame's attributes and latents for next frame's residual encoding
        if frame_idx != n_frames:
            # Used for residual encoding of next frame
            with torch.no_grad():
                for att_name in gaussians.get_atts:
                    prev_atts = gaussians.get_decoded_atts[att_name].clone()
                    prev_latents = gaussians.get_atts[att_name].clone()
                    gaussians.prev_atts[att_name] = prev_atts
                    gaussians.prev_latents[att_name] = prev_latents
                    gaussians.prev_atts[att_name].requires_grad_(False)
                    gaussians.prev_latents[att_name].requires_grad_(False)
                    gaussians.prev_atts_initial[att_name] = prev_atts.clone()
            cur_frame_views = next_frame_views
            cur_train_images = next_train_images
            cur_train_paths = next_train_paths
            prev_xyz = gaussians._xyz.clone()

        if dataset.timed:
            torch.cuda.synchronize()
        frame_time += time.time()-frame_start
        frame_time_io += time.time()-frame_start

        rendering_time = eval_render_time
        rendering_fps = rendering_frames / rendering_time if rendering_time > 0.0 else 0.0

        # Latency buckets (frame-boundary syncs only): update measured above as one
        # reconstruction span; E2E = decode + update + render.
        e2e_time = decode_time + update_time + render_time
        latency_metrics = {
            "Decode time": round(decode_time, 6),
            "Update time": round(update_time, 6),
            "Render time (heldout)": round(render_time, 6),
            "E2E time": round(e2e_time, 6),
        }

        # Collect frame metrics for logging
        if test_image_dataset.n_cams > 0:
            frame_metrics = {
                "Frame index": frame_idx,
                "Loss": round(ema_loss_for_log,5),
                "Loss (Test)": round(metrics['test']['loss'].item(),5),
                "Loss (Val)": round(metrics['val']['loss'].item(),5),
                "Num points": gaussians._xyz.shape[0],
                "Update points": f"{torch.count_nonzero(gaussians.mask_xyz)}" \
                                    if frame_idx>1 else f"{gaussians._xyz.shape[0]}",
                "Size (MB)": round(cur_size,2),
                "PSNR (Test)": round(metrics['test']['psnr'].item(),2),
                "PSNR (Val)": round(metrics['val']['psnr'].item(),2),
                "Frame time": round(frame_time,2),
                "Frame time IO": round(frame_time_io,2),
                "Rendering time": round(rendering_time, 4),
                "Rendering frames": rendering_frames,
                "Rendering FPS": round(rendering_fps, 2),
                **latency_metrics,
                "Static init time": round(static_init_time, 6) if frame_idx == 1 else None,
                "Training time elapsed": round(net_elapsed_time,2),
            }
        else:
            # Not using test cameras
            frame_metrics = {
                "Frame index": frame_idx,
                "Loss": round(ema_loss_for_log,5),
                "Loss (Val)": round(metrics['val']['loss'].item(),5),
                "Num points": gaussians._xyz.shape[0],
                "Update points": f"{torch.count_nonzero(gaussians.mask_xyz)}" \
                                    if frame_idx>1 else f"{gaussians._xyz.shape[0]}",
                "Size (MB)": round(cur_size,2),
                "PSNR (Val)": round(metrics['val']['psnr'].item(),2),
                "Frame time": round(frame_time,2),
                "Frame time IO": round(frame_time_io,2),
                "Rendering time": round(rendering_time, 4),
                "Rendering frames": rendering_frames,
                "Rendering FPS": round(rendering_fps, 2),
                **latency_metrics,
                "Static init time": round(static_init_time, 6) if frame_idx == 1 else None,
                "Training time elapsed": round(net_elapsed_time,2),
            }

        training_metrics.append(frame_metrics)
        
        # Log to wandb if enabled
        if wandb_enabled:
            wandb_log = {
                "frame/test/loss_viewpoint/psnr": metrics['test']['psnr'].item(),
                       "frame/test/loss_viewpoint/loss": metrics['test']['loss'].item(),
                       "frame/val/loss_viewpoint/psnr": metrics['val']['psnr'].item(),
                       "frame/val/loss_viewpoint/loss": metrics['val']['loss'].item(),
                       "frame/size": cur_size,
                       "frame/num_points": gaussians._xyz.shape[0],
                       "frame/update_points": torch.count_nonzero(gaussians.mask_xyz) \
                                              if frame_idx>1 else gaussians._xyz.shape[0],
                       "frame/iter_time": frame_time,
                       "frame/iter_time_io": frame_time_io,
                       "frame/rendering_time": rendering_time,
                       "frame/rendering_frames": rendering_frames,
                       "frame/rendering_fps": rendering_fps,
                       "frame/elapsed": net_elapsed_time,
                       "frame/num_iterations": opt.iterations if frame_idx>1 else 0,
                       "frame_idx": frame_idx}
            wandb_log["frame/latency/decode"] = decode_time
            wandb_log["frame/latency/update"] = update_time
            wandb_log["frame/latency/render"] = render_time
            wandb_log["frame/latency/e2e"] = e2e_time
            wandb.log(wandb_log)


        # Compute and display average metrics
        if test_image_dataset.n_cams > 0:
            avg_metrics = {
                "Loss (Test)": round(sum([fm["Loss (Test)"] for fm in training_metrics])/len(training_metrics),5),
                "Loss (Val)": round(sum([fm["Loss (Val)"] for fm in training_metrics])/len(training_metrics),5),
                "PSNR (Test)": round(sum([fm["PSNR (Test)"] for fm in training_metrics])/len(training_metrics),2),
                "PSNR (Val)": round(sum([fm["PSNR (Val)"] for fm in training_metrics])/len(training_metrics),2),
                "Size (MB)": round(sum([fm["Size (MB)"] for fm in training_metrics])),
                "Frame time": round(sum([fm["Frame time"] for fm in training_metrics])/len(training_metrics),2),
                "Rendering time": round(sum([fm["Rendering time"] for fm in training_metrics]), 4),
                "Rendering FPS": round(
                    sum([fm["Rendering frames"] for fm in training_metrics])
                    / max(sum([fm["Rendering time"] for fm in training_metrics]), EPS), 2),
                "Elapsed time": round(frame_metrics["Training time elapsed"],2),
            }
        else:
            avg_metrics = {
                "Loss (Val)": round(sum([fm["Loss (Val)"] for fm in training_metrics])/len(training_metrics),5),
                "PSNR (Val)": round(sum([fm["PSNR (Val)"] for fm in training_metrics])/len(training_metrics),2),
                "Size (MB)": round(sum([fm["Size (MB)"] for fm in training_metrics])),
                "Frame time": round(sum([fm["Frame time"] for fm in training_metrics])/len(training_metrics),2),
                "Rendering time": round(sum([fm["Rendering time"] for fm in training_metrics]), 4),
                "Rendering FPS": round(
                    sum([fm["Rendering frames"] for fm in training_metrics])
                    / max(sum([fm["Rendering time"] for fm in training_metrics]), EPS), 2),
                "Elapsed time": round(frame_metrics["Training time elapsed"],2),
            }

        # Update progress display
        del frame_metrics["Training time elapsed"]
        postfix_metrics = {k: v for k, v in frame_metrics.items() if v is not None}
        if enable_tqdm:
            progress_bar_frame.set_postfix(postfix_metrics)
            progress_bar_frame.update(1)
        else:
            frame_counter += 1
            print(f"frame {frame_counter} frame_metrics: {frame_metrics}")

        # End frame index loop
          
    with open(os.path.join(args.model_path,'training_metrics.json'),'w') as f:
        json.dump(training_metrics, f, indent=4)

    with open(os.path.join(args.model_path, 'avg_metrics.json'),'w') as f:
        json.dump(avg_metrics, f)

    # Dedicated latency tables: initialization (frame 1, incl. static 3DGS init) vs
    # steady-state per-frame updates (frames 2..N). E2E = decode + update + render.
    init_frame = next((fm for fm in training_metrics if fm["Frame index"] == 1), None)
    steady_frames = [fm for fm in training_metrics if fm["Frame index"] > 1]
    initialization_latency = {} if init_frame is None else {
        "frame_idx": 1,
        "static_init_time": init_frame.get("Static init time"),
        "decode_time": init_frame["Decode time"],
        "update_time": init_frame["Update time"],
        "render_time": init_frame["Render time (heldout)"],
        "e2e_time": init_frame["E2E time"],
    }
    steady_state_latency = {
        "decode": _latency_stats([fm["Decode time"] for fm in steady_frames]),
        "update": _latency_stats([fm["Update time"] for fm in steady_frames]),
        "render": _latency_stats([fm["Render time (heldout)"] for fm in steady_frames]),
        "e2e": _latency_stats([fm["E2E time"] for fm in steady_frames]),
    }
    latency_summary = {
        "timed": bool(dataset.timed),
        "initialized_from_checkpoint": checkpoint if checkpoint is not None else False,
        "note": "Meaningful only with --timed (num_workers=0, CUDA syncs). E2E = decode + update + render."
                + (" Frame 1 loaded from checkpoint: static_init_time is ply-load time and update_time~0."
                   if checkpoint is not None else ""),
        "initialization": initialization_latency,
        "steady_state": steady_state_latency,
    }
    with open(os.path.join(args.model_path, 'latency.json'), 'w') as f:
        json.dump(latency_summary, f, indent=4)

    # Print the two latency tables.
    print('\nLatency tables (seconds)' + ('' if dataset.timed else '  [NOTE: run with --timed for accurate numbers]'))
    if initialization_latency:
        print('  Initialization (frame 1, loaded from checkpoint):' if checkpoint is not None
              else '  Initialization (frame 1, one-time):')
        _si_label = 'ply_load' if checkpoint is not None else 'static_init'
        print(f"    {_si_label:>12} {'decode':>10} {'update':>10} {'render':>10} {'e2e':>10}")
        _si = initialization_latency["static_init_time"]
        print(f"    {(_si if _si is not None else 0.0):>12.4f} "
              f"{initialization_latency['decode_time']:>10.4f} {initialization_latency['update_time']:>10.4f} "
              f"{initialization_latency['render_time']:>10.4f} {initialization_latency['e2e_time']:>10.4f}")
    if steady_frames:
        print(f'  Steady-state per-frame updates (frames 2..N, n={len(steady_frames)}):')
        print(f"    {'bucket':>8} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
        for _name in ["decode", "update", "render", "e2e"]:
            _s = steady_state_latency[_name]
            print(f"    {_name:>8} {_s['mean']:>10.4f} {_s['std']:>10.4f} {_s['min']:>10.4f} {_s['max']:>10.4f}")

    if enable_tqdm:
        progress_bar_frame.close()

    # Display final results
    print('\nFinal average training metrics:')
    for k,v in avg_metrics.items():
        print(k+":"+ str(v))

    # Log final metrics to wandb
    if wandb_enabled:
        summary_dict = {
            'average/test/loss_viewpoint/psnr': avg_metrics.get("PSNR (Test)", 0),
            'average/test/loss_viewpoint/loss': avg_metrics.get("Loss (Test)", 0),
            'average/val/loss_viewpoint/psnr': avg_metrics["PSNR (Val)"],
            'average/val/loss_viewpoint/loss': avg_metrics["Loss (Val)"],
            'average/size': avg_metrics["Size (MB)"],
            'average/frame_time': avg_metrics["Frame time"],
            'total/rendering_time': avg_metrics["Rendering time"],
            'average/rendering_fps': avg_metrics["Rendering FPS"],
            'average/elapsed_time': avg_metrics["Elapsed time"]
        }
        for _name in ["decode", "update", "render", "e2e"]:
            summary_dict[f"steady_state/latency/{_name}_mean"] = steady_state_latency[_name]["mean"]
            summary_dict[f"steady_state/latency/{_name}_std"] = steady_state_latency[_name]["std"]
        if initialization_latency:
            summary_dict["init/static_init_time"] = initialization_latency["static_init_time"]
            summary_dict["init/e2e_time"] = initialization_latency["e2e_time"]
        wandb.run.summary.update(summary_dict)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")


    # Create wandb logger
    if WANDB_FOUND and args.use_wandb:
        wandb_project = args.wandb_project
        wandb_run_name = args.wandb_run_name
        wandb_entity = args.wandb_entity
        wandb_mode = args.wandb_mode
        if not wandb_run_name:
            scene_name = os.path.basename(args.source_path.rstrip("/")) or os.path.basename(args.model_path.rstrip("/"))
            wandb_run_name = f"{scene_name}_{time.strftime('%Y%m%d-%H%M%S')}"
        id = hashlib.md5(wandb_run_name.encode('utf-8')).hexdigest()
        name = wandb_run_name
        wandb.init(
            project=wandb_project,
            name=name,
            entity=wandb_entity,
            config=args,
            sync_tensorboard=False,
            dir=args.model_path,
            mode=wandb_mode,
            id=id,
            resume="allow"
        )

    return tb_writer

def training_report(tb_writer, wandb_enabled, model_args, frame_idx, iteration, Ll1, loss, l1_loss, size, 
                    elapsed, is_test, scene : Scene, renderFunc, renderArgs, prev_report=None, report_alpha=False, max_iterations=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('elapsed', elapsed, iteration)
        tb_writer.add_scalar('size', size, iteration)

    if wandb_enabled:
        frame_str = f"{str(frame_idx).zfill(4)}"
        iter_metric = "iter_"+frame_str
        frame_str = "frame_"+frame_str

        # Log iterwise metrics only for the first few frames
        if frame_idx <= 2 and iteration % model_args.wandb_log_interval == 0:

            wandb.log({frame_str+"/train_loss_patches/l1_loss": Ll1.item(), 
                       frame_str+"/train_loss_patches/total_loss": loss.item(), 
                       frame_str+"/num_points": scene.gaussians.get_xyz.shape[0],
                       frame_str+"/update_points": f"{torch.count_nonzero(scene.gaussians.mask_xyz)}",
                       frame_str+"/elapsed": elapsed,
                       frame_str+"/size": size,
                       iter_metric: iteration
                       })
        
    # Report test and samples of training set
    if is_test:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'val', 'cameras' : scene.getTrainCameras()[0:1]})  # hack: hardcoded val views indices

        report = {"_render_time": 0.0, "_render_count": 0}
        for config in validation_configs:
            metrics = {}
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                if model_args.log_images:
                    os.makedirs(os.path.join(model_args.model_path,config['name'],"gt"),exist_ok=True)
                    os.makedirs(os.path.join(model_args.model_path,config['name'],"renders"),exist_ok=True)
                for idx, viewpoint in enumerate(config['cameras']):
                    if model_args.timed:
                        torch.cuda.synchronize()
                    render_start = time.time()
                    rendered = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    if model_args.timed:
                        torch.cuda.synchronize()
                    report["_render_time"] += time.time() - render_start
                    report["_render_count"] += 1
                    image = torch.clamp(rendered["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), 
                                             image[None], global_step=iteration)
                        if prev_report is None: # First time logging
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), 
                                                 gt_image[None], global_step=iteration)
                    if wandb_enabled and model_args.wandb_log_error_maps and frame_idx <= model_args.wandb_log_error_map_frames and max_iterations is not None and iteration == max_iterations:
                        from utils.image_utils import create_error_map_canvas

                        error_canvas = create_error_map_canvas(
                            gt_image,
                            image,
                            vmin=model_args.wandb_error_map_vmin,
                            vmax=model_args.wandb_error_map_vmax,
                            cmap_name=model_args.wandb_error_map_cmap
                        )

                        wandb.log({
                            "frame/" + config['name'] + "_view_{}/error_map".format(camName_from_Path(viewpoint.image_path)):
                                wandb.Image(error_canvas.permute(1, 2, 0).detach().cpu().numpy(),
                                           caption=f"Frame {frame_idx} | GT | Rendered | Error Diff")
                        })
                    if model_args.log_images:
                        # Not logging GTs
                        os.makedirs(os.path.join(model_args.model_path,config['name'],"renders", 
                                                 camName_from_Path(viewpoint.image_path)),exist_ok=True)
                        if prev_report is None:
                            os.makedirs(os.path.join(model_args.model_path,config['name'],"gt", 
                                                     camName_from_Path(viewpoint.image_path)),exist_ok=True)
                            if os.path.exists(os.path.join(
                                                model_args.model_path,config['name'],"gt", 
                                                camName_from_Path(viewpoint.image_path),str(model_args.start_idx+frame_idx).zfill(4)+".png"
                                                )
                                            ):
                                os.remove(os.path.join(model_args.model_path,config['name'],"gt", 
                                                       camName_from_Path(viewpoint.image_path),str(model_args.start_idx+frame_idx).zfill(4)+".png"))
                            os.symlink(viewpoint.image_path,os.path.join(model_args.model_path,config['name'],"gt", 
                                                                         camName_from_Path(viewpoint.image_path),str(model_args.start_idx+frame_idx).zfill(4)+".png"))

                        save_image(image,os.path.join(model_args.model_path,config['name'],"renders",
                                                      camName_from_Path(viewpoint.image_path),str(model_args.start_idx+frame_idx).zfill(4)+".png"))

                        # Save error maps locally if enabled
                        if model_args.wandb_log_error_maps:
                            from utils.image_utils import create_error_map_canvas

                            error_map_dir = os.path.join(model_args.model_path,config['name'],"error_maps",
                                                        camName_from_Path(viewpoint.image_path))
                            os.makedirs(error_map_dir, exist_ok=True)

                            error_canvas = create_error_map_canvas(
                                gt_image,
                                image,
                                vmin=model_args.wandb_error_map_vmin,
                                vmax=model_args.wandb_error_map_vmax,
                                cmap_name=model_args.wandb_error_map_cmap
                            )

                            save_image(error_canvas, os.path.join(error_map_dir,
                                                                  str(model_args.start_idx+frame_idx).zfill(4)+".png"))

                        # visualize alpha mask if possible
                        if report_alpha:
                            if viewpoint.original_alpha_mask is None:
                                if verbose:
                                    print(f"training_report(): since the dataloader does not provide `original_alpha_mask`, will skip visualizing alpha.")
                            else:
                                # assert viewpoint.gt_alpha_mask is not None, "assume dataloader provides gt_alpha_mask."
                                alpha = torch.clamp(rendered["alpha"], 0.0, 1.0)  # (1, H, W)
                                gt_alpha = torch.clamp(viewpoint.original_alpha_mask, 0.0, 1.0)  # (1, H, W)

                                def _colormap(_data, vmin, vmax, cmap_name="jet"):
                                    # _data: tensor, (H, W)
                                    _height = _data.shape[0]
                                    _width = _data.shape[1]
                                    _data_np = _data.detach().cpu().numpy()  # (H, W)
                                    _vis = value2color(_data_np.ravel(), vmin=vmin, vmax=vmax, cmap_name=cmap_name)  # (HW, 3)
                                    _vis = np.transpose(_vis.reshape((_height, _width, 3)), (2, 0, 1))  # (3, H, W)
                                    _vis = torch.from_numpy(_vis).float().to(_data.device)
                                    return _vis

                                # vis range for alpha: [0, 1]
                                # vis range for alpha diff: choose to use [0, 1] as well for consistent color mapping
                                vis_alpha = _colormap(alpha[0], vmin=0.0, vmax=1.0, cmap_name="jet")
                                vis_gt_alpha = _colormap(gt_alpha[0], vmin=0.0, vmax=1.0, cmap_name="jet")
                                vis_diff_alpha = _colormap(torch.abs(alpha - gt_alpha)[0], vmin=0.0, vmax=1.0, cmap_name="jet")
                                canvas = torch.cat((vis_gt_alpha, vis_alpha, vis_diff_alpha), dim=2)
                                save_image(
                                    canvas,
                                    os.path.join(
                                        model_args.model_path,config['name'],
                                        "renders",
                                        camName_from_Path(viewpoint.image_path),
                                        "compare_alpha_" + str(model_args.start_idx + frame_idx).zfill(4) + ".png"
                                    )
                                )

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                metrics['l1'] = l1_test
                metrics['psnr'] = psnr_test
                report[config['name']] = metrics

                if wandb_enabled and frame_idx <= 2:
                    wandb.log({frame_str+"/"+config["name"]+"/loss_viewpoint/l1_loss": l1_test, 
                               frame_str+"/"+config["name"]+"/loss_viewpoint/psnr": psnr_test})

        report['iteration'] = iteration
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        return report
    else:
        return None

if __name__ == "__main__":

    print('Running on ', socket.gethostname())
    # Config file is used for argument defaults. Command line arguments override config file.
    config_path = sys.argv[sys.argv.index("--config")+1] if "--config" in sys.argv else None
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
    config = defaultdict(lambda: {}, config)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser, config['model_params'])
    op_i = OptimizationParamsInitial(parser, config['opt_params_initial'])
    op_r = OptimizationParamsRest(parser, config['opt_params_rest'])
    pp = PipelineParams(parser, config['pipe_params'])
    qp = QuantizeParams(parser, config['quantize_params'])

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--scene', type=str, default=None,
                        help='Scene name. If set, source_path is resolved to <data_root>/<scene> and model_path defaults to output/<scene> when unset.')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_format", type=str, default='ply')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--init_from_ply', action='store_true', default=False,
                        help='On/off toggle: initialize frame 1 from <source_path>/init_3dgs.ply and skip the static 3DGS training. The filename is fixed (init_3dgs.ply) per scene.')
    parser.add_argument('--use_xyz_legacy', action='store_true', default=False, help='If set, use legacy xyz decoding in GaussianModel (_xyz_legacy) to reproduce paper numbers. To save compressed pkl\'s, leave unset or set to False. Default: False (use _xyz_fixed).')
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)

    # If --scene is set, resolve source_path and (optionally) model_path from data_root
    if args.scene:
        if not args.data_root:
            raise ValueError("--scene requires data_root to be set (in yaml model_params.data_root or via --data_root).")
        if not args.source_path:
            args.source_path = os.path.join(args.data_root, args.scene)
        if not args.model_path:
            args.model_path = os.path.join("output/n3dv_origin", args.scene)

    # On/off toggle for ply init: derive the fixed per-scene path <source_path>/init_3dgs.ply.
    if args.init_from_ply:
        if not args.source_path:
            raise ValueError("--init_from_ply requires source_path (set --scene or --source_path).")
        args.start_checkpoint = os.path.join(args.source_path, "init_3dgs.ply")

    # Merge optimization args for initial and rest and change accordingly
    op = OptimizationParams(op_i.extract(args), op_r.extract(args))

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    lp_args = lp.extract(args)
    pp_args = pp.extract(args)
    qp_args = qp.extract(args)

    # Check for incompatible options
    if args.use_xyz_legacy and getattr(lp_args, 'log_compressed', False):
        print('Error: must use xyz_fixed with log_compressed (do not use --use-xyz-legacy with --log_compressed)')
        sys.exit(1)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp_args, op, pp_args, qp_args, args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
