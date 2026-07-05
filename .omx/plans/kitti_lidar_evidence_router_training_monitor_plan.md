# KITTI LiDAR Evidence Router Training Monitor Plan

See the project-facing copy at:

`todo/kitti_lidar_evidence_router_training_monitor_plan.md`

This OMX plan mirrors the same execution criteria:

- Train only the new LiDAR evidence path first.
- Monitor loss, resource usage, LiDAR token validity, evidence coverage, and gate movement.
- Save GT / LiDAR overlay / `result/KITTI.ckpt` baseline / trained-model panels on route-disjoint hardcases.
- Treat current `test1` as in-route diagnostic only; use `test2` or a geofence split for valid generalization because train/test1 share drives.
- Use `tools/build_kitti_gps_buffer_split.py` for stricter GPS-buffer splits. Current 30m split keeps 17055 train frames, removes 900 train frames near the test2 route, and writes `dataset/kitti_raw_sat_lidar_geofence_test2_buffer30/`.
- The old 500-step run is a smoke checkpoint only: metadata still points validation at `test1`, and test2 panels show weak LiDAR effect with `lidar_gate_mean` almost unchanged from 0.001.
- `lidar_evidence_router_geofence_s1_gate0p01_2000` finished 2000 steps but is a low-LR control run: optimizer grouping also scaled new LiDAR UNet modules by `lidar_unet_lr_scale=0.05`. The optimizer is now fixed so new LiDAR modules use base lr while old partial-unfreeze UNet params use the low lr.
- `lidar_evidence_router_geofence_s1_fixedlr_gate0p01_2000` finished 2000 steps. The LiDAR gate moved from 0.010000 to 0.010071, but test2 panels still show very weak normal-vs-shift response, so frozen-backbone Stage 1 is insufficient and partial-unfreeze is the next run.
- `lidar_evidence_router_geofence_s2_partial_warmfixed_2000` has now run from step 2000 to step 9000 on the geofence train split. It saved `step_009000.pt` and `last.pt`.
- 9000-step metrics: 140 log records from step 2050-9000, loss last10 mean `0.398759`, `lidar_gate_mean` `0.010126 -> 0.010253`, LiDAR token validity last10 mean `0.543506`, hit coverage last10 mean `0.279701`.
- Foreground training signal was present overall, but not uniformly: 89 / 140 logged batches had `num_dynamic_boxes > 0`, while the last 20 logged batches had 0 dynamic boxes.
- Route-disjoint foreground panels are saved for the key checkpoints: `sample_vis_step003000_test2/panels/`, `sample_vis_step006000_test2/panels/`, and `sample_vis_last9000_test2/panels/`; 9000-step zoom panels are under `sample_vis_last9000_test2/zoom_panels/`.
- Key checkpoint trend: step 3000 has `normal-zero` mean `10.620` and `normal-shift_x` mean `0.433`; step 6000 has `6.909` and `0.234`; step 9000 has `14.873` and `0.572`. This consistently shows reaction to LiDAR presence but almost no reaction to LiDAR position.
- 9000-step acceptance result: failed the core geometry-control criterion. Visual panels confirm the right foreground vehicle and center cyclist/vehicles are not corrected by LiDAR evidence.
- Strong-injection S3 run completed: `lidar_evidence_router_geofence_s3_strength_gate0p05_lr3_2000`, warmstarted from the 9000-step S2 checkpoint, forced all LiDAR gates to `0.05`, used LiDAR context/new LiDAR UNet lr `1.5e-4`, kept old partial-unfreeze UNet lr at `2.5e-6`, and ran 2000 steps with batch size 1. Training stayed stable; final `lidar_gate_mean` was about `0.04979`, peak CUDA allocation about `9130 MB`.
- S3 foreground hardcase panels are saved under `sample_vis_last_test2/panels/`, with zoom panels under `sample_vis_last_test2/zoom_panels/`.
- S3 metrics: `normal-zero` mean dropped from S2 `14.873` to `2.994`; `normal-shift_x` mean rose from `0.572` to `1.135`. This means stronger injection increases LiDAR perturbation sensitivity, but visual inspection still shows mostly texture/edge changes instead of foreground topology changes.
- S3 acceptance result: still failed. Normal/zero/shift outputs share the same KITTI street prior; foreground vehicles/pedestrian-like structures are not forced to follow LiDAR evidence.
- Current diagnosis: the LiDAR path is wired into denoise, but foreground RGB/instance regions are not explicitly supervised strongly enough. Ignoring LiDAR-supported foreground therefore has little loss cost, so the model keeps using the strong CS2S/KITTI prior.
- Updated next objective: add an offline foreground instance/union mask cache, preferably from SAM3 on GT images. Training reads `foreground_mask` only; SAM3 is not run in the diffusion loop. Use the mask for stronger foreground RGB/x0/LPIPS supervision, optionally intersected or weighted by LiDAR hit/near-field evidence.
- Tracklet-projected boxes and local `bbox_2d` may be used only as plumbing/smoke fallbacks. The formal foreground RGB supervision target is a high-quality instance segmentation mask cache.
- Foreground mask supervision is now wired and smoke-tested: dataloader loads `{sample_id}_foreground.png`, the train script exposes foreground loss weights, diffusion loss supports foreground eps/x0/RGB/crop-LPIPS terms, and `tools/build_kitti_foreground_mask_cache.py` can build SAM3 masks or bbox2d-only smoke masks. Smoke runs passed under `foreground_mask_loss_smoke_2step` and `foreground_mask_lpips_smoke_1step`.
- Accept the run only if LiDAR perturbations cause localized foreground/evidence-region changes without destroying CS2S global layout.
- Next run should not simply extend the same configuration or only raise gates. Add explicit LiDAR perturbation consistency / contrastive loss and foreground/hit-cluster oversampling before considering more LoRA or broader UNet unfreeze. Use LiDAR-derived hit/free-space/unknown masks as primary supervision; do not rely on noisy KITTI bbox labels as the main foreground loss.
