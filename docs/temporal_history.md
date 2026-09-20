# Minimal two-frame conditional diffusion

Implementation: optional 11-channel UNet input = noisy latent(4) + fixed aligned VAE latent(4) + area-averaged valid/measured/estimated(3). All previous history reader/attention/decoder adapter modes are removed. The default unconfigured model keeps its original four-channel single-frame path.

The previous RGB alone supplies history appearance. Existing dense depth, source/target LiDAR and ego pose produce aligned RGB and source masks; target RGB is used only for supervised training. Current satellite and LiDAR conditioning paths are unchanged. No vehicle instance association is assumed.

Training: `tools/train_temporal_pairs.py --settings SETTINGS --manifest MANIFEST --out-dir NEW --rgb-weight VALUE --edge-weight VALUE`. This is a single-GPU reference implementation. Jointly updates the UNet; VAE and sat/LiDAR condition encoders stay frozen. New input channels start at zero. Mild RGB augmentation is shared by both frames. Reference masks alone are area averaged; RGB is not preblurred.

Loss: epsilon MSE + sqrt(alpha_bar) times RGB Charbonnier and Sobel X/Y L1, compared to current GT. VAE decoding retains gradients to the predicted clean latent; predicted RGB is not clipped. These are single-noise-step clean-image estimates, not differentiable full-DDIM terminal losses. Coefficients are explicit experiment settings, not validated gains.

Training uses precomputed reference NPZ files and original train-manifest membership, preserving geographic split. Missing references fail clearly: the stopped producer is not implicitly relaunched. Existing preparation/cache tools remain reusable under their historical filenames. Raw RGB and source-index cache fields are reused, not cached old adapter features.

Inference: `tools/infer_temporal.py --settings SETTINGS --temporal-checkpoint NEW.pt --input PAYLOAD.pt --out OUTPUT.pt`. Payload includes kwargs(current sat/LiDAR conditions), warp_rgb, valid, measured, estimated with batch axes. Without a temporal checkpoint, `--history-off` and payload shape run the original base. With a new checkpoint, `--history-off` gives the updated UNet control. Target RGB is not required for inference.

New checkpoints are strict `temporal_concat_rgb_v1` full UNet weights tied to base identity. Legacy adapter checkpoints are rejected. They do not include optimizer resume state. Default keeps last two checkpoints, saving every20 epochs and at completion. No training auto-launch occurs after code changes.

Verification and smoke evidence: `/mnt/shizhm/CS2S_run_control/temporal_minimal_20260920`.
