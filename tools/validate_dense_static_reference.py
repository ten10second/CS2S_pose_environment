"""Inference-only fixed-pair dense static reference validation.

Target RGB is opened only after all reference conditions are constructed.
This script changes no generative model or training configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.temporal_history_geometry import get_geometry, load_velodyne, rebase_kitti_path, consecutive_rows
from tools.temporal_static_geometry import build_static_pair
from tools.temporal_dense_reprojection import build_dense_reference, calibrate_depth, _sparse_depth_map
from tools.temporal_dense_eval import (
    sift_known_pose_matches, project_source_pixels_with_depth, masked_image_diagnostics,
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2))


def save_image(path, array):
    a = np.asarray(array)
    Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255 + .5)).save(path)


def panel(images, labels, columns=2):
    h, w = images[0].shape[:2]
    result = Image.new("RGB", (columns * w, ((len(images) + columns - 1) // columns) * (h + 26)), "#161b22")
    draw = ImageDraw.Draw(result)
    for i, (im, label) in enumerate(zip(images, labels)):
        x, y = (i % columns) * w, (i // columns) * (h + 26)
        draw.text((x + 8, y + 5), label, fill="white")
        result.paste(Image.fromarray(np.uint8(np.clip(im, 0, 1) * 255 + .5)), (x, y + 26))
    return result


def error_summary(errors, mask):
    e = np.asarray(errors)[mask]
    return {"n": int(len(e)), "median_px": float(np.median(e)) if len(e) else None,
            "p90_px": float(np.percentile(e, 90)) if len(e) else None,
            "mean_px": float(e.mean()) if len(e) else None,
            "within_2px": float((e <= 2).mean()) if len(e) else None}


def depth_summary(pred, measured, mask):
    ok = mask & np.isfinite(measured) & (measured > 1) & (measured < 80) & np.isfinite(pred) & (pred > 0)
    delta = np.abs(pred[ok] - measured[ok])
    return {"n": int(ok.sum()), "median_abs_m": float(np.median(delta)) if len(delta) else None,
            "abs_rel": float((delta / measured[ok]).mean()) if len(delta) else None}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--settings", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--vendor", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:4")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--depth-cache", default="")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cache = Path(args.depth_cache) if args.depth_cache else out / "native_depth"
    cache.mkdir(parents=True, exist_ok=True)
    settings = json.loads(Path(args.settings).read_text())["args"]
    entries = json.loads(Path(args.pairs).read_text())["pairs"]
    wanted = {entry[k] for entry in entries for k in ["previous", "current"]}
    manifests = {}
    for split, key in [("train", "train_manifest"), ("heldout", "val_manifest")]:
        with open(settings[key]) as stream:
            manifests[split] = {r["sample_id"]: r for r in map(json.loads, stream) if r["sample_id"] in wanted}
    kitti_root = settings["kitti_root"]
    width, height = args.width, args.height
    image_size = (width, height)
    yy, xx = np.indices((height, width))
    upper = yy < height // 2
    fit_partition = ((xx * 73856093 + yy * 19349663) % 5) != 0
    ckpt_hash = sha256(args.checkpoint)
    model = None
    records = []
    all_matches = []
    write_json(out / "settings.json", {**vars(args), "checkpoint_sha256": ckpt_hash,
        "source_commit": "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
        "depth_model": "Depth Anything V2 Metric VKITTI Small", "input_size": 518,
        "depth_max_m": 80, "source_calibration_fit_fraction": .8,
        "depth_tolerance_m": .75, "target_rgb_for_condition": False,
        "manual_regions": False, "training": False,
        "upper_image_note": "Top half of image, NOT semantic building segmentation",
        "matches_note": "Mutual-ratio SIFT with known-pose epipolar filtering; noisy diagnostic, not ground truth"})
    write_json(out / "pairs.json", entries)
    for entry in entries:
        folder = out / entry["name"]
        folder.mkdir()
        prev = manifests[entry["split"]][entry["previous"]]
        cur = manifests[entry["split"]][entry["current"]]
        if not consecutive_rows(prev, cur):
            raise ValueError("pair must be consecutive within the original manifest")
        def path(row, key):
            return rebase_kitti_path(row[key], kitti_root)
        geom = get_geometry(path(prev, "calib_dir"))
        pose = geom.relative_velo_pose(path(prev, "oxts_path"), path(cur, "oxts_path"))
        prev_points = load_velodyne(path(prev, "velodyne_path"))
        cur_points = load_velodyne(path(cur, "velodyne_path"))
        with Image.open(path(prev, "image_02_path")) as im:
            if im.size != geom.image_size:
                raise ValueError("source RGB and calibration image sizes differ")
            native_rgb = np.array(im.convert("RGB"))
            prev_rgb = np.array(im.convert("RGB").resize(image_size, Image.BILINEAR), dtype=np.float32) / 255
        cache_id = hashlib.sha256((entry["previous"] + ckpt_hash).encode()).hexdigest()[:24]
        depth_path = cache / (cache_id + ".npy")
        if depth_path.exists():
            native_depth = np.load(depth_path)
        else:
            import torch
            if str(args.device).startswith("cuda"):
                torch.cuda.set_device(torch.device(args.device))
            if model is None:
                sys.path.insert(0, str(Path(args.vendor) / "metric_depth"))
                from depth_anything_v2.dpt import DepthAnythingV2
                model = DepthAnythingV2(encoder="vits", features=64,
                    out_channels=[48, 96, 192, 384], max_depth=80)
                model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
                model = model.to(args.device).eval()
            with torch.no_grad():
                native_depth = model.infer_image(native_rgb[:, :, ::-1].copy(), input_size=518)
            np.save(depth_path, native_depth)
        # Metric model predicts effective image_02 depth; compare in rect_cam0 z convention.
        camera_z_shift = float((np.linalg.inv(geom.p_rect_02[:, :3]) @ geom.p_rect_02[:, 3])[2])
        depth_raw = cv2.resize(native_depth, image_size, interpolation=cv2.INTER_LINEAR) - camera_z_shift
        source_depth = _sparse_depth_map(prev_points, geom, image_size)
        target_depth = _sparse_depth_map(cur_points, geom, image_size)
        depth_scaled, fit = calibrate_depth(depth_raw, source_depth, fit_partition)
        depth_anchored = np.where(np.isfinite(source_depth), source_depth, depth_scaled)
        sparse = build_static_pair(prev, cur, kitti_root, image_size=image_size)
        references = {}
        depths = {"raw": depth_raw, "scaled": depth_scaled, "anchored": depth_anchored}
        for name, depth in depths.items():
            references[name] = build_dense_reference(prev_rgb, depth, prev_points, cur_points, geom, pose)
        # Save inference-side artifact BEFORE opening any target RGB.
        candidate = references["anchored"]
        np.savez_compressed(folder / "reference.npz", warped_rgb=candidate["warped_rgb"],
            support_mask=candidate["support_mask"], measured_mask=candidate["measured_mask"],
            estimated_mask=candidate["estimated_mask"], target_conflict_mask=candidate["target_conflict_mask"],
            source_uv=candidate["source_uv"], depth_raw=depth_raw, depth_scaled=depth_scaled,
            depth_anchored=depth_anchored, source_depth=source_depth, target_depth=target_depth)
        # Evaluation begins here. Target image is not a construction input.
        with Image.open(path(cur, "image_02_path")) as im:
            target_rgb = np.array(im.convert("RGB").resize(image_size, Image.BILINEAR), dtype=np.float32) / 255
        matches = sift_known_pose_matches(prev_rgb, target_rgb, geom, pose,
            max_sampson_px=1.5, source_lidar_mask=np.isfinite(source_depth))
        src, dst = matches["source_xy"], matches["target_xy"]
        projections = {name: project_source_pixels_with_depth(src, depth, geom, pose) for name, depth in depths.items()}
        projections["rotation_only"] = project_source_pixels_with_depth(src, np.full_like(depth_raw, 1e7), geom, pose)
        projections["identity"] = {"target_xy": src, "valid": np.ones(len(src), bool)}
        joint_valid = np.logical_and.reduce([v["valid"] for v in projections.values()])
        sx = np.clip(np.rint(src[:, 0]).astype(int), 0, width - 1)
        sy = np.clip(np.rint(src[:, 1]).astype(int), 0, height - 1)
        groups = {"all": joint_valid, "upper_image": joint_valid & (dst[:, 1] < height / 2),
                  "source_no_lidar": joint_valid & ~np.isfinite(source_depth[sy, sx])}
        feature_metrics = {}
        match_arrays = {"source_xy": src, "target_xy": dst, "common_valid": joint_valid}
        for name, proj in projections.items():
            err = np.linalg.norm(proj["target_xy"] - dst, axis=1)
            feature_metrics[name] = {g: error_summary(err, mask) for g, mask in groups.items()}
            match_arrays[name + "_error"] = err
            match_arrays[name + "_xy"] = proj["target_xy"]
        for name, mask in groups.items():
            match_arrays[name + "_mask"] = mask
        np.savez_compressed(folder / "matches.npz", **match_arrays)
        all_matches.append(match_arrays)
        sparse_rgb = sparse["warped_rgb"].transpose(1, 2, 0)
        sparse_strict = sparse["strict_mask"][0]
        common = sparse_strict.copy()
        dense_common = np.ones((height, width), bool)
        for value in references.values():
            common &= value["support_mask"]
            dense_common &= value["support_mask"]
        image_metrics = {}
        compare_images = {"identity": prev_rgb, "sparse": sparse_rgb}
        compare_images.update({k: v["warped_rgb"] for k, v in references.items()})
        for name, im in compare_images.items():
            image_metrics[name] = {"sparse_common": masked_image_diagnostics(im, target_rgb, common)}
            if name != "sparse":
                image_metrics[name]["dense_common"] = masked_image_diagnostics(im, target_rgb, dense_common)
                image_metrics[name]["upper_dense_common"] = masked_image_diagnostics(im, target_rgb, dense_common & upper)
                image_metrics[name]["target_no_lidar_common"] = masked_image_diagnostics(im, target_rgb, dense_common & ~np.isfinite(target_depth))
        coverage = {"sparse_strict": float(sparse_strict.mean()),
                    "sparse_support": float(sparse["support_mask"].mean())}
        for name, ref in references.items():
            coverage[name] = {key: float(ref[key].mean()) for key in ["support_mask", "measured_mask", "estimated_mask", "target_conflict_mask"]}
            coverage[name]["upper_support"] = float(ref["support_mask"][upper].mean())
            coverage[name]["new_support_vs_sparse_strict"] = float((ref["support_mask"] & ~sparse_strict).mean())
        coverage["sparse_upper_strict"] = float(sparse_strict[upper].mean())
        record = {"name": entry["name"], "split": entry["split"], "image_size": image_size,
            "ego_translation_m": float(np.linalg.norm(pose[:3, 3])), "fit": fit,
            "source_lidar_holdout": {name: depth_summary(depth, source_depth, ~fit_partition)
                for name, depth in [("raw", depth_raw), ("scaled", depth_scaled)]},
            "coverage": coverage, "matching": matches["diagnostics"],
            "feature_metrics": feature_metrics, "image_metrics": image_metrics,
            "dense_diagnostics": {k: v["diagnostics"] for k, v in references.items()},
            "dynamic_exclusion_available": sparse["diagnostics"]["dynamic_exclusion_available"]}
        write_json(folder / "metrics.json", record)
        records.append(record)
        colors = np.zeros_like(prev_rgb)
        colors[candidate["estimated_mask"]] = [.15, .45, .95]
        colors[candidate["measured_mask"]] = [.1, .9, .3]
        colors[candidate["target_conflict_mask"]] = [.95, .15, .15]
        overlay = target_rgb.copy()
        mask = candidate["support_mask"]
        overlay[mask] = .5 * target_rgb[mask] + .5 * candidate["warped_rgb"][mask]
        for name, im in [("previous", prev_rgb), ("target", target_rgb), ("sparse", sparse_rgb),
                         ("raw", references["raw"]["warped_rgb"]), ("anchored", candidate["warped_rgb"]),
                         ("support_types", colors), ("overlay", overlay)]:
            save_image(folder / (name + ".png"), im)
        panel([prev_rgb, target_rgb, sparse_rgb, candidate["warped_rgb"], colors, overlay],
              [entry["name"] + " | Previous GT", "Target GT (evaluation only)", "Sparse LiDAR reference",
               "Dense depth + source LiDAR reference", "Green: both-depth checked | Blue: estimated | Red: conflict",
               "50/50 overlay (target + reference; not generation)"]).save(folder / "panel.jpg", quality=94)
        print(json.dumps({"pair": entry["name"], "scale": fit["scale"], "coverage": coverage["anchored"],
                          "feature_error": feature_metrics["anchored"]}, ensure_ascii=False), flush=True)
        write_json(out / "results.json", records)
    pooled = {}
    for method in ["identity", "rotation_only", "raw", "scaled", "anchored"]:
        pooled[method] = {}
        for group in ["all", "upper_image", "source_no_lidar"]:
            errors = np.concatenate([r[method + "_error"][r[group + "_mask"]] for r in all_matches])
            pooled[method][group] = error_summary(errors, np.ones(len(errors), bool))
    write_json(out / "pooled_features.json", pooled)
    write_json(out / "done.json", {"done": True, "pairs": len(records), "training": False,
        "target_rgb_for_condition": False, "checkpoint_sha256": ckpt_hash})


if __name__ == "__main__":
    main()
