import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.kitti_raw_lidar_utils import read_jsonl  # noqa: E402
from dataloader.kitti_raw_lidar_utils import load_raw_calibration, load_velodyne_points, project_velo_to_image  # noqa: E402


DEFAULT_PROMPTS = ("car", "van", "truck", "pedestrian", "cyclist")


def parse_args():
    parser = argparse.ArgumentParser(description="Build KITTI raw foreground mask cache for supervised RGB loss.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--source",
        choices=["sam3", "sam2_box", "sam2_yolo_box", "sam2_auto_yolo", "bbox2d"],
        default="sam3",
    )
    parser.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS))
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--box-score-threshold", type=float, default=0.1)
    parser.add_argument("--min-box-area", type=float, default=16.0)
    parser.add_argument("--max-boxes", type=int, default=64)
    parser.add_argument("--sam2-repo", default="third_party/sam2")
    parser.add_argument("--sam2-checkpoint", default="/home/shizhm/Downloads/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-auto-points-per-side", type=int, default=16)
    parser.add_argument("--sam2-auto-pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--sam2-auto-stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--sam2-auto-min-mask-region-area", type=int, default=64)
    parser.add_argument("--yolo-model", default="yolo11x.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.10)
    parser.add_argument("--yolo-iou", type=float, default=0.6)
    parser.add_argument(
        "--yolo-classwise-nms",
        action="store_true",
        help="Use class-wise YOLO NMS. By default sam2_auto_yolo uses agnostic NMS to suppress duplicate class labels for the same object.",
    )
    parser.add_argument(
        "--sam2-auto-yolo-classes",
        default="person,bicycle,car,motorcycle,bus,truck",
        help="Comma-separated YOLO class names used to keep class-agnostic SAM2 automatic masks.",
    )
    parser.add_argument(
        "--sam2-auto-mask-overlap-threshold",
        type=float,
        default=0.3,
        help="Keep a SAM2 auto mask if at least this fraction of the mask overlaps target YOLO boxes.",
    )
    parser.add_argument(
        "--sam2-auto-box-coverage-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional extra keep rule for small SAM2 masks that cover this fraction of a YOLO box. "
            "Use 0 to disable; relying on box coverage alone can admit large background masks."
        ),
    )
    parser.add_argument(
        "--sam2-auto-max-mask-area-frac",
        type=float,
        default=0.4,
        help="Drop SAM2 auto masks larger than this fraction of the image before YOLO-overlap filtering.",
    )
    parser.add_argument(
        "--sam2-auto-max-mask-to-box-area-ratio",
        type=float,
        default=1.25,
        help="Drop SAM2 auto masks whose area is much larger than the matched YOLO box.",
    )
    parser.add_argument(
        "--sam2-auto-min-box-coverage-threshold",
        type=float,
        default=0.05,
        help="Drop SAM2 auto masks that cover too little of the matched YOLO box.",
    )
    parser.add_argument(
        "--sam2-auto-topk-per-box",
        type=int,
        default=1,
        help="Keep only the top-k SAM2 automatic masks per YOLO box to avoid accumulating background fragments.",
    )
    parser.add_argument("--restrict-to-box", action="store_true", default=True)
    parser.add_argument("--no-restrict-to-box", dest="restrict_to_box", action="store_false")
    parser.add_argument("--box-padding-pixels", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--lidar-box-support-filter",
        action="store_true",
        help="For sam2_yolo_box, keep only detector boxes that contain projected LiDAR hits.",
    )
    parser.add_argument("--lidar-box-support-min-points", type=int, default=1)
    parser.add_argument("--lidar-box-support-max-depth", type=float, default=60.0)
    parser.add_argument("--lidar-box-support-padding-pixels", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--suffix", default="_foreground.png")
    parser.add_argument("--metadata-name", default="foreground_masks.jsonl")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def safe_sample_id(sample_id: str) -> str:
    return sample_id.replace("/", "__")


def output_path(out_root: Path, sample_id: str, suffix: str) -> Path:
    return out_root / f"{safe_sample_id(sample_id)}{suffix}"


def read_bbox2d_boxes(record, class_names=None, score_threshold=0.0, min_area=0.0, max_boxes=0):
    image_path = Path(record["image_02_path"])
    drive_dir = image_path.parents[2]
    bbox_path = drive_dir / "bbox_2d" / f"{image_path.stem}.txt"
    boxes = []
    if not bbox_path.is_file():
        return boxes
    class_names = {name.lower() for name in class_names} if class_names else None
    for line in bbox_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        y0, y1, x0, x1 = [float(v) for v in parts[:4]]
        cls = parts[4].lower()
        score = float(parts[5])
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if class_names is not None and cls not in class_names:
            continue
        if score < score_threshold or area < min_area:
            continue
        boxes.append({"class": cls, "score": score, "xyxy": [x0, y0, x1, y1], "box_area": area})
    boxes.sort(key=lambda row: row["box_area"], reverse=True)
    if max_boxes > 0:
        boxes = boxes[:max_boxes]
    return boxes


def read_bbox2d_mask(record, image_size, class_names=None, score_threshold=0.0, min_area=0.0, max_boxes=0):
    mask = Image.new("L", image_size, 0)
    boxes = read_bbox2d_boxes(record, class_names, score_threshold, min_area, max_boxes)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.rectangle(box["xyxy"], fill=255)
    return mask, boxes


def build_yolo_box_detector(
    yolo_model_name,
    device,
    yolo_conf=0.10,
    yolo_iou=0.6,
    yolo_agnostic_nms=True,
    yolo_classes=None,
    min_area=0.0,
    max_boxes=0,
):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("YOLO foreground detection requires ultralytics.") from exc

    yolo_model = YOLO(yolo_model_name)
    names = {int(idx): str(name).lower() for idx, name in yolo_model.names.items()}
    requested = {str(name).strip().lower() for name in (yolo_classes or []) if str(name).strip()}
    class_ids = [idx for idx, name in names.items() if name in requested]
    if not class_ids:
        raise ValueError(
            f"No YOLO classes matched {sorted(requested)}. Available classes include: {sorted(names.values())[:20]}"
        )

    def run(image):
        image_arr = np.asarray(image.convert("RGB"))
        yolo_device = 0 if str(device).startswith("cuda") else device
        yolo_results = yolo_model.predict(
            image_arr,
            classes=class_ids,
            conf=float(yolo_conf),
            iou=float(yolo_iou),
            agnostic_nms=bool(yolo_agnostic_nms),
            max_det=int(max_boxes) if int(max_boxes) > 0 else 300,
            device=yolo_device,
            verbose=False,
        )
        boxes = []
        if yolo_results:
            result = yolo_results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                cls = result.boxes.cls.detach().cpu().numpy().astype(int)
                conf = result.boxes.conf.detach().cpu().numpy()
                for box, class_id, score in zip(xyxy, cls, conf):
                    x0, y0, x1, y1 = [float(v) for v in box]
                    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                    if area < float(min_area):
                        continue
                    boxes.append(
                        {
                            "class": names.get(int(class_id), str(class_id)),
                            "score": float(score),
                            "xyxy": [x0, y0, x1, y1],
                            "box_area": area,
                        }
                    )
        boxes.sort(key=lambda row: row["score"], reverse=True)
        if max_boxes > 0:
            boxes = boxes[: int(max_boxes)]
        return boxes

    return run


def filter_boxes_by_lidar_support(
    record,
    image_size,
    boxes,
    min_points=1,
    max_depth=60.0,
    padding_pixels=4,
):
    if not boxes:
        return boxes
    if "velodyne_path" not in record or "calib_dir" not in record:
        return boxes
    velodyne_path = Path(record["velodyne_path"])
    if not velodyne_path.is_file():
        return boxes
    calib = load_raw_calibration(record["calib_dir"])
    points = load_velodyne_points(str(velodyne_path))
    uv, depth, valid = project_velo_to_image(points[:, :3], calib, output_size=(image_size[1], image_size[0]))
    valid = valid & np.isfinite(depth) & (depth > 0.0) & (depth <= float(max_depth))
    if not np.any(valid):
        return []
    uv = uv[valid]
    depth = depth[valid]
    pad = int(padding_pixels)
    kept = []
    for row in boxes:
        x0, y0, x1, y1 = [float(v) for v in row["xyxy"]]
        in_box = (
            (uv[:, 0] >= x0 - pad)
            & (uv[:, 0] <= x1 + pad)
            & (uv[:, 1] >= y0 - pad)
            & (uv[:, 1] <= y1 + pad)
        )
        hit_count = int(in_box.sum())
        if hit_count < int(min_points):
            continue
        supported = dict(row)
        supported["lidar_box_hit_count"] = hit_count
        supported["lidar_box_min_depth"] = float(depth[in_box].min()) if hit_count > 0 else 0.0
        supported["lidar_box_mean_depth"] = float(depth[in_box].mean()) if hit_count > 0 else 0.0
        kept.append(supported)
    return kept


def projected_lidar_support_mask(
    record,
    image_size,
    max_depth=60.0,
    dilation_pixels=0,
):
    if "velodyne_path" not in record or "calib_dir" not in record:
        return None
    velodyne_path = Path(record["velodyne_path"])
    if not velodyne_path.is_file():
        return None
    calib = load_raw_calibration(record["calib_dir"])
    points = load_velodyne_points(str(velodyne_path))
    uv, depth, valid = project_velo_to_image(points[:, :3], calib, output_size=(image_size[1], image_size[0]))
    valid = valid & np.isfinite(depth) & (depth > 0.0) & (depth <= float(max_depth))
    mask = np.zeros((image_size[1], image_size[0]), dtype=bool)
    if np.any(valid):
        xy = np.rint(uv[valid]).astype(np.int64)
        xy[:, 0] = np.clip(xy[:, 0], 0, image_size[0] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, image_size[1] - 1)
        mask[xy[:, 1], xy[:, 0]] = True
    dilation = int(dilation_pixels)
    if dilation > 0 and mask.any():
        padded = np.pad(mask, dilation, mode="constant", constant_values=False)
        dilated = np.zeros_like(mask)
        kernel = 2 * dilation + 1
        for dy in range(kernel):
            for dx in range(kernel):
                dilated |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        mask = dilated
    return mask


def as_numpy_masks(masks):
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    return masks


def build_sam3_runner(device):
    try:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except Exception as exc:
        raise RuntimeError(
            "SAM3 is not importable in this environment. Create/use the SAM3 env, then rerun with --source sam3."
        ) from exc

    model = build_sam3_image_model()
    if device and hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    processor = Sam3Processor(model)

    def run(image, prompts, threshold):
        union = Image.new("L", image.size, 0)
        rows = []
        state = processor.set_image(image)
        for prompt in prompts:
            with torch.no_grad():
                output = processor.set_text_prompt(state=state, prompt=prompt)
            masks = as_numpy_masks(output.get("masks", []))
            scores = output.get("scores", [])
            boxes = output.get("boxes", [])
            if hasattr(scores, "detach"):
                scores = scores.detach().cpu().numpy()
            if hasattr(boxes, "detach"):
                boxes = boxes.detach().cpu().numpy()
            for idx, mask_arr in enumerate(masks):
                score = float(scores[idx]) if idx < len(scores) else 1.0
                if score < threshold:
                    continue
                mask_img = Image.fromarray((mask_arr > 0).astype(np.uint8) * 255, mode="L").resize(image.size, Image.NEAREST)
                union = Image.fromarray(np.maximum(np.asarray(union), np.asarray(mask_img)).astype(np.uint8), mode="L")
                row = {"class": prompt, "score": score}
                if idx < len(boxes):
                    row["xyxy"] = [float(v) for v in boxes[idx]]
                rows.append(row)
        return union, rows

    return run


def build_sam2_box_runner(repo_path, checkpoint, model_cfg, device, restrict_to_box=True, box_padding_pixels=8):
    repo = Path(repo_path)
    if str(repo.resolve()) not in sys.path:
        sys.path.insert(0, str(repo.resolve()))
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise RuntimeError(
            f"SAM2 is not importable. Expected official repo at {repo}. "
            "Use --sam2-repo third_party/sam2 or install facebookresearch/sam2."
        ) from exc

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)

    def box_clip_mask(image_size, xyxy):
        mask = Image.new("L", image_size, 0)
        x0, y0, x1, y1 = xyxy
        pad = int(box_padding_pixels)
        x0 = max(0, int(x0) - pad)
        y0 = max(0, int(y0) - pad)
        x1 = min(image_size[0] - 1, int(x1) + pad)
        y1 = min(image_size[1] - 1, int(y1) + pad)
        ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)
        return np.asarray(mask) > 0

    def run(image, boxes, support_mask=None, support_min_points=1):
        union_arr = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        if not boxes:
            return Image.fromarray(union_arr, mode="L"), []
        box_array = np.asarray([row["xyxy"] for row in boxes], dtype=np.float32)
        enabled_autocast = str(device).startswith("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled_autocast):
            predictor.set_image(np.asarray(image.convert("RGB")))
            masks, scores, _ = predictor.predict(box=box_array, multimask_output=False)
        masks = as_numpy_masks(masks)
        scores = np.asarray(scores).reshape(-1)
        instances = []
        for idx, row in enumerate(boxes):
            if idx >= len(masks):
                continue
            mask_arr = masks[idx] > 0
            if restrict_to_box:
                mask_arr = mask_arr & box_clip_mask(image.size, row["xyxy"])
            if support_mask is not None:
                support_hits = int(np.logical_and(mask_arr, support_mask).sum())
                if support_hits < int(support_min_points):
                    continue
            else:
                support_hits = 0
            union_arr = np.maximum(union_arr, mask_arr.astype(np.uint8) * 255)
            instances.append(
                {
                    **row,
                    "sam2_score": float(scores[idx]) if idx < len(scores) else 0.0,
                    "mask_area": int(mask_arr.sum()),
                    "lidar_mask_hit_count": support_hits,
                }
            )
        return Image.fromarray(union_arr, mode="L"), instances

    return run


def build_sam2_auto_yolo_runner(
    repo_path,
    checkpoint,
    model_cfg,
    device,
    points_per_side=16,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.92,
    min_mask_region_area=64,
    yolo_model_name="yolo11x.pt",
    yolo_conf=0.10,
    yolo_iou=0.6,
    yolo_agnostic_nms=True,
    yolo_classes=None,
    mask_overlap_threshold=0.2,
    box_coverage_threshold=0.15,
    max_mask_area_frac=0.4,
    max_mask_to_box_area_ratio=1.25,
    min_box_coverage_threshold=0.05,
    topk_per_box=1,
):
    repo = Path(repo_path)
    if str(repo.resolve()) not in sys.path:
        sys.path.insert(0, str(repo.resolve()))
    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "SAM2 automatic + YOLO source requires the official SAM2 repo and ultralytics."
        ) from exc

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=int(points_per_side),
        pred_iou_thresh=float(pred_iou_thresh),
        stability_score_thresh=float(stability_score_thresh),
        min_mask_region_area=int(min_mask_region_area),
        output_mode="binary_mask",
    )
    yolo_model = YOLO(yolo_model_name)
    names = {int(idx): str(name).lower() for idx, name in yolo_model.names.items()}
    requested = {str(name).strip().lower() for name in (yolo_classes or []) if str(name).strip()}
    class_ids = [idx for idx, name in names.items() if name in requested]
    if not class_ids:
        raise ValueError(
            f"No YOLO classes matched {sorted(requested)}. Available classes include: {sorted(names.values())[:20]}"
        )

    def xyxy_to_mask(image_size, xyxy):
        mask = np.zeros((image_size[1], image_size[0]), dtype=bool)
        x0, y0, x1, y1 = xyxy
        x0 = max(0, int(np.floor(x0)))
        y0 = max(0, int(np.floor(y0)))
        x1 = min(image_size[0], int(np.ceil(x1)))
        y1 = min(image_size[1], int(np.ceil(y1)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
        return mask

    def run(image, prompts):
        image_rgb = image.convert("RGB")
        image_arr = np.asarray(image_rgb)
        yolo_device = 0 if str(device).startswith("cuda") else device
        yolo_results = yolo_model.predict(
            image_arr,
            classes=class_ids,
            conf=float(yolo_conf),
            iou=float(yolo_iou),
            agnostic_nms=bool(yolo_agnostic_nms),
            device=yolo_device,
            verbose=False,
        )
        boxes = []
        if yolo_results:
            result = yolo_results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                cls = result.boxes.cls.detach().cpu().numpy().astype(int)
                conf = result.boxes.conf.detach().cpu().numpy()
                for box, class_id, score in zip(xyxy, cls, conf):
                    boxes.append(
                        {
                            "class": names.get(int(class_id), str(class_id)),
                            "score": float(score),
                            "xyxy": [float(v) for v in box],
                        }
                    )
        if not boxes:
            return Image.new("L", image.size, 0), []

        box_masks = [xyxy_to_mask(image.size, row["xyxy"]) for row in boxes]
        box_areas = [max(float(mask.sum()), 1.0) for mask in box_masks]
        candidates_by_box = [[] for _ in boxes]
        with torch.inference_mode():
            anns = generator.generate(image_arr)
        image_area = float(max(image.size[0] * image.size[1], 1))
        for ann_idx, ann in enumerate(anns):
            mask_arr = np.asarray(ann["segmentation"], dtype=bool)
            mask_area = max(float(mask_arr.sum()), 1.0)
            if mask_area / image_area > float(max_mask_area_frac):
                continue
            for box_idx, box_mask in enumerate(box_masks):
                if mask_area > box_areas[box_idx] * float(max_mask_to_box_area_ratio):
                    continue
                inter = float(np.logical_and(mask_arr, box_mask).sum())
                mask_overlap = inter / mask_area
                box_coverage = inter / box_areas[box_idx]
                if box_coverage < float(min_box_coverage_threshold):
                    continue
                box_coverage_ok = (
                    float(box_coverage_threshold) > 0.0
                    and box_coverage >= float(box_coverage_threshold)
                    and mask_overlap >= float(mask_overlap_threshold) * 0.5
                )
                if mask_overlap >= float(mask_overlap_threshold) or box_coverage_ok:
                    # Prefer masks that are mostly inside the detector box and cover
                    # a meaningful part of it; this suppresses tiny/background SAM2
                    # fragments that happen to sit inside a large vehicle box.
                    score = mask_overlap * box_coverage
                    candidates_by_box[box_idx].append(
                        {
                            **boxes[box_idx],
                            "match_score": score,
                            "mask_overlap": mask_overlap,
                            "box_coverage": box_coverage,
                            "mask_arr": mask_arr,
                            "sam2_auto_index": int(ann_idx),
                            "sam2_predicted_iou": float(ann.get("predicted_iou", 0.0)),
                            "sam2_stability_score": float(ann.get("stability_score", 0.0)),
                            "mask_area": int(mask_arr.sum()),
                            "sam2_bbox_xywh": [float(v) for v in ann.get("bbox", [])],
                        }
                    )

        union_arr = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        instances = []
        max_keep = max(int(topk_per_box), 0)
        selected_ann_indices = set()
        for candidates in candidates_by_box:
            candidates.sort(
                key=lambda row: (
                    row["match_score"],
                    row["box_coverage"],
                    row["sam2_predicted_iou"],
                    row["sam2_stability_score"],
                ),
                reverse=True,
            )
            kept_for_box = 0
            for candidate in candidates:
                if max_keep > 0 and kept_for_box >= max_keep:
                    break
                ann_idx = candidate["sam2_auto_index"]
                if ann_idx in selected_ann_indices:
                    continue
                selected_ann_indices.add(ann_idx)
                kept_for_box += 1
                union_arr = np.maximum(union_arr, candidate["mask_arr"].astype(np.uint8) * 255)
                instance = {key: value for key, value in candidate.items() if key != "mask_arr"}
                instances.append(instance)
        return Image.fromarray(union_arr, mode="L"), instances

    return run


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.manifest)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    prompts = [item.strip() for item in args.prompts.split(",") if item.strip()]
    sam3_run = build_sam3_runner(args.device) if args.source == "sam3" else None
    sam2_run = (
        build_sam2_box_runner(
            args.sam2_repo,
            args.sam2_checkpoint,
            args.sam2_config,
            args.device,
            restrict_to_box=args.restrict_to_box,
            box_padding_pixels=args.box_padding_pixels,
        )
        if args.source in {"sam2_box", "sam2_yolo_box"}
        else None
    )
    yolo_box_detect = (
        build_yolo_box_detector(
            args.yolo_model,
            args.device,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_agnostic_nms=not args.yolo_classwise_nms,
            yolo_classes=[item.strip() for item in args.sam2_auto_yolo_classes.split(",")],
            min_area=args.min_box_area,
            max_boxes=args.max_boxes,
        )
        if args.source == "sam2_yolo_box"
        else None
    )
    sam2_auto_yolo_run = (
        build_sam2_auto_yolo_runner(
            args.sam2_repo,
            args.sam2_checkpoint,
            args.sam2_config,
            args.device,
            points_per_side=args.sam2_auto_points_per_side,
            pred_iou_thresh=args.sam2_auto_pred_iou_thresh,
            stability_score_thresh=args.sam2_auto_stability_score_thresh,
            min_mask_region_area=args.sam2_auto_min_mask_region_area,
            yolo_model_name=args.yolo_model,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_agnostic_nms=not args.yolo_classwise_nms,
            yolo_classes=[item.strip() for item in args.sam2_auto_yolo_classes.split(",")],
            mask_overlap_threshold=args.sam2_auto_mask_overlap_threshold,
            box_coverage_threshold=args.sam2_auto_box_coverage_threshold,
            max_mask_area_frac=args.sam2_auto_max_mask_area_frac,
            max_mask_to_box_area_ratio=args.sam2_auto_max_mask_to_box_area_ratio,
            min_box_coverage_threshold=args.sam2_auto_min_box_coverage_threshold,
            topk_per_box=args.sam2_auto_topk_per_box,
        )
        if args.source == "sam2_auto_yolo"
        else None
    )
    metadata_path = out_root / args.metadata_name

    with metadata_path.open("w") as meta_file:
        for idx, record in enumerate(records):
            image = Image.open(record["image_02_path"]).convert("RGB")
            mask_path = output_path(out_root, record["sample_id"], args.suffix)
            if args.skip_existing and mask_path.is_file():
                mask = Image.open(mask_path).convert("L")
                row = {
                    "index": idx,
                    "sample_id": record["sample_id"],
                    "image_02_path": record["image_02_path"],
                    "mask_path": str(mask_path),
                    "source": args.source,
                    "skipped_existing": True,
                    "num_instances": -1,
                    "mask_coverage": float(np.asarray(mask).mean() / 255.0),
                    "instances": [],
                }
                meta_file.write(json.dumps(row, sort_keys=True) + "\n")
                continue
            if args.source == "bbox2d":
                mask, instances = read_bbox2d_mask(
                    record,
                    image.size,
                    class_names=prompts,
                    score_threshold=args.box_score_threshold,
                    min_area=args.min_box_area,
                    max_boxes=args.max_boxes,
                )
            elif args.source == "sam2_box":
                boxes = read_bbox2d_boxes(
                    record,
                    class_names=prompts,
                    score_threshold=args.box_score_threshold,
                    min_area=args.min_box_area,
                    max_boxes=args.max_boxes,
                )
                mask, instances = sam2_run(image, boxes)
            elif args.source == "sam2_yolo_box":
                boxes = yolo_box_detect(image)
                support_mask = None
                if args.lidar_box_support_filter:
                    support_mask = projected_lidar_support_mask(
                        record,
                        image.size,
                        max_depth=args.lidar_box_support_max_depth,
                        dilation_pixels=args.lidar_box_support_padding_pixels,
                    )
                    boxes = filter_boxes_by_lidar_support(
                        record,
                        image.size,
                        boxes,
                        min_points=args.lidar_box_support_min_points,
                        max_depth=args.lidar_box_support_max_depth,
                        padding_pixels=args.lidar_box_support_padding_pixels,
                    )
                mask, instances = sam2_run(
                    image,
                    boxes,
                    support_mask=support_mask,
                    support_min_points=args.lidar_box_support_min_points,
                )
            elif args.source == "sam2_auto_yolo":
                mask, instances = sam2_auto_yolo_run(image, prompts)
            else:
                mask, instances = sam3_run(image, prompts, args.score_threshold)
            mask.save(mask_path)
            row = {
                "index": idx,
                "sample_id": record["sample_id"],
                "image_02_path": record["image_02_path"],
                "mask_path": str(mask_path),
                "source": args.source,
                "num_instances": len(instances),
                "mask_coverage": float(np.asarray(mask).mean() / 255.0),
                "instances": instances,
            }
            meta_file.write(json.dumps(row, sort_keys=True) + "\n")
            if idx % 100 == 0:
                print(json.dumps({"index": idx, "sample_id": record["sample_id"], "mask_coverage": row["mask_coverage"]}))

    print(json.dumps({"complete": True, "num_records": len(records), "out_root": str(out_root), "metadata": str(metadata_path)}))


if __name__ == "__main__":
    main()
