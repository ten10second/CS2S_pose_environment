import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


DYNAMIC_CLASS_TO_ID = {
    "Car": 1,
    "Van": 2,
    "Truck": 3,
    "Pedestrian": 4,
    "Person (sitting)": 5,
    "Person_sitting": 5,
    "Cyclist": 6,
    "Tram": 7,
}

IGNORED_TRACKLET_CLASSES = {"Misc", "DontCare"}
CONDITION_CHANNELS_BY_MODE = {
    "none": 4,
    "bbox_dynamic": 4,
    "dynamic_points": 4,
    "raw_lidar": 4,
    "dynamic_full": 4,
}


def lidar_condition_channels(mode: str) -> int:
    mode = mode.lower()
    if mode not in CONDITION_CHANNELS_BY_MODE:
        raise ValueError(f"Unsupported lidar condition mode: {mode}")
    return CONDITION_CHANNELS_BY_MODE[mode]


def lidar_condition_gate_channel(mode: str) -> int:
    return -1


@dataclass(frozen=True)
class TrackletBox:
    object_type: str
    class_id: int
    frame_id: int
    h: float
    w: float
    l: float
    tx: float
    ty: float
    tz: float
    rx: float
    ry: float
    rz: float
    state: int = -1
    occlusion: int = -1
    truncation: float = -1.0

    @property
    def center(self) -> np.ndarray:
        return np.asarray([self.tx, self.ty, self.tz + self.h / 2.0], dtype=np.float32)


def _parse_float(node: ET.Element, key: str, default: float = 0.0) -> float:
    text = node.findtext(key)
    return default if text is None else float(text)


def _parse_int(node: ET.Element, key: str, default: int = -1) -> int:
    text = node.findtext(key)
    if text is None:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def parse_tracklet_xml(xml_path: str) -> Dict[int, List[TrackletBox]]:
    """Parse KITTI raw tracklet_labels.xml into frame-indexed dynamic boxes."""
    xml_file = Path(xml_path)
    if not xml_path or not xml_file.is_file():
        return {}

    root = ET.parse(xml_file).getroot()
    tracklets = root.find("tracklets")
    if tracklets is None:
        return {}

    boxes_by_frame: Dict[int, List[TrackletBox]] = {}
    for item in tracklets.findall("item"):
        object_type = item.findtext("objectType")
        if not object_type or object_type in IGNORED_TRACKLET_CLASSES:
            continue
        class_id = DYNAMIC_CLASS_TO_ID.get(object_type)
        if class_id is None:
            continue

        h = _parse_float(item, "h")
        w = _parse_float(item, "w")
        l = _parse_float(item, "l")
        first_frame = _parse_int(item, "first_frame", 0)
        poses = item.find("poses")
        if poses is None:
            continue

        pose_index = 0
        for pose in poses.findall("item"):
            if pose.find("tx") is None:
                continue
            frame_id = first_frame + pose_index
            pose_index += 1
            box = TrackletBox(
                object_type=object_type,
                class_id=class_id,
                frame_id=frame_id,
                h=h,
                w=w,
                l=l,
                tx=_parse_float(pose, "tx"),
                ty=_parse_float(pose, "ty"),
                tz=_parse_float(pose, "tz"),
                rx=_parse_float(pose, "rx"),
                ry=_parse_float(pose, "ry"),
                rz=_parse_float(pose, "rz"),
                state=_parse_int(pose, "state"),
                occlusion=_parse_int(pose, "occlusion"),
                truncation=_parse_float(pose, "truncation", -1.0),
            )
            boxes_by_frame.setdefault(frame_id, []).append(box)

    return boxes_by_frame


def boxes_to_jsonable(boxes: Iterable[TrackletBox]) -> List[dict]:
    return [asdict(box) for box in boxes]


def read_calib_file(path: str) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            data[key] = np.asarray([float(x) for x in value.split()], dtype=np.float32)
        except ValueError:
            continue
    return data


def _transform_from_rt(calib: Dict[str, np.ndarray]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = calib["R"].reshape(3, 3)
    transform[:3, 3] = calib["T"].reshape(3)
    return transform


def _camera_center_from_projection(projection: np.ndarray) -> np.ndarray:
    intrinsics = projection[:3, :3]
    translation = np.linalg.inv(intrinsics) @ projection[:, 3]
    return (-translation).astype(np.float32)


def load_raw_calibration(calib_dir: str) -> Dict[str, np.ndarray]:
    calib_path = Path(calib_dir)
    if not (calib_path / "calib_cam_to_cam.txt").exists():
        for candidate in calib_path.rglob("calib_cam_to_cam.txt"):
            if (candidate.parent / "calib_velo_to_cam.txt").exists():
                calib_path = candidate.parent
                break
    cam = read_calib_file(str(calib_path / "calib_cam_to_cam.txt"))
    velo = read_calib_file(str(calib_path / "calib_velo_to_cam.txt"))
    imu_path = calib_path / "calib_imu_to_velo.txt"
    imu = read_calib_file(str(imu_path)) if imu_path.exists() else {}

    p_rect_02 = cam["P_rect_02"].reshape(3, 4)
    r_rect_00 = cam.get("R_rect_00", np.eye(3, dtype=np.float32).reshape(-1)).reshape(3, 3)
    s_rect_02 = cam.get("S_rect_02", np.asarray([1242.0, 375.0], dtype=np.float32))

    tr_velo_to_cam = _transform_from_rt(velo)

    r_rect_00_ext = np.eye(4, dtype=np.float32)
    r_rect_00_ext[:3, :3] = r_rect_00

    result = {
        "calib_dir": str(calib_path),
        "P_rect_02": p_rect_02.astype(np.float32),
        "R_rect_00": r_rect_00.astype(np.float32),
        "R_rect_00_ext": r_rect_00_ext.astype(np.float32),
        "Tr_velo_to_cam": tr_velo_to_cam.astype(np.float32),
        "S_rect_02": s_rect_02.astype(np.float32),
    }
    if imu:
        tr_imu_to_velo = _transform_from_rt(imu)
        tr_imu_to_rect_00 = r_rect_00_ext @ tr_velo_to_cam @ tr_imu_to_velo
        cam0_rect_in_imu = np.linalg.inv(tr_imu_to_rect_00) @ np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        cam2_rect_in_rect0 = _camera_center_from_projection(p_rect_02)
        cam2_rect_in_imu = np.linalg.inv(tr_imu_to_rect_00) @ np.concatenate(
            [cam2_rect_in_rect0, np.asarray([1.0], dtype=np.float32)]
        )
        result.update(
            {
                "Tr_imu_to_velo": tr_imu_to_velo.astype(np.float32),
                "Tr_imu_to_rect_00": tr_imu_to_rect_00.astype(np.float32),
                "cam0_rect_in_imu": cam0_rect_in_imu[:3].astype(np.float32),
                "cam2_rect_in_rect0": cam2_rect_in_rect0.astype(np.float32),
                "cam2_rect_in_imu": cam2_rect_in_imu[:3].astype(np.float32),
                "cam2_imu_forward_right": np.asarray(
                    [cam2_rect_in_imu[0], -cam2_rect_in_imu[1]],
                    dtype=np.float32,
                ),
            }
        )
    return result


def scaled_camera_k(calib: Dict[str, np.ndarray], output_size: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = output_size
    src_w, src_h = calib["S_rect_02"]
    p2 = calib["P_rect_02"]
    k = np.asarray(
        [
            [p2[0, 0] * out_w / src_w, 0.0, p2[0, 2] * out_w / src_w],
            [0.0, p2[1, 1] * out_h / src_h, p2[1, 2] * out_h / src_h],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return k


def load_velodyne_points(path: str) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return points.reshape(-1, 4)


def velo_to_rect(points_xyz: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    if points_xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts_h = np.concatenate(
        [points_xyz[:, :3], np.ones((points_xyz.shape[0], 1), dtype=np.float32)],
        axis=1,
    ).T
    rect = (calib["R_rect_00_ext"] @ calib["Tr_velo_to_cam"] @ pts_h).T
    return rect[:, :3].astype(np.float32)


def project_velo_to_image(
    points_xyz: np.ndarray,
    calib: Dict[str, np.ndarray],
    output_size: Tuple[int, int] = (128, 512),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_xyz.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )

    out_h, out_w = output_size
    src_w, src_h = calib["S_rect_02"]

    rect_xyz = velo_to_rect(points_xyz, calib)
    rect_h = np.concatenate(
        [rect_xyz, np.ones((rect_xyz.shape[0], 1), dtype=np.float32)],
        axis=1,
    ).T
    pix = calib["P_rect_02"] @ rect_h

    depth = pix[2]
    safe_depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    uv = (pix[:2] / safe_depth).T
    uv[:, 0] *= out_w / src_w
    uv[:, 1] *= out_h / src_h

    valid = (
        (depth > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < out_w)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < out_h)
    )
    return uv.astype(np.float32), depth.astype(np.float32), valid


def box_corners_velo(box: TrackletBox) -> np.ndarray:
    x = np.asarray([box.l / 2, box.l / 2, -box.l / 2, -box.l / 2, box.l / 2, box.l / 2, -box.l / 2, -box.l / 2])
    y = np.asarray([box.w / 2, -box.w / 2, -box.w / 2, box.w / 2, box.w / 2, -box.w / 2, -box.w / 2, box.w / 2])
    z = np.asarray([-box.h / 2, -box.h / 2, -box.h / 2, -box.h / 2, box.h / 2, box.h / 2, box.h / 2, box.h / 2])
    local = np.stack([x, y, z], axis=0).astype(np.float32)
    c = math.cos(box.rz)
    s = math.sin(box.rz)
    rz = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rz @ local).T + box.center[None, :]


def points_in_boxes(points_xyz: np.ndarray, boxes: List[TrackletBox]) -> Tuple[np.ndarray, np.ndarray]:
    inside_any = np.zeros((points_xyz.shape[0],), dtype=bool)
    class_ids = np.zeros((points_xyz.shape[0],), dtype=np.int64)
    if points_xyz.size == 0 or not boxes:
        return inside_any, class_ids

    for box in boxes:
        diff = points_xyz[:, :3] - box.center[None, :]
        c = math.cos(box.rz)
        s = math.sin(box.rz)
        local_x = c * diff[:, 0] + s * diff[:, 1]
        local_y = -s * diff[:, 0] + c * diff[:, 1]
        local_z = diff[:, 2]
        inside = (
            (np.abs(local_x) <= box.l / 2.0)
            & (np.abs(local_y) <= box.w / 2.0)
            & (np.abs(local_z) <= box.h / 2.0)
        )
        inside_any |= inside
        class_ids[inside] = box.class_id
    return inside_any, class_ids


def rasterize_box_masks(
    boxes: List[TrackletBox],
    calib: Dict[str, np.ndarray],
    output_size: Tuple[int, int] = (128, 512),
) -> Tuple[np.ndarray, np.ndarray]:
    out_h, out_w = output_size
    mask_img = Image.new("F", (out_w, out_h), 0.0)
    class_img = Image.new("F", (out_w, out_h), 0.0)
    mask_draw = ImageDraw.Draw(mask_img)
    class_draw = ImageDraw.Draw(class_img)

    for box in boxes:
        corners = box_corners_velo(box)
        uv, _, valid = project_velo_to_image(corners, calib, output_size)
        if not np.any(valid):
            continue
        valid_uv = uv[valid]
        x0 = max(0, int(np.floor(valid_uv[:, 0].min())))
        y0 = max(0, int(np.floor(valid_uv[:, 1].min())))
        x1 = min(out_w - 1, int(np.ceil(valid_uv[:, 0].max())))
        y1 = min(out_h - 1, int(np.ceil(valid_uv[:, 1].max())))
        if x1 <= x0 or y1 <= y0:
            continue
        mask_draw.rectangle([x0, y0, x1, y1], fill=1.0)
        class_draw.rectangle([x0, y0, x1, y1], fill=float(box.class_id) / float(len(DYNAMIC_CLASS_TO_ID)))

    return np.asarray(mask_img, dtype=np.float32), np.asarray(class_img, dtype=np.float32)


def project_dynamic_boxes(
    boxes: List[TrackletBox],
    calib: Dict[str, np.ndarray],
    output_size: Tuple[int, int] = (128, 512),
    max_boxes: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    out_h, out_w = output_size
    projected = np.zeros((max_boxes, 5), dtype=np.float32)
    valid_mask = np.zeros((max_boxes,), dtype=np.float32)
    out_idx = 0
    for box in boxes:
        if out_idx >= max_boxes:
            break
        corners = box_corners_velo(box)
        uv, _, valid = project_velo_to_image(corners, calib, output_size)
        if not np.any(valid):
            continue
        valid_uv = uv[valid]
        x0 = max(0, int(np.floor(valid_uv[:, 0].min())))
        y0 = max(0, int(np.floor(valid_uv[:, 1].min())))
        x1 = min(out_w - 1, int(np.ceil(valid_uv[:, 0].max())))
        y1 = min(out_h - 1, int(np.ceil(valid_uv[:, 1].max())))
        if x1 <= x0 or y1 <= y0:
            continue
        projected[out_idx] = np.asarray([x0, y0, x1, y1, int(box.class_id)], dtype=np.float32)
        valid_mask[out_idx] = 1.0
        out_idx += 1
    return projected, valid_mask


def _rasterize_points(
    uv: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    output_size: Tuple[int, int],
    max_depth: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    out_h, out_w = output_size
    point_mask = np.zeros((out_h, out_w), dtype=np.float32)
    depth_map = np.zeros((out_h, out_w), dtype=np.float32)
    nearest = np.full((out_h, out_w), np.inf, dtype=np.float32)

    valid_indices = np.nonzero(valid)[0]
    for idx in valid_indices:
        x = int(np.clip(round(float(uv[idx, 0])), 0, out_w - 1))
        y = int(np.clip(round(float(uv[idx, 1])), 0, out_h - 1))
        d = float(depth[idx])
        if d < nearest[y, x]:
            nearest[y, x] = d
            point_mask[y, x] = 1.0
            depth_map[y, x] = min(d, max_depth) / max_depth
    return point_mask, depth_map, int(valid_indices.size)


def generate_lidar_condition(
    velodyne_path: str,
    boxes: List[TrackletBox],
    calib: Dict[str, np.ndarray],
    output_size: Tuple[int, int] = (128, 512),
    mode: str = "dynamic_full",
    max_depth: float = 80.0,
    max_dynamic_boxes: int = 32,
) -> Dict[str, np.ndarray]:
    mode = mode.lower()
    if mode not in CONDITION_CHANNELS_BY_MODE:
        raise ValueError(f"Unsupported lidar condition mode: {mode}")

    out_h, out_w = output_size
    cond = np.zeros((lidar_condition_channels(mode), out_h, out_w), dtype=np.float32)
    points = load_velodyne_points(velodyne_path)
    points_xyz = points[:, :3] if points.size else np.zeros((0, 3), dtype=np.float32)
    dynamic_class_hist = np.zeros((max(DYNAMIC_CLASS_TO_ID.values()) + 1,), dtype=np.float32)
    for box in boxes:
        dynamic_class_hist[int(box.class_id)] += 1.0

    box_mask, class_map = rasterize_box_masks(boxes, calib, output_size)
    dynamic_boxes, dynamic_box_valid = project_dynamic_boxes(boxes, calib, output_size, max_dynamic_boxes)
    dynamic_mask = box_mask[None, :, :].astype(np.float32)

    all_uv, all_depth, all_valid = project_velo_to_image(points_xyz, calib, output_size)
    valid_point_mask, _, valid_projected_count = _rasterize_points(all_uv, all_depth, all_valid, output_size, max_depth)

    dynamic_point_count = 0
    if mode != "none":
        if mode in {"bbox_dynamic", "dynamic_full"}:
            cond[0] = box_mask
            cond[3] = class_map

        if mode == "raw_lidar":
            point_mask, depth_map, dynamic_point_count = _rasterize_points(
                all_uv, all_depth, all_valid, output_size, max_depth
            )
            cond[1] = point_mask
            cond[2] = depth_map
        elif mode in {"dynamic_points", "dynamic_full"}:
            inside, _ = points_in_boxes(points_xyz, boxes)
            dyn_uv, dyn_depth, dyn_valid = project_velo_to_image(points_xyz[inside], calib, output_size)
            point_mask, depth_map, dynamic_point_count = _rasterize_points(
                dyn_uv, dyn_depth, dyn_valid, output_size, max_depth
            )
            cond[1] = point_mask
            cond[2] = depth_map

    return {
        "lidar_cond": cond,
        "dynamic_mask": dynamic_mask,
        "dynamic_boxes": dynamic_boxes,
        "dynamic_box_valid": dynamic_box_valid,
        "lidar_valid_mask": valid_point_mask[None, :, :].astype(np.float32),
        "dynamic_class_hist": dynamic_class_hist,
        "num_projected_lidar_points": np.asarray(valid_projected_count, dtype=np.int64),
        "num_projected_dynamic_points": np.asarray(dynamic_point_count, dtype=np.int64),
        "num_dynamic_boxes": np.asarray(len(boxes), dtype=np.int64),
    }


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: str) -> List[dict]:
    with Path(path).open("r") as f:
        return [json.loads(line) for line in f if line.strip()]
