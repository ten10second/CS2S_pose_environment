"""Assemble and validate paired geometry-history rollout videos.

This tool is intentionally diagnostic. It verifies that a history-enabled
rollout and a history-disabled rollout are a fair 32-frame pair before writing
side-by-side artifacts for visual inspection.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EXPECTED_FRAMES = 32
SELECTED_OFFSETS = (0, 1, 8, 16, 24, 31)
LABEL_HEIGHT = 20
HEADER_HEIGHT = 24
ROW_LABELS = ("GT", "HISTORY OFF", "HISTORY ON")
RECORD_FIELDS = (
    "sample_id",
    "frame_index",
    "sequence_id",
    "is_observed_initial_frame",
    "has_history",
    "history_disabled",
    "history_input_sha256",
    "history_output_sha256",
    "initial_noise_sha256",
    "denoiser_batch_sizes",
    "history_attention_steps",
    "normal_path",
)
SUMMARY_FIELDS = (
    "args",
    "step_noise_sha256",
    "cfg_batch_factor",
    "uncond_cfg",
    "hist_ckpt",
    "base_ckpt",
    "hist_step",
    "block_indices",
    "first_rgb",
    "future_rgb_condition_check_passed",
)
PAIRED_ARG_FIELDS = (
    "ddim_steps",
    "eta",
    "temperature",
    "seed",
    "start_index",
    "num_samples",
)


class ComparisonError(ValueError):
    """Raised when the two rollout directories are not a valid comparison."""


def read_json(path: Path):
    if not path.exists():
        raise ComparisonError(f"missing required file: {path}")
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def safe_id(sample_id: str) -> str:
    return str(sample_id).replace("/", "__")


def ensure_fresh_out_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory must be fresh or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def require_fields(payload: dict, fields: Iterable[str], label: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ComparisonError(f"{label} missing fields: {missing}")


def as_bool(value, label: str) -> bool:
    if not isinstance(value, bool):
        raise ComparisonError(f"{label} must be boolean, got {value!r}")
    return value


def load_records(run_dir: Path, label: str) -> List[dict]:
    records = read_json(run_dir / "records.json")
    if not isinstance(records, list):
        raise ComparisonError(f"{label} records.json must be a list")
    if len(records) != EXPECTED_FRAMES:
        raise ComparisonError(f"{label} must contain exactly {EXPECTED_FRAMES} records, got {len(records)}")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ComparisonError(f"{label} record {index} must be an object")
        require_fields(record, RECORD_FIELDS, f"{label} record {index}")
    return records


def load_summary(run_dir: Path, label: str) -> dict:
    summary = read_json(run_dir / "run_summary.json")
    if not isinstance(summary, dict):
        raise ComparisonError(f"{label} run_summary.json must be an object")
    require_fields(summary, SUMMARY_FIELDS, f"{label} run_summary")
    return summary


def require_summary(summary: dict, label: str, *, expect_disabled: bool) -> None:
    if summary["cfg_batch_factor"] != 2:
        raise ComparisonError(f"{label} cfg_batch_factor must be 2")
    if float(summary["uncond_cfg"]) != 3.0:
        raise ComparisonError(f"{label} uncond_cfg must be 3")
    if int(summary["hist_step"]) <= 0:
        raise ComparisonError(f"{label} hist_step must be positive")
    if summary["first_rgb"] is not True:
        raise ComparisonError(f"{label} first_rgb must be true")
    if summary["future_rgb_condition_check_passed"] is not True:
        raise ComparisonError(f"{label} future_rgb_condition_check_passed must be true")
    args = summary.get("args")
    if not isinstance(args, dict):
        raise ComparisonError(f"{label} args must be an object")
    if bool(args.get("disable_history")) is not expect_disabled:
        raise ComparisonError(f"{label} args.disable_history mismatch")


def comparable_summary_values(summary: dict) -> dict:
    values = {
        "step_noise_sha256": summary["step_noise_sha256"],
        "cfg_batch_factor": summary["cfg_batch_factor"],
        "uncond_cfg": float(summary["uncond_cfg"]),
        "hist_ckpt": summary["hist_ckpt"],
        "base_ckpt": summary["base_ckpt"],
        "hist_step": int(summary["hist_step"]),
        "block_indices": summary["block_indices"],
    }
    args = summary["args"]
    for field in PAIRED_ARG_FIELDS:
        if field in args:
            values[f"args.{field}"] = args[field]
    return values


def require_same_summaries(on_summary: dict, off_summary: dict) -> None:
    on_values = comparable_summary_values(on_summary)
    off_values = comparable_summary_values(off_summary)
    for field in PAIRED_ARG_FIELDS:
        key = f"args.{field}"
        on_present = key in on_values
        off_present = key in off_values
        if on_present != off_present:
            raise ComparisonError(f"run summaries are not paired: {key} present in only one run")
    mismatches = {
        key: (on_values[key], off_values[key])
        for key in on_values
        if on_values[key] != off_values[key]
    }
    if mismatches:
        raise ComparisonError(f"run summaries are not paired: {mismatches}")


def require_contiguous(records: Sequence[dict], label: str) -> None:
    first_sequence = records[0]["sequence_id"]
    first_frame = records[0]["frame_index"]
    if not isinstance(first_frame, int):
        raise ComparisonError(f"{label} frame_index must be integer")
    for index, record in enumerate(records):
        if record["sequence_id"] != first_sequence:
            raise ComparisonError(f"{label} sequence_id changed at offset {index}")
        expected_frame = first_frame + index
        if record["frame_index"] != expected_frame:
            raise ComparisonError(
                f"{label} frame gap at offset {index}: expected {expected_frame}, got {record['frame_index']}"
            )


def require_same_records(on_records: Sequence[dict], off_records: Sequence[dict]) -> None:
    for index, (on_record, off_record) in enumerate(zip(on_records, off_records)):
        for field in ("sample_id", "frame_index", "sequence_id"):
            if on_record[field] != off_record[field]:
                raise ComparisonError(f"ON/OFF {field} mismatch at offset {index}")
        if index == 0:
            if on_record["initial_noise_sha256"] is not None or off_record["initial_noise_sha256"] is not None:
                raise ComparisonError("first RGB bootstrap frame must not have initial_noise_sha256")
        elif on_record["initial_noise_sha256"] != off_record["initial_noise_sha256"]:
            raise ComparisonError(f"initial_noise_sha256 mismatch at offset {index}")
        elif not on_record["initial_noise_sha256"]:
            raise ComparisonError(f"generated frame {index} missing initial_noise_sha256")


def require_record_semantics(
    records: Sequence[dict],
    label: str,
    *,
    expect_disabled: bool,
    block_count: int,
    ddim_steps: int | None,
) -> None:
    for index, record in enumerate(records):
        observed = as_bool(record["is_observed_initial_frame"], f"{label} observed flag {index}")
        has_history = as_bool(record["has_history"], f"{label} has_history {index}")
        disabled = as_bool(record["history_disabled"], f"{label} history_disabled {index}")
        if observed != (index == 0):
            raise ComparisonError(f"{label} observed initial frame flag is wrong at offset {index}")
        if disabled is not expect_disabled:
            raise ComparisonError(f"{label} history_disabled is wrong at offset {index}")
        if index == 0:
            if has_history:
                raise ComparisonError(f"{label} first frame must not have history")
            if record["denoiser_batch_sizes"] or record["history_attention_steps"]:
                raise ComparisonError(f"{label} first RGB frame must not run denoiser/history attention")
            continue

        batches = record["denoiser_batch_sizes"]
        if not isinstance(batches, list) or not batches or any(batch != 2 for batch in batches):
            raise ComparisonError(f"{label} generated offset {index} must use CFG denoiser batch size 2")
        if ddim_steps is not None and len(batches) != ddim_steps:
            raise ComparisonError(
                f"{label} generated offset {index} denoiser call count {len(batches)} != ddim_steps {ddim_steps}"
            )
        trace = record["history_attention_steps"]
        if expect_disabled:
            if has_history:
                raise ComparisonError(f"{label} disabled offset {index} unexpectedly has history")
            if trace:
                raise ComparisonError(f"{label} disabled offset {index} unexpectedly read history")
        else:
            if not has_history:
                raise ComparisonError(f"{label} enabled offset {index} missing history")
            expected_trace = len(batches) * block_count
            if not isinstance(trace, list) or len(trace) != expected_trace:
                raise ComparisonError(
                    f"{label} enabled offset {index} history trace length {len(trace)} != {expected_trace}"
                )


def require_history_chain(on_records: Sequence[dict], off_records: Sequence[dict]) -> None:
    if on_records[0]["history_output_sha256"] != off_records[0]["history_output_sha256"]:
        raise ComparisonError("bootstrap output latent sha256 differs between ON and OFF")
    for index in range(1, len(on_records)):
        expected = on_records[index - 1]["history_output_sha256"]
        if on_records[index]["history_input_sha256"] != expected:
            raise ComparisonError(f"ON history chain broken at offset {index}")


def image_path(run_dir: Path, record: dict, kind: str) -> Path:
    if kind == "normal":
        recorded = Path(record["normal_path"])
        local = run_dir / "images" / "normal" / recorded.name
        candidates = [local, recorded]
    elif kind == "gt":
        candidates = [run_dir / "images" / "gt" / f"{safe_id(record['sample_id'])}.png"]
    else:
        raise ValueError(f"unknown image kind: {kind}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ComparisonError(f"missing {kind} image for {record['sample_id']}: tried {candidates}")


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_array(path: Path) -> np.ndarray:
    return np.asarray(load_rgb(path), dtype=np.uint8)


def require_same_shape(arrays: Sequence[np.ndarray], label: str) -> Tuple[int, int]:
    shapes = {tuple(array.shape) for array in arrays}
    if len(shapes) != 1:
        raise ComparisonError(f"{label} image shape mismatch: {sorted(shapes)}")
    height, width, channels = next(iter(shapes))
    if channels != 3:
        raise ComparisonError(f"{label} images must be RGB")
    return height, width


def require_first_rgb_matches_gt(on_dir: Path, off_dir: Path, on_records: Sequence[dict], off_records: Sequence[dict]) -> None:
    on_first = image_array(image_path(on_dir, on_records[0], "normal"))
    off_first = image_array(image_path(off_dir, off_records[0], "normal"))
    gt_first = image_array(image_path(on_dir, on_records[0], "gt"))
    off_gt_first = image_array(image_path(off_dir, off_records[0], "gt"))
    require_same_shape([on_first, off_first, gt_first, off_gt_first], "first frame")
    if not np.array_equal(gt_first, off_gt_first):
        raise ComparisonError("first GT image differs between ON and OFF runs")
    if not np.array_equal(on_first, gt_first):
        raise ComparisonError("history ON first RGB frame is not pixel-identical to GT")
    if not np.array_equal(off_first, gt_first):
        raise ComparisonError("history OFF first RGB frame is not pixel-identical to GT")


def require_gt_images_match(on_dir: Path, off_dir: Path, on_records: Sequence[dict], off_records: Sequence[dict]) -> None:
    for index, (on_record, off_record) in enumerate(zip(on_records, off_records)):
        on_gt = image_array(image_path(on_dir, on_record, "gt"))
        off_gt = image_array(image_path(off_dir, off_record, "gt"))
        require_same_shape([on_gt, off_gt], f"GT offset {index}")
        if not np.array_equal(on_gt, off_gt):
            raise ComparisonError(f"GT image differs between ON and OFF at offset {index}")


def validate_pair(on_dir: Path, off_dir: Path) -> dict:
    on_records = load_records(on_dir, "ON")
    off_records = load_records(off_dir, "OFF")
    on_summary = load_summary(on_dir, "ON")
    off_summary = load_summary(off_dir, "OFF")

    require_summary(on_summary, "ON", expect_disabled=False)
    require_summary(off_summary, "OFF", expect_disabled=True)
    require_same_summaries(on_summary, off_summary)
    require_contiguous(on_records, "ON")
    require_contiguous(off_records, "OFF")
    require_same_records(on_records, off_records)

    block_count = len(on_summary["block_indices"])
    if block_count <= 0:
        raise ComparisonError("block_indices must be non-empty")
    ddim_steps = on_summary["args"].get("ddim_steps")
    if ddim_steps is not None:
        ddim_steps = int(ddim_steps)
        if ddim_steps <= 0:
            raise ComparisonError("args.ddim_steps must be positive when present")
    require_record_semantics(
        on_records,
        "ON",
        expect_disabled=False,
        block_count=block_count,
        ddim_steps=ddim_steps,
    )
    require_record_semantics(
        off_records,
        "OFF",
        expect_disabled=True,
        block_count=block_count,
        ddim_steps=ddim_steps,
    )
    require_history_chain(on_records, off_records)
    require_first_rgb_matches_gt(on_dir, off_dir, on_records, off_records)
    require_gt_images_match(on_dir, off_dir, on_records, off_records)

    return {
        "on_records": on_records,
        "off_records": off_records,
        "on_summary": on_summary,
        "off_summary": off_summary,
    }


def compute_differences(on_dir: Path, off_dir: Path, on_records: Sequence[dict], off_records: Sequence[dict]) -> dict:
    per_frame = []
    for index, (on_record, off_record) in enumerate(zip(on_records, off_records)):
        on_array = image_array(image_path(on_dir, on_record, "normal")).astype(np.int16)
        off_array = image_array(image_path(off_dir, off_record, "normal")).astype(np.int16)
        require_same_shape([on_array, off_array], f"offset {index}")
        diff = np.abs(on_array - off_array).astype(np.float32)
        per_frame.append(
            {
                "offset": index,
                "sample_id": on_record["sample_id"],
                "frame_index": on_record["frame_index"],
                "mean_abs_rgb_0_255": float(diff.mean()),
                "max_abs_rgb_0_255": int(diff.max()),
            }
        )
    future = per_frame[1:]
    return {
        "on_minus_off_abs": {
            "overall_excluding_first_mean_abs_rgb_0_255": float(np.mean([row["mean_abs_rgb_0_255"] for row in future])),
            "overall_excluding_first_max_abs_rgb_0_255": int(max(row["max_abs_rgb_0_255"] for row in future)),
            "per_frame": per_frame,
        }
    }


def draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fill=(255, 255, 255)) -> None:
    draw.text(xy, text, fill=fill, font=ImageFont.load_default())


def fit_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = int(round(image.height * (width / image.width)))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def comparison_frame(
    gt: Image.Image,
    off: Image.Image,
    on: Image.Image,
    *,
    offset: int,
    sample_id: str,
    frame_index: int,
    width: int = 1024,
) -> Image.Image:
    rows = [fit_width(image, width) for image in (gt, off, on)]
    row_height = rows[0].height
    for row in rows:
        if row.height != row_height:
            raise ComparisonError("comparison images do not share aspect ratio")
    canvas = Image.new("RGB", (width, HEADER_HEIGHT + len(rows) * (LABEL_HEIGHT + row_height)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (8, 6), f"offset={offset:02d} frame={frame_index} sample={sample_id}")
    y = HEADER_HEIGHT
    for label, row in zip(ROW_LABELS, rows):
        draw.rectangle((0, y, width, y + LABEL_HEIGHT), fill=(32, 32, 32))
        draw_text(draw, (8, y + 5), label)
        canvas.paste(row, (0, y + LABEL_HEIGHT))
        y += LABEL_HEIGHT + row_height
    return canvas


def iter_comparison_frames(on_dir: Path, off_dir: Path, on_records: Sequence[dict], off_records: Sequence[dict]):
    for index, (on_record, off_record) in enumerate(zip(on_records, off_records)):
        yield comparison_frame(
            load_rgb(image_path(on_dir, on_record, "gt")),
            load_rgb(image_path(off_dir, off_record, "normal")),
            load_rgb(image_path(on_dir, on_record, "normal")),
            offset=index,
            sample_id=on_record["sample_id"],
            frame_index=on_record["frame_index"],
        )


def choose_h264_encoder(encoder_listing: str):
    names = {parts[1] for line in encoder_listing.splitlines()
             if len(parts := line.split()) >= 2}
    return next((name for name in ('libx264', 'libopenh264') if name in names), None)


def write_video_ffmpeg(frames: Sequence[Image.Image], path: Path, fps: int):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    encoders = subprocess.run([ffmpeg, '-hide_banner', '-encoders'], capture_output=True, text=True, check=True)
    codec = choose_h264_encoder(encoders.stdout)
    if codec is None:
        return None
    width, height = frames[0].size
    cmd = [
        ffmpeg,
        '-hide_banner',
        '-loglevel',
        'error',
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        codec,
        '-b:v',
        '6M',
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    # communicate(input=...) drains stderr and reports encoder failures instead
    # of losing their cause behind BrokenPipeError on an unsupported codec.
    proc = subprocess.run(cmd, input=b''.join(np.asarray(frame, dtype=np.uint8).tobytes() for frame in frames),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {path} with code {proc.returncode}:\n"
            f"{proc.stdout.decode(errors='replace')}\n{proc.stderr.decode(errors='replace')}"
        )
    return codec


def write_video_cv2(frames: Sequence[Image.Image], path: Path, fps: int) -> None:
    import cv2

    width, height = frames[0].size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {path}")
    try:
        for frame in frames:
            rgb = np.asarray(frame, dtype=np.uint8)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_video(frames: Sequence[Image.Image], path: Path, fps: int) -> str:
    if not frames:
        raise ValueError("cannot write empty video")
    codec = write_video_ffmpeg(frames, path, fps)
    if codec:
        return f"ffmpeg/{codec}"
    write_video_cv2(frames, path, fps)
    return "opencv/mp4v"


def make_selected_montage(
    on_dir: Path,
    off_dir: Path,
    on_records: Sequence[dict],
    off_records: Sequence[dict],
    out_path: Path,
) -> None:
    cells = []
    for offset in SELECTED_OFFSETS:
        cells.append(
            comparison_frame(
                load_rgb(image_path(on_dir, on_records[offset], "gt")),
                load_rgb(image_path(off_dir, off_records[offset], "normal")),
                load_rgb(image_path(on_dir, on_records[offset], "normal")),
                offset=offset,
                sample_id=on_records[offset]["sample_id"],
                frame_index=on_records[offset]["frame_index"],
                width=512,
            )
        )
    make_grid(cells, columns=len(SELECTED_OFFSETS), out_path=out_path)


def signed_difference_image(on_image: Image.Image, off_image: Image.Image) -> Image.Image:
    on_array = np.asarray(on_image.convert("RGB"), dtype=np.int16)
    off_array = np.asarray(off_image.convert("RGB"), dtype=np.int16)
    require_same_shape([on_array, off_array], "difference montage")
    signed = np.clip((on_array - off_array) * 8 + 128, 0, 255).astype(np.uint8)
    return Image.fromarray(signed, mode="RGB")


def make_difference_montage(
    on_dir: Path,
    off_dir: Path,
    on_records: Sequence[dict],
    off_records: Sequence[dict],
    out_path: Path,
) -> None:
    cells = []
    for offset in SELECTED_OFFSETS:
        diff = signed_difference_image(
            load_rgb(image_path(on_dir, on_records[offset], "normal")),
            load_rgb(image_path(off_dir, off_records[offset], "normal")),
        )
        diff = fit_width(diff, 512)
        cell = Image.new("RGB", (512, HEADER_HEIGHT + LABEL_HEIGHT + diff.height), (18, 18, 18))
        draw = ImageDraw.Draw(cell)
        draw_text(draw, (8, 6), f"offset={offset:02d} frame={on_records[offset]['frame_index']}")
        draw.rectangle((0, HEADER_HEIGHT, 512, HEADER_HEIGHT + LABEL_HEIGHT), fill=(32, 32, 32))
        draw_text(draw, (8, HEADER_HEIGHT + 5), "ON-OFF x8, zero=gray")
        cell.paste(diff, (0, HEADER_HEIGHT + LABEL_HEIGHT))
        cells.append(cell)
    make_grid(cells, columns=len(SELECTED_OFFSETS), out_path=out_path)


def make_grid(cells: Sequence[Image.Image], *, columns: int, out_path: Path) -> None:
    if not cells:
        raise ValueError("cannot make montage with no cells")
    width = max(cell.width for cell in cells)
    height = max(cell.height for cell in cells)
    rows = int(np.ceil(len(cells) / columns))
    canvas = Image.new("RGB", (columns * width, rows * height), (0, 0, 0))
    for index, cell in enumerate(cells):
        x = (index % columns) * width
        y = (index // columns) * height
        canvas.paste(cell, (x, y))
    canvas.save(out_path)


def assemble_comparison(on_dir: Path, off_dir: Path, out_dir: Path) -> dict:
    ensure_fresh_out_dir(out_dir)
    validated = validate_pair(on_dir, off_dir)
    on_records = validated["on_records"]
    off_records = validated["off_records"]
    frames = list(iter_comparison_frames(on_dir, off_dir, on_records, off_records))
    codec = write_video(frames, out_dir / "comparisons.mp4", fps=10)
    slow_codec = write_video(frames, out_dir / "comparisons_slow.mp4", fps=5)
    make_selected_montage(on_dir, off_dir, on_records, off_records, out_dir / "montage_selected.png")
    make_difference_montage(on_dir, off_dir, on_records, off_records, out_dir / "difference_montage.png")

    verification = {
        "passed": True,
        "num_frames": len(on_records),
        "sequence_id": on_records[0]["sequence_id"],
        "first_frame_index": on_records[0]["frame_index"],
        "last_frame_index": on_records[-1]["frame_index"],
        "sample_ids": [record["sample_id"] for record in on_records],
        "cfg_batch_factor": validated["on_summary"]["cfg_batch_factor"],
        "uncond_cfg": validated["on_summary"]["uncond_cfg"],
        "hist_step": validated["on_summary"]["hist_step"],
        "block_indices": validated["on_summary"]["block_indices"],
        "step_noise_sha256": validated["on_summary"]["step_noise_sha256"],
        "video_codec": codec,
        "slow_video_codec": slow_codec,
        "artifacts": {
            "comparison_video": str(out_dir / "comparisons.mp4"),
            "slow_comparison_video": str(out_dir / "comparisons_slow.mp4"),
            "selected_montage": str(out_dir / "montage_selected.png"),
            "difference_montage": str(out_dir / "difference_montage.png"),
        },
    }
    verification.update(compute_differences(on_dir, off_dir, on_records, off_records))
    write_json(out_dir / "verification.json", verification)
    return verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on-dir", type=Path, required=True, help="History-enabled rollout directory.")
    parser.add_argument("--off-dir", type=Path, required=True, help="History-disabled rollout directory.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Fresh output directory for videos and diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verification = assemble_comparison(args.on_dir, args.off_dir, args.out_dir)
    print(json.dumps(verification, sort_keys=True))


if __name__ == "__main__":
    main()
