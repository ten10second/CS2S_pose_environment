#!/usr/bin/env python3
"""Select fixed consecutive KITTI test clips for qualitative monitoring."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Record = Dict[str, object]
Clip = List[Record]


def read_jsonl(path: Path) -> List[Record]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id(record: Record) -> str:
    value = record.get("sample_id")
    if value is not None:
        return str(value)
    date, drive, frame_index = record_key(record)
    return f"{date}/{drive}/{frame_index:010d}"


def record_key(record: Record) -> Tuple[str, str, int]:
    try:
        return str(record["date"]), str(record["drive"]), int(record["frame_index"])
    except KeyError as exc:
        raise ValueError(f"Record is missing required field: {exc.args[0]}") from exc


def drive_key(record: Record) -> Tuple[str, str]:
    date, drive, _ = record_key(record)
    return date, drive


def group_by_drive(records: Sequence[Record]) -> Dict[Tuple[str, str], List[Record]]:
    grouped: Dict[Tuple[str, str], List[Record]] = defaultdict(list)
    for record in records:
        grouped[drive_key(record)].append(record)
    for drive_records in grouped.values():
        drive_records.sort(key=lambda item: record_key(item)[2])
    return dict(grouped)


def consecutive_runs(records: Sequence[Record]) -> Iterable[List[Record]]:
    current: List[Record] = []
    previous_frame = None
    for record in records:
        frame_index = record_key(record)[2]
        if previous_frame is None or frame_index == previous_frame + 1:
            current.append(record)
        else:
            if current:
                yield current
            current = [record]
        previous_frame = frame_index
    if current:
        yield current


def select_middle_segment(run: Sequence[Record], frames_per_clip: int) -> Clip:
    start = (len(run) - frames_per_clip) // 2
    return list(run[start : start + frames_per_clip])


def select_test_clips(
    test_records: Sequence[Record],
    train_records: Sequence[Record],
    clips: int = 2,
    frames_per_clip: int = 16,
) -> List[Clip]:
    if clips <= 0:
        raise ValueError("--clips must be positive")
    if frames_per_clip <= 0:
        raise ValueError("--frames-per-clip must be positive")

    validate_test_records(test_records)
    train_sample_ids = {sample_id(record) for record in train_records}
    train_drives = {drive_key(record) for record in train_records}

    selected: List[Clip] = []
    records_by_drive = group_by_drive(test_records)
    for key in sorted(records_by_drive):
        if key in train_drives:
            continue
        for run in consecutive_runs(records_by_drive[key]):
            if len(run) < frames_per_clip:
                continue
            candidate = select_middle_segment(run, frames_per_clip)
            if any(sample_id(record) in train_sample_ids for record in candidate):
                continue
            selected.append(candidate)
            break
        if len(selected) >= clips:
            break

    if len(selected) < clips:
        raise RuntimeError(
            f"Found {len(selected)} eligible consecutive test clips; "
            f"need {clips} clips x {frames_per_clip} frames."
        )
    validate_selected_clips(selected, train_records, expected_clips=clips, frames_per_clip=frames_per_clip)
    return selected


def validate_test_records(records: Sequence[Record]) -> None:
    for record in records:
        split = str(record.get("split", ""))
        if split != "test2":
            raise ValueError(f"Expected split=test2 for {sample_id(record)}, got {split!r}")


def validate_selected_clips(
    clips: Sequence[Clip],
    train_records: Sequence[Record],
    expected_clips: int,
    frames_per_clip: int,
) -> None:
    if len(clips) != expected_clips:
        raise ValueError(f"Expected {expected_clips} clips, got {len(clips)}")

    train_sample_ids = {sample_id(record) for record in train_records}
    train_drives = {drive_key(record) for record in train_records}
    seen_sample_ids = set()
    for clip_index, clip in enumerate(clips):
        if len(clip) != frames_per_clip:
            raise ValueError(f"Clip {clip_index} has {len(clip)} frames; expected {frames_per_clip}")
        validate_test_records(clip)
        clip_drive = drive_key(clip[0])
        if clip_drive in train_drives:
            raise ValueError(f"Clip {clip_index} drive overlaps train split: {clip_drive}")
        previous_frame = None
        for record in clip:
            if drive_key(record) != clip_drive:
                raise ValueError(f"Clip {clip_index} mixes drives")
            sid = sample_id(record)
            if sid in train_sample_ids:
                raise ValueError(f"Clip {clip_index} sample overlaps train split: {sid}")
            if sid in seen_sample_ids:
                raise ValueError(f"Repeated selected sample: {sid}")
            seen_sample_ids.add(sid)
            frame_index = record_key(record)[2]
            if previous_frame is not None and frame_index != previous_frame + 1:
                raise ValueError(
                    f"Clip {clip_index} is not strictly consecutive at {sid}: "
                    f"{previous_frame} -> {frame_index}"
                )
            previous_frame = frame_index


def records_by_sample_id(records: Sequence[Record], label: str) -> Dict[str, Record]:
    by_id = {}
    for record in records:
        sid = sample_id(record)
        if sid in by_id:
            raise ValueError(f"{label} contains repeated sample_id: {sid}")
        by_id[sid] = record
    return by_id


def validate_selected_records_match_test_manifest(
    selected_records: Sequence[Record],
    test_records: Sequence[Record],
) -> None:
    validate_test_records(test_records)
    test_by_id = records_by_sample_id(test_records, "test manifest")
    for record in selected_records:
        sid = sample_id(record)
        authoritative = test_by_id.get(sid)
        if authoritative is None:
            raise ValueError(f"Selected sample is not present in authoritative test manifest: {sid}")
        if record != authoritative:
            raise ValueError(f"Selected sample does not match authoritative test manifest: {sid}")


def flatten_clips(clips: Sequence[Clip]) -> List[Record]:
    return [record for clip in clips for record in clip]


def chunks(records: Sequence[Record], size: int) -> List[Clip]:
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    if len(records) % size != 0:
        raise ValueError(f"Flat manifest has {len(records)} records, not a multiple of {size}")
    return [list(records[index : index + size]) for index in range(0, len(records), size)]


def validate_fixed_test_sequence_manifest(
    selected_manifest: Path,
    train_manifest: Path,
    test_manifest: Path,
    report_json: Path = None,
    expected_clips: int = 2,
    frames_per_clip: int = 16,
) -> List[Clip]:
    """Load and validate the fixed test-sequence manifest before qualitative inference."""

    selected_manifest = Path(selected_manifest)
    train_manifest = Path(train_manifest)
    test_manifest = Path(test_manifest)
    selected_records = read_jsonl(selected_manifest)
    train_records = read_jsonl(train_manifest)
    test_records = read_jsonl(test_manifest)
    validate_selected_records_match_test_manifest(selected_records, test_records)
    clips = chunks(selected_records, int(frames_per_clip))
    validate_selected_clips(
        clips,
        train_records,
        expected_clips=int(expected_clips),
        frames_per_clip=int(frames_per_clip),
    )

    if report_json is not None:
        report_path = Path(report_json)
        report = json.loads(report_path.read_text())
        selected_ids = [sample_id(record) for record in selected_records]
        if report.get("test_manifest_sha256") != sha256_file(test_manifest):
            raise ValueError("Report test_manifest_sha256 does not match authoritative test manifest")
        if report.get("train_manifest_sha256") != sha256_file(train_manifest):
            raise ValueError("Report train_manifest_sha256 does not match train manifest")
        if report.get("num_clips") != expected_clips:
            raise ValueError(f"Report num_clips mismatch: {report.get('num_clips')} != {expected_clips}")
        if report.get("frames_per_clip") != frames_per_clip:
            raise ValueError(
                f"Report frames_per_clip mismatch: {report.get('frames_per_clip')} != {frames_per_clip}"
            )
        if report.get("selected_sample_ids") != selected_ids:
            raise ValueError("Report selected_sample_ids do not match fixed manifest order")
        if report.get("selected_drive_train_overlap"):
            raise ValueError("Report contains selected_drive_train_overlap")
        if report.get("selected_sample_train_overlap"):
            raise ValueError("Report contains selected_sample_train_overlap")
    return clips


def build_report(
    clips: Sequence[Clip],
    train_records: Sequence[Record],
    test_manifest: Path,
    train_manifest: Path,
    output_manifest: Path,
) -> Dict[str, object]:
    selected_records = flatten_clips(clips)
    selected_samples = [sample_id(record) for record in selected_records]
    train_samples = {sample_id(record) for record in train_records}
    selected_drives = {"/".join(drive_key(record)) for record in selected_records}
    train_drives = {"/".join(drive_key(record)) for record in train_records}
    return {
        "selection_policy": "first_sorted_test2_drives_with_middle_segment_of_first_strictly_consecutive_run",
        "test_manifest": str(test_manifest),
        "test_manifest_sha256": sha256_file(test_manifest),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": sha256_file(train_manifest),
        "output_manifest": str(output_manifest),
        "num_clips": len(clips),
        "frames_per_clip": len(clips[0]) if clips else 0,
        "num_selected_frames": len(selected_records),
        "selected_sample_ids": selected_samples,
        "selected_drive_train_overlap": sorted(selected_drives & train_drives),
        "selected_sample_train_overlap": sorted(set(selected_samples) & train_samples),
        "clips": [clip_report(index, clip) for index, clip in enumerate(clips)],
    }


def clip_report(index: int, clip: Clip) -> Dict[str, object]:
    frames = [record_key(record)[2] for record in clip]
    date, drive = drive_key(clip[0])
    return {
        "clip_index": index,
        "date": date,
        "drive": drive,
        "start_frame_index": frames[0],
        "end_frame_index": frames[-1],
        "num_frames": len(clip),
        "sample_ids": [sample_id(record) for record in clip],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select fixed held-out consecutive KITTI test2 clips for qualitative visualization."
    )
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--report-json", default="")
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_manifest = Path(args.test_manifest)
    train_manifest = Path(args.train_manifest)
    output_manifest = Path(args.output_manifest)
    report_path = Path(args.report_json) if args.report_json else output_manifest.with_suffix(".report.json")

    test_records = read_jsonl(test_manifest)
    train_records = read_jsonl(train_manifest)
    clips = select_test_clips(
        test_records,
        train_records,
        clips=int(args.clips),
        frames_per_clip=int(args.frames_per_clip),
    )
    write_jsonl(output_manifest, flatten_clips(clips))
    report = build_report(clips, train_records, test_manifest, train_manifest, output_manifest)
    if report["selected_drive_train_overlap"] or report["selected_sample_train_overlap"]:
        raise RuntimeError("Selected clips overlap train split; refusing to write report.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
