import argparse
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PATH_KEYS = (
    "image_02_path",
    "satellite_path",
    "velodyne_path",
    "oxts_path",
    "calib_cam_to_cam_path",
    "calib_velo_to_cam_path",
)
OPTIONAL_PATH_KEYS = ("tracklet_xml_path",)
CALIB_OPTIONAL_FILES = ("calib_imu_to_velo.txt",)


def parse_args():
    parser = argparse.ArgumentParser(description="Copy KITTI raw files referenced by a manifest to a local cache.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root", default="/media/shizhm/Lenovo/KITTI_RAW")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0, help="Skip this many manifest rows before caching.")
    parser.add_argument("--no-tracklets", action="store_true", help="Blank tracklet_xml_path in cached records.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _relative_to_source(path: Path, source_root: Path) -> Path:
    try:
        return path.resolve().relative_to(source_root.resolve())
    except ValueError:
        return Path(path.name)


def _copy_file(src: Path, dst: Path, overwrite: bool) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and not overwrite and dst.stat().st_size == src.stat().st_size:
        return True
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    return True


def _cache_path(src_value: str, source_root: Path, cache_root: Path) -> Path:
    src = Path(src_value)
    return cache_root / _relative_to_source(src, source_root)


def _cache_record(record: dict, source_root: Path, cache_root: Path, overwrite: bool, no_tracklets: bool):
    cached = dict(record)
    copied = []
    missing = []

    for key in PATH_KEYS:
        src_value = record.get(key, "")
        if not src_value:
            missing.append(key)
            continue
        src = Path(src_value)
        dst = _cache_path(src_value, source_root, cache_root)
        if _copy_file(src, dst, overwrite):
            cached[key] = str(dst)
            copied.append(key)
        else:
            missing.append(key)

    calib_dir = record.get("calib_dir", "")
    if calib_dir:
        src_calib_dir = Path(calib_dir)
        dst_calib_dir = _cache_path(calib_dir, source_root, cache_root)
        cached["calib_dir"] = str(dst_calib_dir)
        for file_name in CALIB_OPTIONAL_FILES:
            src = src_calib_dir / file_name
            if src.exists():
                _copy_file(src, dst_calib_dir / file_name, overwrite)

    if no_tracklets:
        cached["tracklet_xml_path"] = ""
        cached["has_dynamic_xml"] = False
    else:
        for key in OPTIONAL_PATH_KEYS:
            src_value = record.get(key, "")
            if not src_value:
                continue
            src = Path(src_value)
            dst = _cache_path(src_value, source_root, cache_root)
            if _copy_file(src, dst, overwrite):
                cached[key] = str(dst)

    return cached, copied, missing


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    cache_root = Path(args.cache_root)
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    cached_count = 0
    skipped = 0
    with out_manifest.open("w") as out:
        for row_index, record in enumerate(_read_jsonl(Path(args.manifest))):
            if row_index < args.start:
                continue
            if args.limit and total >= args.limit:
                break
            total += 1
            try:
                cached, copied, missing = _cache_record(
                    record,
                    source_root=source_root,
                    cache_root=cache_root,
                    overwrite=bool(args.overwrite),
                    no_tracklets=bool(args.no_tracklets),
                )
            except Exception as exc:
                skipped += 1
                print(json.dumps({"skipped": record.get("sample_id", ""), "error": str(exc)}, ensure_ascii=False))
                continue
            if missing:
                skipped += 1
                print(json.dumps({"skipped": record.get("sample_id", ""), "missing": missing}, ensure_ascii=False))
                continue
            out.write(json.dumps(cached, ensure_ascii=False, sort_keys=True) + "\n")
            cached_count += 1
            if cached_count == 1 or cached_count % 100 == 0:
                print(json.dumps({"cached": cached_count, "seen": total, "last_sample": record.get("sample_id", "")}))

    print(
        json.dumps(
            {
                "complete": True,
                "manifest": str(args.manifest),
                "out_manifest": str(out_manifest),
                "cache_root": str(cache_root),
                "seen": total,
                "cached": cached_count,
                "skipped": skipped,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
