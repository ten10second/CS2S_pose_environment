"""Reformat saved samples on CPU, including outputs from an already-running trainer."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset
from tools.generate_kitti_raea_samples import make_panel, safe_sample_id


LAYOUT = "satellite-lidar_rgb-gt-cfg-v1"


def reformat_step(step_dir, dataset, records_by_id, guidance_scale):
    records_path = step_dir / "records.json"
    source = records_path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    marker_path = step_dir / "presentation_records.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if marker.get("layout") == LAYOUT and marker.get("source_records_sha256") == digest:
            if all(Path(row["panel_path"]).is_file() for row in marker["records"]):
                return 0
    output_records = []
    for item in json.loads(source):
        sample_id = item["sample_id"]
        if "trained:normal" not in item:
            continue
        sat_path = step_dir / "images" / "satellite" / f"{safe_sample_id(sample_id)}.png"
        if not sat_path.exists():
            record = records_by_id[sample_id]
            calib = dataset._calib(record["calib_dir"])
            with Image.open(record["satellite_path"]) as source_image:
                # Reproduce the actual camera-aligned satellite input, including
                # its rotation/crop, without loading LiDAR features or any model.
                satellite = dataset._satellite_image(source_image, record["oxts_path"], calib)
            sat_path.parent.mkdir(parents=True, exist_ok=True)
            satellite.save(sat_path)
            satellite.close()
        overlay = next(
            item[key] for key in (
                "LiDAR projection on RGB", "LiDAR depth on GT", "LiDAR depth (near red, far blue)"
            ) if key in item
        )
        image_paths = {
            "Satellite input": sat_path,
            "LiDAR projection on RGB": overlay,
            "GT": item["GT"],
            "trained:normal": item["trained:normal"],
        }
        panel_path = step_dir / "panels" / f"{safe_sample_id(sample_id)}.png"
        backup = step_dir / "panels_previous_layout" / panel_path.name
        if panel_path.exists() and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(panel_path, backup)
        make_panel(step_dir, sample_id, image_paths, guidance_scale=guidance_scale)
        output_records.append({
            **item, **{key: str(value) for key, value in image_paths.items()},
            "panel_path": str(panel_path), "guidance_scale": guidance_scale,
        })
    marker = {"layout": LAYOUT, "source_records_sha256": digest, "records": output_records}
    temp = marker_path.with_suffix(".tmp")
    temp.write_text(json.dumps(marker, indent=2))
    os.replace(temp, marker_path)
    return len(output_records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--kitti-root", required=True)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    dataset = SatLidarRawDataset(
        manifest=args.manifest, kitti_root=args.kitti_root,
        include_range_image=False, include_tracklets=False,
    )
    records_by_id = {record["sample_id"]: record for record in dataset.records}
    while True:
        for path in sorted((Path(args.run_dir) / "samples").glob("step_*/records.json")):
            try:
                count = reformat_step(path.parent, dataset, records_by_id, args.guidance_scale)
                if count:
                    print(json.dumps({"step": path.parent.name, "panels": count, "layout": LAYOUT}), flush=True)
            except (OSError, ValueError, KeyError, StopIteration) as exc:
                if not args.watch:
                    raise
                print(json.dumps({"step": path.parent.name, "retry": str(exc)}), flush=True)
        if not args.watch:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
