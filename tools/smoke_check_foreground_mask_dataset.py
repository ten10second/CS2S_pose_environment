import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-check KITTI foreground mask cache loading.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--foreground-mask-root", required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = SatLidarRawDataset(args.manifest, foreground_mask_root=args.foreground_mask_root)
    for idx in range(min(args.num_samples, len(dataset))):
        item = dataset[idx]
        print(
            idx,
            item["sample_id"],
            float(item["foreground_mask_available"].item()),
            float(item["foreground_mask"].mean().item()),
            tuple(item["foreground_mask"].shape),
        )


if __name__ == "__main__":
    main()
