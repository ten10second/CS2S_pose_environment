import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.kitti_raw_lidar_utils import read_jsonl  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build DINOv2 image semantic feature cache for KITTI raw frames.")
    parser.add_argument("--manifest", default="dataset/kitti_raw_sat_lidar/train_manifest.jsonl")
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--path-rewrite",
        default="",
        help="Optional OLD=NEW replacement for manifest paths, e.g. /media/a=/media/b.",
    )
    parser.add_argument(
        "--kitti-root",
        default="",
        help="Optional KITTI_RAW root used to rebase machine-specific manifest paths.",
    )
    parser.add_argument("--model", default="dinov2_vits14")
    parser.add_argument("--weights", default="", help="Optional local DINOv2 backbone checkpoint.")
    parser.add_argument(
        "--dinov2-root",
        default=str(REPO_ROOT / "third_party" / "dinov2"),
        help="Local DINOv2 repository used by torch.hub; defaults to the vendored third_party copy.",
    )
    parser.add_argument("--input-height", type=int, default=224)
    parser.add_argument("--input-width", type=int, default=896)
    parser.add_argument("--token-height", type=int, default=8)
    parser.add_argument("--token-width", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--preview-every", type=int, default=0)
    parser.add_argument("--preview-root", default="")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def apply_path_rewrite(path: str, rule: str) -> str:
    if not rule:
        return path
    if "=" not in rule:
        raise ValueError("--path-rewrite must be OLD=NEW")
    old, new = rule.split("=", 1)
    return path.replace(old, new, 1)


def rewrite_path(path: str, args) -> str:
    value = apply_path_rewrite(path, args.path_rewrite)
    if args.kitti_root:
        marker = "/KITTI_RAW/"
        if marker in value:
            value = str(Path(args.kitti_root) / value.split(marker, 1)[1])
    return value


def safe_sample_id(sample_id: str) -> str:
    return str(sample_id).replace("/", "__")


def load_dinov2(model_name: str, device: torch.device, dinov2_root: str, weights: str = ""):
    repo_root = Path(dinov2_root).expanduser().resolve()
    if not (repo_root / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv2 hub repository not found: {repo_root}")
    load_kwargs = {"pretrained": True, "source": "local"}
    if weights:
        weights_path = Path(weights).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"DINOv2 weights not found: {weights_path}")
        load_kwargs["weights"] = str(weights_path)
    model = torch.hub.load(str(repo_root), model_name, **load_kwargs)
    model.eval().to(device)
    return model


def atomic_savez(path: Path, **arrays) -> None:
    temp_path = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    temp_path.unlink(missing_ok=True)
    try:
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def extract_patch_tokens(model, image_tensor: torch.Tensor):
    with torch.no_grad():
        if hasattr(model, "forward_features"):
            features = model.forward_features(image_tensor)
            if isinstance(features, dict):
                if "x_norm_patchtokens" in features:
                    return features["x_norm_patchtokens"]
                if "patch_tokens" in features:
                    return features["patch_tokens"]
        out = model(image_tensor)
        if out.ndim == 3:
            return out
    raise RuntimeError("Could not extract patch tokens from DINO model output.")


def pca_rgb(feat_chw: np.ndarray) -> Image.Image:
    c, h, w = feat_chw.shape
    x = feat_chw.reshape(c, h * w).T.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    rgb = x @ vh[:3].T
    rgb = rgb.reshape(h, w, 3)
    lo = np.percentile(rgb, 1, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 99, axis=(0, 1), keepdims=True)
    rgb = (rgb - lo) / np.maximum(hi - lo, 1e-6)
    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb).resize((w * 16, h * 16), Image.NEAREST)


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    preview_root = Path(args.preview_root) if args.preview_root else out_root / "_preview"
    if int(args.preview_every) > 0:
        preview_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_dinov2(args.model, device, args.dinov2_root, args.weights)
    patch_size = int(getattr(model, "patch_size", 14))
    input_h = int(args.input_height)
    input_w = int(args.input_width)
    if input_h % patch_size != 0 or input_w % patch_size != 0:
        raise ValueError(f"input size must be divisible by patch size {patch_size}, got {input_h}x{input_w}")
    grid_h = input_h // patch_size
    grid_w = input_w // patch_size
    transform = transforms.Compose(
        [
            transforms.Resize((input_h, input_w), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    records = list(enumerate(read_jsonl(args.manifest)))[args.shard_index :: args.num_shards]
    if int(args.limit) > 0:
        records = records[: int(args.limit)]
    written = 0
    skipped = 0
    failed = 0
    for idx, record in records:
        sample_id = record["sample_id"]
        out_path = out_root / f"{safe_sample_id(sample_id)}.npz"
        if args.skip_existing and out_path.is_file():
            skipped += 1
            continue
        image_path = Path(rewrite_path(record["image_02_path"], args))
        try:
            with Image.open(image_path) as img:
                tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
            patch_tokens = extract_patch_tokens(model, tensor)
            feat = patch_tokens.reshape(1, grid_h, grid_w, -1).permute(0, 3, 1, 2)
            feat = F.adaptive_avg_pool2d(feat.float(), (int(args.token_height), int(args.token_width)))
            feat_chw = feat[0].detach().cpu().numpy().astype(np.float32)
            atomic_savez(
                out_path,
                dino_feat=feat_chw.transpose(1, 2, 0),
                image_semantic_mask=np.ones((int(args.token_height), int(args.token_width)), dtype=np.float32),
                sample_id=np.asarray(sample_id),
                model=np.asarray(args.model),
                input_size=np.asarray([input_h, input_w], dtype=np.int32),
                token_size=np.asarray([int(args.token_height), int(args.token_width)], dtype=np.int32),
            )
            if int(args.preview_every) > 0 and written % int(args.preview_every) == 0:
                pca_rgb(feat_chw).save(preview_root / f"{safe_sample_id(sample_id)}_dino_pca.png")
            written += 1
            if written == 1 or written % 100 == 0:
                print(json.dumps({"written": written, "idx": idx, "sample_id": sample_id, "out": str(out_path)}))
        except Exception as exc:
            failed += 1
            print(json.dumps({"failed": failed, "idx": idx, "sample_id": sample_id, "error": str(exc)}))
    summary = {
        "complete": failed == 0,
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "out_root": str(out_root),
    }
    print(json.dumps(summary))
    if failed:
        raise SystemExit(f"DINO cache generation failed for {failed} samples; rerun with --skip-existing after fixing errors.")


if __name__ == "__main__":
    main()
