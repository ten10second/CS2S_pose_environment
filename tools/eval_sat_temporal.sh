#!/usr/bin/env bash
# One-shot evaluation of a sat_temporal_net run: flicker ratio + attention stats.
# Usage: bash tools/eval_sat_temporal.sh <run_dir>
set -u
RUN=$1
PY=/home/shizhm/miniconda3/envs/ControlS2S/bin/python
REPO=/mnt/shizhm/CS2S_pose_environment_sat-lidar-ray-posterior-evidence

echo "=== frames: $(ls $RUN/images/normal 2>/dev/null | wc -l)/300 ==="

$PY - <<EOF
import json
import numpy as np
from pathlib import Path
recs = json.load(open("$RUN/records.json"))
confs = [r["noise"].get("sat_attn_max_mean") for r in recs
         if r["noise"].get("sat_attn_max_mean") is not None]
print(f"=== sat attention focus (frames with history) ===")
if confs:
    print(f"  mean={np.mean(confs):.4f}  p50={np.percentile(confs,50):.4f}  "
          f"p95={np.percentile(confs,95):.4f}  max={max(confs):.4f}  (uniform baseline 0.111)")
else:
    print("  no stats recorded")

EOF

echo "=== flicker ratio (gen/GT consecutive-frame LPIPS) ==="
CUDA_VISIBLE_DEVICES=${EVAL_GPU:-5} $PY - <<EOF
import json, numpy as np, torch, lpips
from pathlib import Path
import PIL.Image as I
run = Path("$RUN")
dev = torch.device("cuda:0")
fn = lpips.LPIPS(net="alex").to(dev).eval()

def t(p):
    a = np.asarray(I.open(p).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(dev)

def pair_lpips(pa, pb):
    # P2-01: scale BOTH inputs to [-1,1] BEFORE calling LPIPS; never touch its output
    return float(fn(t(pa) * 2 - 1, t(pb) * 2 - 1))

gen = sorted((run / "images" / "normal").glob("*.png"))
tg, tt, pairs = [], [], []
with torch.no_grad():
    for a, b in zip(gen, gen[1:]):
        ga, gb = run / "images" / "gt" / a.name, run / "images" / "gt" / b.name
        if not (run/"images"/"normal"/a.name).exists() or not gb.exists():
            continue  # P2-02 (partial): skip unpaired frames, count them
        tg.append(pair_lpips(run/"images"/"normal"/a.name, run/"images"/"normal"/b.name))
        tt.append(pair_lpips(ga, gb))
        pairs.append(a.name)
mg, mt = float(np.mean(tg)), float(np.mean(tt))
print(f"  tLPIPS gen={mg:.4f}  gt={mt:.4f}  RATIO={mg/mt:.3f}  (n={len(tg)}, skipped {len(gen)-1-len(tg)})")
print("  baselines: per_frame 1.630 | warp2 1.305 | autoreg 0.820 | instance 1.302 | target <1.3")
EOF
