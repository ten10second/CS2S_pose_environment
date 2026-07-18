#!/usr/bin/env python3
"""Build meeting-ready plots from a KITTI RAEA training run."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont


DEFAULT_RUN = Path(
    "/media/shizhm/sda2/CS2S_results/kitti_raea_utonia_dino_curriculum/"
    "raea_utonia_dino_full100_memmap_b2_resume_200k"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phd_proposal/stage_reports/20260717_sat_lidar_raea/assets"),
    )
    parser.add_argument("--smooth", type=int, default=50)
    return parser.parse_args()


def load_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def values(rows, key):
    return np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)


def moving_average(array, window):
    if window <= 1:
        return array
    valid = np.isfinite(array)
    numerator = np.convolve(np.where(valid, array, 0.0), np.ones(window), mode="same")
    denominator = np.convolve(valid.astype(np.float64), np.ones(window), mode="same")
    return numerator / np.maximum(denominator, 1.0)


def plot_series(ax, steps, array, label, color, window, linewidth=2.2):
    ax.plot(steps, array, color=color, alpha=0.08, linewidth=0.7)
    ax.plot(steps, moving_average(array, window), color=color, label=label, linewidth=linewidth)


def style_axis(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#d8dee9", linewidth=0.7, alpha=0.65)
    ax.axvline(200000, color="#5e6673", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.text(
        202000,
        0.97,
        "memmap B2 continuation",
        transform=ax.get_xaxis_transform(),
        va="top",
        fontsize=8.5,
        color="#4d5561",
    )


def build_training_figure(rows, output_path, window):
    steps = values(rows, "step")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        "KITTI full-train diagnostics: RAEA + Utonia point features + DINO alignment",
        fontsize=17,
        fontweight="bold",
    )

    ax = axes[0, 0]
    plot_series(ax, steps, values(rows, "loss_total"), "total", "#b23a48", window)
    plot_series(ax, steps, values(rows, "loss_eps_base"), "base epsilon", "#1f5a94", window)
    style_axis(ax, "A. Diffusion optimization", "loss")
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    plot_series(
        ax,
        steps,
        values(rows, "loss_lidar_hit_eps"),
        "LiDAR-hit epsilon",
        "#376b8a",
        window,
    )
    plot_series(
        ax,
        steps,
        values(rows, "loss_lidar_hit_x0"),
        "LiDAR-hit latent x0",
        "#d9822b",
        window,
    )
    plot_series(
        ax,
        steps,
        values(rows, "loss_lidar_hit_image_l1"),
        "LiDAR-hit RGB L1",
        "#2d8f64",
        window,
    )
    style_axis(ax, "B. LiDAR-supported reconstruction", "raw loss")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    plot_series(
        ax,
        steps,
        values(rows, "window_lidar_depth_log_l1_mean"),
        "sparse log-depth",
        "#5b4bb7",
        window,
    )
    plot_series(
        ax,
        steps,
        values(rows, "window_lidar_bottleneck_depth_log_l1_mean"),
        "bottleneck log-depth",
        "#be5f3d",
        window,
    )
    plot_series(
        ax,
        steps,
        values(rows, "window_lidar_semantic_alignment_loss_mean"),
        "DINO cosine alignment",
        "#168aad",
        window,
    )
    style_axis(ax, "C. Geometry and semantic supervision", "loss")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    plot_series(
        ax,
        steps,
        values(rows, "ray_evidence_sat_weight_mean"),
        "satellite evidence",
        "#247a53",
        window,
    )
    plot_series(
        ax,
        steps,
        values(rows, "ray_evidence_lidar_weight_mean"),
        "LiDAR evidence",
        "#d1495b",
        window,
    )
    plot_series(
        ax,
        steps,
        values(rows, "ray_evidence_null_weight_mean"),
        "null/prior evidence",
        "#6d7580",
        window,
    )
    style_axis(ax, "D. Ray-aligned evidence usage", "mean attention weight")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="center left")

    entropy_ax = ax.twinx()
    entropy = moving_average(values(rows, "lidar_attn_entropy_norm_mean"), window)
    entropy_ax.plot(steps, entropy, color="#6f2da8", linewidth=1.8, linestyle=":")
    entropy_ax.set_ylabel("local LiDAR attention entropy", color="#6f2da8")
    entropy_ax.tick_params(axis="y", colors="#6f2da8")
    entropy_ax.set_ylim(0.0, 1.02)

    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def build_raea_fusion_figure(output_path):
    fig, ax = plt.subplots(figsize=(16, 7.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    colors = {
        "sat_fill": "#D7EEE4",
        "sat_edge": "#2D7F5E",
        "lidar_fill": "#F7D3C1",
        "lidar_edge": "#B9562E",
        "latent_fill": "#E8EDF2",
        "latent_edge": "#4A6072",
        "null_fill": "#ECEFF2",
        "null_edge": "#7B8794",
        "raea_fill": "#FFF0C7",
        "raea_edge": "#A57414",
    }

    def box(x, y, w, h, title, detail, fill, edge, title_size=12.5):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=2.1))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=title_size, fontweight="bold")
        if detail:
            ax.text(x + w / 2, y + h * 0.29, detail, ha="center", va="center", fontsize=9.5, color="#3F4B57")

    def arrow(x1, y1, x2, y2, color, width=2.0, dashed=False):
        style = "--" if dashed else "-"
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=width,
                linestyle=style,
                color=color,
                shrinkA=1,
                shrinkB=1,
            )
        )

    ax.text(0.35, 6.86, "RAEA Fusion Inside One Denoising Transformer Block", fontsize=22, fontweight="bold", color="#17212B")
    ax.text(
        0.37,
        6.53,
        "Satellite and LiDAR references are computed in parallel from the same x_base, then selected per target camera ray.",
        fontsize=11.5,
        color="#52606D",
    )

    box(0.45, 4.85, 2.55, 0.9, "Satellite context", "CS2S encoder + pose/calibration", colors["sat_fill"], colors["sat_edge"])
    box(3.8, 4.85, 2.75, 0.9, "Pose-aligned Sat Cross-Attn", "x_base queries footprint-aligned tokens", colors["sat_fill"], colors["sat_edge"], 11.5)
    box(7.25, 4.85, 2.05, 0.9, "Satellite evidence", "delta_sat(u,v)", "#CDE8DC", colors["sat_edge"])

    box(0.45, 3.05, 2.55, 0.9, "Noisy street latent", "x_t(u,v)", colors["latent_fill"], colors["latent_edge"])
    box(3.8, 3.05, 2.75, 0.9, "Self-attention", "x_base(u,v)", colors["latent_fill"], colors["latent_edge"])
    box(7.25, 3.05, 2.05, 0.9, "Ray-conditioned query", "Wq[x_base + ray_PE(u,v)]", "#DDE5EC", colors["latent_edge"], 11.7)

    box(0.45, 1.25, 2.55, 0.9, "Ray-depth LiDAR tokens", "3D/Utonia feature on 8x32 rays", "#FDEBDD", "#C6682F", 11.7)
    box(3.8, 1.25, 2.75, 0.9, "Local LiDAR Cross-Attn", "x_base queries aligned 3x3 ray tokens", "#F9E2D3", "#C6682F", 11.5)
    box(7.25, 1.25, 2.05, 0.9, "LiDAR evidence", "delta_lidar(u,v)", colors["lidar_fill"], colors["lidar_edge"])

    box(10.25, 2.35, 2.55, 2.1, "RAEA", "Attn(q, {sat, LiDAR, null})\nselection per target camera ray", colors["raea_fill"], colors["raea_edge"], 17)
    box(10.55, 1.15, 1.95, 0.65, "Null / prior", "learned evidence", colors["null_fill"], colors["null_edge"], 11.5)
    box(13.45, 3.0, 2.25, 1.0, "Residual update + FFN", "x' = x_base +\ndelta_ray", colors["latent_fill"], colors["latent_edge"], 11.7)

    arrow(3.0, 5.3, 3.8, 5.3, colors["sat_edge"])
    arrow(6.55, 5.3, 7.25, 5.3, colors["sat_edge"])
    arrow(9.3, 5.3, 10.85, 4.45, colors["sat_edge"], 2.4)

    arrow(3.0, 3.5, 3.8, 3.5, colors["latent_edge"])
    arrow(6.55, 3.5, 7.25, 3.5, colors["latent_edge"])
    arrow(9.3, 3.5, 10.25, 3.5, colors["latent_edge"], 2.4)

    arrow(3.0, 1.7, 3.8, 1.7, "#C6682F")
    arrow(6.55, 1.7, 7.25, 1.7, colors["lidar_edge"])
    arrow(9.3, 1.7, 10.85, 2.35, colors["lidar_edge"], 2.4)
    arrow(11.52, 1.8, 11.52, 2.35, colors["null_edge"])
    arrow(12.8, 3.5, 13.45, 3.5, colors["raea_edge"], 2.8)

    arrow(5.15, 3.95, 5.15, 4.85, colors["latent_edge"], 1.4, dashed=True)
    arrow(5.15, 3.05, 5.15, 2.15, colors["latent_edge"], 1.4, dashed=True)
    ax.text(5.25, 4.39, "same x_base query", fontsize=8.8, color="#4A6072", va="center")
    ax.text(5.25, 2.6, "same x_base query", fontsize=8.8, color="#4A6072", va="center")

    ax.text(
        8.0,
        0.48,
        "Repeated across UNet transformer blocks. No decoder feedback, ControlNet residual, or scalar LiDAR gate.",
        ha="center",
        fontsize=10.5,
        color="#6B7280",
    )
    fig.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def trailing_mean(rows, key, count=500):
    array = values(rows[-count:], key)
    return float(np.nanmean(array))


def leading_mean(rows, key, count=500):
    array = values(rows[:count], key)
    return float(np.nanmean(array))


def build_summary(rows, output_path):
    tracked = [
        "loss_total",
        "loss_eps_base",
        "loss_lidar_hit_image_l1",
        "window_lidar_depth_log_l1_mean",
        "window_lidar_bottleneck_depth_log_l1_mean",
        "window_lidar_semantic_alignment_loss_mean",
        "ray_evidence_sat_weight_mean",
        "ray_evidence_lidar_weight_mean",
        "ray_evidence_null_weight_mean",
        "lidar_attn_entropy_norm_mean",
    ]
    summary = {
        "row_count": len(rows),
        "first_step": int(rows[0]["step"]),
        "last_step": int(rows[-1]["step"]),
        "leading_500_mean": {key: leading_mean(rows, key) for key in tracked},
        "trailing_500_mean": {key: trailing_mean(rows, key) for key in tracked},
        "last_record": {key: rows[-1].get(key) for key in tracked},
        "last_cuda_max_allocated_mb": rows[-1].get("cuda_max_allocated_mb"),
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n")


def font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def add_header(image, title, subtitle=None):
    header_height = 72 if subtitle else 48
    canvas = Image.new("RGB", (image.width, image.height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 8), title, font=font(23), fill="#18212b")
    if subtitle:
        draw.text((18, 39), subtitle, font=font(15), fill="#52606d")
    return canvas


def build_latest_effect(samples_dir, output_path):
    latest_steps = sorted(
        int(path.name.split("_")[-1])
        for path in samples_dir.glob("step_*")
        if (path / "panels").is_dir()
    )
    latest = latest_steps[-1]
    panel_dir = samples_dir / f"step_{latest:06d}" / "panels"
    panel_paths = sorted(panel_dir.glob("*.png"))
    images = []
    for panel_path in panel_paths[:2]:
        frame_id = panel_path.stem.split("__")[-1]
        panel = Image.open(panel_path).convert("RGB")
        images.append(add_header(panel, f"step {latest:,} | frame {frame_id}"))
    canvas = Image.new("RGB", (max(img.width for img in images), sum(img.height for img in images)), "white")
    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.height
    canvas.save(output_path, quality=95)


def closest_steps(available, targets):
    selected = []
    for target in targets:
        candidate = min(available, key=lambda step: abs(step - target))
        if candidate not in selected:
            selected.append(candidate)
    return selected


def build_progression(samples_dir, output_path):
    available = sorted(
        int(path.name.split("_")[-1])
        for path in samples_dir.glob("step_*")
        if (path / "panels").is_dir()
    )
    selected = closest_steps(available, [200000, 250000, 300000, 350000, 400000, 445000])
    frame_name = "2011_09_26__2011_09_26_drive_0023_sync__0000000248.png"
    rows = []
    for step in selected:
        path = samples_dir / f"step_{step:06d}" / "panels" / frame_name
        if path.exists():
            rows.append(add_header(Image.open(path).convert("RGB"), f"training progression | step {step:,}"))
    canvas = Image.new("RGB", (max(img.width for img in rows), sum(img.height for img in rows)), "white")
    y = 0
    for img in rows:
        canvas.paste(img, (0, y))
        y += img.height
    canvas.save(output_path, quality=95)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_dir / "metrics" / "train_metrics.jsonl"
    rows = load_rows(metrics_path)
    build_training_figure(rows, args.output_dir / "training_diagnostics.png", args.smooth)
    build_raea_fusion_figure(args.output_dir / "raea_fusion_detail.png")
    build_summary(rows, args.output_dir / "training_summary.json")
    build_latest_effect(args.run_dir / "samples", args.output_dir / "effect_latest_two_frames.png")
    build_progression(args.run_dir / "samples", args.output_dir / "effect_progression_frame0248.png")
    print(f"wrote report assets to {args.output_dir}")


if __name__ == "__main__":
    main()
