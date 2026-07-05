#!/usr/bin/env python3
"""Generate detailed method/backbone-style figures for the thesis proposal.

V4 intentionally removes most decorative "AI poster" elements and uses a
conference-method-diagram style: tensors, repeated blocks, branching heads,
state banks, explicit losses, and train/infer paths.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from generate_opening_report_figures_v3 import PALETTE, SVG


W, H = 2400, 1350
OUT_DIR = Path("/home/shizhm/codespace/CS2S_pose_environment/research_figures/opening_report_backbone_v4")

COL = {
    "ink": "#17202A",
    "muted": "#5C6675",
    "line": "#93A1B2",
    "blue": "#2563EB",
    "blue_l": "#EFF6FF",
    "green": "#16A34A",
    "green_l": "#F0FDF4",
    "orange": "#D97706",
    "orange_l": "#FFF7ED",
    "red": "#DC2626",
    "red_l": "#FEF2F2",
    "purple": "#6D5BD0",
    "purple_l": "#F4F1FF",
    "gray": "#64748B",
    "gray_l": "#F8FAFC",
    "cyan": "#0891B2",
    "cyan_l": "#ECFEFF",
}


def t(svg: SVG, x: float, y: float, s: str, size: int = 18, color: str = COL["ink"], weight: int | str = 600, anchor: str = "start") -> None:
    svg.text(x, y, s, size=size, fill=color, weight=weight, anchor=anchor)


def mono(svg: SVG, x: float, y: float, s: str, size: int = 16, color: str = COL["muted"], weight: int | str = 500, anchor: str = "start") -> None:
    svg.text(x, y, s, size=size, fill=color, weight=weight, anchor=anchor, extra='class="mono"')


def header(svg: SVG, title: str, subtitle: str) -> None:
    t(svg, 60, 58, title, 30, COL["ink"], 800)
    t(svg, 60, 92, subtitle, 18, COL["muted"], 500)
    svg.line(60, 110, W - 60, 110, COL["line"], 1.4, arrow=False, opacity=0.7)


def group(svg: SVG, x: int, y: int, w: int, h: int, tag: str, name: str, stroke: str, fill: str = "#FFFFFF") -> None:
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.8, rx=12, shadow=False)
    svg.rect(x, y, w, 38, fill=fill, stroke=stroke, sw=0, rx=12, opacity=0.96)
    t(svg, x + 16, y + 26, tag, 17, stroke, 800)
    t(svg, x + 58, y + 26, name, 18, COL["ink"], 800)


def block(svg: SVG, x: int, y: int, w: int, h: int, title: str, detail: str = "", stroke: str = COL["line"], fill: str = "#FFFFFF", title_size: int = 17) -> None:
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.4, rx=8)
    t(svg, x + w / 2, y + 26, title, title_size, COL["ink"], 800, anchor="middle")
    if detail:
        lines = detail.split("\n")
        for i, line in enumerate(lines):
            mono(svg, x + w / 2, y + 52 + i * 22, line, 14, COL["muted"], 500, anchor="middle")


def tensor(svg: SVG, x: int, y: int, w: int, h: int, name: str, shape: str, stroke: str = COL["blue"], fill: str = COL["blue_l"], depth: int = 3) -> None:
    for i in range(depth, 0, -1):
        svg.rect(x + i * 5, y - i * 5, w, h, fill=fill, stroke=stroke, sw=1.0, rx=6, opacity=0.9)
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.3, rx=6)
    t(svg, x + w / 2, y + 25, name, 16, COL["ink"], 800, anchor="middle")
    mono(svg, x + w / 2, y + h - 15, shape, 13, COL["muted"], 500, anchor="middle")


def op_chain(svg: SVG, xs: list[int], y: int, labels: list[str], color: str, fill: str) -> None:
    for i, (x, label) in enumerate(zip(xs, labels)):
        block(svg, x, y, 135, 58, label, stroke=color, fill=fill, title_size=15)
        if i < len(xs) - 1:
            svg.line(x + 135, y + 29, xs[i + 1] - 8, y + 29, COL["line"], 2.4)


def arrow(svg: SVG, x1: int, y1: int, x2: int, y2: int, color: str = COL["line"], dash: str | None = None, sw: float = 2.6) -> None:
    svg.line(x1, y1, x2, y2, color=color, sw=sw, arrow=True, dash=dash)


def loss_box(svg: SVG, x: int, y: int, w: int, h: int, losses: list[tuple[str, str]]) -> None:
    svg.rect(x, y, w, h, fill="#FFFFFF", stroke=COL["gray"], sw=1.2, rx=10, dash="7 6")
    t(svg, x + 18, y + 28, "Training losses / constraints", 17, COL["ink"], 800)
    cell_w = (w - 36) / len(losses)
    for i, (name, eq) in enumerate(losses):
        cx = x + 18 + i * cell_w
        t(svg, cx + 8, y + 64, name, 14, COL["ink"], 700)
        mono(svg, cx + 8, y + 87, eq, 12, COL["muted"])


def fig1() -> str:
    svg = SVG(W, H)
    header(svg, "研究内容一：GeoSensor4DGS 算法 Backbone", "World-frame 4DGS from satellite anchors, multi-sensor evidence, and forward Gaussian heads")

    group(svg, 60, 145, 380, 1030, "(a)", "Inputs", COL["blue"], COL["gray_l"])
    tensor(svg, 95, 205, 130, 82, "Multi-view RGB", "M x 3 x H x W", COL["blue"], COL["blue_l"])
    tensor(svg, 265, 205, 118, 82, "LiDAR/Depth", "N_pts / HxW", COL["cyan"], COL["cyan_l"])
    tensor(svg, 95, 340, 130, 82, "Satellite tile", "z_xy + height", COL["green"], COL["green_l"])
    tensor(svg, 265, 340, 118, 82, "Pose packet", "K,T,LLA", COL["orange"], COL["orange_l"])
    block(svg, 95, 492, 288, 86, "Coordinate chain", "LLA -> ECEF -> ENU\nT_cam, T_lidar, T_ego", COL["orange"], "#FFFFFF")
    block(svg, 95, 628, 288, 88, "Static prior", "road/building/height prior\nfrom satellite branch", COL["green"], "#FFFFFF")
    block(svg, 95, 766, 288, 88, "Dynamic cues", "motion residual + SAM2 mask\ninstance proposals", COL["red"], "#FFFFFF")
    mono(svg, 95, 945, "Goal: ego pose is only a query;\nGaussians live in ENU world frame.", 15, COL["muted"])

    group(svg, 495, 145, 600, 1030, "(b)", "Encoders and Geo-normalization", COL["blue"], "#FFFFFF")
    tensor(svg, 535, 210, 145, 80, "Image tokens", "M x h x w x C", COL["blue"], COL["blue_l"])
    block(svg, 725, 204, 190, 92, "DINO/VGGT\nBackbone", "alt-attn / FPN", COL["blue"], "#FFFFFF")
    tensor(svg, 955, 210, 100, 80, "F_img", "N x C", COL["blue"], COL["blue_l"])
    arrow(svg, 680, 250, 725, 250)
    arrow(svg, 915, 250, 955, 250)
    tensor(svg, 535, 365, 145, 80, "Depth tokens", "N_pts x C", COL["cyan"], COL["cyan_l"])
    block(svg, 725, 360, 190, 92, "Depth/LiDAR\nEncoder", "sparse conv / MLP", COL["cyan"], "#FFFFFF")
    tensor(svg, 955, 365, 100, 80, "F_geo", "N x C", COL["cyan"], COL["cyan_l"])
    arrow(svg, 680, 405, 725, 405)
    arrow(svg, 915, 405, 955, 405)
    tensor(svg, 535, 520, 145, 80, "Map tokens", "tile x C", COL["green"], COL["green_l"])
    block(svg, 725, 515, 190, 92, "Satellite\nEncoder", "layout + height", COL["green"], "#FFFFFF")
    tensor(svg, 955, 520, 100, 80, "F_map", "N x C", COL["green"], COL["green_l"])
    arrow(svg, 680, 560, 725, 560)
    arrow(svg, 915, 560, 955, 560)
    group(svg, 535, 690, 520, 355, "(b1)", "World anchor construction", COL["orange"], "#FFFFFF")
    op_chain(svg, [575, 735, 895], 755, ["project", "sample", "anchor"], COL["orange"], COL["orange_l"])
    block(svg, 575, 855, 210, 78, "Geo covariance gate", "pose noise -> Sigma_mu", COL["orange"], "#FFFFFF")
    block(svg, 825, 855, 190, 78, "static/dynamic split", "mask + residual", COL["red"], "#FFFFFF")
    mono(svg, 575, 995, "A_t = {mu_i^ENU, f_i, sigma_i, source_i}_{i=1..N}", 16, COL["muted"])

    group(svg, 1150, 145, 650, 1030, "(c)", "Gaussian Token Backbone", COL["purple"], COL["purple_l"])
    tensor(svg, 1195, 220, 145, 80, "Anchor tokens", "N x (3+C)", COL["purple"], "#FFFFFF")
    block(svg, 1390, 204, 210, 92, "Token init", "pos enc + source emb", COL["purple"], "#FFFFFF")
    tensor(svg, 1645, 220, 110, 80, "Z_0", "N x D", COL["purple"], "#FFFFFF")
    arrow(svg, 1340, 260, 1390, 260)
    arrow(svg, 1600, 260, 1645, 260)
    group(svg, 1195, 365, 560, 315, "(c1)", "x L Gaussian World Layers", COL["purple"], "#FFFFFF")
    for i, (label, yy) in enumerate([("Self-Attn", 430), ("Cross-Attn", 500), ("FFN + AddNorm", 570)]):
        block(svg, 1240, yy, 150, 46, label, stroke=COL["purple"], fill="#FFFFFF", title_size=14)
        block(svg, 1430, yy, 150, 46, ["F_img", "F_map/F_geo", "residual"][i], stroke=COL["blue"], fill=COL["gray_l"], title_size=14)
        arrow(svg, 1390, yy + 23, 1430, yy + 23)
    svg.path("M1590,453 C1670,453 1670,615 1590,615", stroke=COL["purple"], sw=2.2, fill="none", arrow=True)
    mono(svg, 1240, 650, "Z_l = Phi_l(Z_{l-1}, F_img, F_geo, F_map)", 15, COL["muted"])
    group(svg, 1195, 745, 560, 300, "(c2)", "Prediction heads", COL["purple"], "#FFFFFF")
    block(svg, 1235, 805, 205, 86, "Geometry Head", "Delta mu, scale, quat\nSigma, depth residual", COL["orange"], "#FFFFFF")
    block(svg, 1485, 805, 205, 86, "Attribute Head", "alpha, SH/color\nsemantic, velocity", COL["green"], "#FFFFFF")
    block(svg, 1235, 935, 205, 72, "Static Gaussians", "G_t^S in ENU", COL["green"], COL["green_l"])
    block(svg, 1485, 935, 205, 72, "Dynamic Gaussians", "G_t^D + v_i", COL["red"], COL["red_l"])
    arrow(svg, 1440, 848, 1485, 848)

    group(svg, 1855, 145, 485, 1030, "(d)", "Render, query, supervision", COL["gray"], "#FFFFFF")
    block(svg, 1900, 225, 185, 88, "World 4DGS", "G_t = G_t^S union G_t^D", COL["purple"], "#FFFFFF")
    block(svg, 2115, 225, 170, 88, "Query pose", "T_query, K", COL["orange"], "#FFFFFF")
    arrow(svg, 2085, 269, 2115, 269)
    block(svg, 1955, 385, 255, 90, "Differentiable rasterizer", "R(G_t, T_query)", COL["gray"], COL["gray_l"])
    arrow(svg, 2030, 313, 2030, 385)
    block(svg, 1905, 540, 135, 72, "I_hat", "RGB", COL["blue"], COL["blue_l"])
    block(svg, 2050, 540, 135, 72, "D_hat", "depth", COL["cyan"], COL["cyan_l"])
    block(svg, 2195, 540, 105, 72, "S_hat", "sem", COL["green"], COL["green_l"])
    arrow(svg, 2080, 475, 1975, 540)
    arrow(svg, 2080, 475, 2115, 540)
    arrow(svg, 2080, 475, 2240, 540)
    loss_box(
        svg,
        1895,
        705,
        390,
        170,
        [
            ("rgb", "||I_hat-I||"),
            ("depth", "||D_hat-D||"),
            ("geo", "||mu-pi(tile)||"),
            ("temp", "||G_t-U(G_t-1)||"),
        ],
    )
    block(svg, 1900, 940, 385, 88, "Inference path", "memory write / novel-view / occupancy cue", COL["gray"], "#FFFFFF")
    mono(svg, 1900, 1105, "Key detail: satellite prior anchors static layout;\nlocal sensors correct geometry and dynamics.", 15, COL["muted"])

    arrow(svg, 440, 660, 495, 660)
    arrow(svg, 1095, 660, 1150, 660)
    arrow(svg, 1800, 660, 1855, 660)
    return svg.final()


def fig2() -> str:
    svg = SVG(W, H)
    header(svg, "研究内容二：Geo GaussianMemory 算法 Backbone", "Geo-addressed bounded memory with static tiles, dynamic tracks, fusion, erasure, compression, and LOD")

    group(svg, 60, 145, 350, 1040, "(a)", "Streaming short window", COL["blue"], COL["gray_l"])
    for i, tt in enumerate(["t-k", "...", "t-1", "t"]):
        y = 220 + i * 150
        tensor(svg, 95, y, 105, 62, f"G_{tt}", "N x D", COL["purple"], "#FFFFFF", depth=2)
        block(svg, 230, y - 2, 125, 66, "pose/key", "p_t, tile_id", COL["orange"], "#FFFFFF", title_size=14)
        if i < 3:
            arrow(svg, 177, y + 76, 177, y + 125, COL["blue"])
    block(svg, 95, 850, 260, 82, "Window 4DGS producer", "from GeoSensor4DGS\nstrictly feed-forward", COL["purple"], "#FFFFFF")
    mono(svg, 95, 1035, "short-window state is disposable;\nonly fused memory persists", 15, COL["muted"])

    group(svg, 465, 145, 495, 1040, "(b)", "Addressing and read", COL["green"], "#FFFFFF")
    block(svg, 505, 220, 185, 78, "ENU tiling", "floor(mu/r)", COL["green"], COL["green_l"])
    block(svg, 735, 220, 170, 78, "Hash index", "tile -> slots", COL["green"], COL["green_l"])
    arrow(svg, 690, 259, 735, 259, COL["green"])
    block(svg, 505, 360, 400, 92, "Memory read", "M_{t-1}[tile +/- radius]\nstatic slots + dynamic track slots", COL["green"], "#FFFFFF")
    block(svg, 505, 520, 185, 80, "Alignment", "warp by Delta T", COL["orange"], "#FFFFFF")
    block(svg, 720, 520, 185, 80, "Association", "Mahalanobis / IoU", COL["orange"], "#FFFFFF")
    arrow(svg, 690, 560, 720, 560)
    block(svg, 505, 680, 185, 82, "Consist. check", "free-space conflict\nmulti-view support", COL["red"], "#FFFFFF")
    block(svg, 720, 680, 185, 82, "Importance score", "visibility x age x task", COL["red"], "#FFFFFF")
    arrow(svg, 690, 721, 720, 721)
    mono(svg, 505, 940, "key = (tile_x, tile_y, lod, layer)\nvalue = Gaussian slots + metadata", 15, COL["muted"])

    group(svg, 1015, 145, 620, 1040, "(c)", "Static tile update branch", COL["green"], COL["green_l"])
    tensor(svg, 1055, 225, 125, 72, "old tile", "S x D", COL["green"], "#FFFFFF")
    tensor(svg, 1215, 225, 125, 72, "new obs", "N x D", COL["blue"], COL["blue_l"])
    block(svg, 1380, 215, 190, 92, "Change classifier", "keep / update\nemerge / erase", COL["red"], "#FFFFFF")
    arrow(svg, 1180, 261, 1215, 261)
    arrow(svg, 1340, 261, 1380, 261)
    group(svg, 1055, 365, 515, 335, "(c1)", "Local Gaussian operators", COL["green"], "#FFFFFF")
    op_chain(svg, [1090, 1250, 1410], 430, ["match", "fuse", "replace"], COL["green"], COL["green_l"])
    op_chain(svg, [1090, 1250, 1410], 535, ["add", "erase", "prune"], COL["red"], COL["red_l"])
    mono(svg, 1090, 650, "S'_tile = Compress(Fuse(S_tile, G_t), budget_tile)", 15, COL["muted"])
    group(svg, 1055, 760, 515, 265, "(c2)", "Bounded maintenance", COL["gray"], "#FFFFFF")
    block(svg, 1090, 825, 130, 62, "LOD down", "far tiles", COL["gray"], COL["gray_l"], 14)
    block(svg, 1250, 825, 130, 62, "merge", "similar GS", COL["gray"], COL["gray_l"], 14)
    block(svg, 1410, 825, 130, 62, "evict", "low score", COL["gray"], COL["gray_l"], 14)
    mono(svg, 1090, 965, "sum_i |G_i| <= B, per-tile quota + global quota", 15, COL["muted"])

    group(svg, 1695, 145, 645, 1040, "(d)", "Dynamic instance memory branch", COL["red"], COL["red_l"])
    tensor(svg, 1735, 225, 130, 72, "track bank", "K x T x D", COL["red"], "#FFFFFF")
    block(svg, 1905, 215, 165, 92, "Predict", "x_t = x + v dt", COL["red"], "#FFFFFF")
    block(svg, 2105, 215, 170, 92, "Associate", "mask / 3D IoU", COL["red"], "#FFFFFF")
    arrow(svg, 1865, 261, 1905, 261)
    arrow(svg, 2070, 261, 2105, 261)
    group(svg, 1735, 365, 540, 245, "(d1)", "Track lifecycle state machine", COL["red"], "#FFFFFF")
    states = [("birth", COL["green"]), ("persist", COL["blue"]), ("split/merge", COL["purple"]), ("vanish", COL["orange"]), ("retire", COL["red"])]
    for i, (name, color) in enumerate(states):
        x = 1770 + i * 98
        block(svg, x, 450, 78, 54, name, stroke=color, fill="#FFFFFF", title_size=13)
        if i < len(states) - 1:
            arrow(svg, x + 78, 477, x + 96, 477, COL["line"], sw=2)
    mono(svg, 1770, 570, "dynamic memory is instance-level, not tile-only", 14, COL["muted"])
    group(svg, 1735, 675, 540, 250, "(d2)", "Write policy", COL["gray"], "#FFFFFF")
    block(svg, 1770, 735, 130, 62, "TTL", "short-lived", COL["gray"], COL["gray_l"], 14)
    block(svg, 1930, 735, 130, 62, "velocity", "linear prior", COL["gray"], COL["gray_l"], 14)
    block(svg, 2090, 735, 130, 62, "compact", "top-k GS", COL["gray"], COL["gray_l"], 14)
    mono(svg, 1770, 875, "D'_j = Update(D_j, G_t^inst, obs_support)", 15, COL["muted"])
    block(svg, 1770, 990, 450, 80, "Memory readout", "local 4DGS for render/query/planning cue", COL["purple"], "#FFFFFF")

    arrow(svg, 410, 660, 465, 660)
    arrow(svg, 960, 620, 1015, 620)
    arrow(svg, 1635, 620, 1695, 620)
    return svg.final()


def fig3() -> str:
    svg = SVG(W, H)
    header(svg, "研究内容三：前馈流式累积增强感知 Backbone", "Weak-temporal streaming perception: world-state accumulation instead of full-history feature propagation")

    group(svg, 60, 145, 330, 1030, "(a)", "Heavy temporal baseline", COL["gray"], COL["gray_l"])
    for i, name in enumerate(["F_{t-4}", "F_{t-3}", "F_{t-2}", "F_{t-1}"]):
        tensor(svg, 105, 225 + i * 105, 95, 56, name, "Cxhxw", COL["gray"], "#FFFFFF", depth=2)
        block(svg, 230, 223 + i * 105, 95, 58, "cache", "attn", COL["gray"], "#FFFFFF", 13)
        if i < 3:
            arrow(svg, 153, 288 + i * 105, 153, 320 + i * 105, COL["gray"], sw=2)
    block(svg, 100, 735, 230, 88, "Cost source", "full history attention\nquery propagation", COL["gray"], "#FFFFFF")
    mono(svg, 100, 975, "V4 contrasts this with a compact\nworld-frame state update.", 15, COL["muted"])

    group(svg, 445, 145, 500, 1030, "(b)", "Current observation encoder", COL["blue"], "#FFFFFF")
    tensor(svg, 485, 225, 125, 70, "I_t", "M x 3 x H x W", COL["blue"], COL["blue_l"])
    tensor(svg, 645, 225, 115, 70, "D/L_t", "sparse", COL["cyan"], COL["cyan_l"])
    tensor(svg, 795, 225, 95, 70, "p_t", "SE(3)", COL["orange"], COL["orange_l"])
    block(svg, 500, 380, 170, 80, "Shared encoder", "CNN/ViT + FPN", COL["blue"], "#FFFFFF")
    block(svg, 715, 380, 170, 80, "Geometry lift", "K,T -> ENU", COL["orange"], "#FFFFFF")
    arrow(svg, 610, 295, 580, 380)
    arrow(svg, 760, 295, 800, 380)
    tensor(svg, 565, 560, 125, 70, "Obs_t", "N x D", COL["blue"], COL["blue_l"])
    tensor(svg, 735, 560, 110, 70, "A_t", "N x 3", COL["orange"], COL["orange_l"])
    block(svg, 565, 745, 280, 78, "Obs reliability", "visibility, mask support, pose cov", COL["red"], "#FFFFFF")
    mono(svg, 500, 990, "No multi-frame encoder is required;\ncurrent evidence enters once.", 15, COL["muted"])

    group(svg, 1000, 145, 815, 1030, "(c)", "Feed-forward state update layers", COL["purple"], COL["purple_l"])
    tensor(svg, 1045, 215, 130, 74, "S_{t-1}", "N_s x D", COL["green"], COL["green_l"])
    tensor(svg, 1215, 215, 125, 74, "Obs_t", "N_o x D", COL["blue"], COL["blue_l"])
    tensor(svg, 1380, 215, 105, 74, "A_t", "N_o x 3", COL["orange"], COL["orange_l"])
    block(svg, 1530, 205, 210, 94, "Update inputs", "state + observation + pose", COL["purple"], "#FFFFFF")
    arrow(svg, 1175, 252, 1215, 252)
    arrow(svg, 1340, 252, 1380, 252)
    arrow(svg, 1485, 252, 1530, 252)
    group(svg, 1045, 365, 695, 390, "(c1)", "x L streaming update block", COL["purple"], "#FFFFFF")
    layers = [
        ("Ego align", "S^- = Warp(S_{t-1}, Delta T)", COL["orange"], COL["orange_l"]),
        ("Obs cross-attn", "A = Attn(S^-, Obs_t)", COL["blue"], COL["blue_l"]),
        ("Geo gate", "w = f(vis, cov, age)", COL["green"], COL["green_l"]),
        ("Uncertainty update", "sigma_t = Decay + Residual", COL["red"], COL["red_l"]),
        ("FFN refine", "S_t = AddNorm(S^- + wA)", COL["purple"], COL["purple_l"]),
    ]
    for i, (name, detail, stroke, fill) in enumerate(layers):
        x = 1080 + (i % 3) * 205
        y = 430 + (i // 3) * 120
        block(svg, x, y, 170, 74, name, detail, stroke, fill, title_size=14)
        if i in [0, 1]:
            arrow(svg, x + 170, y + 37, x + 197, y + 37, COL["line"], sw=2)
        if i == 2:
            arrow(svg, x + 85, y + 74, 1180, y + 116, COL["line"], sw=2)
        if i == 3:
            arrow(svg, x + 170, y + 37, x + 197, y + 37, COL["line"], sw=2)
    mono(svg, 1080, 705, "single-pass; no replay of {F_{t-k},...,F_{t-1}}", 15, COL["muted"])
    group(svg, 1045, 830, 695, 230, "(c2)", "Theoretical handles", COL["gray"], "#FFFFFF")
    block(svg, 1085, 900, 180, 62, "stability", "||e_t|| <= rho||e_t-1||+eta", COL["gray"], COL["gray_l"], 13)
    block(svg, 1300, 900, 180, 62, "bounded error", "pose/noise terms", COL["gray"], COL["gray_l"], 13)
    block(svg, 1515, 900, 180, 62, "latency", "O(|S|+|Obs|)", COL["gray"], COL["gray_l"], 13)
    mono(svg, 1085, 1025, "Temporal gain is attributed to persistent world state, not a large temporal network.", 15, COL["muted"])

    group(svg, 1870, 145, 470, 1030, "(d)", "Perception decoders and feedback", COL["red"], "#FFFFFF")
    tensor(svg, 1915, 225, 125, 72, "S_t", "N_s x D", COL["purple"], COL["purple_l"])
    block(svg, 2075, 215, 190, 92, "GS-to-BEV/Occ", "rasterize or pool", COL["red"], "#FFFFFF")
    arrow(svg, 2040, 261, 2075, 261)
    group(svg, 1915, 390, 350, 300, "(d1)", "Task heads", COL["red"], COL["red_l"])
    block(svg, 1950, 455, 130, 62, "Occupancy", "P(occ)", COL["blue"], COL["blue_l"], 14)
    block(svg, 2105, 455, 130, 62, "Semantic", "P(cls)", COL["green"], COL["green_l"], 14)
    block(svg, 1950, 555, 130, 62, "Motion", "flow/v", COL["orange"], COL["orange_l"], 14)
    block(svg, 2105, 555, 130, 62, "Uncertainty", "sigma", COL["red"], COL["red_l"], 14)
    group(svg, 1915, 745, 350, 205, "(d2)", "Losses", COL["gray"], "#FFFFFF")
    mono(svg, 1950, 810, "L = L_occ + L_sem + L_flow\n  + lambda L_render + beta L_stab", 15, COL["muted"])
    block(svg, 1950, 1015, 285, 72, "Feedback to memory", "write compact S_t, discard raw history", COL["purple"], "#FFFFFF")

    arrow(svg, 390, 660, 445, 660)
    arrow(svg, 945, 660, 1000, 660)
    arrow(svg, 1815, 660, 1870, 660)
    return svg.final()


def write_assets() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assets = {
        "fig1_backbone_v4_geosensor4dgs": fig1(),
        "fig2_backbone_v4_geo_gaussian_memory": fig2(),
        "fig3_backbone_v4_feedforward_streaming": fig3(),
    }
    for stem, content in assets.items():
        (OUT_DIR / f"{stem}.svg").write_text(content, encoding="utf-8")

    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        print(f"SVG files written to {OUT_DIR}; Chrome not found, PNG export skipped.")
        return
    for stem in assets:
        svg_path = OUT_DIR / f"{stem}.svg"
        for scale, suffix in [(1, ".png"), (2, "_2x.png")]:
            out_path = OUT_DIR / f"{stem}{suffix}"
            cmd = [
                chrome,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-software-rasterizer",
                f"--user-data-dir=/tmp/chrome-backbone-v4-{scale}",
                f"--window-size={W},{H}",
                f"--force-device-scale-factor={scale}",
                f"--screenshot={out_path}",
                svg_path.as_uri(),
            ]
            env = os.environ.copy()
            env.setdefault("LANG", "zh_CN.UTF-8")
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            except subprocess.CalledProcessError as exc:
                print(f"Chrome PNG export skipped for {stem}{suffix}: {exc}")
                break


if __name__ == "__main__":
    write_assets()
    print(f"wrote backbone assets to {OUT_DIR}")
