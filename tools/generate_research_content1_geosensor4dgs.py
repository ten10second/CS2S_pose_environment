#!/usr/bin/env python3
"""Draw a focused method/backbone figure for research content 1.

Style target:
- Skyfall-GS-like dual satellite/ground evidence
- ReconDrive-like Gaussian center/parameter heads
- StreamSplat-like differentiable rendering supervision box
"""

from __future__ import annotations

from pathlib import Path

from generate_opening_report_backbone_v4_png import Canvas, COL, block, group, tensor, title


OUT_DIR = Path("/home/shizhm/codespace/CS2S_pose_environment/research_figures/research_content1_geosensor4dgs")


def line(c: Canvas, x1: int, y1: int, x2: int, y2: int, color: str = COL["line"], dash: bool = False, w: int = 3) -> None:
    c.line(x1, y1, x2, y2, color, w, True, dash)


def mini_map(c: Canvas, x: int, y: int, w: int, h: int) -> None:
    c.rect(x, y, w, h, "#E9F8EC", COL["green"], 2, 4)
    # Roads and blocks, kept schematic but evidence-like.
    c.d.rectangle([c.sc(x + 15), c.sc(y + 12), c.sc(x + 76), c.sc(y + 44)], fill="#BDE7B8")
    c.d.rectangle([c.sc(x + 112), c.sc(y + 18), c.sc(x + 180), c.sc(y + 52)], fill="#ADD99F")
    c.d.rectangle([c.sc(x + 35), c.sc(y + 95), c.sc(x + 125), c.sc(y + 128)], fill="#CDE5B8")
    c.d.line([c.sc(x + 8), c.sc(y + 82), c.sc(x + w - 8), c.sc(y + 50)], fill="#6B7280", width=c.sc(9))
    c.d.line([c.sc(x + 8), c.sc(y + 82), c.sc(x + w - 8), c.sc(y + 50)], fill="#F8FAFC", width=c.sc(2))
    c.text(x + 12, y + h - 18, "satellite / GIS tile", 13, COL["green"], True)


def mini_camera(c: Canvas, x: int, y: int, w: int, h: int, label: str) -> None:
    c.rect(x, y, w, h, "#EFF6FF", COL["blue"], 2, 4)
    c.d.polygon(
        [
            (c.sc(x + w * 0.43), c.sc(y + h * 0.42)),
            (c.sc(x + w * 0.58), c.sc(y + h * 0.42)),
            (c.sc(x + w * 0.78), c.sc(y + h * 0.86)),
            (c.sc(x + w * 0.21), c.sc(y + h * 0.86)),
        ],
        fill="#6B7280",
    )
    c.d.polygon(
        [
            (c.sc(x + w * 0.48), c.sc(y + h * 0.42)),
            (c.sc(x + w * 0.52), c.sc(y + h * 0.42)),
            (c.sc(x + w * 0.57), c.sc(y + h * 0.86)),
            (c.sc(x + w * 0.43), c.sc(y + h * 0.86)),
        ],
        fill="#FFFFFF",
    )
    c.d.line([c.sc(x + 10), c.sc(y + h * 0.39), c.sc(x + w - 10), c.sc(y + h * 0.39)], fill="#78977A", width=c.sc(5))
    c.text(x + 10, y + 20, label, 12, COL["blue"], True)


def mini_depth(c: Canvas, x: int, y: int, w: int, h: int) -> None:
    c.rect(x, y, w, h, "#ECFEFF", COL["cyan"], 2, 4)
    for i in range(72):
        px = x + 14 + (i * 29) % (w - 28)
        py = y + 22 + (i * 41) % (h - 40)
        r = 2 + (i % 3)
        color = ["#2563EB", "#0891B2", "#F97316", "#CBD5E1"][i % 4]
        c.d.ellipse([c.sc(px - r), c.sc(py - r), c.sc(px + r), c.sc(py + r)], fill=color)
    c.text(x + 10, y + 20, "LiDAR / depth", 12, COL["cyan"], True)


def small_grid(c: Canvas, x: int, y: int, w: int, h: int, label: str, color: str) -> None:
    c.rect(x, y, w, h, "#FFFFFF", color, 2, 4)
    colors = ["#BFDBFE", "#C7F9CC", "#FDE68A", "#FBCFE8", "#DDD6FE", "#E2E8F0"]
    cols, rows = 7, 4
    cw, rh = (w - 20) / cols, (h - 36) / rows
    for r in range(rows):
        for col in range(cols):
            x0 = x + 10 + col * cw
            y0 = y + 28 + r * rh
            c.d.rectangle(
                [c.sc(x0), c.sc(y0), c.sc(x0 + cw - 4), c.sc(y0 + rh - 4)],
                fill=colors[(r * 2 + col) % len(colors)],
            )
    c.text(x + 10, y + 20, label, 12, color, True)


def draw_figure(c: Canvas) -> None:
    title(
        c,
        "研究内容一：地理坐标锚定的传感器约束 4DGS 表征框架",
        "Dual-evidence GeoSensor4DGS backbone: satellite anchors + vehicle sensors -> Gaussian heads -> differentiable supervision",
    )

    # Dual evidence inputs.
    group(c, 55, 145, 500, 1080, "(a)", "Dual evidence inputs", COL["blue"], COL["gray_l"])
    group(c, 85, 205, 440, 355, "(a1)", "Satellite / GIS evidence", COL["green"], "#FFFFFF")
    mini_map(c, 120, 265, 185, 145)
    small_grid(c, 325, 265, 160, 145, "height prior", COL["green"])
    tensor(c, 120, 445, 138, 58, "layout tokens", "N_map x C", COL["green"], COL["green_l"], 2)
    tensor(c, 330, 445, 138, 58, "anchor prior", "N_a x 3", COL["orange"], COL["orange_l"], 2)
    line(c, 258, 474, 330, 474)

    group(c, 85, 610, 440, 440, "(a2)", "On-board sensor evidence", COL["blue"], "#FFFFFF")
    mini_camera(c, 120, 670, 155, 90, "front cam")
    mini_camera(c, 300, 670, 82, 90, "left")
    mini_camera(c, 398, 670, 82, 90, "right")
    mini_depth(c, 120, 795, 180, 118)
    small_grid(c, 320, 795, 160, 118, "mask / residual", COL["red"])
    tensor(c, 120, 950, 138, 58, "image tokens", "N_img x C", COL["blue"], COL["blue_l"], 2)
    tensor(c, 330, 950, 138, 58, "depth tokens", "N_geo x C", COL["cyan"], COL["cyan_l"], 2)
    line(c, 258, 979, 330, 979)
    c.multiline(105, 1100, "双输入证据：卫星静态锚点\n+ 车载局部校正", 16, COL["muted"])

    # Coordinate and sensor constraints.
    group(c, 620, 145, 500, 1080, "(b)", "Geo-sensor normalization", COL["orange"], "#FFFFFF")
    block(c, 665, 230, 185, 82, "LLA -> ECEF", "GPS/IMU packet", COL["orange"], COL["orange_l"])
    block(c, 890, 230, 185, 82, "ECEF -> ENU", "world frame", COL["orange"], COL["orange_l"])
    line(c, 850, 271, 890, 271, COL["orange"])
    block(c, 665, 375, 185, 82, "Extrinsics", "T_cam,T_lidar,T_ego", COL["blue"], "#FFFFFF")
    block(c, 890, 375, 185, 82, "Intrinsics", "K, distortion, FoV", COL["blue"], "#FFFFFF")
    block(c, 665, 520, 185, 82, "Time sync", "t_img,t_lidar,t_pose", COL["cyan"], "#FFFFFF")
    block(c, 890, 520, 185, 82, "Pose cov gate", "Sigma_pose -> Sigma_mu", COL["red"], "#FFFFFF")
    group(c, 665, 695, 410, 305, "(b1)", "World anchor tokens", COL["orange"], COL["orange_l"])
    block(c, 705, 765, 110, 56, "project", "pi_ENU", COL["orange"], "#FFFFFF", 13)
    block(c, 850, 765, 110, 56, "sample", "sat/sensor", COL["orange"], "#FFFFFF", 13)
    block(c, 995, 765, 58, 56, "gate", "cov", COL["red"], "#FFFFFF", 13)
    line(c, 815, 793, 850, 793, COL["orange"])
    line(c, 960, 793, 995, 793, COL["orange"])
    tensor(c, 735, 880, 145, 60, "A_t", "N x (mu,f,sigma)", COL["orange"], "#FFFFFF", 2)
    tensor(c, 910, 880, 130, 60, "valid mask", "N x 1", COL["red"], COL["red_l"], 2)
    c.text(675, 1080, "x_world = T_ENU<-ECEF T_ECEF<-LLA T_ego T_cam x_cam", 14, COL["muted"], False, True)

    # Feature encoders and fusion backbone.
    group(c, 1185, 145, 610, 1080, "(c)", "Forward Gaussian backbone", COL["purple"], COL["purple_l"])
    tensor(c, 1230, 230, 110, 68, "F_sat", "N x C", COL["green"], "#FFFFFF", 2)
    tensor(c, 1230, 345, 110, 68, "F_img", "N x C", COL["blue"], "#FFFFFF", 2)
    tensor(c, 1230, 460, 110, 68, "F_geo", "N x C", COL["cyan"], "#FFFFFF", 2)
    block(c, 1390, 275, 165, 86, "Token init", "source emb + ENU pos enc", COL["purple"], "#FFFFFF")
    block(c, 1390, 435, 165, 86, "Static/dynamic split", "motion residual + mask", COL["red"], "#FFFFFF")
    line(c, 1340, 264, 1390, 315, COL["line"])
    line(c, 1340, 379, 1390, 315, COL["line"])
    line(c, 1340, 494, 1390, 478, COL["line"])
    tensor(c, 1605, 340, 120, 70, "Z_0", "N x D", COL["purple"], "#FFFFFF", 3)
    line(c, 1555, 318, 1605, 370, COL["purple"])
    line(c, 1555, 478, 1605, 370, COL["purple"])

    group(c, 1230, 625, 520, 310, "(c1)", "x L Gaussian fusion layers", COL["purple"], "#FFFFFF")
    block(c, 1270, 695, 150, 52, "Self-Attn", "Gaussian tokens", COL["purple"], "#FFFFFF", 13)
    block(c, 1465, 695, 150, 52, "Cross-Attn", "sat/sensor feats", COL["blue"], "#FFFFFF", 13)
    block(c, 1270, 790, 150, 52, "Geo-gated FFN", "cov + valid mask", COL["orange"], "#FFFFFF", 13)
    block(c, 1465, 790, 150, 52, "Add & Norm", "residual update", COL["purple"], "#FFFFFF", 13)
    line(c, 1420, 721, 1465, 721)
    c.line(1540, 747, 1540, 770, COL["line"], 2, False)
    line(c, 1540, 770, 1420, 816)
    line(c, 1420, 816, 1465, 816)
    c.text(1270, 900, "Z_l = Phi_l(Z_{l-1}, F_sat, F_img, F_geo, A_t)", 14, COL["muted"], False, True)

    # Dual heads and outputs.
    group(c, 1845, 145, 500, 1080, "(d)", "Dual Gaussian heads + supervision", COL["red"], "#FFFFFF")
    group(c, 1885, 225, 420, 250, "(d1)", "Dual-head Gaussian prediction", COL["purple"], COL["purple_l"])
    block(c, 1925, 295, 165, 86, "Geometry Head", "mu^ENU, scale, quat\ncov, depth residual", COL["orange"], "#FFFFFF", 13)
    block(c, 2115, 295, 155, 86, "Attribute Head", "alpha, color, sem\nvelocity, lifespan", COL["green"], "#FFFFFF", 13)
    tensor(c, 1935, 410, 135, 52, "G_t^S", "static GS", COL["green"], COL["green_l"], 2)
    tensor(c, 2125, 410, 125, 52, "G_t^D", "dynamic GS", COL["red"], COL["red_l"], 2)

    group(c, 1885, 535, 420, 245, "(d2)", "Differentiable rasterization", COL["gray"], "#FFFFFF")
    block(c, 1925, 605, 165, 70, "Query pose", "K,T_query", COL["orange"], "#FFFFFF", 13)
    block(c, 2115, 605, 155, 70, "Rasterizer", "R(G_t,T_query)", COL["gray"], COL["gray_l"], 13)
    line(c, 2090, 640, 2115, 640)
    small_grid(c, 1925, 700, 95, 55, "RGB", COL["blue"])
    small_grid(c, 2045, 700, 95, 55, "Depth", COL["cyan"])
    small_grid(c, 2165, 700, 95, 55, "Sem", COL["green"])

    group(c, 1885, 840, 420, 245, "(d3)", "Training supervision", COL["red"], "#FFFFFF")
    c.text(1915, 900, "L = L_rgb + L_depth + L_mask", 14, COL["muted"], False, True)
    c.text(1915, 932, "  + lambda_geo L_geo + lambda_t L_temp", 14, COL["muted"], False, True)
    block(c, 1915, 980, 110, 48, "sensor", "FoV/range", COL["blue"], "#FFFFFF", 12)
    block(c, 2050, 980, 110, 48, "geo", "tile/ENU", COL["orange"], "#FFFFFF", 12)
    block(c, 2185, 980, 90, 48, "time", "smooth", COL["purple"], "#FFFFFF", 12)
    c.multiline(1895, 1138, "solid arrows: inference data flow\ndotted paths: training constraints", 14, COL["muted"])

    # Cross-panel arrows and train constraint paths.
    line(c, 555, 745, 620, 745)
    line(c, 1120, 745, 1185, 745)
    line(c, 1795, 745, 1845, 745)
    line(c, 1730, 372, 1885, 330, COL["purple"])
    c.line(2270, 980, 2270, 790, COL["red"], 2, True, True)
    c.line(2115, 780, 2035, 840, COL["red"], 2, True, True)


def render(scale: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    c = Canvas(scale)
    draw_figure(c)
    suffix = "_2x.png" if scale == 2 else ".png"
    c.save(OUT_DIR / f"research_content1_geosensor4dgs_backbone{suffix}")


if __name__ == "__main__":
    render(1)
    render(2)
    print(f"wrote research content 1 figure to {OUT_DIR}")
