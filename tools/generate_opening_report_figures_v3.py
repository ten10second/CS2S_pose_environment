#!/usr/bin/env python3
"""Generate V3 thesis-opening report figures as SVG and PNG assets.

The figures intentionally mimic dense AI-conference method diagrams:
evidence thumbnails, colored module regions, state timelines, supervision
paths, and intermediate representations.
"""

from __future__ import annotations

import html
import os
import shutil
import subprocess
import textwrap
from pathlib import Path


OUT_DIR = Path("/home/shizhm/Downloads/开题报告_AI顶会风格插图_v3")
W, H = 2200, 1240


PALETTE = {
    "ink": "#18202B",
    "muted": "#5F6B7A",
    "line": "#9CA7B5",
    "bg": "#F8FAFC",
    "sensor": "#E9F3FF",
    "sensor_s": "#2775C8",
    "memory": "#EAF8EF",
    "memory_s": "#2F9D5B",
    "gaussian": "#FFF3D6",
    "gaussian_s": "#D49319",
    "pred": "#FDECEC",
    "pred_s": "#D84A4A",
    "purple": "#F1EEFF",
    "purple_s": "#7257C7",
    "gray": "#EEF2F6",
    "gray_s": "#64748B",
    "cyan": "#E8F8FA",
    "cyan_s": "#1593A6",
}


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


class SVG:
    def __init__(self, width: int = W, height: int = H) -> None:
        self.width = width
        self.height = height
        self.items: list[str] = []

    def add(self, raw: str) -> None:
        self.items.append(raw)

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: int = 26,
        fill: str = PALETTE["ink"],
        weight: int | str = 500,
        anchor: str = "start",
        opacity: float = 1.0,
        extra: str = "",
    ) -> None:
        self.add(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}" opacity="{opacity}" {extra}>'
            f"{esc(text)}</text>"
        )

    def wrap(
        self,
        x: float,
        y: float,
        text: str,
        width_chars: int,
        size: int = 22,
        fill: str = PALETTE["muted"],
        weight: int | str = 400,
        line_gap: int = 1,
    ) -> None:
        for i, line in enumerate(textwrap.wrap(text, width=width_chars, break_long_words=False)):
            self.text(x, y + i * (size + line_gap + 6), line, size=size, fill=fill, weight=weight)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "white",
        stroke: str = PALETTE["line"],
        sw: float = 1.5,
        rx: float = 14,
        opacity: float = 1.0,
        dash: str | None = None,
        shadow: bool = False,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        filter_attr = ' filter="url(#softShadow)"' if shadow else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"'
            f"{dash_attr}{filter_attr}/>"
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = PALETTE["line"],
        sw: float = 3,
        arrow: bool = True,
        dash: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" opacity="{opacity}"'
            f"{dash_attr}{marker}/>"
        )

    def path(
        self,
        d: str,
        stroke: str = PALETTE["line"],
        sw: float = 3,
        fill: str = "none",
        arrow: bool = True,
        dash: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"'
            f"{dash_attr}{marker}/>"
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        fill: str,
        stroke: str = "white",
        sw: float = 1.0,
        opacity: float = 1.0,
    ) -> None:
        self.add(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}"/>'
        )

    def ellipse(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        fill: str,
        stroke: str = "white",
        sw: float = 1.0,
        opacity: float = 1.0,
        rotate: float = 0,
    ) -> None:
        transform = f' transform="rotate({rotate} {cx} {cy})"' if rotate else ""
        self.add(
            f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{transform}/>'
        )

    def final(self) -> str:
        defs = f"""
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE['line']}"/>
  </marker>
  <filter id="softShadow" x="-10%" y="-10%" width="120%" height="130%">
    <feDropShadow dx="0" dy="8" stdDeviation="8" flood-color="#334155" flood-opacity="0.12"/>
  </filter>
  <pattern id="fineGrid" width="28" height="28" patternUnits="userSpaceOnUse">
    <path d="M 28 0 L 0 0 0 28" fill="none" stroke="#CBD5E1" stroke-width="1" opacity="0.55"/>
  </pattern>
  <linearGradient id="depthGrad" x1="0%" x2="100%" y1="0%" y2="100%">
    <stop offset="0%" stop-color="#0F172A"/>
    <stop offset="55%" stop-color="#2563EB"/>
    <stop offset="100%" stop-color="#F97316"/>
  </linearGradient>
</defs>
"""
        style = """
<style>
  text { font-family: "Noto Sans CJK SC", "Source Han Sans SC", "Arial", sans-serif; }
  .mono { font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace; }
</style>
"""
        body = "\n".join(self.items)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n{defs}\n{style}\n'
            f'<rect width="{self.width}" height="{self.height}" fill="{PALETTE["bg"]}"/>\n{body}\n</svg>\n'
        )


def title(svg: SVG, cn: str, en: str) -> None:
    svg.text(60, 64, cn, 34, PALETTE["ink"], 700)
    svg.text(60, 101, en, 20, PALETTE["muted"], 500)
    svg.line(60, 120, 2140, 120, PALETTE["line"], 1.5, arrow=False, opacity=0.65)


def panel(svg: SVG, x: int, y: int, w: int, h: int, tag: str, name: str, fill: str, stroke: str) -> None:
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=2, rx=18, shadow=True)
    svg.text(x + 24, y + 38, tag, 20, stroke, 800)
    svg.text(x + 72, y + 38, name, 24, PALETTE["ink"], 700)


def pill(svg: SVG, x: int, y: int, w: int, h: int, text: str, fill: str, stroke: str, size: int = 19) -> None:
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.4, rx=h / 2)
    svg.text(x + w / 2, y + h / 2 + size / 3, text, size, stroke, 700, anchor="middle")


def tiny_label(svg: SVG, x: int, y: int, text: str, color: str = PALETTE["muted"]) -> None:
    svg.text(x, y, text, 16, color, 600)


def draw_satellite(svg: SVG, x: int, y: int, w: int, h: int, label: str = "Geo map tile") -> None:
    svg.rect(x, y, w, h, "#DCEEDB", "#7AA37A", 1.3, rx=10)
    patches = [
        (x + 18, y + 18, w * 0.34, h * 0.25, "#B9D7A5"),
        (x + w * 0.58, y + 20, w * 0.30, h * 0.22, "#A8CFA2"),
        (x + 28, y + h * 0.62, w * 0.28, h * 0.22, "#C7D5B5"),
        (x + w * 0.54, y + h * 0.58, w * 0.34, h * 0.27, "#B8C1A4"),
    ]
    for px, py, pw, ph, c in patches:
        svg.rect(px, py, pw, ph, c, c, 0, rx=8, opacity=0.92)
    svg.path(f"M{x+10},{y+h*0.42} C{x+w*0.25},{y+h*0.28} {x+w*0.54},{y+h*0.70} {x+w-8},{y+h*0.48}", "#EDF2F7", 15, arrow=False)
    svg.path(f"M{x+12},{y+h*0.42} C{x+w*0.25},{y+h*0.28} {x+w*0.54},{y+h*0.70} {x+w-8},{y+h*0.48}", "#64748B", 2, arrow=False, dash="10 10")
    for i in range(1, 4):
        svg.line(x + i * w / 4, y, x + i * w / 4, y + h, "#FFFFFF", 1, arrow=False, opacity=0.45)
    for i in range(1, 3):
        svg.line(x, y + i * h / 3, x + w, y + i * h / 3, "#FFFFFF", 1, arrow=False, opacity=0.45)
    svg.text(x + 16, y + h - 18, label, 17, "#315B31", 700)


def draw_camera(svg: SVG, x: int, y: int, w: int, h: int, label: str, accent: str = "#2563EB") -> None:
    svg.rect(x, y, w, h, "#DDEAF8", "#7BA2CE", 1.2, rx=8)
    svg.add(f'<polygon points="{x+w*0.42},{y+h*0.48} {x+w*0.60},{y+h*0.48} {x+w*0.83},{y+h*0.88} {x+w*0.18},{y+h*0.88}" fill="#5B6470" opacity="0.82"/>')
    svg.add(f'<polygon points="{x+w*0.48},{y+h*0.48} {x+w*0.52},{y+h*0.48} {x+w*0.56},{y+h*0.88} {x+w*0.44},{y+h*0.88}" fill="#F8FAFC" opacity="0.9"/>')
    svg.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h*0.48}" fill="#B7D3ED" opacity="0.85" rx="8"/>')
    svg.line(x + w * 0.1, y + h * 0.46, x + w * 0.9, y + h * 0.46, "#2D5B2D", 6, arrow=False, opacity=0.45)
    svg.text(x + 10, y + 22, label, 15, accent, 800)


def draw_depth(svg: SVG, x: int, y: int, w: int, h: int, label: str = "Depth / LiDAR") -> None:
    svg.rect(x, y, w, h, "url(#depthGrad)", "#334155", 1.2, rx=8)
    for i in range(60):
        px = x + 16 + ((i * 37) % int(w - 32))
        py = y + 20 + ((i * 53) % int(h - 42))
        color = "#E0F2FE" if i % 3 else "#FDBA74"
        svg.circle(px, py, 2.4 + (i % 4) * 0.4, color, "none", opacity=0.78)
    svg.text(x + 10, y + 23, label, 15, "#EAF2FF", 800)


def draw_bev(svg: SVG, x: int, y: int, w: int, h: int, label: str, mode: str = "occ") -> None:
    svg.rect(x, y, w, h, "#F8FAFC", "#94A3B8", 1.2, rx=8)
    cols, rows = 8, 5
    cw, rh = w / cols, h / rows
    colors = ["#E2E8F0", "#93C5FD", "#86EFAC", "#FCA5A5", "#FDE68A", "#C4B5FD"]
    for r in range(rows):
        for c in range(cols):
            idx = (r * 3 + c * 5 + len(label)) % len(colors)
            op = 0.28 + ((r + c) % 3) * 0.12
            svg.rect(x + c * cw + 3, y + r * rh + 3, cw - 6, rh - 6, colors[idx], colors[idx], 0, rx=4, opacity=op)
    if mode == "road":
        svg.path(f"M{x+w*0.16},{y+h*0.88} C{x+w*0.45},{y+h*0.55} {x+w*0.55},{y+h*0.35} {x+w*0.88},{y+h*0.10}", "#64748B", 9, arrow=False, opacity=0.78)
        svg.path(f"M{x+w*0.22},{y+h*0.90} C{x+w*0.48},{y+h*0.58} {x+w*0.58},{y+h*0.37} {x+w*0.91},{y+h*0.14}", "#F8FAFC", 2, arrow=False, dash="8 8")
    svg.text(x + 10, y + 22, label, 15, "#334155", 800)


def draw_gaussians(svg: SVG, x: int, y: int, w: int, h: int, label: str, dynamic: bool = True) -> None:
    svg.rect(x, y, w, h, "#101827", "#334155", 1.2, rx=10)
    for i in range(55):
        px = x + 18 + ((i * 47) % int(w - 36))
        py = y + 26 + ((i * 71) % int(h - 48))
        if i % 7 == 0 and dynamic:
            c = "#F87171"
        elif i % 5 == 0:
            c = "#FBBF24"
        else:
            c = "#60A5FA"
        svg.ellipse(px, py, 7 + (i % 4), 3.5 + (i % 3), c, "white", 0.5, 0.65, rotate=(i * 23) % 180)
    for gx in range(1, 5):
        svg.line(x + gx * w / 5, y, x + gx * w / 5, y + h, "#475569", 0.8, arrow=False, opacity=0.55)
    for gy in range(1, 4):
        svg.line(x, y + gy * h / 4, x + w, y + gy * h / 4, "#475569", 0.8, arrow=False, opacity=0.55)
    svg.text(x + 12, y + 24, label, 16, "#E2E8F0", 800)


def draw_feature_grid(svg: SVG, x: int, y: int, w: int, h: int, label: str) -> None:
    svg.rect(x, y, w, h, "#F8FAFC", "#94A3B8", 1.2, rx=8)
    cols, rows = 9, 6
    colors = ["#BFDBFE", "#C4B5FD", "#A7F3D0", "#FDE68A", "#FBCFE8"]
    for r in range(rows):
        for c in range(cols):
            svg.rect(
                x + 8 + c * ((w - 16) / cols),
                y + 30 + r * ((h - 42) / rows),
                (w - 22) / cols,
                (h - 48) / rows,
                colors[(r * 2 + c) % len(colors)],
                "white",
                0.7,
                rx=3,
                opacity=0.75,
            )
    svg.text(x + 10, y + 22, label, 15, "#334155", 800)


def draw_equation_strip(svg: SVG, x: int, y: int, w: int, h: int, items: list[tuple[str, str]]) -> None:
    svg.rect(x, y, w, h, "#FFFFFF", "#CBD5E1", 1.2, rx=14, dash="8 8")
    n = len(items)
    gap = 18
    iw = (w - gap * (n + 1)) / n
    for i, (name, eq) in enumerate(items):
        ix = x + gap + i * (iw + gap)
        svg.rect(ix, y + 22, iw, h - 44, "#F8FAFC", "#CBD5E1", 1.0, rx=12)
        svg.text(ix + 16, y + 52, name, 18, PALETTE["ink"], 800)
        svg.text(ix + 16, y + 84, eq, 17, PALETTE["muted"], 500, extra='class="mono"')


def fig1() -> str:
    svg = SVG()
    title(svg, "研究内容一：地理坐标锚定的传感器约束 4DGS 表征框架", "Geo-anchored sensor-constrained 4D Gaussian representation")

    panel(svg, 60, 150, 520, 575, "(a)", "多源观测证据", PALETTE["sensor"], PALETTE["sensor_s"])
    draw_satellite(svg, 90, 205, 205, 150, "satellite tile / GPS")
    draw_camera(svg, 320, 205, 210, 96, "front cam")
    draw_camera(svg, 320, 317, 100, 88, "left")
    draw_camera(svg, 430, 317, 100, 88, "right")
    draw_depth(svg, 90, 382, 205, 122)
    draw_bev(svg, 320, 430, 210, 125, "sparse BEV", mode="road")
    pill(svg, 92, 535, 190, 38, "pose stamp p_t", "#FFFFFF", PALETTE["sensor_s"])
    pill(svg, 302, 535, 225, 38, "tile id + timestamp", "#FFFFFF", PALETTE["sensor_s"])
    svg.wrap(92, 606, "raw images, LiDAR/depth, map tiles, GPS/IMU pose are kept as visible evidence rather than hidden inputs", 43, 19)

    panel(svg, 625, 150, 430, 575, "(b)", "坐标与传感器约束", "#F1F7FF", "#3B82F6")
    svg.rect(665, 215, 350, 170, "url(#fineGrid)", "#93C5FD", 1.4, rx=12)
    svg.line(730, 340, 910, 250, "#EF4444", 4, arrow=True)
    svg.line(730, 340, 740, 235, "#22C55E", 4, arrow=True)
    svg.line(730, 340, 930, 350, "#3B82F6", 4, arrow=True)
    svg.text(916, 245, "E", 18, "#EF4444", 800)
    svg.text(746, 234, "N", 18, "#22C55E", 800)
    svg.text(936, 356, "U", 18, "#3B82F6", 800)
    draw_feature_grid(svg, 665, 415, 160, 120, "calib tokens")
    draw_feature_grid(svg, 850, 415, 165, 120, "pose prior")
    pill(svg, 665, 570, 150, 38, "time sync", "#FFFFFF", "#3B82F6")
    pill(svg, 832, 570, 183, 38, "uncertainty gate", "#FFFFFF", "#3B82F6")
    svg.text(670, 660, "x_world = T_geo T_ego T_cam x_cam", 19, PALETTE["ink"], 600, extra='class="mono"')
    svg.text(670, 690, "validity: FoV, range, pose covariance", 18, PALETTE["muted"], 500)

    panel(svg, 1100, 150, 540, 575, "(c)", "Backbone 与双预测头", "#F5F3FF", PALETTE["purple_s"])
    draw_feature_grid(svg, 1135, 220, 185, 135, "DINO/VGGT features")
    draw_feature_grid(svg, 1135, 390, 185, 135, "dense geo features")
    svg.line(1325, 286, 1400, 286, PALETTE["purple_s"], 3.2)
    svg.line(1325, 456, 1400, 456, PALETTE["purple_s"], 3.2)
    svg.rect(1408, 215, 190, 145, "#FFFFFF", PALETTE["purple_s"], 1.6, rx=14)
    svg.text(1428, 250, "Geometry Head", 22, PALETTE["ink"], 800)
    svg.text(1428, 286, "center / scale", 18, PALETTE["muted"], 500)
    svg.text(1428, 316, "rotation / depth", 18, PALETTE["muted"], 500)
    svg.rect(1408, 385, 190, 145, "#FFFFFF", PALETTE["purple_s"], 1.6, rx=14)
    svg.text(1428, 420, "Attribute Head", 22, PALETTE["ink"], 800)
    svg.text(1428, 456, "color / opacity", 18, PALETTE["muted"], 500)
    svg.text(1428, 486, "semantics / velocity", 18, PALETTE["muted"], 500)
    svg.rect(1150, 570, 448, 92, "#FFFFFF", "#C4B5FD", 1.2, rx=12, dash="7 7")
    svg.text(1170, 604, "image / mask / feature / Gaussian index mapping", 18, PALETTE["ink"], 700)
    for i, label in enumerate(["I", "M", "F", "G"]):
        px = 1180 + i * 95
        svg.rect(px, 622, 58, 26, ["#BFDBFE", "#FCA5A5", "#C4B5FD", "#FDE68A"][i], "white", 0.8, rx=5)
        svg.text(px + 29, 642, label, 15, PALETTE["ink"], 800, anchor="middle")

    panel(svg, 1690, 150, 450, 575, "(d)", "Geo-anchored 4DGS", PALETTE["gaussian"], PALETTE["gaussian_s"])
    draw_gaussians(svg, 1725, 220, 380, 260, "static + dynamic Gaussians")
    svg.rect(1725, 512, 180, 110, "#FFFFFF", PALETTE["gaussian_s"], 1.2, rx=12)
    draw_bev(svg, 1738, 530, 154, 74, "render")
    svg.rect(1925, 512, 180, 110, "#FFFFFF", PALETTE["gaussian_s"], 1.2, rx=12)
    draw_bev(svg, 1938, 530, 154, 74, "semantics")
    svg.line(1815, 485, 1815, 510, PALETTE["gaussian_s"], 3)
    svg.line(2015, 485, 2015, 510, PALETTE["gaussian_s"], 3)
    svg.text(1730, 668, "G_t={mu,Sigma,alpha,c,s,v}_geo", 18, PALETTE["ink"], 600, extra='class="mono"')

    svg.line(580, 440, 625, 440, PALETTE["line"], 3.4)
    svg.line(1055, 440, 1100, 440, PALETTE["line"], 3.4)
    svg.line(1640, 440, 1690, 440, PALETTE["line"], 3.4)
    svg.path("M1815,725 C1815,785 1510,790 1510,755", PALETTE["line"], 2.2, arrow=True, dash="8 8", opacity=0.8)

    panel(svg, 60, 785, 2080, 360, "(e)", "训练监督与可微闭环", "#FFFFFF", "#CBD5E1")
    draw_equation_strip(
        svg,
        95,
        855,
        1260,
        142,
        [
            ("reprojection", "L_img = ||R(G_t)-I_t||"),
            ("depth", "L_d = ||D(G_t)-D_t||"),
            ("geo", "L_geo = ||pi_geo(mu)-tile||"),
            ("temporal", "L_temp = ||G_t-U(G_t-1)||"),
        ],
    )
    svg.rect(1405, 855, 690, 142, "#F8FAFC", "#CBD5E1", 1.3, rx=14, dash="8 8")
    svg.text(1430, 890, "sensor-constrained differentiable rasterizer", 22, PALETTE["ink"], 800)
    for i, name in enumerate(["RGB", "depth", "mask", "semantic", "pose"]):
        pill(svg, 1430 + i * 124, 920, 102, 38, name, "#FFFFFF", ["#2563EB", "#F97316", "#EF4444", "#7C3AED", "#16A34A"][i], 16)
    svg.text(1430, 980, "dotted paths denote weak/auxiliary constraints", 17, PALETTE["muted"], 500)
    svg.path("M2050,1005 C2065,1080 1700,1115 1490,1015", "#CBD5E1", 2.2, arrow=True, dash="8 8")
    svg.text(96, 1070, "Differentiable constraints tie sensor evidence to geo-anchored Gaussian parameters.", 20, PALETTE["muted"], 600)

    return svg.final()


def fig2() -> str:
    svg = SVG()
    title(svg, "研究内容二：地理地址化的有界 4DGS 流式记忆架构", "Bounded geo-addressed streaming memory for static map and dynamic Gaussian tracks")

    panel(svg, 60, 150, 410, 835, "(a)", "流式输入窗口", PALETTE["sensor"], PALETTE["sensor_s"])
    times = ["t-3", "t-2", "t-1", "t"]
    for i, t in enumerate(times):
        yy = 220 + i * 165
        draw_camera(svg, 95, yy, 145, 72, f"cam {t}", "#2563EB")
        draw_bev(svg, 255, yy, 170, 72, f"BEV {t}", mode="road")
        svg.rect(96, yy + 85, 328, 42, "#FFFFFF", PALETTE["sensor_s"], 1.0, rx=9)
        svg.text(112, yy + 112, f"p_{t} | tile(x,y,l) | age", 14, PALETTE["ink"], 600, extra='class="mono"')
        if i < len(times) - 1:
            svg.line(260, yy + 134, 260, yy + 158, PALETTE["sensor_s"], 2.2)
    svg.wrap(95, 920, "bounded temporal slice; older evidence is stored as geo-addressed state", 34, 17)

    panel(svg, 520, 150, 1050, 835, "(b)", "Bounded Geo-addressed Memory", PALETTE["memory"], PALETTE["memory_s"])
    svg.rect(555, 215, 980, 78, "#FFFFFF", PALETTE["memory_s"], 1.4, rx=14)
    svg.text(578, 248, "key = (tile_x, tile_y, level, semantic_layer, time_bucket)", 22, PALETTE["ink"], 700, extra='class="mono"')
    for i, name in enumerate(["dirty bit", "confidence", "TTL", "LOD", "budget"]):
        pill(svg, 582 + i * 178, 262, 140, 28, name, "#F8FAFC", PALETTE["memory_s"], 14)

    svg.rect(555, 325, 980, 285, "#FFFFFF", "#8AC99D", 1.4, rx=16)
    svg.text(580, 360, "Static Gaussian Map Bank: old/new alignment + local update", 20, PALETTE["ink"], 800)
    draw_gaussians(svg, 585, 385, 180, 145, "old tile G")
    draw_depth(svg, 790, 385, 165, 145, "new LiDAR")
    svg.line(765, 458, 790, 458, PALETTE["line"], 3)
    svg.line(955, 458, 990, 458, PALETTE["line"], 3)
    svg.rect(990, 385, 180, 145, "#FEF2F2", "#EF4444", 1.3, rx=12)
    svg.text(1012, 420, "Change Detect", 21, PALETTE["ink"], 800)
    for i, (cx, cy, col, txt) in enumerate(
        [(1034, 466, "#22C55E", "emerge"), (1090, 490, "#EF4444", "vanish"), (1126, 454, "#F59E0B", "shift")]
    ):
        svg.circle(cx, cy, 13, col, "white", 1.2)
        svg.text(cx - 24, cy + 38, txt, 14, col, 700)
    svg.line(1170, 458, 1205, 458, PALETTE["line"], 3)
    svg.rect(1205, 385, 145, 145, "#F8FAFC", PALETTE["memory_s"], 1.3, rx=12)
    svg.text(1222, 422, "Update", 21, PALETTE["ink"], 800)
    svg.text(1222, 455, "add / erase", 17, PALETTE["muted"], 500)
    svg.text(1222, 484, "fuse / prune", 17, PALETTE["muted"], 500)
    svg.line(1350, 458, 1380, 458, PALETTE["line"], 3)
    draw_gaussians(svg, 1380, 385, 125, 145, "new G'")
    svg.text(585, 575, "old/new alignment makes dirty regions explicit before tile-level update", 18, PALETTE["muted"], 500)

    svg.rect(555, 645, 980, 250, "#FFFFFF", "#8AC99D", 1.4, rx=16)
    svg.text(580, 680, "Dynamic Gaussian Track Bank: persistent / emerging / vanishing lifecycle", 22, PALETTE["ink"], 800)
    states = [
        ("birth", "#22C55E", 610),
        ("associate", "#2563EB", 770),
        ("deform", "#F59E0B", 930),
        ("merge", "#7C3AED", 1090),
        ("retire", "#EF4444", 1250),
    ]
    for i, (name, col, xx) in enumerate(states):
        svg.rect(xx, 720, 125, 78, "#F8FAFC", col, 1.5, rx=12)
        svg.circle(xx + 32, 758, 14, col, "white", 1)
        svg.text(xx + 58, 752, name, 18, PALETTE["ink"], 800)
        svg.text(xx + 58, 780, f"track {i}", 15, PALETTE["muted"], 500, extra='class="mono"')
        if i < len(states) - 1:
            svg.line(xx + 125, 759, states[i + 1][2] - 8, 759, PALETTE["line"], 3)
    svg.rect(610, 825, 765, 40, "#F8FAFC", "#CBD5E1", 1.0, rx=10)
    svg.text(630, 851, "static bank stores long-lived geometry; dynamic bank stores short-lived tracks with bounded TTL", 17, PALETTE["muted"], 500)

    panel(svg, 1620, 150, 520, 835, "(c)", "查询、读出与预算控制", PALETTE["gaussian"], PALETTE["gaussian_s"])
    svg.rect(1660, 220, 420, 110, "#FFFFFF", PALETTE["gaussian_s"], 1.3, rx=14)
    svg.text(1685, 258, "query pose + geo range", 23, PALETTE["ink"], 800)
    svg.text(1685, 294, "Q = (p_t, radius, task)", 20, PALETTE["muted"], 500, extra='class="mono"')
    svg.line(1835, 330, 1835, 370, PALETTE["gaussian_s"], 3)
    draw_gaussians(svg, 1660, 370, 190, 160, "local 4DGS")
    draw_bev(svg, 1888, 370, 190, 160, "local BEV", mode="road")
    svg.line(1850, 450, 1888, 450, PALETTE["line"], 3)
    svg.rect(1660, 565, 420, 130, "#FFFFFF", PALETTE["gaussian_s"], 1.3, rx=14)
    svg.text(1685, 602, "render / occupancy / downstream cue", 23, PALETTE["ink"], 800)
    for i, name in enumerate(["novel view", "occ", "flow"]):
        pill(svg, 1685 + i * 125, 625, 105, 38, name, "#F8FAFC", PALETTE["gaussian_s"], 15)
    svg.rect(1660, 735, 420, 160, "#FFFFFF", "#64748B", 1.3, rx=14, dash="9 8")
    svg.text(1685, 772, "budget controller", 23, PALETTE["ink"], 800)
    for i, name in enumerate(["evict", "compress", "LOD", "refresh"]):
        pill(svg, 1685 + (i % 2) * 180, 800 + (i // 2) * 48, 145, 36, name, "#F8FAFC", "#64748B", 15)

    svg.line(470, 560, 520, 560, PALETTE["line"], 3.5)
    svg.line(1570, 560, 1620, 560, PALETTE["line"], 3.5)
    svg.path("M2080,850 C2140,1030 850,1110 345,985", "#94A3B8", 2.0, arrow=True, dash="9 9", opacity=0.75)

    panel(svg, 60, 1030, 2080, 115, "(d)", "更新策略与边界", "#FFFFFF", "#CBD5E1")
    svg.text(95, 1092, "Local evidence updates only changed tiles and tracks under a bounded memory budget.", 24, PALETTE["muted"], 600)
    return svg.final()


def fig3() -> str:
    svg = SVG()
    title(svg, "研究内容三：前馈流式约束下的时序累积增强感知", "Feed-forward streaming state update for temporally accumulated perception")

    panel(svg, 60, 160, 310, 585, "(a)", "灰色基线：重历史回放", PALETTE["gray"], PALETTE["gray_s"])
    for i, t in enumerate(["t-4", "t-3", "t-2", "t-1"]):
        yy = 235 + i * 78
        draw_bev(svg, 100, yy, 95, 52, t)
        svg.rect(215, yy, 95, 52, "#FFFFFF", "#94A3B8", 1.0, rx=8)
        svg.text(236, yy + 32, "queue", 16, PALETTE["gray_s"], 700)
        if i < 3:
            svg.line(155, yy + 56, 155, yy + 73, PALETTE["gray_s"], 2)
    svg.rect(95, 570, 220, 72, "#FFFFFF", PALETTE["gray_s"], 1.2, rx=12)
    svg.text(124, 600, "recurrent / replay", 20, PALETTE["ink"], 800)
    svg.text(124, 628, "high memory cost", 17, PALETTE["muted"], 500)
    svg.wrap(95, 700, "contrast only: bounded state avoids full replay", 26, 17)

    panel(svg, 425, 160, 440, 230, "(b)", "当前观测注入", PALETTE["sensor"], PALETTE["sensor_s"])
    draw_camera(svg, 455, 220, 175, 88, "current frame")
    draw_bev(svg, 650, 220, 175, 88, "current BEV", mode="road")
    pill(svg, 455, 327, 165, 36, "Obs_t", "#FFFFFF", PALETTE["sensor_s"])
    pill(svg, 640, 327, 185, 36, "pose p_t", "#FFFFFF", PALETTE["sensor_s"])

    panel(svg, 910, 160, 435, 230, "(c)", "上一时刻有界状态", PALETTE["memory"], PALETTE["memory_s"])
    draw_gaussians(svg, 945, 215, 180, 115, "G_{t-1}")
    svg.rect(1150, 215, 150, 115, "#FFFFFF", PALETTE["memory_s"], 1.2, rx=12)
    svg.text(1172, 250, "memory", 22, PALETTE["ink"], 800)
    svg.text(1172, 282, "tiles + tracks", 18, PALETTE["muted"], 500)
    svg.text(1172, 312, "bounded TTL", 18, PALETTE["muted"], 500)

    panel(svg, 1390, 160, 750, 230, "(d)", "增强感知输出", PALETTE["pred"], PALETTE["pred_s"])
    outputs = [("occupancy", "#93C5FD"), ("motion flow", "#FCA5A5"), ("semantic map", "#A7F3D0"), ("novel view", "#FDE68A")]
    for i, (name, fill) in enumerate(outputs):
        xx = 1430 + i * 170
        svg.rect(xx, 225, 135, 90, "#FFFFFF", PALETTE["pred_s"], 1.1, rx=10)
        draw_bev(svg, xx + 10, 240, 115, 52, name)
        svg.text(xx + 67, 342, name, 17, PALETTE["ink"], 800, anchor="middle")
    svg.text(1430, 370, "multi-task heads share the accumulated state", 17, PALETTE["muted"], 500)

    panel(svg, 425, 455, 1285, 320, "(e)", "Feed-forward Accumulative State Update", "#FFFFFF", PALETTE["purple_s"])
    core = [
        ("ego-motion align", "#DBEAFE", "#2563EB", 465),
        ("observation inject", "#E0F2FE", "#0891B2", 710),
        ("uncertainty decay", "#FEF3C7", "#D97706", 955),
        ("state refine", "#DCFCE7", "#16A34A", 1200),
        ("single-pass readout", "#FEE2E2", "#DC2626", 1445),
    ]
    for i, (name, fill, stroke, xx) in enumerate(core):
        svg.rect(xx, 545, 190, 100, fill, stroke, 1.5, rx=14)
        if name == "single-pass readout":
            svg.text(xx + 95, 582, "single-pass", 18, PALETTE["ink"], 800, anchor="middle")
            svg.text(xx + 95, 608, "readout", 18, PALETTE["ink"], 800, anchor="middle")
        else:
            svg.text(xx + 95, 588, name, 19, PALETTE["ink"], 800, anchor="middle")
        svg.text(xx + 95, 620, ["T_ego", "Obs_t", "sigma_t", "U(.)", "Y_t"][i], 18, PALETTE["muted"], 600, anchor="middle", extra='class="mono"')
        if i < len(core) - 1:
            svg.line(xx + 190, 595, core[i + 1][3] - 8, 595, PALETTE["line"], 3.2)
    svg.rect(520, 675, 1075, 58, "#F8FAFC", "#CBD5E1", 1.0, rx=12, dash="8 8")
    svg.text(545, 712, "G_t = U(G_{t-1}, p_t, Obs_t, sigma_t), no full history replay", 19, PALETTE["ink"], 600, extra='class="mono"')
    svg.path("M645,390 C650,430 585,455 585,540", PALETTE["sensor_s"], 2.8, arrow=True)
    svg.path("M1130,390 C1130,430 1045,455 1045,540", PALETTE["memory_s"], 2.8, arrow=True)
    svg.path("M1538,645 C1570,690 1590,735 1618,780 C1648,835 1668,845 1710,800", PALETTE["pred_s"], 2.6, arrow=True, dash="8 8")

    panel(svg, 1760, 455, 380, 320, "(f)", "约束与优势", PALETTE["gaussian"], PALETTE["gaussian_s"])
    for i, (name, col) in enumerate(
        [("bounded state", "#16A34A"), ("no replay", "#2563EB"), ("geo prior", "#D97706"), ("uncertainty aware", "#DC2626")]
    ):
        pill(svg, 1800, 530 + i * 52, 290, 38, name, "#FFFFFF", col, 17)
    svg.text(1800, 742, "causal time band + feed-forward core", 17, PALETTE["muted"], 500)

    panel(svg, 60, 855, 2080, 260, "(g)", "连续时间轴：history / current / future", "#FFFFFF", "#CBD5E1")
    timeline_x = [150, 465, 780, 1095, 1410, 1725]
    timeline_t = ["t-3", "t-2", "t-1", "t", "t+1", "t+2"]
    for i, (xx, tt) in enumerate(zip(timeline_x, timeline_t)):
        draw_bev(svg, xx, 925, 205, 105, tt, mode="road")
        color = PALETTE["sensor_s"] if tt == "t" else (PALETTE["pred_s"] if "+" in tt else PALETTE["gray_s"])
        pill(svg, xx + 42, 1046, 120, 34, "current" if tt == "t" else ("future" if "+" in tt else "history"), "#FFFFFF", color, 15)
        svg.text(xx + 102, 904, tt, 22, color, 800, anchor="middle", extra='class="mono"')
        if i < len(timeline_x) - 1:
            svg.line(xx + 205, 978, timeline_x[i + 1] - 8, 978, PALETTE["line"], 3)
    svg.path("M1198,855 C1198,820 1120,790 1120,735", PALETTE["purple_s"], 2.4, arrow=True, dash="8 8")
    svg.text(95, 1094, "Causal streaming state carries compact history while current observation updates future perception.", 20, PALETTE["muted"], 600)

    svg.line(370, 455, 425, 595, PALETTE["gray_s"], 2.6, arrow=True, dash="8 8", opacity=0.75)
    svg.line(865, 275, 910, 275, PALETTE["line"], 3.2)
    svg.line(1345, 275, 1390, 275, PALETTE["line"], 3.2)
    return svg.final()


def write_assets() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assets = {
        "fig1_v3_geo_sensor_4dgs": fig1(),
        "fig2_v3_geo_addressed_memory": fig2(),
        "fig3_v3_feedforward_streaming": fig3(),
    }
    for stem, content in assets.items():
        (OUT_DIR / f"{stem}.svg").write_text(content, encoding="utf-8")

    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        print("SVG files written. Chrome not found; PNG export skipped.")
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
                f"--window-size={W},{H}",
                f"--force-device-scale-factor={scale}",
                f"--screenshot={out_path}",
                svg_path.as_uri(),
            ]
            env = os.environ.copy()
            env.setdefault("LANG", "zh_CN.UTF-8")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)


if __name__ == "__main__":
    write_assets()
    print(f"wrote assets to {OUT_DIR}")
