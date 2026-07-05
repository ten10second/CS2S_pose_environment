#!/usr/bin/env python3
"""Raster PNG renderer for V4 backbone figures.

This is intentionally independent from Chrome/SVG conversion because the
current sandbox may block headless Chrome. It draws a paper-backbone style
directly with PIL.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 2400, 1350
OUT_DIR = Path("/home/shizhm/codespace/CS2S_pose_environment/research_figures/opening_report_backbone_v4")

REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

COL = {
    "ink": "#17202A",
    "muted": "#5C6675",
    "line": "#93A1B2",
    "bg": "#FFFFFF",
    "panel": "#F8FAFC",
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
    "cyan": "#0891B2",
    "cyan_l": "#ECFEFF",
    "gray": "#64748B",
    "gray_l": "#F8FAFC",
}


class Canvas:
    def __init__(self, scale: int = 1) -> None:
        self.s = scale
        self.img = Image.new("RGB", (W * scale, H * scale), COL["bg"])
        self.d = ImageDraw.Draw(self.img)

    def sc(self, v: float) -> int:
        return int(round(v * self.s))

    def font(self, size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
        path = MONO if mono else (BOLD if bold else REG)
        return ImageFont.truetype(path, self.sc(size))

    def text(self, x: float, y: float, s: str, size: int = 18, fill: str = COL["ink"], bold: bool = False, mono: bool = False, anchor: str = "la") -> None:
        self.d.text((self.sc(x), self.sc(y)), s, font=self.font(size, bold, mono), fill=fill, anchor=anchor)

    def multiline(self, x: float, y: float, s: str, size: int = 15, fill: str = COL["muted"], bold: bool = False, mono: bool = False, spacing: int = 5) -> None:
        self.d.multiline_text(
            (self.sc(x), self.sc(y)),
            s,
            font=self.font(size, bold, mono),
            fill=fill,
            spacing=self.sc(spacing),
        )

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "#FFFFFF", outline: str = COL["line"], width: int = 2, r: int = 10) -> None:
        box = [self.sc(x), self.sc(y), self.sc(x + w), self.sc(y + h)]
        if hasattr(self.d, "rounded_rectangle"):
            self.d.rounded_rectangle(box, radius=self.sc(r), fill=fill, outline=outline, width=self.sc(width))
        else:
            self.d.rectangle(box, fill=fill, outline=outline)
            if width > 1:
                for i in range(1, self.sc(width)):
                    self.d.rectangle([box[0] + i, box[1] + i, box[2] - i, box[3] - i], outline=outline)

    def line(self, x1: float, y1: float, x2: float, y2: float, fill: str = COL["line"], width: int = 3, arrow: bool = True, dash: bool = False) -> None:
        p1 = (self.sc(x1), self.sc(y1))
        p2 = (self.sc(x2), self.sc(y2))
        if dash:
            self._dash_line(p1, p2, fill, self.sc(width))
        else:
            self.d.line([p1, p2], fill=fill, width=self.sc(width))
        if arrow:
            self._arrow_head(x1, y1, x2, y2, fill, max(8, width * 4))

    def _dash_line(self, p1: tuple[int, int], p2: tuple[int, int], fill: str, width: int) -> None:
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        step = self.sc(18)
        gap = self.sc(10)
        cur = 0
        while cur < length:
            end = min(cur + step, length)
            self.d.line(
                [(int(x1 + ux * cur), int(y1 + uy * cur)), (int(x1 + ux * end), int(y1 + uy * end))],
                fill=fill,
                width=width,
            )
            cur += step + gap

    def _arrow_head(self, x1: float, y1: float, x2: float, y2: float, fill: str, size: int) -> None:
        ang = math.atan2(y2 - y1, x2 - x1)
        sx = self.sc(size)
        tip = (self.sc(x2), self.sc(y2))
        left = (int(tip[0] - sx * math.cos(ang - 0.45)), int(tip[1] - sx * math.sin(ang - 0.45)))
        right = (int(tip[0] - sx * math.cos(ang + 0.45)), int(tip[1] - sx * math.sin(ang + 0.45)))
        self.d.polygon([tip, left, right], fill=fill)

    def save(self, path: Path) -> None:
        self.img.save(path)


def title(c: Canvas, cn: str, en: str) -> None:
    c.text(60, 52, cn, 31, COL["ink"], True)
    c.text(60, 88, en, 18, COL["muted"])
    c.line(60, 112, W - 60, 112, COL["line"], 2, arrow=False)


def group(c: Canvas, x: int, y: int, w: int, h: int, tag: str, name: str, color: str, fill: str = "#FFFFFF") -> None:
    c.rect(x, y, w, h, fill, color, 2, 12)
    c.text(x + 18, y + 27, tag, 17, color, True)
    c.text(x + 62, y + 27, name, 18, COL["ink"], True)


def block(c: Canvas, x: int, y: int, w: int, h: int, name: str, detail: str = "", color: str = COL["line"], fill: str = "#FFFFFF", size: int = 15) -> None:
    c.rect(x, y, w, h, fill, color, 2, 8)
    c.text(x + w / 2, y + 25, name, size, COL["ink"], True, anchor="ma")
    if detail:
        c.multiline(x + 12, y + 45, detail, 12, COL["muted"], mono=True, spacing=3)


def tensor(c: Canvas, x: int, y: int, w: int, h: int, name: str, shape: str, color: str, fill: str, depth: int = 3) -> None:
    for i in range(depth, 0, -1):
        c.rect(x + i * 6, y - i * 6, w, h, fill, color, 1, 6)
    c.rect(x, y, w, h, fill, color, 2, 6)
    c.text(x + w / 2, y + 25, name, 14, COL["ink"], True, anchor="ma")
    c.text(x + w / 2, y + h - 14, shape, 11, COL["muted"], False, True, anchor="ma")


def op_chain(c: Canvas, xs: list[int], y: int, names: list[str], color: str, fill: str) -> None:
    for i, (x, name) in enumerate(zip(xs, names)):
        block(c, x, y, 130, 54, name, "", color, fill, 13)
        if i < len(xs) - 1:
            c.line(x + 130, y + 27, xs[i + 1] - 10, y + 27, COL["line"], 3)


def dashed_back(c: Canvas, x1: int, y1: int, x2: int, y2: int) -> None:
    c.line(x1, y1, x2, y2, COL["gray"], 2, True, dash=True)


def fig1(c: Canvas) -> None:
    title(c, "研究内容一：GeoSensor4DGS 算法 Backbone", "World-frame 4DGS with satellite anchors, ENU coordinates, multi-sensor evidence, and Gaussian heads")

    group(c, 55, 150, 370, 1040, "(a)", "Inputs", COL["blue"], COL["gray_l"])
    tensor(c, 90, 210, 125, 74, "RGB cams", "M x 3 x H x W", COL["blue"], COL["blue_l"])
    tensor(c, 250, 210, 120, 74, "LiDAR/Depth", "N_pts / HxW", COL["cyan"], COL["cyan_l"])
    tensor(c, 90, 335, 125, 74, "Sat. tile", "z_xy + height", COL["green"], COL["green_l"])
    tensor(c, 250, 335, 120, 74, "Pose packet", "K,T,LLA", COL["orange"], COL["orange_l"])
    block(c, 90, 485, 280, 90, "Coordinate chain", "LLA -> ECEF -> ENU\nT_cam,T_lidar,T_ego", COL["orange"])
    block(c, 90, 625, 280, 88, "Static prior", "road/building/height\nsatellite branch", COL["green"])
    block(c, 90, 765, 280, 88, "Dynamic cues", "motion residual + mask\ninstance proposals", COL["red"])
    c.multiline(90, 960, "ego pose is query-only;\nGaussians live in ENU\nworld frame", 15)

    group(c, 475, 150, 610, 1040, "(b)", "Encoders + Geo-normalization", COL["blue"])
    rows = [
        ("Image tokens", "DINO/VGGT", "F_img", COL["blue"], COL["blue_l"], 220),
        ("Depth tokens", "LiDAR encoder", "F_geo", COL["cyan"], COL["cyan_l"], 375),
        ("Map tokens", "Satellite encoder", "F_map", COL["green"], COL["green_l"], 530),
    ]
    for name, enc, out, color, fill, y in rows:
        tensor(c, 520, y, 130, 70, name, "N x C", color, fill)
        block(c, 705, y - 5, 170, 80, enc, "FPN / MLP", color)
        tensor(c, 930, y, 90, 70, out, "N x C", color, fill)
        c.line(650, y + 35, 705, y + 35)
        c.line(875, y + 35, 930, y + 35)
    group(c, 520, 700, 500, 360, "(b1)", "World anchor construction", COL["orange"])
    op_chain(c, [560, 720, 880], 765, ["project", "sample", "anchor"], COL["orange"], COL["orange_l"])
    block(c, 560, 865, 200, 74, "Covariance gate", "pose noise -> Sigma_mu", COL["orange"])
    block(c, 810, 865, 170, 74, "Static/dynamic split", "mask + residual", COL["red"])
    c.text(560, 1000, "A_t={mu_i^ENU, f_i, sigma_i, source_i}", 15, COL["muted"], False, True)

    group(c, 1135, 150, 645, 1040, "(c)", "Gaussian token backbone", COL["purple"], COL["purple_l"])
    tensor(c, 1180, 225, 135, 72, "Anchor tokens", "N x (3+C)", COL["purple"], "#FFFFFF")
    block(c, 1370, 215, 190, 92, "Token init", "pos enc + source emb", COL["purple"])
    tensor(c, 1610, 225, 100, 72, "Z_0", "N x D", COL["purple"], "#FFFFFF")
    c.line(1315, 261, 1370, 261)
    c.line(1560, 261, 1610, 261)
    group(c, 1180, 365, 535, 315, "(c1)", "x L Gaussian world layers", COL["purple"])
    for y, name, side in [(430, "Self-Attn", "tokens"), (500, "Cross-Attn", "F_img/F_map"), (570, "FFN+AddNorm", "residual")]:
        block(c, 1230, y, 145, 46, name, "", COL["purple"], "#FFFFFF", 13)
        block(c, 1425, y, 145, 46, side, "", COL["blue"], COL["gray_l"], 13)
        c.line(1375, y + 23, 1425, y + 23)
    c.text(1230, 652, "Z_l = Phi_l(Z_{l-1},F_img,F_geo,F_map)", 14, COL["muted"], False, True)
    group(c, 1180, 750, 535, 295, "(c2)", "Prediction heads", COL["purple"])
    block(c, 1220, 815, 200, 82, "Geometry Head", "Delta mu, scale, quat\nSigma, depth residual", COL["orange"])
    block(c, 1475, 815, 200, 82, "Attribute Head", "alpha, color/SH\nsemantics, velocity", COL["green"])
    block(c, 1220, 940, 200, 68, "G_t^static", "world static", COL["green"], COL["green_l"])
    block(c, 1475, 940, 200, 68, "G_t^dynamic", "instance velocity", COL["red"], COL["red_l"])

    group(c, 1840, 150, 505, 1040, "(d)", "Rasterize + supervision", COL["gray"])
    block(c, 1885, 225, 185, 88, "World 4DGS", "G_t^S union G_t^D", COL["purple"])
    block(c, 2110, 225, 175, 88, "Query pose", "T_query,K", COL["orange"])
    c.line(2070, 269, 2110, 269)
    block(c, 1950, 390, 260, 86, "Differentiable rasterizer", "R(G_t,T_query)", COL["gray"], COL["gray_l"])
    c.line(2030, 313, 2030, 390)
    for x, name, color, fill in [(1885, "I_hat", COL["blue"], COL["blue_l"]), (2035, "D_hat", COL["cyan"], COL["cyan_l"]), (2185, "S_hat", COL["green"], COL["green_l"])]:
        block(c, x, 540, 110, 65, name, "", color, fill, 14)
    block(c, 1885, 700, 405, 175, "Losses", "L_rgb + L_depth + L_geo\n+ L_temp + L_mask", COL["gray"])
    block(c, 1885, 950, 405, 82, "Inference path", "memory write / novel view / occ cue", COL["purple"])
    c.multiline(1885, 1090, "satellite prior anchors static layout;\nlocal sensors correct geometry/dynamics", 14)

    c.line(425, 660, 475, 660)
    c.line(1085, 660, 1135, 660)
    c.line(1780, 660, 1840, 660)


def fig2(c: Canvas) -> None:
    title(c, "研究内容二：Geo GaussianMemory 算法 Backbone", "Geo-addressed bounded memory: static tiles, dynamic tracks, fusion, erasure, compression, and LOD")

    group(c, 55, 150, 350, 1040, "(a)", "Streaming short window", COL["blue"], COL["gray_l"])
    for i, tt in enumerate(["t-k", "...", "t-1", "t"]):
        y = 230 + i * 145
        tensor(c, 90, y, 100, 58, f"G_{tt}", "N x D", COL["purple"], "#FFFFFF", 2)
        block(c, 220, y - 2, 130, 62, "pose/key", "p_t,tile_id", COL["orange"], "#FFFFFF", 13)
        if i < 3:
            c.line(140, y + 70, 140, y + 115, COL["blue"], 2)
    block(c, 90, 840, 260, 82, "Window 4DGS producer", "GeoSensor4DGS\nstrictly feed-forward", COL["purple"])
    c.multiline(90, 1020, "short-window state is disposable;\nonly fused memory persists", 14)

    group(c, 455, 150, 505, 1040, "(b)", "Addressing + read", COL["green"])
    block(c, 500, 225, 180, 76, "ENU tiling", "floor(mu/r)", COL["green"], COL["green_l"])
    block(c, 735, 225, 170, 76, "Hash index", "tile -> slots", COL["green"], COL["green_l"])
    c.line(680, 263, 735, 263, COL["green"])
    block(c, 500, 365, 405, 92, "Memory read", "M_{t-1}[tile +/- radius]\nstatic slots + track slots", COL["green"])
    block(c, 500, 525, 185, 76, "Alignment", "warp by Delta T", COL["orange"])
    block(c, 720, 525, 185, 76, "Association", "Mahalanobis / IoU", COL["orange"])
    c.line(685, 563, 720, 563)
    block(c, 500, 680, 185, 84, "Consist. check", "free-space conflict\nmulti-view support", COL["red"])
    block(c, 720, 680, 185, 84, "Importance score", "visibility x age x task", COL["red"])
    c.text(500, 950, "key=(tile_x,tile_y,lod,layer)", 14, COL["muted"], False, True)
    c.text(500, 980, "value=Gaussian slots + metadata", 14, COL["muted"], False, True)

    group(c, 1015, 150, 620, 1040, "(c)", "Static tile update branch", COL["green"], COL["green_l"])
    tensor(c, 1055, 230, 120, 68, "old tile", "S x D", COL["green"], "#FFFFFF")
    tensor(c, 1210, 230, 120, 68, "new obs", "N x D", COL["blue"], COL["blue_l"])
    block(c, 1375, 220, 190, 88, "Change classifier", "keep/update\nemerge/erase", COL["red"])
    c.line(1175, 264, 1210, 264)
    c.line(1330, 264, 1375, 264)
    group(c, 1055, 365, 515, 330, "(c1)", "Local Gaussian operators", COL["green"])
    op_chain(c, [1090, 1250, 1410], 430, ["match", "fuse", "replace"], COL["green"], COL["green_l"])
    op_chain(c, [1090, 1250, 1410], 535, ["add", "erase", "prune"], COL["red"], COL["red_l"])
    c.text(1090, 650, "S'_tile=Compress(Fuse(S_tile,G_t),budget_tile)", 13, COL["muted"], False, True)
    group(c, 1055, 760, 515, 265, "(c2)", "Bounded maintenance", COL["gray"])
    block(c, 1090, 825, 130, 62, "LOD down", "far tiles", COL["gray"], COL["gray_l"], 13)
    block(c, 1250, 825, 130, 62, "merge", "similar GS", COL["gray"], COL["gray_l"], 13)
    block(c, 1410, 825, 130, 62, "evict", "low score", COL["gray"], COL["gray_l"], 13)
    c.text(1090, 965, "sum_i |G_i| <= B, per-tile quota + global quota", 13, COL["muted"], False, True)

    group(c, 1690, 150, 650, 1040, "(d)", "Dynamic instance memory branch", COL["red"], COL["red_l"])
    tensor(c, 1735, 230, 130, 68, "track bank", "K x T x D", COL["red"], "#FFFFFF")
    block(c, 1905, 220, 165, 88, "Predict", "x_t=x+v dt", COL["red"])
    block(c, 2110, 220, 170, 88, "Associate", "mask / 3D IoU", COL["red"])
    c.line(1865, 264, 1905, 264)
    c.line(2070, 264, 2110, 264)
    group(c, 1735, 365, 540, 245, "(d1)", "Track lifecycle", COL["red"])
    states = [("birth", COL["green"]), ("persist", COL["blue"]), ("split", COL["purple"]), ("vanish", COL["orange"]), ("retire", COL["red"])]
    for i, (name, color) in enumerate(states):
        x = 1770 + i * 100
        block(c, x, 450, 78, 54, name, "", color, "#FFFFFF", 12)
        if i < 4:
            c.line(x + 78, 477, x + 96, 477, COL["line"], 2)
    c.text(1770, 570, "dynamic memory is instance-level, not tile-only", 13, COL["muted"])
    group(c, 1735, 675, 540, 250, "(d2)", "Write policy", COL["gray"])
    for x, name, detail in [(1770, "TTL", "short-lived"), (1930, "velocity", "linear prior"), (2090, "compact", "top-k GS")]:
        block(c, x, 735, 130, 62, name, detail, COL["gray"], COL["gray_l"], 13)
    c.text(1770, 875, "D'_j=Update(D_j,G_t^inst,obs_support)", 14, COL["muted"], False, True)
    block(c, 1770, 990, 450, 80, "Memory readout", "local 4DGS for render/query/planning cue", COL["purple"])

    c.line(405, 660, 455, 660)
    c.line(960, 620, 1015, 620)
    c.line(1635, 620, 1690, 620)


def fig3(c: Canvas) -> None:
    title(c, "研究内容三：前馈流式累积增强感知 Backbone", "Weak-temporal streaming perception: world-state accumulation instead of full-history feature propagation")

    group(c, 55, 150, 330, 1040, "(a)", "Heavy temporal baseline", COL["gray"], COL["gray_l"])
    for i, name in enumerate(["F_{t-4}", "F_{t-3}", "F_{t-2}", "F_{t-1}"]):
        y = 230 + i * 105
        tensor(c, 100, y, 95, 54, name, "Cxhxw", COL["gray"], "#FFFFFF", 2)
        block(c, 225, y - 2, 95, 58, "cache", "attn", COL["gray"], "#FFFFFF", 12)
        if i < 3:
            c.line(148, y + 64, 148, y + 95, COL["gray"], 2)
    block(c, 100, 735, 230, 88, "Cost source", "full-history attention\nquery propagation", COL["gray"])
    c.multiline(100, 975, "contrast: compact world-state\nupdate replaces feature replay", 14)

    group(c, 445, 150, 500, 1040, "(b)", "Current observation encoder", COL["blue"])
    tensor(c, 485, 230, 125, 66, "I_t", "M x 3 x H x W", COL["blue"], COL["blue_l"])
    tensor(c, 645, 230, 115, 66, "D/L_t", "sparse", COL["cyan"], COL["cyan_l"])
    tensor(c, 795, 230, 95, 66, "p_t", "SE(3)", COL["orange"], COL["orange_l"])
    block(c, 500, 380, 170, 80, "Shared encoder", "CNN/ViT + FPN", COL["blue"])
    block(c, 715, 380, 170, 80, "Geometry lift", "K,T -> ENU", COL["orange"])
    tensor(c, 565, 560, 125, 68, "Obs_t", "N x D", COL["blue"], COL["blue_l"])
    tensor(c, 735, 560, 110, 68, "A_t", "N x 3", COL["orange"], COL["orange_l"])
    block(c, 565, 745, 280, 78, "Obs reliability", "visibility, mask support, pose cov", COL["red"])
    c.multiline(500, 995, "No multi-frame encoder;\ncurrent evidence enters once.", 14)

    group(c, 1000, 150, 815, 1040, "(c)", "Feed-forward state update layers", COL["purple"], COL["purple_l"])
    tensor(c, 1045, 220, 130, 70, "S_{t-1}", "N_s x D", COL["green"], COL["green_l"])
    tensor(c, 1215, 220, 125, 70, "Obs_t", "N_o x D", COL["blue"], COL["blue_l"])
    tensor(c, 1380, 220, 105, 70, "A_t", "N_o x 3", COL["orange"], COL["orange_l"])
    block(c, 1530, 210, 210, 90, "Update inputs", "state + observation + pose", COL["purple"])
    c.line(1175, 255, 1215, 255)
    c.line(1340, 255, 1380, 255)
    c.line(1485, 255, 1530, 255)
    group(c, 1045, 365, 695, 390, "(c1)", "x L streaming update block", COL["purple"])
    modules = [
        (1080, 430, "Ego align", "S^-=Warp(S,DeltaT)", COL["orange"], COL["orange_l"]),
        (1285, 430, "Obs cross-attn", "Attn(S^-,Obs_t)", COL["blue"], COL["blue_l"]),
        (1490, 430, "Geo gate", "vis,cov,age", COL["green"], COL["green_l"]),
        (1180, 550, "Uncertainty", "sigma_t=decay+res", COL["red"], COL["red_l"]),
        (1385, 550, "FFN refine", "S_t=AddNorm(...)", COL["purple"], COL["purple_l"]),
    ]
    for x, y, name, detail, color, fill in modules:
        block(c, x, y, 170, 74, name, detail, color, fill, 13)
    c.line(1250, 467, 1285, 467)
    c.line(1455, 467, 1490, 467)
    c.line(1575, 504, 1265, 550)
    c.line(1350, 587, 1385, 587)
    c.text(1080, 705, "single-pass; no replay of {F_{t-k},...,F_{t-1}}", 14, COL["muted"], False, True)
    group(c, 1045, 830, 695, 230, "(c2)", "Theoretical handles", COL["gray"])
    block(c, 1085, 900, 180, 62, "stability", "||e_t|| <= rho||e_t-1||+eta", COL["gray"], COL["gray_l"], 12)
    block(c, 1300, 900, 180, 62, "bounded error", "pose/noise terms", COL["gray"], COL["gray_l"], 12)
    block(c, 1515, 900, 180, 62, "latency", "O(|S|+|Obs|)", COL["gray"], COL["gray_l"], 12)
    c.text(1085, 1025, "Temporal gain comes from persistent world state, not a large temporal net.", 13, COL["muted"])

    group(c, 1870, 150, 470, 1040, "(d)", "Perception decoders + feedback", COL["red"])
    tensor(c, 1915, 230, 125, 70, "S_t", "N_s x D", COL["purple"], COL["purple_l"])
    block(c, 2075, 220, 190, 88, "GS-to-BEV/Occ", "rasterize or pool", COL["red"])
    c.line(2040, 264, 2075, 264)
    group(c, 1915, 390, 350, 300, "(d1)", "Task heads", COL["red"], COL["red_l"])
    for x, y, name, detail, color, fill in [
        (1950, 455, "Occupancy", "P(occ)", COL["blue"], COL["blue_l"]),
        (2105, 455, "Semantic", "P(cls)", COL["green"], COL["green_l"]),
        (1950, 555, "Motion", "flow/v", COL["orange"], COL["orange_l"]),
        (2105, 555, "Uncertainty", "sigma", COL["red"], COL["red_l"]),
    ]:
        block(c, x, y, 130, 62, name, detail, color, fill, 13)
    group(c, 1915, 745, 350, 205, "(d2)", "Losses", COL["gray"])
    c.multiline(1950, 810, "L = L_occ + L_sem + L_flow\n  + lambda L_render + beta L_stab", 14, COL["muted"], mono=True)
    block(c, 1950, 1015, 285, 72, "Feedback to memory", "write compact S_t; discard raw history", COL["purple"])

    c.line(385, 660, 445, 660)
    c.line(945, 660, 1000, 660)
    c.line(1815, 660, 1870, 660)


def render_all(scale: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = [
        ("fig1_backbone_v4_geosensor4dgs", fig1),
        ("fig2_backbone_v4_geo_gaussian_memory", fig2),
        ("fig3_backbone_v4_feedforward_streaming", fig3),
    ]
    for stem, fn in figs:
        c = Canvas(scale)
        fn(c)
        suffix = "_2x.png" if scale == 2 else ".png"
        c.save(OUT_DIR / f"{stem}{suffix}")


if __name__ == "__main__":
    render_all(1)
    render_all(2)
    print(f"wrote PIL PNG assets to {OUT_DIR}")
