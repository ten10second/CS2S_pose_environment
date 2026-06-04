# CS2S + KITTI Raw Geometry LiDAR Plan

## 0. Current Direction

当前研究方向改为单阶段 raw LiDAR 几何注入：

```text
satellite static prior + projected raw LiDAR geometry
-> single-stage diffusion denoising
-> RGB street view with current-frame foreground objects
```

不再追两条旧路线：

- 不做局部后处理。
- 不做单层弱控制分支。

当前要证明的不是 LiDAR 直接解码 RGB，而是：

```text
冻结的 satellite-to-street diffusion prior 提供街景外观分布；
projected raw LiDAR geometry 改变局部 denoising trajectory；
当前帧前景物体由 raw LiDAR presence / geometry 激活，由 diffusion prior 补 appearance。
```

## 1. Local Dataset Facts

- KITTI raw root: `/media/shizhm/Lenovo/KITTI_RAW`
- 可用环境: `ControlS2S`
- 当前机器 GPU: RTX 4090 24GB
- 当前首批优先日期: `2011_09_26`
- 当前数据中找到 `36` 个带 `tracklet_labels.xml` 的 drive segment，均在 `2011_09_26` 下。
- `2011_09_29` 下目前没有找到 `tracklet_labels.xml`，可做 raw LiDAR projection，但不能直接从 XML 自动筛 object points。

KITTI raw segment 内可用字段：

- `image_02/data/*.png`
- `satellite/*.png`
- `velodyne_points/data/*.bin`
- `oxts/data/*.txt`
- 部分 drive 根目录有 `tracklet_labels.xml`

## 2. Static Satellite Path

原 CS2S KITTI 路径仍作为静态街景 prior：

- 原配置: `configs/Boost_Sat2Den/KITTI_geo_ldm.yaml`
- 原训练配置: `configs/Boost_Sat2Den/train/KITTI_geo_ldm.yaml`
- 原 dataloader: `dataloader.KITTI_wo_loc.SatGrdDataset`
- satellite 默认北向。
- 原 dataloader 会根据 OXTS heading 旋转 satellite map 到相机方向，再做 camera GPS shift 和 center crop。

重要含义：

```text
satellite-only 主要学静态场景；
当前帧车辆、行人、骑行者等前景物体不能指望从 satellite-only 稳定出现。
```

## 3. Object LiDAR-RGB Alignment Contract

前景可移动物体候选来自每个 drive segment 内的 `tracklet_labels.xml`。

这里的 object / foreground 不是“正在运动”的物理定义，而是：

```text
satellite map 中不稳定存在、但当前相机/LiDAR 帧中真实可见的 movable objects。
```

因此停在路边的车、等红灯的车、静止行人等，只要有 tracklet 3D box 且 LiDAR 点落入 box，也应被视为 object LiDAR geometry。

每个 tracklet item 读取：

- `objectType`
- `h`, `w`, `l`
- `first_frame`
- `poses/item/tx`
- `poses/item/ty`
- `poses/item/tz`
- `poses/item/rx`
- `poses/item/ry`
- `poses/item/rz`

实际帧号：

```text
frame_idx = first_frame + pose_idx
```

object 类别第一版定义为 movable foreground：

- `Car`
- `Van`
- `Truck`
- `Pedestrian`
- `Person (sitting)`
- `Cyclist`
- `Tram`

XML pose 使用 Velodyne 坐标：

- `x`: forward
- `y`: left
- `z`: up
- `rz`: yaw around Velodyne z-axis

当前按底部参考点处理：

```text
box_center_z = tz + h / 2
```

LiDAR point-in-box：

```text
center = [tx, ty, tz + h / 2]
p_local = Rz(-rz) * (p_velo - center)

inside =
  abs(p_local.x) <= l / 2
  abs(p_local.y) <= w / 2
  abs(p_local.z) <= h / 2
```

Velodyne 到 RGB `image_02` 投影：

```text
pixel = P_rect_02 * R_rect_00 * Tr_velo_to_cam * point_velodyne
```

只保留：

- depth > 0
- `0 <= u < image_width`
- `0 <= v < image_height`

## 4. Active Code Paths

当前保留的有效入口：

- `dataloader/kitti_raw_lidar_utils.py`
  - XML tracklet parser
  - KITTI raw calibration loader
  - Velodyne -> `image_02` projection
  - point-in-box object LiDAR filtering
  - LiDAR condition generation
- `dataloader/KITTI_raw_sat_lidar.py`
  - KITTI raw satellite + image_02 + LiDAR dataset
  - 返回原 CS2S 字段 plus LiDAR/object fields
- `tools/build_kitti_raw_sat_lidar_manifest.py`
  - 按 date/drive segment 生成 manifest
  - `--require-tracklet` 只保留有 XML 的 segment
- `tools/check_kitti_tracklets.py`
  - 检查 XML 展帧、投影框、object points、overlay
- `tools/smoke_kitti_raw_sat_lidar.py`
  - dataloader / model backward smoke
- `tools/train_kitti_raw_sat_lidar_control.py`
  - 单卡训练入口
  - 当前只走 multi-scale LiDAR control
- `tools/eval_kitti_raw_sat_lidar_runs.py`
  - 固定样本评估、object 区域 metrics、LiDAR overlay
- `models/KITTI_geo_ldm/lidar_condition_model.py`
  - `LidarMultiScaleControl`
- `configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_dynamic.yaml`
  - 当前 LiDAR object config

当前已删除的旧入口：

- 局部后处理相关脚本。
- 单层弱控制分支类和训练参数。

## 5. Current Raw LiDAR Condition

注意：当前代码字段名仍沿用历史 `dynamic_*` / `dyn_*` 命名；在本计划中它们语义上应理解为 movable foreground / object，而不是必须正在运动。

当前主线 `condition_mode=raw_lidar_geometry` 输出 10 通道：

1. `confidence`
2. `x_cam_norm`
3. `y_cam_norm`
4. `z_cam_norm`
5. `near_depth`
6. `intensity`
7. `local_density`
8. `sparse_seed_mask`
9. `depth_edge`
10. `final_gate`

XML object-only 的 `dynamic_geometry` 输出 8 通道，保留为 sanity / upper-bound 对照。


输出尺寸：

```text
H = 128
W = 512
```

同时返回：

- `dynamic_mask`，历史命名；语义为 object/foreground mask
- `dynamic_boxes`，历史命名；语义为 object/foreground boxes
- `dynamic_box_valid`，历史命名
- `lidar_valid_mask`
- `dynamic_class_hist`，历史命名
- `num_dynamic_boxes`，历史命名；语义为 object box 数量
- `num_projected_lidar_points`
- `num_projected_dynamic_points`，历史命名；语义为 projected object LiDAR points

当前主线坚持“直接从 LiDAR 几何做”，输入应是纯 LiDAR geometry field，不把类别/text/object token 作为模型条件。核心几何分量包括：

- object point occupancy
- metric depth / inverse depth
- camera-frame xyz
- reflectance / intensity
- local point density
- distance transform / splatted confidence
- depth edge / geometry saliency
- final residual gate

## 6. PointPillars-like BEV Geometry Support

新增实验分支：

```text
condition_mode=raw_lidar_bev_geometry
```

目标不是引入类别检测器，而是让 raw LiDAR 先在更自然的 BEV / pillar 空间里形成几何 support，再投回 RGB 对齐 condition。

输出尺寸：

```text
lidar_cond.shape = [16, 128, 512]
```

前 10 通道完全复用 `raw_lidar_geometry`：

1. `confidence`
2. `x_cam_norm`
3. `y_cam_norm`
4. `z_cam_norm`
5. `near_depth`
6. `intensity`
7. `local_density`
8. `sparse_seed_mask`
9. `depth_edge`
10. `raw_final_gate`

新增 6 个 BEV / pillar 通道：

11. `pillar_objectness`
12. `pillar_non_ground`
13. `pillar_height_range`
14. `pillar_density`
15. `pillar_compactness`
16. `pillar_support_gate`

其中 `pillar_support_gate` 是新的 residual gate：

```text
gate_channel = 15
```

处理逻辑：

- 使用 rectified camera-frame `x-z` 平面做 BEV pillarization。
- 每个 pillar 统计点数、局部 `y` 高度范围、密度。
- 根据高度范围 / 点数筛出非平坦几何候选。
- 用 BEV 连通域过滤过长墙面 / 大片背景结构，只保留 compact support。
- 将 per-point pillar feature 投影回 RGB，并做小半径 nearest-fill 生成 `pillar_support_gate`。

当前 B64 object-rich 诊断集上 gate 覆盖率：

```text
raw_lidar_geometry final_gate > 0: about 65%
raw_lidar_bev_geometry support_gate > 0: about 25%
```

已完成 smoke：

- condition shape `[16,128,512]`
- `LidarMultiScaleControl(in_channels=16, gate_channel=15)` forward 输出 13 个 residual
- zero gate residual max abs = 0
- model backward loss finite
- 2-step train smoke loss finite，`static_teacher_gate_channel=15`

## 6. Current Model Path

当前有效模型路径：

```text
lidar_cond
-> LidarMultiScaleControl
-> multi-scale residuals for UNet skips + middle feature
-> frozen or lightly-unfrozen satellite-to-street diffusion prior
```

`LidarMultiScaleControl` 输出 residual list，UNet 在 denoising 中把这些 residual 加到 feature 上。前景物体注入不是贴图，而是改变 denoising 方向：

```text
noise + satellite condition -> static denoising trajectory
noise + satellite condition + LiDAR residuals -> foreground-object trajectory
```

下一轮重点不是恢复旧实验，而是设计：

- 更强的 LiDAR geometry representation
- mask-gated multi-scale residual injection
- object-region weighted objective
- static background preservation objective
- 不依赖 text/object token 的前景物体激活实验
- raw LiDAR 全量输入 + `final_gate` 控制 residual 修改权限
- satellite-only epsilon teacher consistency 保护非 gate 区域

## 7. Minimal Verified Commands

Manifest generation:

```bash
conda run -n ControlS2S python tools/build_kitti_raw_sat_lidar_manifest.py \
  --kitti-root /media/shizhm/Lenovo/KITTI_RAW \
  --date 2011_09_26 \
  --out-dir dataset/kitti_raw_sat_lidar \
  --val-drives \
    2011_09_26_drive_0002_sync \
    2011_09_26_drive_0014_sync \
    2011_09_26_drive_0020_sync \
    2011_09_26_drive_0029_sync \
    2011_09_26_drive_0046_sync \
    2011_09_26_drive_0059_sync \
    2011_09_26_drive_0079_sync \
    2011_09_26_drive_0093_sync
```

XML / projection check:

```bash
conda run -n ControlS2S python tools/check_kitti_tracklets.py \
  --drive /media/shizhm/Lenovo/KITTI_RAW/2011_09_26/2011_09_26_drive_0005_sync \
  --frame-id 0000000000 \
  --output-overlay results/sat_lidar_dynamic/smoke/drive0005_frame0000000000_overlay.png
```

Data smoke:

```bash
conda run -n ControlS2S python tools/smoke_kitti_raw_sat_lidar.py \
  --condition-mode raw_lidar_geometry \
  --batch-size 2
```

Model backward smoke:

```bash
conda run -n ControlS2S python tools/smoke_kitti_raw_sat_lidar.py \
  --condition-mode raw_lidar_geometry \
  --batch-size 1 \
  --model-backward
```

## 8. Next Plan Questions

新的 plan 应先回答这些问题：

1. 是否彻底去掉 `dyn_class_norm`，只保留 LiDAR geometry。
2. 是否用 tracklet XML 只做 object filtering / loss mask，而不把类别属性输入模型。
3. LiDAR sparse points 应该如何 splat 成连续 field。
4. residual 是否需要 object mask/confidence gating，避免破坏静态背景。
5. 评估成功标准应偏 object presence / geometry alignment，而不是 GT 颜色纹理重建。
