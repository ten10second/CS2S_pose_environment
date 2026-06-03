# CS2S + KITTI Raw Dynamic LiDAR Plan

## 0. Current Direction

当前研究方向改为单阶段动态 LiDAR 几何注入：

```text
satellite static prior + projected dynamic LiDAR geometry
-> single-stage diffusion denoising
-> RGB street view with dynamic objects
```

不再追两条旧路线：

- 不做局部后处理。
- 不做单层弱控制分支。

当前要证明的不是 LiDAR 直接解码 RGB，而是：

```text
冻结的 satellite-to-street diffusion prior 提供街景外观分布；
projected dynamic LiDAR geometry 改变局部 denoising trajectory；
动态目标由 LiDAR presence / geometry 激活，由 diffusion prior 补 appearance。
```

## 1. Local Dataset Facts

- KITTI raw root: `/media/shizhm/Lenovo/KITTI_RAW`
- 可用环境: `ControlS2S`
- 当前机器 GPU: RTX 4090 24GB
- 当前首批优先日期: `2011_09_26`
- 当前数据中找到 `36` 个带 `tracklet_labels.xml` 的 drive segment，均在 `2011_09_26` 下。
- `2011_09_29` 下目前没有找到 `tracklet_labels.xml`，可做 raw LiDAR projection，但不能直接从 XML 自动筛 dynamic points。

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
动态车、人、骑行者不能指望从 satellite-only 稳定出现。
```

## 3. Dynamic LiDAR-RGB Alignment Contract

动态目标来自每个 drive segment 内的 `tracklet_labels.xml`。

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

动态类别第一版定义为 movable foreground：

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
  - point-in-box dynamic LiDAR filtering
  - LiDAR condition generation
- `dataloader/KITTI_raw_sat_lidar.py`
  - KITTI raw satellite + image_02 + LiDAR dataset
  - 返回原 CS2S 字段 plus LiDAR/dynamic fields
- `tools/build_kitti_raw_sat_lidar_manifest.py`
  - 按 date/drive segment 生成 manifest
  - `--require-tracklet` 只保留有 XML 的 segment
- `tools/check_kitti_tracklets.py`
  - 检查 XML 展帧、投影框、dynamic points、overlay
- `tools/smoke_kitti_raw_sat_lidar.py`
  - dataloader / model backward smoke
- `tools/train_kitti_raw_sat_lidar_control.py`
  - 单卡训练入口
  - 当前只走 multi-scale LiDAR control
- `tools/eval_kitti_raw_sat_lidar_runs.py`
  - 固定样本评估、动态区域 metrics、LiDAR overlay
- `models/KITTI_geo_ldm/lidar_condition_model.py`
  - `LidarMultiScaleControl`
- `configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_dynamic.yaml`
  - 当前 LiDAR dynamic config

当前已删除的旧入口：

- 局部后处理相关脚本。
- 单层弱控制分支类和训练参数。

## 5. Current LiDAR Condition

当前 `lidar_cond` 为 4 通道：

1. `dyn_box_mask`
2. `dyn_point_mask`
3. `dyn_depth_norm`
4. `dyn_class_norm`

输出尺寸：

```text
H = 128
W = 512
```

同时返回：

- `dynamic_mask`
- `dynamic_boxes`
- `dynamic_box_valid`
- `lidar_valid_mask`
- `dynamic_class_hist`
- `num_dynamic_boxes`
- `num_projected_lidar_points`
- `num_projected_dynamic_points`

注意：下一轮 plan 需要重新审视这个 4ch 表征是否过于弱或过于语义化。若坚持“直接从 LiDAR 几何做”，应考虑把输入改成更纯的 LiDAR geometry field，例如:

- dynamic point occupancy
- metric depth / inverse depth
- camera-frame xyz
- reflectance / intensity
- local point density
- distance transform / splatted confidence

## 6. Current Model Path

当前有效模型路径：

```text
lidar_cond
-> LidarMultiScaleControl
-> multi-scale residuals for UNet skips + middle feature
-> frozen or lightly-unfrozen satellite-to-street diffusion prior
```

`LidarMultiScaleControl` 输出 residual list，UNet 在 denoising 中把这些 residual 加到 feature 上。动态目标注入不是贴图，而是改变 denoising 方向：

```text
noise + satellite condition -> static denoising trajectory
noise + satellite condition + LiDAR residuals -> dynamic-object trajectory
```

下一轮重点不是恢复旧实验，而是设计：

- 更强的 LiDAR geometry representation
- mask-gated multi-scale residual injection
- dynamic-region weighted objective
- static background preservation objective
- 不依赖 text/object token 的动态目标激活实验

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
  --condition-mode dynamic_full \
  --batch-size 2
```

Model backward smoke:

```bash
conda run -n ControlS2S python tools/smoke_kitti_raw_sat_lidar.py \
  --condition-mode dynamic_full \
  --batch-size 1 \
  --model-backward
```

## 8. Next Plan Questions

新的 plan 应先回答这些问题：

1. 是否彻底去掉 `dyn_class_norm`，只保留 LiDAR geometry。
2. 是否用 tracklet XML 只做 dynamic filtering / loss mask，而不把类别属性输入模型。
3. LiDAR sparse points 应该如何 splat 成连续 field。
4. residual 是否需要 dynamic mask gating，避免破坏静态背景。
5. 评估成功标准应偏 object presence / geometry alignment，而不是 GT 颜色纹理重建。
