> 历史实验记录：以下配置、结论和命令对应当时版本，不是当前启动指南。
> 当前方案见 [时序说明](temporal_history.md) 和 [全量 B/C 协议](temporal_full_bc.md)。
> 部分旧实验脚本已退役，原源码保留于服务器 `CS2S_run_control/temporal_cleanup_commit_20260920/before_source.tar.gz`。

# 建筑与背景：稠密历史重投影验证

## 结论

**支持将稠密重投影作为带有效性标记的历史参考继续做小样本实验；不支持把所有稠密对应直接作为硬拷贝或强像素监督。**

这次只验证几何与可见历史，不经过生成模型、没有训练。它证明能向建筑/背景提供更多且多数位置更准确的历史参考，尚未证明 decoder 会利用它，也未证明生成视频已经稳定。

## 实验设置

- 固定上一轮的 10 对相邻帧（原 train 清单 6 对、heldout 清单 4 对），沿用 GPS30m 清单，不改划分。
- 前帧 RGB 估计深度：Depth Anything V2 Metric VKITTI Small，原生图像输入，input_size=518。前帧 LiDAR 的固定 80% 投影点拟合全局中位比例；剩余 20% 用于源深度诊断。
- 候选方法：原始深度、比例校准深度、校准后在源 LiDAR 像素使用实测深度。最终候选为第三种。比例拟合不读取后帧 RGB 或后帧 LiDAR。
- 完整自车位姿、标定和 P_rect_02 投影，z-buffer 处理前向可见性，后帧 LiDAR 排除已知冲突。深度容差沿用 0.75m；尚未证明最优。
- 后帧 RGB 在 reference.npz 写出后才读取，仅用于图像误差和 SIFT 对应评估。没有用后帧图像预测深度、估计光流或构造历史。
- 测试 512×128（原生成分辨率）和 1024×256（细节诊断）。原生深度缓存复用，没有额外训练。

## 覆盖：新增部分主要是估计支持

512×128，10 对的逐对均值：

| 指标 | 原稀疏参考 | 稠密＋LiDAR 候选 |
|---|---:|---:|
| 两帧深度检查通过的覆盖 | 10.16% | 10.12% |
| 可提供参考的位置（稠密列含未验证估计） | 10.16% | 72.65% |
| 图像上半部参考覆盖（同样含估计） | 5.19% | 81.77% |

这里的上半部是固定图像分区，包含建筑、天空、植被，**不是建筑语义分割**。新增覆盖不能解释为新增实测几何，深度通过也不等于已确认静态。

## 对齐精度：同一批独立特征对应

在两帧 GT 上做 SIFT 双向比例匹配，按已知位姿的对极约束筛选；各方法使用完全相同的匹配点和有效集合，不按某个方法的误差挑点。真实匹配仍可能包含重复窗户、植被和运动目标，指标是诊断代理，不是精确三维真值。

1024×256 图像上半部，共 3660 个保留匹配点，合并统计：

| 方法 | 中位误差 px | P90 误差 px | ≤2px 比例 |
|---|---:|---:|---:|
| 原位置不动 | 8.41 | 32.47 | 8.5% |
| 纯旋转近似 | 2.94 | 16.19 | 40.2% |
| 单目深度＋完整位姿 | 1.10 | 7.03 | 66.6% |
| 加前帧 LiDAR 比例校准 | 1.07 | 6.87 | 68.4% |
| 再使用源 LiDAR 实测深度 | 1.06 | 6.83 | 69.0% |

在源像素没有 LiDAR 的 5801 个匹配上，中位误差从纯旋转的 5.01px 降到 1.16px。这说明改善包含 LiDAR 未采到的位置，并非只重画已有点。

## 逐对结果与失败边界

| 帧对 | 稠密支持覆盖（512） | 上半部旋转误差中位（1024 px） | 上半部候选误差中位（1024 px） |
|---|---:|---:|---:|
| train_01 | 65.9% | 5.89 | 0.73 |
| train_05 | 74.6% | 2.28 | 2.18 |
| train_06 | 69.0% | 3.98 | 1.27 |
| train_07 | 80.8% | 3.01 | 1.05 |
| train_09 | 72.9% | 1.74 | 1.14 |
| train_15 | 67.7% | 2.26 | 0.75 |
| heldout_00 | 70.0% | 5.85 | 5.33 |
| heldout_01 | 76.4% | 4.34 | 0.85 |
| heldout_02 | 82.7% | 1.51 | 0.33 |
| heldout_06 | 66.6% | 5.16 | 0.88 |

- 所有 10 对的上半部中位误差都低于纯旋转，但 train_05 和 heldout_00 的改善很小。
- heldout_00 仍有 5.33px 中位误差，P90 9.70px；候选比未校准单目深度的 5.15px 略差。全局 LiDAR 比例校准不能保证高层/远处每个表面都更准确。
- 合并上半部约 31% 匹配误差超过 2px，P90 为 6.83px。不能把这些参考强制复制到目标中。
- 可视化中的黑色网格是逐像素前向投影放大造成的未命中位置；没有补洞、没有把目标 GT 填回参考，不是生成模型产生的模糊。灰色叠加图包含目标 GT，只用于查看错位，不能作为推理效果图。
- 人工查看 heldout_01、train_01 的立面窗户，大体跟随自车变化；heldout_00 的树木边界和部分结构仍可见重影。全量对照均保留在交互页面，未只选成功例。
- 缺少可靠动态实例排除和稠密可见性真值；车辆不在这次结论范围内。

## 下一步建议

1. 保留独立的实测验证/估计支持/已知冲突标记，将候选作为软历史条件；原实测可靠对应应优先保留。
2. 历史编码按支持归一化处理投影空洞，验证高频立面纹理是否进入特征；不能把黑色占位当作建筑颜色。
3. 接入现有静态分支做匹配小样本验证，以建筑外观继承、目标几何和清晰度共同验收。若参考正确但生成仍换立面，再定位网络利用历史的问题。
4. 不启动全量训练、不添加基于所有估计像素的强一致性损失。较大深度模型、表面补全或动态关联需要独立验证；本轮只测试了 Small 这一候选。

## 复现与代码

服务器仓库：`/mnt/shizhm/CS2S_pose_environment_temporal_v22`。
输出：`/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/dense_reprojection_20260919`。
控制记录/日志：`/mnt/shizhm/CS2S_run_control/temporal_static_p0p2_20260919/dense_probe.log` 和 `dense_probe512.log`。
新增代码：`tools/temporal_dense_reprojection.py`、`tools/temporal_dense_eval.py`、`tools/validate_dense_static_reference.py` 及两份对应测试。没有修改训练、单帧或已有静态模块。
13 个几何/评估测试通过，Python 编译和 diff 检查通过，两次 10 对推理验证正常结束。官方权重 SHA256 校验通过；环境未安装新 Python 包。
模型代码 commit：`a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`；权重 revision：`c725b8589bdf6ab04072cab74c0467830db80d6d`。
权重 SHA256：`9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb`。
模型来源：[Depth Anything V2 官方 metric depth](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)。

```bash
cd /mnt/shizhm/CS2S_pose_environment_temporal_v22
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/shizhm/miniconda3/envs/ControlS2S/bin/python tools/validate_dense_static_reference.py \
  --settings /mnt/shizhm/CS2S_run_control/temporal_static_p0p2_20260919/base_settings.json \
  --pairs /mnt/shizhm/CS2S_run_control/temporal_static_p0p2_20260919/sensor_pairs.json \
  --vendor /mnt/shizhm/third_party/Depth-Anything-V2-probe-a561b849 \
  --checkpoint /mnt/shizhm/models/depth_anything_v2/depth_anything_v2_metric_vkitti_vits.pth \
  --out /mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/dense_reprojection_20260919/reproduction1024 \
  --device cuda:4 --width 1024 --height 256
```

重跑 512 时加 `--width 512 --height 128`，并通过 `--depth-cache` 指向已存的 `run1024/native_depth`。输出目录须为新目录，避免覆盖历史实验。

[全部帧对交互对照](/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/dense_reprojection_20260919/index.html) · [建筑局部对照](/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/dense_reprojection_20260919/building_crops.jpg)
