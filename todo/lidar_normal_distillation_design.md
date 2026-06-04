# RGB→LiDAR 跨模态法向蒸馏：设计文档

> Train-time only cross-modal distillation。
> 核心 idea：RGB 看到的世界比稀疏 LiDAR 密 ~100×，用这个密度差做"免费"的局部几何蒸馏，
> 让稀疏 LiDAR encoder 获得接近 dense 的局部结构理解。**推理时不需要 RGB**。

---

## 0. 一句话故事

> "RGB foundation model 提供密集 per-point surface normal 作为 pseudo-label，
> 蒸馏给稀疏 LiDAR encoder；训练用 RGB，推理纯 LiDAR。"

- 比纯 LiDAR 自监督**强得多**（有密集、指向真实几何的局部监督）。
- 比需要配对训练的跨模态方法**轻得多**（foundation model pseudo-label 现成、推理不依赖 RGB）。
- 同时解掉前期两个根因：
  - 绕过 **全图弱 epsilon 缺方向监督**（normal 蒸馏给密集局部、指向真实几何的梯度）。
  - 绕过 **手工 BEV 不可学**（learned normal 替代手工 compactness/objectness）。

---

## 1. 蒸馏目标：per-point surface normal（首选）

为什么是 normal 而不是 depth：
- LiDAR **本来就有精确深度**，蒸 RGB 绝对深度没意义。
- 要蒸的是 **RGB 才有的局部结构**：法向、边界、表面连续性。
- per-point surface normal 是 LiDAR 稀疏点**最缺**的局部几何 → 价值最高。

可选扩展（后续做）：
- per-point relative depth **残差**（不是绝对深度）。
- per-point semantic/part embedding（SAM / DINO feature 投影）。
- per-point learned "objectness"，替代手工 objectness-like 通道。

---

## 2. 关键设计：四个必须处理的技术坑

### 2.1 坐标系统一（最容易翻车）
- RGB-derived normal 在**相机坐标系**，LiDAR 在**激光雷达/世界坐标系**。
- 直接监督会学到被外参污染的 normal（学到的是相机姿态而非几何）。
- **务必在投影/取 label 前对齐**：
  - 把 RGB normal 旋到 LiDAR 系：`n_lidar = R^{-1} @ n_cam`，或
  - 全部统一到世界系。

### 2.2 方向损失（不能用 L2）
- normal 是单位向量，且有符号歧义（n 与 -n 同一平面）。
- 用 cosine 形式并取绝对值消歧义：

```
L_normal = mean_i [ w_i * (1 - | <n_hat_i, n_gt_i> | ) ]
```

- 直接 MSE 会因符号翻转产生假梯度。

### 2.3 pseudo-label 置信加权
- RGB normal 在**无纹理区、远处、反光面**会乱。
- 每个 pseudo-label 配置信 `w_i`：来源可用模型不确定度 / depth gradient 一致性 /
  多模型交叉验证夹角。
- 否则 encoder 会被天空、路面反光的错 normal 带偏。

### 2.4 投影遮挡可见性剔除
- LiDAR 点投到图像后，被遮挡的点会取到错误 RGB normal。
- 投影后做 **z-buffer / 深度一致性检验**：
  仅保留 LiDAR 深度 ≈ RGB metric depth 的点。
- 这一步直接决定 pseudo-label 质量。

---

## 3. Normal 估计器选型

场景约束：自动驾驶街景 + 大规模 per-point pseudo-label 生成。
核心诉求 = **几何准 + 室外泛化 + 批量快**，而非单张最锐。

| 模型 | 关键点 | 速度 | 角色 |
|---|---|---|---|
| **Metric3D v2** | 同时出 metric depth + normal，室外/驾驶泛化强，depth/normal 自洽 | 前馈，快 | **主力 pseudo-label 生成器** |
| **DSINE** | per-pixel ray direction 归纳偏置，边界锐、轻量（CVPR24 Oral）| 快（前馈）| 次选 / 交叉验证 |
| **StableNormal** | diffusion，降方差、最锐最稳，但慢 | 慢（扩散）| 离线高质量子集精修 |
| **Marigold-Normal** | SD 微调，质量好但多步 ensemble 慢 | 慢 | 不推荐大规模 |
| 单步 fine-tuned SD | WACV25 证明单步确定性模型超过 Marigold，快且准 | 快 | 备选 |

### 主力选 Metric3D v2 的三个理由
1. **depth 与 normal 同源且度量自洽** → 可用其 metric depth 做 z-buffer 遮挡剔除，
   再取同模型 normal，两者不打架。
2. **室外/驾驶泛化是其主打**（DSINE / Marigold 强项偏 object / 室内）。
3. **前馈速度** → 适合整个数据集批量生成 per-point label，
   不像 StableNormal / Marigold 扩散多步拖垮预处理。

### DSINE 作为"第二意见"
- ray-direction 归纳偏置在物体边界更锐。
- 在抽样子集同时跑 Metric3D v2 + DSINE，两者 normal 夹角大的点 → 降低 `w_i`。
- 白捡一个 free 的 label 质量过滤器（接 2.3 的置信加权）。

### 新动向（仅关注，暂不用）
- RoSE（ICLR 2026 Oral）：用 image-to-video 生成模型预测 shading 序列估 normal，
  object 基准新 SOTA，但偏 object-level 且更重。
  适合后期精修少量关键帧，不适合街景大规模预处理。

---

## 4. 接入顺序（落地步骤）

1. **批量出 normal+depth**：用 Metric3D v2 对全集生成 dense normal + metric depth。
2. **可见性剔除**：用 metric depth 做投影 z-buffer 检验，
   仅保留 LiDAR 点深度 ≈ Metric3D depth 的点。
3. **坐标对齐取 label**：normal 经相机外参旋到 LiDAR / 世界系，取 per-point n_gt。
4. **交叉验证置信**：DSINE 在子集上跑，与 Metric3D normal 夹角差异 → 置信 w_i。
5. **蒸馏训练**：方向损失
   `L_normal = mean_i [ w_i * (1 - | <n_hat_i, n_gt_i> | ) ]`
   监督 LiDAR encoder 的 per-point normal 预测头。

### 训练 / 推理对比

```
训练:
  RGB → Metric3D v2 → dense depth + normal
  LiDAR 点投影 → 可见性剔除 → per-point pseudo-label (normal)
  LiDAR encoder → 预测 per-point normal → L_normal 强局部监督
  (并行) 原 diffusion 主线 control 分支照常训练

推理:
  仅 LiDAR encoder → 局部几何特征已学好 → 注入 frozen UNet → 生成 RGB object
  不需要 RGB 输入
```

---

## 5. 分阶段训练目标

### Stage 0：4 帧 overfit

- 目标：先确认坐标、pseudo-label、loss、网络容量没有明显错误。
- 只用少量帧反复训练，要求 loss 能明显下降。
- 通过标准：
  - weighted cosine normal loss 明显下降；
  - sign-invariant median angular error 比 untrained baseline 至少下降 50%；
  - pred normal overlay 在路面、车体、建筑边缘上能贴近 GT normal。

### Stage 1：小规模 1k-5k 帧

- 目标：检查 Metric3D pseudo-label 质量、visibility filtering 和 normal overlay 是否在不同 drive/date 上稳定。
- 从多个 drive/date 采样，不随机按相邻 frame 泄漏验证。
- 通过标准：
  - val loss 稳定下降；
  - 可视化中路面、建筑、车辆表面 normal 有结构差异；
  - 训练没有只记住少数帧。

### Stage 2：KITTI RAW 全量

- normal encoder 训练不需要 satellite、不需要 XML、不需要动态框。
- 全量数据只要求 `image_02 + velodyne_points + calib_cam_to_cam + calib_velo_to_cam`。
- split 按 `date/drive`，不能随机按 frame split，避免相邻帧泄漏。
- 第一版不做 object/class reweight；如果 object 表面学得弱，下一轮再加非地面点、深度边界点、高曲率点采样加权。

---

## 6. 待确认 / 下一步

- 外参格式：KITTI 标准 `T_cam_lidar` vs 自定义 pipeline 格式 → 决定剔除+旋转脚本写法。
- 是否把 per-point normal 蒸馏与现有 gate / residual 修复（consistency gate / residual gate 解耦）并行做。
- normal 预测头挂在 LiDAR encoder 哪个尺度（建议高分辨率 BEV / per-point 分支）。
