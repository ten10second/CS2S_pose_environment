# 时序实验历史与实测记录（归档）

这份文档记录**已经删除的时序方案**和 v3 几何 history 的全部实测结果。
其中 v0 / v1 / v2 的代码已于 2026-09-11 删除（见 git `5ddc47c` 及更早），
这份文档是它们唯一的设计与结果记录。当前 live 设计见
`docs/temporal_design_map.md`。

结论速览：

| 方案 | 结论 |
| --- | --- |
| v0 推断期 latent 复用 / 几何 warp / 目标级 transport | 自回归最强但过平滑（ratio 0.820，锁定历史）；几何 warp 增益更小；几何 transport 无法绑定慢速运动目标 |
| v1 学习门控时序 evidence + route-C 卫星 token | 无结果记录，被 v2 取代 |
| v2 route-history（条件感知的 history 交叉注意力） | 训练集获益增长而留出集变负；固定探针下 history 在任何 (split, t) 都无收益，架构暂停 |
| v3 几何 history（Stage A–F） | A 通过几何可行性 gate；B/C 未过 RGB gate；D 定位 depth 耦合；E bilinear skip 失败；F 当前 on/off 无差别，验收 gate 未过 |

---

## 1. v0：推断期时序（2026-08-31 冻结）

在收敛的 500k 单帧 checkpoint 上做推理实验，不是训练出来的时序模型。

- checkpoint：`step_500000.pt`
- 指标：相邻生成帧 LPIPS 均值 ÷ 相邻 GT 帧 LPIPS 均值。接近 1 表示帧间变化量与
  真实序列相当。该指标不衡量语义正确性、几何、身份保持或感知质量。
- 产出这些数字的脚本（`analyze_temporal_flicker.py`，硬编码 v0 运行目录，不接受
  命令行输入）已随 v0 一起删除。需要复算时按上式重写一个 `--gen-dir/--gt-dir`
  的 CLI 即可，不需要旧代码。
- 结果只覆盖一个 seed 和两个 clip，不是置信区间。

推断变体：`per_frame`（每帧独立噪声）、`shared`（全序列共享初始噪声）、
`autoregressive`（上一帧 latent 加噪去噪，`ar_strength=0.7`）、
`posewarp`（OXTS 自运动 + LiDAR 拟合地面单应 warp，丢弃动态格，`ar_strength=0.5`）、
`posewarp2`（只在局部 LiDAR 仿射流与单应一致处 warp）。

这不是等算力消融：`per_frame`/`shared` 跑满 50 步 DDIM，`autoregressive` 从第二帧
起只跑 15 步，`posewarp`/`posewarp2` 跑 25 步。也不是完全 seed 受控的消融：
`ddim_KITTI.py` 在模块导入时就创建逐步噪声，早于脚本里的 `torch.manual_seed`。

| 测试 clip | 帧数 | 方法 | 生成 tLPIPS | GT tLPIPS | 比值 |
| --- | ---: | --- | ---: | ---: | ---: |
| drive 0057 | 58 | per-frame | 0.2157 | 0.1090 | 1.978 |
| drive 0057 | 58 | shared noise | 0.2153 | 0.1090 | 1.975 |
| drive 0057 | 58 | autoregressive | 0.1164 | 0.1090 | 1.068 |
| drive 0057 | 58 | posewarp | 0.1739 | 0.1090 | 1.595 |
| drive 0020 | 300 | per-frame | 0.3025 | 0.1856 | 1.630 |
| drive 0020 | 300 | autoregressive | 0.1523 | 0.1858 | 0.820 |
| drive 0020 | 300 | posewarp | 0.2641 | 0.1858 | 1.421 |
| drive 0020 | 300 | posewarp2 | 0.2425 | 0.1858 | 1.305 |

输出快照：`/mnt/shizhm/DATA/KITTI/CS2S_results/kitti_ray_posterior/ray_posterior_utonia_dino_sd14_4gpu_20260812/inference/temporal_flicker_analysis.json`

结论：

1. drive 0057 上只共享初始噪声把比值从 1.978 变到 1.975，可忽略；由于逐步 DDIM
   噪声未跨进程固定，这不能隔离"共享初始噪声"的作用，需要多 seed 受控重测。
2. 自回归 latent 复用是最强的推断期稳定器，58 帧比值接近 1；但 300 帧比值 0.820
   低于真实序列，说明过平滑或历史锁定，存在拖影和运动目标误差累积的风险。
3. 位姿引导 warp 有帮助但增益更小。单个地面单应无法正确 warp 建筑、植被、独立
   运动目标和新生可见区域；稀疏 LiDAR 也限制了动态剔除掩码。
4. 这些实验确认时序信息有用，但没有确立最终时序架构，也没有确立整体质量提升。
   图像质量与条件保真度必须与 tLPIPS 一起评估。
5. 运行元数据没有独立保存 checkpoint、manifest 切片、采样参数和 git commit。
   以后的正式实验必须把这些写进 `run_summary.json`。

### 1.1 instance transport 轮次（2026-08-31）

- `lidar_object_association.py`（无标签跨帧聚类关联）+ `instance` 噪声模式：
  匹配到的运动目标沿自身图像位移继承内容，而不是被动态重置。
- drive_0020、300 帧、`instance` @ `ar_strength=0.5`：tLPIPS 比值 1.302
  （posewarp2 1.305、per-frame 1.630、autoregressive 0.820）。全局闪烁不变，
  符合预期：被 transport 的区域很小。
- 关联在 100/300 帧上触发（131 个目标实例）。验证叠加图确认聚类/匹配机制正确；
  主要误报是树叶和玻璃回波（在 transport 下无害）。
- 关键负面结果：每帧位移低于约 0.35 m（3.5 m/s）的目标低于 OXTS 位姿噪声和
  距离量化给出的欧氏容差下限，因此典型慢速城市交通会被当作背景做单应 transport，
  身份仍被重新采样。目标身份 CLIP 指标（`analyze_object_identity.py`）这一轮
  只有 n=5 对可比样本，不足以评分。
- 结论：仅靠几何的推断期 transport 无法绑定慢速运动目标的身份。这是
  噪声/latent 初始化族的边界，也是转向学习式时序 evidence 流的具体动机。

## 2. v1：学习门控时序 evidence + route-C

在冻结的单帧 ray-posterior backbone 上，给每个 ray-posterior fusion 模块注入
零初始化的时序门（第三条 evidence 流：上一帧 fused posterior，经地面单应 transport
并按可靠性掩码）；只训练门控，目标函数保持单帧扩散损失不变。route-C 是它的
孪生分支：用上一帧的卫星 patch token 做时空局部参考注意力。

- 训练脚本 `train_kitti_temporal.py`，采样 `--noise-mode temporal_net|sat_temporal_net`。
- 仓库里**没有这一代的结论记录**：它被 v2 的 trainer 取代，v2 findings 开头直接
  进入 v2 的 Phase A。
- 遗留产物（代码已删，权重仍在服务器）：
  `temporal_v1_4gpu_20260904/temporal_step_20000.pt`（仅门控，20k 步）、
  `sat_temporal_v2_2gpu_20260905/temporal_step_13000.pt`（route-C，13.6k 步停止）。
  注意后者的目录名叫 v2，实际是 route-C 卫星 token 注意力。

## 3. v2：route-history（已明确暂停）

设计差异：history = 上一帧的最终 latent（训练时 teacher-forced，用 GT 编码），
不是上一帧的卫星 token 也不是上一帧的生成结果；history 注意力的 query 显式接收
当前条件的 fused summary，以便只在条件无法解释处读 history；目标函数仍是单帧
去噪损失。

**Phase A 多样本轮（2026-09-08）**：48 训练对 + 16 留出连续对（同一段 drive），
1200 步，lr 1e-4，双卡，teacher-forced GT history，纯去噪目标。探针对比
correct / disabled / wrong history（same-drive-far、other-drive）。

结果：训练获益增长（+0.05 → +0.10），留出获益为负且恶化（-0.002 → -0.038）；
留出对上的 correct-vs-wrong 判别始终在噪声水平（±0.003）。checkpoint：
`hist_v2_step_{400,800,1200}.pt`；指标：`hist_v2_multisample_20260907/metrics.jsonl`。

判定：该配置下（48 对、25.4M history 参数、1200 步）history 模块记住的是训练对
而不是学会通用地使用 history；在未见对上条件化有害且随训练恶化。go/no-go 判据 5
（correct 必须优于 disabled）在留出对上失败——有效性主张暂停。

**固定探针复评（2026-09-08，评审后）**：对 fresh/400/800/1200 四个 checkpoint
用全部四项评审要求的修正重评（每对冻结 history latent、拦截固定 timestep
250/750、每个 (pair, t, draw) 在各 history 条件下使用相同噪声、完整分量拆解、
逐行 JSONL：`temporal_v2_reeval.jsonl`，2048 行）。

框架自检：fresh（未训练）参考在每 split、每 timestep 上四个条件**逐位相同**。

固定 t 下的分量均值（val t=250, step_1200）：

| condition | denoise | depth | point | total |
| --- | ---: | ---: | ---: | ---: |
| disabled | 0.1889 | 0.0147 | 0.1738 | 0.3774 |
| correct | 0.2351 | 0.0154 | 0.2256 | 0.4761 |
| wrong:other_drive | 0.2320 | 0.0152 | 0.2218 | 0.4690 |
| wrong:same_drive_far | 0.2272 | 0.0153 | 0.2165 | 0.4590 |

1. history 条件化在**任何** (split, t) 上都没有收益：训练集 t=250（固定）下
   denoise 为 0.1925（correct）对 0.1886（disabled），略差而非更好。训练循环探针
   的 +0.10 训练"获益"在固定 timestep 和固定 history latent 下不成立，是
   timestep/latent 采样不受控的产物。
2. 留出对上的损害集中在 DENOISING 项（t=250 时 +0.046）和 point-region 项
   （+0.052）；LiDAR depth 项可忽略（+0.0007）。评审假设 1（depth 驱动的辅助优化）
   因此被排除为主要机制。
3. correct history 在留出对上**一致地比 wrong history 更有害**——学到的修正与
   48 个训练对场景耦合，在别处误导。注意力聚焦度增长（0.11 → 0.285），但它读到的
   内容无法迁移。

决策：这不是训练目标问题，也不是训练时长问题。按此配置的条件感知残差适配器学到的是
场景耦合的修正，没有拒绝/泛化行为。**暂停该架构**。任何延续都需要显式的机制变更
（对应关系约束、容量降到只有瓶颈块、或换 history 载体），并在任何长训练之前用上述
固定探针协议评估。

## 4. v3：几何 history（Stage A–F）

### 4.1 几何核心与 Stage A 门

`tools/temporal_history_geometry.build_pair_geometry(prev_row, cur_row, kitti_root, grid=(16, 64))`
返回：

- `history_grid`：`(H, W, 2)` float32，上一帧特征图的 grid-sample 坐标
  （`align_corners=False`）。
- `history_valid`：`(H, W)` bool。无效表示未知，不回退到 identity。
- `metrics`：可 JSON 序列化的覆盖率与运动诊断。

投影核心使用完整的 `P_rect_02` 3×4 矩阵（含平移列）、`R_rect_00`、Velodyne 标定、
OXTS 位姿和上一帧扫描的 z-buffer 支持。

审计命令：

```bash
python tools/audit_temporal_history_geometry.py \
  --manifest /path/to/train_manifest.jsonl \
  --kitti-root /path/to/KITTI_RAW \
  --out-dir /tmp/geometry_audit \
  --num-pairs 32
```

预注册的可行性 gate（判断 Stage B 是否值得跑，不是关于语义建筑稳定性的主张）：

- `pilot_gate_valid_coverage_pass`：平均 `coverage_all >= 0.05`
- `pilot_gate_non_ground_proxy_all_pass`：平均 `coverage_non_ground_proxy_all >= 0.05`
- `pilot_gate_rgb_motion_pass`：至少 16 个带运动的对
  （`prev_to_cur_translation_m >= 0.2`）且
  `rgb_error_identity_minus_mapped_mean > 0`

**A 实测（64 个带运动的对，跨 drive 轮转）**：合成 identity、偏心 identity、平移、
旋转、`P_rect` 平移、地面符号、遮挡和未知支持测试全部通过。

| 指标 | 结果 |
| --- | --- |
| 有效 query 覆盖率 | 43.63% |
| 地面代理支持格 / 全网格 | 20.99% |
| 非地面代理支持格 / 全网格 | 22.64% |
| 逐点 RGB 误差（identity 采样） | 0.10729 |
| 逐点 RGB 误差（投影采样） | 0.05575 |
| 每对误差缩减中位数 | 0.05063 |

三个预注册 gate 全过，另目视检查了两张投影叠加图。这是几何可行性结果，不是生成视频
质量结果。非地面不是语义建筑掩码，稀疏深度一致性也无法可靠排除每个慢速运动车辆。
lookup 使用被支持 query 格上实测点的位移，是局部近似而非稠密表面重建。

### 4.2 Stage B：adapter 与四卡 smoke

- `GeometryHistoryAttention`：条件感知 Query、投影局部 3×3 history K/V、可学习 null
  key、固定零 null value、bias-free 零初始化 out。
- 显式初始融合索引 `2,12`；history 维度 64，4 头每头 32 维。encoder 和 attention
  是唯一可训练模块。
- 无 history 时在同一 DDP wrapper 内保持图连通。backbone 保持 eval；卫星 dropout 显式
  施加，以覆盖 CFG 且不掉 LiDAR/history。
- 原始训练目标保留并记录分解。没有未对齐的时序图像损失、频率分裂、长期记忆或
  原始 AR 初始化。
- 前 20 步强制在第 2 步走无 history、第 3 步走无卫星条件。要求 step-0 与基线精确相等、
  梯度有限、所有可训练参数在图中。
- 首次四卡尝试（`stage_b_smoke`）到第 1 步后所有 rank 在强制的无 history 第 2 步同步
  因非有限梯度中止（保留失败日志，不算成功 smoke）。改用 AMP 初始 scale 1024
  （而非 65536）从全新初始化重试（`stage_b_smoke_scale1024`）后不再复现。
- 到第 20 步：encoder 梯度 0.00039974、condition Query 0.00005588、output 0.0108774；
  固定 disabled 探针在 0/10/20 步精确不变。rank-0 留出总损失获益在 t=250 为
  +0.00013188、t=750 为 -0.00009403——**混合且极小，不是时序有效性的证据**。

### 4.3 Stage C：有界多 drive pilot

- 全部训练 drive 的连续对 + epoch 洗牌；整个 drive 独立留出。
- 训练分区 10605 对；留出分区 3858 对（来自五个独立 drive）。每 rank 一个固定训练
  探针和一个留出探针，只覆盖 4+4 对，不是完整验证基准。
- 预算最多 1000 个优化步，每 100 步查探针。探针缓存 history latent 并钉住当前 VAE
  采样、timestep (250,750) 和扩散噪声，对比 disabled/correct/wrong geometry/wrong
  history。wrong geometry 保持目标有效性和源坐标集合不变。
- 启动后所有 rank 到第 31 步：同步梯度范数 encoder 0.0009646243、condition Query
  0.0001363657、output 0.0195291123；可训练参数 608576；索引 2 和 12 宽度都是 640。
- 四卡 smoke 在第 20 步的平均留出获益为 +0.00008006（t=250）和 -0.00000973（t=750）。
- **C 在 `stage_c_pilot/geometry_history_step_1000.pt` 未通过 RGB 有效性 gate**：
  留出 RGB 去噪略差于 disabled history；depth 改善；correct history 在 RGB 上优于
  wrong history/geometry。

### 4.4 Stage D：瓶颈后耦合对照

冻结 checkpoint 的梯度审计发现 t=750 处 encoder 局部冲突，以及融合索引 12 处
depth 梯度为零——该位置在 `openaimodel.py` 里位于瓶颈深度预测之后。该审计只是快照，
不证明"去掉 depth 损失就能通过 RGB vs disabled"。

代码变更（不得续训 2,12 权重）：

- geometry history 默认改为 `after_bottleneck`：最细的 640 维 decoder 融合块，
  位于 `lidar_bottleneck_depth_head` 之后。encoder/middle 融合索引被拒绝。
- 移除失败的局部 3×3 copy 和 bilinear skip 路径；history 变成由对应关系门控的
  上一帧 RGB latent 记忆。
- 新运行目录 `stage_d_post_bottleneck`，全新 adapter，同样 1000 步预算和同样八个
  固定探针。
- 若放置正确，预期诊断是：瓶颈 depth 指标在 disabled/correct/wrong 下完全相同，
  只看 RGB `loss_eps_base`。不得用总损失宣布时序成功。

该轮回答了"encoder 侧 depth 耦合是否导致 on>off"；结果归档后 launcher 被删除。

### 4.5 Stage E：通过对应关系门做外观传输

Stage D 的几何 lookup 正确，但 on/off 视频几乎相同：零初始化注意力残差太弱，改不动
像素。Stage E 不改进投影，而是加入 history token 在投影格上的 bilinear appearance
skip，带非零 `to_skip` map；无效格和 `has_history=False` 保持精确零，注意力 `to_out`
仍零初始化。

- checkpoint mode：`geometry_history_v1_transport`。不得续训 Stage D 的
  `geometry_history_v1` 权重。
- 放置仍是 `after_bottleneck`，冻结 backbone 不变。
- step 0 不再所有条件相同：correct history 必须与 disabled 不同；disabled 探针必须
  跨步逐位不变。
- 判 RGB `loss_eps_base`，以及生成的 on/off 视频是否在对应表面上分叉；depth 必须在
  history 条件下不变。
- 新运行目录 `stage_e_appearance_transport`，全新 adapter，同样 1000 步和八个固定探针。

结果：bilinear skip 在失败 clip 上没有产生可泛化的外观继承。该路径已移除，
不得续训 Stage E 权重。

### 4.6 Stage F：让生成器参与外观记忆

GT-history 探针和 skip transport 都未能在失败 clip 上继承外观：correct geometry 优于
scrambled geometry，但 correct history 不优于 disabled history。冻结的单帧去噪加
epsilon MSE 不会训练生成器去"画出检索到的颜色"。

Stage F 保持当前卫星和 LiDAR 作为几何条件，history 注意力是以对应关系门控的
上一帧 RGB latent K/V 记忆；history 注入之后的 feed-forward 尾部解冻。仍是单步训练，
不是展开的采样器。

- masked x0 项是 `MSE(x_start, pred_x0)` 在对应格上：**当前帧 GT latent 重建，
  不是上一帧颜色一致性损失**。
- 探针必须对所有条件使用相同损失项；分别比较 `loss_eps_base` 和外观 x0 项，
  不要混用"带 x0"与"不带 x0"。
- 采样时只要设了 `unfreeze_host` 就必须恢复 `history_host`（`ff`/`norm3`）。
- 第一道验收 gate：`--history-source gt` + 完整 DDIM 采样，每帧用上一帧 GT RGB 作
  history。只有当对应表面颜色误差优于 disabled history，生成式 history rollout 才能
  被当作时序测试。
- launcher：`tools/run_geometry_history_generator.sh`（`--unfreeze-host`
  `--appearance-x0-weight 1.0`，1000 步，每 100 步探针）。不得续训 Stage D/E 权重。
  本阶段不做 scheduled sampling。

**开关统一（2026-09-11）**：rollout 现在只有一个 history on/off 决策
（`resolve_history_switch`）。`--disable-history` 不再清空 temporal hub，而是构造与训练
探针 `disabled` 条件相同的 `has_history=False` payload，使配对采样基线和已训练的无
history 条件共用一条代码路径。history 注意力在两种模式下每个去噪步都执行，且除非
无 history 帧报告 null attendance 1.0、valid fraction 0.0、residual 0.0，否则 rollout
中止。这不改变任何数值：被删除的 `hub.clear()` 路径和保留的 null-token 路径都给出
精确零残差。

**当前状态**：history 开与关仍无法区分，验收 gate 未过。根因判定见
`docs/temporal_design_map.md` 第 4 节。

### 4.7 冻结底座与安全边界

- 服务器实验根目录：
  `/mnt/shizhm/DATA/KITTI/CS2S_results/geometry_history_20260908`
- 冻结底座：`base/cfg_step_250000.pt`，从完整的当前 CFG 运行 step checkpoint 硬链接
  （不是 `last.pt`），训练运行的保留策略无法删除它。
  SHA256 `45f2d561302efec8817d7ac1f1c5307ba89546d0f1c38f250f058144c2ed183c`
- 成对的原始 config/args 复制到 `base/cfg_run_config.yaml` 和 `base/cfg_run_args.json`
- 只允许物理 GPU 4,5,6,7 运行该实验；GPU 0–3 与既有 CFG 训练不得停止、重启或重配。
- 不引入新依赖；不复用失败的 v2 权重。

---

## 附录：资产与产物路径

| 路径 | 角色 |
| --- | --- |
| `/mnt/shizhm/BasicModel/checkpoints/sd-v1-4.ckpt` | SD VAE 底座（4.27 GB） |
| `…/ray_posterior_utonia_dino_sd14_4gpu_20260812/checkpoints/step_500000.pt` | 冻结单帧 backbone（11.9 GB） |
| `…/kitti_raw_sat_lidar_geofence_test2_buffer30/train_manifest.jsonl` | 训练 manifest（21.7 MB，sha256 前缀 `deadffce21be97cf`） |
| `…/kitti_raw_sat_lidar_geofence_test2_buffer30/test_manifest.jsonl` | 测试 manifest（9.1 MB，sha256 前缀 `4e8461c4e0cf84c5`） |
| `/mnt/shizhm/DATA/KITTI/CS2S_cache_memmap/utonia_ray_depth_all_fp16` | LiDAR ray 特征缓存（29 GB） |
| `/mnt/shizhm/DATA/KITTI/CS2S_cache_memmap/dino_vits14_8x32_all_fp16` | DINO 语义特征缓存（4.8 GB） |
| `/mnt/shizhm/DATA/KITTI/KITTI_RAW` | KITTI raw 根目录 |
| `…/CS2S_results/geometry_history_20260908/base/cfg_step_250000.pt` | Stage F 冻结底座（见 4.7） |

**已孤立**（加载它们的代码已删除，权重仍在服务器上）：

| 路径 | 原属 |
| --- | --- |
| `…/temporal_v1_4gpu_20260904/temporal_step_20000.pt` | v1 门控（仅门控，20k 步） |
| `…/temporal_v1_4gpu_20260904/temporal_metrics.jsonl` | v1 训练日志 |
| `…/sat_temporal_v2_2gpu_20260905/temporal_step_13000.pt` | route-C 卫星 token 注意力（目录名误导，非 v2） |
| `…/sat_temporal_v2_2gpu_20260905/train.log` | route-C 训练日志 |
| `…/inference/sat_temporal_net_s7000_300f/records.json` | route-C 300 帧评测 |

历史遗留告警：s7000 的 flicker 1.644 是在评测脚本存在 P2-01 LPIPS 符号 bug 时算出的，
报告中的比值使用了单独核验的 GT 均值（0.1858），在严格配对复评前只能当作近似值。
