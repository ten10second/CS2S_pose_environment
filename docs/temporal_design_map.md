# 时序设计地图（当前 live 设计）

最后核对：2026-09-11，分支 `temporal-model`。
历史方案与实测记录见 `docs/temporal_history.md`。

---

## 0. 一句话

**唯一在工作的时序设计是「几何对应 history」**（历史文档里叫 v3 / Stage F）。

上一帧 latent 被编码成 token，按 LiDAR + 位姿算出的像素对应关系，注入 UNet 的
**第 12 个 ray-posterior fusion block（decoder / 640 通道 / 8×32 网格）**。
v0 / v1 / v2 三代方案的代码已经全部删除，只保留在历史文档和 git 历史里。

---

## 1. Live 调用链

```
configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea_cfgdrop10.yaml
  ray_fusion_mode: ray_posterior, use_lidar_cross_attention: true
        │
        ├─ 底座：CFG 单帧 backbone（冻结）
        │        tools/train_kitti_geometry_history.py:320-329
        │
        ├─ 几何：tools/temporal_history_geometry.py:467  build_pair_geometry()
        │        grid=(16,64)，当前 LiDAR 点 + OXTS 相对位姿，
        │        depth_tol=0.75m 的上一帧 z-buffer 支持检查
        │        输出 history_grid (16,64,2) + history_valid (16,64)
        │
        ├─ 注入：tools/temporal_history.py  resolve_history_block_indices()
        │        16 个 fusion block 中取 dim==640 的最后一个 → index 12
        │        ldm/modules/KITTI_attention.py  _build_history_pack() → _forward()
        │
        ├─ 训练：tools/train_kitti_geometry_history.py
        │        mode = "geometry_history_v1_generator"
        │        可训练 = HistoryLatentEncoder + GeometryHistoryAttention
        │                 +（--unfreeze-host 时）注入 block 的 ff/norm3
        │        判据 = 4 条件固定探针（disabled / correct / wrong_geometry /
        │               wrong_history）× t=250,750
        │        launcher: tools/run_geometry_history_generator.sh
        │
        └─ 采样：tools/generate_kitti_geometry_history.py（无 launcher）
                 history 载体 = 上一帧 **生成** latent
                 （--history-source gt 换成上一帧 GT latent）
                 第一帧 / 断帧 / 换 drive → has_history=False
```

`tools/raea_frame_sampling.py` 提供与设计无关的 `prepare_frame_inputs` /
`sample_frame`；`tools/ar_dyn_utils.py` 提供 `seed_step_noise`。

## 2. 必须记住的一个尺寸事实

几何在 **(16,64)** 建立，注入 block 在 **(8,32)**，所以 `_resize_grid`
（`ldm/modules/temporal_history_attention.py`）每次都会走保守下采样：
`valid` 要求 2×2 四个子格全部 valid。这是"有效覆盖率可能远低于 Stage A 量到的
43.63%"的来源，也是 on/off 无差别的候选解释之一。

## 3. 开关契约（history on/off）

- 只有一个决策点：`tools/generate_kitti_geometry_history.py: resolve_history_switch()`。
- `--disable-history` 不绕过模块，它强制 `has_history=False`，走训练探针
  `disabled` 条件同一条 payload 路径。
- `has_history=False` 时注意力仍执行：null token 过 K/V，残差乘精确 0，
  rollout 会断言 `null_all == 1.0`、`valid_neighbor_fraction == 0.0`、
  `residual_to_x == 0.0`，否则中止。

## 4. 当前状态与未决问题

Stage F 的验收 gate 未通过：**history 开与关的生成结果无法区分**。
两个候选解释，互斥，用同一份 `records.json` 就能判定：

| 候选 | 机制 | 查什么字段 |
| --- | --- | --- |
| (a) 目标函数不要求读 history | `disabled` 就是单帧基线；唯一的附加项 `appearance_x0` 是**当前帧** latent 重建，冻结 backbone 本来就能最小化；AdamW 权重衰减把 history 参数拉回 0 | `metrics_rank*.jsonl` 的 `history_ratio` / `history_memory_ratio` 是否随 step 单调衰减 |
| (b) 注入分辨率上掩码被 AND 掉 | (16,64) 的 valid 在 (8,32) 上要求 2×2 全真；若接近 0，`hist_delta` 逐元素严格为 0 | `records.json` 顶层 `history_valid_frac`（16×64 比例）对比 `history_attention_steps[*].valid_neighbor_fraction`（8×32 实际比例），再看 `residual_to_x` |

---

## 5. 时序相关文件清单

**Live**

| 文件 | 作用 |
| --- | --- |
| `tools/train_kitti_geometry_history.py` | Stage F 训练（4 条件探针） |
| `tools/generate_kitti_geometry_history.py` | 生成式 rollout + 开关契约 |
| `tools/temporal_history.py` | block 选择、注入、host 参数存取 |
| `tools/temporal_history_geometry.py` | 对应关系几何 |
| `tools/audit_temporal_history_geometry.py` | 几何可行性审计（Stage A 工具） |
| `tools/summarize_geometry_history_run.py` | 训练产物汇总与 gate |
| `tools/raea_frame_sampling.py` | 与设计无关的采样 helper |
| `tools/ar_dyn_utils.py` | `seed_step_noise` 等小工具 |
| `tools/run_geometry_history_generator.sh` | Stage F launcher |
| `ldm/modules/temporal_history_attention.py` | `HistoryLatentEncoder` + `GeometryHistoryAttention` |
| `ldm/modules/KITTI_attention.py` | `HistoryEvidenceHub` + 注入点 |

**已删除（2026-09-11，代码见 git `5ddc47c` 及更早）**

| 方案 | 已删除的文件 |
| --- | --- |
| v0 推断期（含 instance transport） | `generate_kitti_raea_noise_modes.py`、`lidar_object_association.py`、`validate_object_association.py`、`pose_warp_utils.py`、`analyze_region_flicker.py`、`analyze_object_identity.py`、`analyze_temporal_flicker.py` |
| v1 学习门控 + route-C | `train_kitti_temporal.py`、`temporal_evidence.py`、`eval_sat_temporal.sh`；`KITTI_attention.py` 里的 `SatTemporalReferenceAttention`、`enable_temporal()`、`_build_temporal`、`_build_sat_temporal` |
| v2 route-history | `train_kitti_temporal_v2.py`、`generate_kitti_temporal_v2.py`、`evaluate_kitti_temporal_v2.py`；`HistoryCrossAttention`、`build_payload`、`HistoryState`、`should_use_history` |

通用工具仍然保留：`tools/generate_kitti_raea_samples.py`（单帧采样）、
`tools/assemble_panel_video.py`（面板拼 mp4）。

## 6. 下一层：根因判定与机制选项

第 4 节的 on/off 问题已经展开成独立文档：`docs/temporal_mechanism_options.md`。
要点：

- 时序条件被使用 ⟺ 它是降损的必经之路；Cyclops 满足（源无颜色），我们不满足
  （卫星条件已含颜色）。
- 我们的闪烁**不是**采样噪声造成的（逐步噪声在进程内本来就是固定 bank），而是条件
  随自运动漂移导致的渲染不稳定。
- 判决实验：`tools/probe_history_necessity.py`（4 条件 × 卫星开/关，配对噪声），
  零代码版本可以直接用 `--disable-history` × `--uncond-cfg` 跑。
- 机制选项 M1–M4，优先免训练、复用主干自身投影的 M1。
