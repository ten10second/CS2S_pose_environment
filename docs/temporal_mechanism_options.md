# 时序一致性：第一性原理、根因判定与机制选项

> 归档说明（2026-09-11）：本文是旧 adapter 退役前的机制讨论，旧训练、生成与
> probe 命令已随 adapter 删除，不能在当前 checkout 执行。实际判决结果见
> [实验归档](temporal_history.md)，当前代码状态见 [设计地图](temporal_design_map.md)。
> 下文的“必要且充分”判据、权重衰减必然令通路归零、以及无收益即可确定读出坏了
> 等说法应视为当时的假设，不是已证明结论；实测残差非零且对历史内容敏感。

面向的约束：单帧生成权重只存在于服务器（`/mnt/shizhm/DATA/KITTI/CS2S_results/...`），
本机没有。所以本机负责改代码与设计，服务器负责跑：完整 DDIM 推理、判决实验、
以及 1000 步量级的短训都在预算内；**不在**预算内的是重训单帧 backbone。

因此这份文档回答三件事：**为什么现在不 work**、**怎么用现有产物判定根因**、
**在我们自己的技术特点下应该改成什么机制**。

---

## 0. 结论摘要

1. **时序条件被使用 ⟺ 它是降低损失的必经之路。** 只要存在"关掉 history 也能达到同样损失"
   的参数配置，带权重衰减的优化器就会把 history 通路压到 0。Cyclops 满足这个条件
   （它的源条件是单通道强度，**没有颜色**，外观只能来自上一帧）；我们不满足
   （卫星条件本身就带颜色和纹理）。这不是实现 bug，是目标函数的性质。
2. **我们的闪烁不是采样随机性造成的。** 逐步 DDIM 噪声在进程内是一个固定 bank
   （`ddim_KITTI.py:26` 的 `default_noise`，每步取同一个 `index`），所有帧共用；
   而只共享初始噪声 `x_T` 在实测里只把比值从 1.978 动到 1.975。所以闪烁来自
   **条件随自运动漂移导致的渲染不稳定**，不是"噪声没对齐"。免训练机制应该作用在
   确定性的、条件驱动的部分，而不是噪声上。
3. **因此机制应该复用主干自己的投影，而不是新增一个需要训练的模块。** 我们目前是
   "冻结主干 + 新参数模块"，这是最坏的组合：新参数需要训练信号，而目标函数又不要求
   使用它。免训练路线（跨帧注意力一族）把上一帧的 token 送进**主干自己的** K/V 投影，
   特征在分布内，主干知道怎么读，没有可以被"学掉"的参数。

---

## 1. Cyclops 的做法（arXiv:2608.16264，2026-08-17）

任务：只用稀疏 NRS-LiDAR 强度合成 RGB 视频（"全天候相机"）。
**代码可得性：没有找到公开实现。** 我查了 arXiv 摘要页/HTML 全文、论文内的链接区、
GitHub 方向的多次检索和几个镜像站，论文没有给出 code release。
所以下面的对齐是基于论文公式，不是基于他们的源码。

| 环节 | 做法 |
| --- | --- |
| Stage I | 冻结的稠密化网络把稀疏强度图变成稠密强度图 `I_dense`（单通道，几何丰富，**无颜色**） |
| Stage II | SDXL VAE 编码 `z_src = E(I_dense)`，用 Latent Bridge Matching 把 `z_src` 输运到 RGB latent `z_gt`，只有 4 步 Euler |
| 时序状态 | `s = (z_{t,m}, z_t^src, z_prev)`；`z_prev` 是上一帧的 **target latent**（teacher forcing 用 GT，否则用生成结果），首帧用可学习 null token `z_∅` |
| 注入方式 | 两种注意力：**source cross-attention** 把 `z_src` 作 K/V 注入**每个分辨率层**（结构对齐）；**temporal attention** 把 `z_prev` 作 K/V 注入到 cross-attention 增强后的特征（颜色延续） |
| 训练目标 | Phase 1：`L_LBM + λ1·LPIPS + λ2·Sobel梯度 + λ3·颜色统计`，teacher-forced。Phase 2：加终末奖励 `R = ω1·R_fid + ω2·R_temp`，沿 4 步 ODE 反传（BPTT），teacher forcing 概率 1.0→0.2 线性退火 |
| 时序奖励 | `R_temp(t) = -||(ẑ_t − z_prev) − (z_t^gt − z_{t−1}^gt)||₁` —— 匹配**位移**而不是直接最小化帧间差 |
| 数据 | 40+ 序列 / 30k+ 同步的 sparse-intensity–RGB 对，按场景划分 |

论文里的桥样本与速度回归：

```
z_τ = (1−τ)·z_src + τ·z_gt + σ·sqrt(τ(1−τ))·ε
L_LBM = E[ || v_θ(z_τ, τ, c) − (z_gt − z_τ)/(1−τ) ||² ]
```

值得注意的是：**他们的时序奖励直接以 `z_prev` 为参照**。这是整个设计里唯一一项
"最小化它必须读上一帧"的损失。Phase 1 的逐帧损失（LBM/LPIPS/grad/color）其实
也可以不看 history 就降低；真正把 history 变成必需的是 Phase 2 的 `R_temp`
加上 teacher forcing 的数据构造。

---

## 2. 必要性判据

设 `θ_h` 是 history 通路的参数，`L` 是训练目标。时序条件会被使用，当且仅当

- 它提供了当前条件给不出的信息（`I(target; history | current conditions) > 0`），**且**
- 存在损失项，其最小值**必须**经过 history；换句话说 `min_θ L(θ_h ≠ 0) < min_θ L(θ_h = 0)`。

只要第二条不成立，`θ_h → 0` 就是最优解，权重衰减还会加速它。

| 维度 | Cyclops | 我们的 Stage F |
| --- | --- | --- |
| 当前条件含什么 | 单通道稠密强度（几何 + 反射率，**无颜色**） | 卫星 RGB（**含颜色/纹理**）+ LiDAR（几何） |
| history 的独有信息 | 外观/颜色的确定性 | 与当前条件高度重叠 |
| 是否存在以 history 为参照的损失项 | **有**（`R_temp` 直接比较 `ẑ_t − z_prev` 与 GT 位移） | **没有**（eps 与 x0 都是当前帧重建） |
| 时序模块与主干 | 联合训练（整个 velocity field 端到端） | 冻结主干，只训新模块 |
| 注入位置 | 每个分辨率的 cross-attention 之后 | 单个 decoder block（8×32 / 640 通道） |
| 连续帧数据 | 30k+ 对，teacher forcing + scheduled sampling | 10.6k 对，但目标函数不需要 history |

推论（可检验）：如果第 2 节判据是主因，那么在**把卫星条件置零**的臂里，颜色只能由
history 提供，`correct` 应该明显优于 `disabled`。如果连那一臂都没有收益，问题在读出
机制而不是必要性。

---

## 3. 我们的闪烁来自哪里（代码级证据）

| 事实 | 位置 | 含义 |
| --- | --- | --- |
| `default_noise` 是模块级 bank，导入时创建一次，所有帧、所有 step 都复用同一个 `index` | `models/KITTI_geo_ldm_diffusion/ddim_KITTI.py:26,630,687` | 进程内逐步噪声**本来就与帧无关** |
| `seed_step_noise` 只在每次运行开始时重播一次这个 bank | `tools/ar_dyn_utils.py` | 同一 run 的所有帧共享同一组逐步噪声 |
| v0 实测：只共享 `x_T`，比值 1.978 → 1.975 | `docs/temporal_history.md` §1 | 初始噪声对闪烁几乎无贡献 |
| v0 实测：per-frame 1.630 → autoregressive 0.820 | 同上 | 复用**累积 latent** 才是有效杠杆 |
| v0 实测：posewarp2 1.305、instance 1.302 | 同上 | 几何感知的复用介于两者之间 |

（诚实标注：v0 的 per_frame 与 shared 是两个独立进程，逐步噪声 bank 可能不同，
所以这三行不能当作严格受控实验；但 2、3 两条来自代码，且差异量级只有 0.003。）

结论：**闪烁的主体是"条件漂移 → 同一 3D 表面被重新渲染成不同外观"**，
而不是"随机数没对齐"。这直接决定了机制选择：

- 不要再去调噪声（共享 `x_T` 已被证明无效）；
- 要在推理期把上一帧的**内容**按几何对应搬进来；
- 搬进来的东西必须能被冻结主干直接消费（否则又变成一个要被"学掉"的新模块）。

---

## 4. 判决实验（判定根因 a / b / c）

三个候选原因：

- **(a) 条件冗余**：目标函数不要求读 history（第 2 节）。
- **(b) 掩码塌缩**：几何在 (16,64) 建立，注入 block 在 (8,32)，`_resize_grid` 要求
  2×2 四个子格全部 valid。若有效比例趋近 0，`hist_delta` 逐元素严格为 0。
- **(c) 读出太弱**：残差幅度小，或训练把它压回了 0。

### 4.1 零代码版本（今天就能在服务器上跑）

采样器已有 `--disable-history` 与 `--uncond-cfg`（把卫星条件置零）：

```bash
# 臂 A：卫星条件 + history 开 / 关
python tools/generate_kitti_geometry_history.py ...                 # (sat on,  hist on)
python tools/generate_kitti_geometry_history.py ... --disable-history  # (sat on,  hist off)

# 臂 B：卫星条件置零（Cyclops 式：只剩几何）+ history 开 / 关
python tools/generate_kitti_geometry_history.py ... --uncond-cfg 3                    # (blind, hist on)
python tools/generate_kitti_geometry_history.py ... --uncond-cfg 3 --disable-history  # (blind, hist off)
```

注意 `--uncond-cfg` 会同时改 guidance scale，所以**跨臂不可比**；但每个臂内部的
on/off 对比是配对的，足够定性。

### 4.2 严格版本（本次新增）

`tools/probe_history_necessity.py`：固定一个 frame pair、固定 timestep、固定扩散噪声、
固定 VAE 采样，跑满 8 个格子（4 条件 × 卫星开/关），并同时打印
**raw 覆盖率（16×64）与注入 block 实际读到的覆盖率（8×32）**：

```bash
python tools/probe_history_necessity.py \
  --config <run>/base/cfg_run_config.yaml \
  --sd-base-ckpt /mnt/shizhm/BasicModel/checkpoints/sd-v1-4.ckpt \
  --ckpt <run>/base/cfg_step_250000.pt \
  --hist-ckpt <run>/stage_f_generator/geometry_history_step_1000.pt \
  --manifest <train_manifest.jsonl> --kitti-root /mnt/shizhm/DATA/KITTI/KITTI_RAW \
  --lidar-ray-feature-cache-root <...> --image-semantic-cache-root <...> \
  --block-indices after_bottleneck --history-dim 64 --heads 4 --dim-head 32 \
  --appearance-x0-weight 1.0 \
  --split train --pair-index 0 --timesteps 250,750 --out probe_necessity.json
```

判读表（`benefit = disabled − correct`，> 0 表示 history 有用）：

| 观察 | 结论 | 下一步 |
| --- | --- | --- |
| 卫星置零臂里 benefit 明显 > 0 | (a) 条件冗余成立 | 做 M1 / M3 |
| 两臂 benefit ≈ 0，且 `effective_fraction_at_block ≈ 0` | (b) 掩码塌缩 | 先修几何分辨率与注入点 |
| 两臂 benefit ≈ 0，但 `effective_fraction_at_block` 正常、`residual_to_condition` 很小 | (c) 读出被压回 0 | 换读出方式，别再加大容量 |
| 只有 `correct > wrong_history` | history 确实被读到了，只是没用于降损 | 说明需要 M3 的目标项 |

### 4.3 训练侧常态版本

`fixed_probe(..., satellite_arms=(False, True))` 已支持这个轴（默认仍是单臂，
训练行为不变）；新增的测试在 `tests/test_geometry_history_training.py`。

---

## 5. 机制选项

按"在我们当前约束下落地的难度"排序。

### M1（免训练）几何偏置的跨帧自注意力
- **问题诊断**：我们加了新模块（新参数 + 冻结主干）。新参数既需要训练信号，又会
  被权重衰减压到 0；而它注入的值经过自己的投影，对主干来说是分布外的。
- **做法**：在 `BasicTransformerBlock.attn1`（本来就是 latent token 上的空间自注意力）
  里，把**上一帧同层、同去噪步**的 token 追加进 K/V，用的就是该 block 自己的
  `to_k` / `to_v`。我们的对应关系作为加性偏置进入 logits：
  `score[cur_i, hist_j] += -||p_i − q_j||² / (2σ²)`，`p_i` 是格 `i` 在上一帧的预测坐标
  （`history_grid`），`q_j` 是上一帧的格中心；`history_valid=False` 的格对 history 部分
  给 `-inf`（等价于该格完全不看上一帧，即当前行为）。
- **为什么可能 work**：K/V 来自主干自己的投影，是 in-distribution；零新参数，
  没有可以被"学掉"的东西；也不需要连续帧训练。
- **代价**：需要上一帧同层同 step 的特征。两条路：(i) 两遍采样（video editing 一脉的
  标准做法：先正常去噪并缓存，再用跨帧注意力重跑一遍，约 2× 成本）；
  (ii) 缓存上一帧每 step 每层的中间特征（16 层 × 50 步 × 1024 token × 320 维 fp16
  ≈ 0.5 GB/帧，单帧可控）。
- **风险**：会退化到"复制上一帧"（v0 的 autoregressive 已经证明会过平滑到 ratio 0.82）。
  缓解：偏置只在有效对应格生效；对动态/新可见区域保持原样（用 `history_valid` 与
  一致性状态掩码）。
- **参考**：Tune-A-Video [arXiv:2212.11565](https://arxiv.org/abs/2212.11565)、
  Rerender-A-Video [arXiv:2306.07954](https://arxiv.org/abs/2306.07954)、
  TokenFlow [arXiv:2307.10373](https://arxiv.org/abs/2307.10373)。

### M2（免训练）条件沉默区的 latent 搬运

- 用对应关系把上一帧 latent 搬到当前格，只在**条件欠定/无效**区域混合，并对搬运结果
  重新注入噪声以保留运动；不引入任何新参数。
- 这是 v0 `posewarp2`/`instance` 的 latent 级改进版。v0 已经量化过它的边界：
  慢速运动目标（< 0.35 m/帧）会被当作背景搬运，身份仍被重采样。
- 优点：立即可跑、可解释、有历史数据支撑（1.630 → 1.305）。
- 缺点：整帧级搬运在动态目标上必然出错，需要与 M1 结合（M1 的偏置天然是逐格的）。

### M3（需要短训，最小改动）让 history 成为必需

两条互补的做法，都不需要新的数据源：

1. **空间化的条件盲区**：把现有的整帧 `satellite_condition_dropout_prob` 改成
   "随机区域置零"，并要求在这些区域重建 —— 该区域的颜色只能来自 history。
   这样 history 从"可选"变成"必需"，判据第 2 节的条件自然满足。
2. **表面级时序一致性项**：Cyclops 的 `R_temp` 在我们这里的对应物不是 latent 位移，
   而是**同一 3D 表面在不同帧的渲染一致**：用 `history_grid` 把上一帧 latent 采样到
   当前格，只在有效对应格上要求 `pred_x0` 与之接近（外观），并允许几何相关的位移。
   这正好用上我们没有别人有的东西：逐格对应关系。

### M4（需要重训生成器）完整 Cyclops 式

联合训练整个生成器 + 时序奖励 + scheduled sampling。权重在服务器上有，但要重训
生成器本身，远超当前预算（现有实验都是"冻结 backbone + 短训适配器"），不建议先走。
如果 M1/M3 证明时序信息确实可用，再评估是否值得动 backbone。

---

## 6. 从我们技术特点出发的设计原则

1. **我们的独有资产是逐格 LiDAR 对应关系 + 彩色卫星条件，不是可训练容量。**
   所以优先把对应关系做成推理期的偏置/算子，而不是训练一个读它的模块。
2. **时序信息只应作用在"条件沉默区"**：天空、植被、上部立面、以及条件无法分辨的
   动态目标外观。这可以用现有的 `history_valid`、地面/非地面代理掩码近似，
   而不需要新的标注。
3. **目标要定义在表面上，而不是像素或 latent 上。** Cyclops 没有跨帧稠密几何对应，
   只能用 latent 位移做时序奖励；我们有对应关系，可以直接写
   "同一个 3D 表面在两帧的渲染应当一致"。这是我们把他们的思路做成自己东西的切入点。
4. **与 Cyclops 的本质差异**：他们的颜色是"发明"的（源无颜色），所以时序注意力是
   颜色的唯一来源，天然必需；我们的颜色是"从地图看到的"，所以时序机制要解决的是
   **条件漂移下的一致性锚定**，而不是提供信息。这决定了我们不该照搬
   "把上一帧当 K/V 让模型自己挑"，而应该**用几何对应约束它在哪儿挑**。
