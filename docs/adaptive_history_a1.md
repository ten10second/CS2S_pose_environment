> 历史实验记录：以下配置、结论和命令对应当时版本，不是当前启动指南。
> 当前方案见 [时序说明](temporal_history.md) 和 [全量 B/C 协议](temporal_full_bc.md)。
> 部分旧实验脚本已退役，原源码保留于服务器 `CS2S_run_control/temporal_cleanup_commit_20260920/before_source.tar.gz`。

# A1：状态感知的静态历史融合

## 固定范围

依据用户贴入方案，仅实现 A1，不增加 A2 第二尺度，不解冻基模，不增加 loss。A0 的冻结主干、固定历史残差和 decoder 层级位置都不是已证实的故障根因；本轮检验融合本身增加可训练能力是否有帮助。

## 结构

保留六通道 RGB/valid/measured/estimated 输入和 6→16→32→64 三层 stride-2 encoder。新增 `static_adaptive` 模式，A0 `static_dense`、旧静态及单帧路径继续可用。

同一个 decoder hook：`len(output_blocks)-(num_res_blocks+1)`。实际生产配置 num_res_blocks=2，channel_mult=[1,2,4,4]，因此 hook 为 output_blocks[9]（共 12 个 block），通道 320，空间 16×64，time embedding 1280。该 block 内已执行 Sat SpatialTransformer；LiDAR 金字塔在 encoder 输入残差与 skip/主路径传到这里，ray posterior 条件也保持原融合方式。

`AdaptiveHistoryFusion`：当前 h 经 GN+1×1 到128；R16 经1×1到128；concat→3×3→加时间投影→SiLU→3×3→SiLU→零初始化1×1，输出 Δh 后与 h 相加。只将最后投影的 weight/bias 置零，没有零乘法门。原 decoder、VAE、Sat/LiDAR 参数全部冻结；后续 decoder 保留 autograd，不 detach h。

## 缓存与接口

推理时在一次 `ddim_sampling` 开始编码历史一次，仅缓存 R16。每个去噪步重新计算融合残差。每次采样主动丢弃传入的旧 dense_features 并从本次 RGB 重建，避免颜色干预读到陈旧缓存。历史 dict 使用局部副本，不污染输入。

训练不缓存跨 optimizer step 的特征；cache 仅允许 eval+no_grad，校验 shape/dtype/device。历史是否存在由原始 dense_valid/ enabled 决定，不从消融后的 mask 输入反推。CFG 仍按 [整批 uncond,整批 cond] 扩展所有历史张量，两分支均接收历史，未改变 CFG 语义。

adapter-only checkpoint A1 使用 v3，保存 mode、input_variant、hidden_dim、fusion_dim、time_embed_dim、底座身份；拒绝与 A0 v2 混装。

## 小样本验证

A1 RGB-only 和完整 types 两组，仍用6个训练/4个留出、B2、LR1e-4、1000步、seed3407、50步DDIM/CFG7.5。复用原 dense references 与 A0 已完成实验作对照，不重跑几何，不启动全量训练。不把1000步称为收敛。

先测初始化完整 ON/OFF 采样相等；检查几次更新后 encoder、fusion、output 均有梯度；确认基模哈希与 OFF 完整采样不变。

额外诊断对 A0/A1 的 RGB-only 和 types 使用相同评估 ROI：heldout_01 左侧建筑 [0,0,96,84]、train_05 中部建筑 [140,0,310,85]。ROI 是人工选定的评估矩形（会含少量其他像素），不是训练静态 mask，不声称语义分割。只在 ROI∩valid 交换历史 R/B；固定 x、t、当前条件记录 encoder→Δh→epsilon uncond/cond/CFG，再做同种子完整采样、输出 ROI/区域外响应与红蓝方向。固定状态探针 x 为固定随机 latent，仅用于通路敏感性，非 GT 去噪误差评估。

上一轮 0.28/255 是留出上半图平均，不是整图也不是语义建筑指标。响应变大不自动等于正确继承，需检查方向、局部性和窗户/几何结构。
