> 2026-09-20：此实验已停止，adaptive/centered A1 实现与训练入口已删除。以下为历史记录，命令不适用于当前代码。

# 全量一轮 B/C：检查小样本记忆是否掩盖历史收益

## 本轮固定

- 新加载相同 200k 单帧底座；历史分支重新初始化，不续用小样本训练权重。
- B：centered A1 历史编码/融合 LR 1e-4，decoder output_blocks 9/10/11 与输出层的活跃参数 LR 1e-5。
- C：相同 decoder 范围和 LR；历史关闭、历史分支冻结。
- 单尺度 16×64，完整 RGB + valid + measured + estimated 六通道，F(h,H,t)−F(h,0,t)。
- 仍为原 epsilon 扩散损失；前后帧同步轻度颜色增强，概率 0.7；不新增 loss/attention。
- 单卡 B8，各训练一次全量数据。B GPU4，C GPU5，独立实验非 DDP。

## 数据和预算

原地理隔离 train manifest 有 17,055 帧，形成 14,463 个相邻候选帧对。排除固定 10 对观察样例涉及的帧后，共 **14,448 个训练帧对**。每组 **1,806 次更新，1 epoch**。清单用 seed 3407 固定打乱；同一 epoch 不重复、不补齐、不丢尾。

沿用固定 train8、地理留出8和独立观察10，用于与上一轮逐图比较；观察样例帧不参加此次训练。训练样例彼此允许相邻帧重叠，这是自然两帧训练数据，不把这种重叠当成独立评估证据。

## 条件数据和空间

旧缓存仅覆盖子集。为全量新建紧凑参考 NPZ：前后 RGB uint8、source index、valid/measured/estimated masks。复用已有重投影代码，不保存全量原生分辨率深度及大体积条件 PT。当前帧 sat/LiDAR 条件由冻结分支在线编码。

GPU5 同时运行轻量深度预处理，按训练清单顺序供应参考，训练按需等待。几何只使用前帧 RGB、两帧 LiDAR 和相对位姿，后帧 RGB 仅作为训练目标。原有实测/估计支持含义保持不变。生产/训练均检查剩余磁盘空间。

## 验证与结果产物

- 训练前、半程 903 步、结束 1806 步：固定 t100/500/900、固定噪声，比较 B correct/off/wrong 与 C off。
- 结束后：26 对固定样例，50 DDIM、CFG7.5、同种子生成。
- 保存半程与最终 checkpoint，各组最多保留2个；checkpoint 为底座+可训练参数组合使用的 probe 权重，不包含优化器续训状态。
- 每步记录样本名、条件/目标latent/噪声/增强哈希、梯度、AMP 重试与 epoch 进度。
- 结束自动检查两组所有 step 的匹配性、完整一轮覆盖和冻结参数哈希。

判断重点：留出样例 B 是否优于 C，正确历史是否优于错误历史，以及建筑立面颜色/纹理是否继承且未变模糊。只看训练 loss 降低不能判定成功。

## 已验证

新增全量读取/遍历测试4项、复用 probe 测试7项通过。B/C 各一次真实 B8 更新与固定验证通过：首 loss 均 0.11398522555828094；数据/条件/增强/目标latent/噪声完全匹配；冻结权重不变；AMP 重试0。独立代码审查未发现阻塞问题。

## 路径

- 代码：`/mnt/shizhm/CS2S_pose_environment_temporal_v22/tools/train_centered_decoder_full.py`
- 数据准备：`tools/prepare_full_centered_history.py`
- 控制/日志/紧凑参考：`/mnt/shizhm/CS2S_run_control/centered_decoder_full_20260920`
- 训练结果：`/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/centered_decoder_full_20260920/{B,C}`
- Supervisor：控制目录 `supervise_full.py`；状态 `supervisor_status.json`，失败写 `supervisor_failed.json`。
- 训练实时状态：结果各组 `status.json`、`metrics.jsonl`；成功完成写 `done.json`。

原单帧训练仓库没有修改。此轮只批准1 epoch，不会自动追加训练轮数。
