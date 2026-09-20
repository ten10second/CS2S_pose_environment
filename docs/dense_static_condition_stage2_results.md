> 历史实验记录：以下配置、结论和命令对应当时版本，不是当前启动指南。
> 当前方案见 [时序说明](temporal_history.md) 和 [全量 B/C 协议](temporal_full_bc.md)。
> 部分旧实验脚本已退役，原源码保留于服务器 `CS2S_run_control/temporal_cleanup_commit_20260920/before_source.tar.gz`。

# 稠密静态历史条件：第二阶段结果（2026-09-19）

## 结论

P1–P4 已实现并完成四组匹配小样本训练。稠密历史条件能改变输出，在部分图像指标上优于稀疏参考；**尚未通过“继承建筑立面纹理”的验收**。RGB-only 在本轮代理指标上最好，来源 mask 的额外价值尚未得到证明。完整条件组对历史颜色变换反应很弱，不能将图像误差下降解释为已经保持建筑身份。

没有继续扩大几何实验、启动全量训练或添加一致性损失。所有本轮进程已结束，GPU 4/5 已释放；旧全量时序训练保持暂停。

## 实现和实验设置

- 复用既有稠密重投影缓存，保存 warp RGB、valid、measured、estimated，并附前后帧身份与缓存来源哈希。没有重新计算或扩大几何验证。
- 独立 `static_dense` 模式：512×128 RGB 与掩码经三层 stride-2 卷积编码，通过零初始化投影注入现有最高分辨率 decoder 位置；每个去噪步都能读取。
- measured/estimated 互斥且并集为 valid；hole 可由 valid 取反得到。四组使用相同六通道网络，消融对应的输入通道置零。
- 冻结原 UNet（含 decoder）、VAE、Sat/LiDAR 编码器，只训练历史编码器和注入投影。保留原单帧与旧静态接口。
- 从同一 200k 单帧底座、新初始化历史参数开始。6 个训练帧对、4 个留出帧对，每组 1000 step，B2，LR=1e-4，seed=3407。这是小样本学习探针，不是全量 epoch。
- 四组初始化、每步样本、扩散时间步/噪声生成方案和卫星 dropout 匹配。GPU 4/5 各跑独立组，没有 DDP。
- 仅使用标准 epsilon 预测损失；未对 estimated 区域添加监督，也未启用 measured-only 一致性项。
- 固定 t=100/500/900 监控去噪误差；所有帧对做相同种子、50 步 DDIM、CFG=7.5 完整生成。额外检查完整条件组固定 mask 的 RGB 干预，并在留出集增加第二个采样种子。

## 匹配比较

以下正值表示相对 OFF 误差降低，负值表示变差。去噪指标对固定时间步和样本取均值。图像上半部包含天空、树木等，**不是建筑语义区域**；RGB L1 不等于纹理或身份一致性。

| 条件 | 训练去噪误差改善 | 留出去噪误差改善 | 留出上半图 RGB L1 改善 |
|---|---:|---:|---:|
| 稀疏 LiDAR + 类型 | +0.990% | -0.616% | +0.26% |
| 稠密 RGB | +1.813% | +0.388% | +7.14% |
| 稠密 RGB + valid | +1.439% | -0.194% | +6.00% |
| 稠密 RGB + valid + 类型 | +1.214% | -0.791% | +4.28% |

稠密 RGB 的留出上半图 RGB L1 从 0.265742 降至 0.246757，但 GT 梯度 L1 从 0.057854 升至 0.058323（较差）。因此不能用颜色/亮度误差的改善代替窗户、边缘和立面保持的证据。

完整条件组在第二个种子下，留出上半图 RGB L1 从 OFF 的 0.261433 降至 0.247957，梯度 L1 从 0.060018 降至 0.059792。这说明图像代理指标改善不只发生在第一个种子；样本仅 4 对，仍不足以宣称泛化或清晰度问题解决。

## 历史 RGB 干预：条件被读取了吗？

仅对 `dense_types` 组进行，valid/measured/estimated、当前条件和采样噪声保持不变：

| 干预 | 训练上半图输出变化 L1 | 留出上半图输出变化 L1 |
|---|---:|---:|
| 历史 RGB 全置黑 | 0.014913 | 0.030499 |
| 历史 RGB 交换红蓝通道 | 0.000710 | 0.001108 |

留出集交换参考红蓝后，输出变化约 0.28/255；输出平均红蓝方向与参考变换同向仅 2/4 对。完整条件组能够响应历史 RGB 的大幅变化，但对这一颜色干预缺乏明显响应。**这支持“历史外观读取仍弱”，不能证明已经学会颜色或立面身份继承。** 置黑是分布外干预，单凭它改变输出也不能证明正确利用历史。

注意：这些干预结论仅针对完整 mask 组，尚未给 RGB-only 组做相同因果检查；不能把它们直接外推到所有组。

## 图像证据与失败例

- [heldout_01：所有条件对照](/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/stage2_20260919/comparisons/heldout_01.jpg)：warp 中左侧红褐色建筑清楚可见，生成仍呈现白/灰立面及不同窗户布局。它说明有参考并不自动等于输出继承参考。
- [heldout_01：固定 mask 的历史干预及第二种子](/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/stage2_20260919/interventions/heldout_01/comparison.jpg)：交换历史红蓝后结果几乎不变，第二种子同样没有恢复参考立面。
- [heldout_02：店面街景](/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/stage2_20260919/comparisons/heldout_02.jpg)：历史组改变了部分外观，但仍不能称为窗户与立面严格保持。
- [train_05：训练样本](/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/stage2_20260919/comparisons/train_05.jpg)：原底座本就相似，较小变化不能独立证明历史分支有效。
- [全部 10 对图像画廊](/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/stage2_20260919/index.html)，包含训练与留出、成功代理指标与失败视觉案例。

## 验证与边界

- 远端静态相关单元测试 50 项通过，包含输入合法性、梯度、空条件/OFF 回退、批量与 checkpoint 兼容；`git diff --check` 与编译通过。
- 四组均完成 1000 step，退出码 0，无 AMP 重试。原模型各部分前后哈希相同。
- 四组固定种子的 OFF 完整采样 PNG 逐像素相同；完整条件组 checkpoint 重载后输出也逐像素复现。
- 目标 RGB 只用于正常训练监督和评估，不用于构建或推理历史条件。
- 数据沿用原 GPS30m 划分来源；此小集合碰巧 train/heldout drive 不交叉，不改变原单帧划分规则。
- measured 是深度检查通过，**不是静态建筑语义标签**。目前没有可靠车辆关联，不宣称车辆颜色/型号问题已解决。
- 当前 Sat/LiDAR 保留原条件分支；前帧 Sat 没有显式参与本次静态参考编码，本轮不是完整五输入模型。
- 分辨率 512×128、6/4 小样本、单训练种子、有限推理种子，无法据此判断大规模表现。

## 下一步建议（尚未执行）

保持已有几何参考，集中检查生成器侧：先在本轮表现较好的 RGB-only 组复用颜色干预，确认是否真正传递外观；再用固定单帧对检查当前注入接口能否拟合指定立面。只有证明这个通路有外观表达能力，才考虑有限解冻或调整注入位置。当前没有足够依据直接追加一致性损失或全量训练。

## 文件与复现入口

服务器代码：`/mnt/shizhm/CS2S_pose_environment_temporal_v22`。

本轮输出：`/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/stage2_20260919`。

进程命令及退出记录：`/mnt/shizhm/CS2S_run_control/temporal_static_stage2_20260919`。

主要修改：`ldm/modules/static_history.py`、`ldm/modules/persistent_history.py`、`models/KITTI_geo_ldm_diffusion/openaimodel.py`、`tools/infer_temporal.py`；新增 `tools/prepare_dense_static_reference.py`、`tools/train_dense_static_history.py`、`tools/eval_dense_history_interventions.py` 和两组测试。没有提交 git commit。

机器可读指标：`monitor_summary.json`、`image_metrics.json`、`image_summary.json`、`intervention_metrics.json`。每组目录保留 args、metrics、monitor、done 和 adapter-only checkpoint。分析脚本在 `analysis/` 下存档（路径默认为本机本次实验目录）。
