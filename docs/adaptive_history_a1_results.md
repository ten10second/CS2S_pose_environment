> 历史实验记录：以下配置、结论和命令对应当时版本，不是当前启动指南。
> 当前方案见 [时序说明](temporal_history.md) 和 [全量 B/C 协议](temporal_full_bc.md)。
> 部分旧实验脚本已退役，原源码保留于服务器 `CS2S_run_control/temporal_cleanup_commit_20260920/before_source.tar.gz`。

# A1 自适应历史融合：实现与小样本结果（2026-09-20）

## 结论

已按用户文档限定实现 A1，保留六通道 encoder、原 decoder 单点、冻结主干和原损失。新增融合能读取当前 h 与时间 embedding，工程与梯度验收通过；**本轮外观保持验收未通过，因此没有进入 A2**。

两组 A1 在1000步后训练去噪误差下降约18%，留出误差反而上升11%–12%，是明显的过拟合迹象。指定建筑的历史颜色变化能传到模型预测，但最终图像的颜色跟随仍极弱。增加可训练融合能力并未在本轮自动产生可靠外观继承。

## 修改范围

- 新模式 `static_adaptive`：h→GroupNorm+1×1→128，历史R16→1×1→128，拼接→3×3→加时间投影→SiLU→3×3→SiLU→零初始化1×1，得到Δh后加回h。
- 只对最终投影 weight/bias 零初始化，无零初始化乘法门。history encoder保留6→16→32→64三层stride-2结构。
- 实际生产 hook 为 `output_blocks[9]`（0起始，共12块），h为[B,320,16,64]，temb宽1280；hook在该block的Sat SpatialTransformer之后，LiDAR金字塔已通过encoder与skip进入当前特征。没有假定decoder天然包含所有条件，已核对具体路径。
- 原UNet（含decoder）、VAE、Sat/LiDAR冻结；通过requires_grad冻结参数，注入后到损失保留autograd，无detach(h)。
- 每次采样编码历史一次，只缓存R16；每个去噪步重新计算fusion。新采样丢弃旧缓存并从当前RGB重建。训练不跨optimizer step缓存特征。
- RGB-only输入mask通道为零，但样本是否有历史仍由真实dense_valid与enabled决定。
- CFG保持历史同时进入[整批uncond,整批cond]，未修改分支语义。新增v3 adapter-only checkpoint，严格检查模式、宽度、variant及底座身份，保留A0 v2和旧静态接口。

## 实验设置与公平性

A1 RGB-only和完整types，各从原200k单帧底座重新初始化历史模块训练1000步，6对训练/4对留出，B2、LR1e-4、seed3407、50步DDIM、CFG7.5。复用A0已完成的相同配置结果；没有重跑或扩大几何实验，没有加新loss、第二尺度或全量训练。

逐步核对样本顺序、t_mean、Sat dropout一致；所有固定噪声OFF监控误差完全一致，10对OFF PNG逐像素一致，基础模型各部分哈希不变。A0/A1结构参数数目不同，不能声称完整参数初始化相同。1000步只作通路探针，不代表充分收敛。

## 去噪误差

相对OFF，正值为改善，负值为变差；固定t=100/500/900与全部帧对取均值。

| 版本 | 训练集去噪误差改善 | 留出集去噪误差改善 |
|---|---:|---:|
| a0_rgb | +1.813% | +0.388% |
| a0_types | +1.214% | -0.791% |
| a1_rgb | +18.324% | -11.838% |
| a1_types | +18.374% | -11.055% |

A1留出图像上半部RGB L1：RGB-only 0.264365、types 0.259482；OFF为0.265742，A0 RGB-only为0.246757。上半部不是语义建筑区域，也不能据此宣称窗户纹理或身份保持。

[heldout_01完整对照](/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/adaptive_a1_20260920/comparisons/heldout_01.jpg)：warp已有红褐色立面，A0与A1依旧输出白灰建筑；A1未修复这个关键失败例。[train_05对照](/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/adaptive_a1_20260920/comparisons/train_05.jpg) 中输出变化也不能证明按参考保持立面。

## 建筑ROI的局部改色诊断

对A0/A1的RGB-only和types均测试。手工选定两个评估矩形：heldout_01左侧建筑[0,0,96,84]与train_05中部建筑[140,0,310,85]。它们不是训练静态mask，会包含少量周边内容。仅对ROI∩valid交换历史红蓝通道；mask、几何、当前条件、种子不变。

在heldout_01中，参考ROI的红减蓝均值变化为 **−0.260497**（RGB范围0–1）。输出结果：

| 版本 | 建筑 ROI 输出变化（0–255 尺度） | 输出红减蓝变化 |
|---|---:|---:|
| a0_rgb | 0.471 | +0.0000588 |
| a0_types | 0.359 | +0.0000118 |
| a1_rgb | 0.534 | -0.0001158 |
| a1_types | 0.363 | +0.0000207 |

A1 RGB-only方向同向，但幅度很小；A1 types仍未跟随正确颜色方向。不能把输出变化稍大视为建筑身份保持。这里是指定建筑ROI的局部干预，与上一轮上半图平均0.28/255不是同一口径。

[局部改色并排图](/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/adaptive_a1_20260920/comparisons/heldout_01_local_color.jpg)：每行依次为改色后的参考、正常参考生成、改色参考生成。窗口及立面的继承仍不足。

### 响应沿哪一步消失？

固定随机latent x、t=500、当前条件，仅改变历史ROI颜色，真实CFG路径的ROI平均绝对变化如下：

| 版本 | encoder特征变化 | Δh变化 | epsilon-u变化 | epsilon-c变化 | CFG预测变化 |
|---|---:|---:|---:|---:|---:|
| a0_rgb | 0.021531 | 0.006613 | 0.000541 | 0.000554 | 0.003196 |
| a0_types | 0.011206 | 0.003074 | 0.000257 | 0.000242 | 0.002352 |
| a1_rgb | 0.002549 | 0.007656 | 0.000738 | 0.000810 | 0.004649 |
| a1_types | 0.002249 | 0.002650 | 0.000518 | 0.000618 | 0.003490 |

所有环节均能检测到非零响应；CFG后响应仍然存在，不能归因为“两个分支相消”。同一次固定状态比较中，历史注入前h逐元素相同，排除了改色同时改变当前条件的混淆。A1的残差还随t/当前h改变，A0固定残差在t=100/500/900保持相同。

**这些中间张量量纲和尺度不同，不能横向除出一个“信息损失百分比”，也不足以锁定唯一瓶颈。** 证据支持通路未断、最终颜色传递弱。结合训练/留出差距，新增融合可能更多拟合了少量训练样本的去噪修正，而没有建立可泛化的历史外观依赖；这是解释性假设，不是已证明根因。

## 验证

- 静态相关61项单元测试通过：零初始化/OFF/空历史、RGB-only真实present、更新数次后梯度、状态/时间敏感性、缓存生命周期与B2 CFG顺序、v2/v3隔离；包含CUDA半精度缓存路径。
- 两组完整训练退出码0，第2–1000步encoder/fusion/output均有非零梯度，无AMP重试。
- 初始化完整ON/OFF采样回退通过；原权重哈希不变，所有OFF完整生成逐像素相同。
- 四组诊断均完成；两对样本checkpoint重载的正常生成与各自原输出逐像素相同。
- `py_compile`与`git diff --check`通过；review发现诊断工具CPU分支未保护CUDA调用，已修复（GPU实测完成，CPU整模型诊断未运行）。

## 下一步边界

本轮A1没有达到外观保持验收，暂不扩A2，不自动解冻或增加loss。若继续验证，优先按用户文档做独立的“同帧历史和目标同步局部改色”诊断，检查模块能否学习明确且可证伪的外观映射；不能只改历史却用原目标监督。该额外训练尚未执行。

## 代码和产物

服务器代码：`/mnt/shizhm/CS2S_pose_environment_temporal_v22`。

修改：`ldm/modules/static_history.py`、`ldm/modules/persistent_history.py`、`models/KITTI_geo_ldm_diffusion/openaimodel.py`、`models/KITTI_geo_ldm_diffusion/ddim_KITTI.py`、`tools/infer_temporal.py`、`tools/train_dense_static_history.py`。
新增：`tools/eval_adaptive_history_path.py`、`tests/test_adaptive_static_history.py`、`tests/test_adaptive_static_probe.py`、`docs/adaptive_history_a1.md`。

输出：`/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/adaptive_a1_20260920`；命令、日志、退出记录：`/mnt/shizhm/CS2S_run_control/temporal_static_a1_20260920`。

实验已结束，GPU4/5释放；原全量训练仍保持暂停。未提交git commit。

[全部对照画廊](/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/adaptive_a1_20260920/index.html)。机器可读文件：`monitor_summary.json`、`image_metrics.json`，每个`*_path/`下有逐t的`path_metrics.json`和ROI`image_metrics.json`；保留所有失败例。分析脚本在`analysis/`。
