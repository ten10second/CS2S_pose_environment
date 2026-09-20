> 历史实验记录：以下配置、结论和命令对应当时版本，不是当前启动指南。
> 当前方案见 [时序说明](temporal_history.md) 和 [全量 B/C 协议](temporal_full_bc.md)。
> 部分旧实验脚本已退役，原源码保留于服务器 `CS2S_run_control/temporal_cleanup_commit_20260920/before_source.tar.gz`。

# 静态两帧历史：P0–P2 实施与验证

## 范围与当前状态

日期：2026-09-19。用户要求先完成接口检查、静态对齐与可视化、静态条件接入和小样本验证。
工作目录 `/mnt/shizhm/CS2S_pose_environment_temporal_v22`，分支 `temporal/v22-causal-appearance`，
HEAD `c0cd74153bec0007de8ca1bd1978a76edd5ee145`，现有时序代码包含未提交修改。
原全量训练保持暂停：最后日志 step 14157，最新完整 checkpoint step 12656；不自动恢复。
此次新分支从 200k 单帧底座初始化，不复用此前时序/decoder 更新。

## P0：实际接口

| 内容 | 实际位置与结论 |
|---|---|
| 底座加载 | `tools/train_temporal_pairs.py:load_base`：严格加载 denoise_model、condition_model_sat、lidar_context_model |
| 基础权重 | `/mnt/shizhm/DATA/KITTI/CS2S_results/kitti_ray_posterior/pixel_v22_step200000_snapshot.pt` |
| 基础 SHA256 | `7047b249e6ed1ff0c015c048210c4ce29da4f23ee48acda2c1c4b38810d30406` |
| 配置 | `configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_pixel_v22_cfgdrop10.yaml` |
| 当前条件 | `encode_conditions`：卫星 context、pixel LiDAR context/evidence、相机参数与几何 mask |
| RGB/VAE | RGB 128×512，PIL Resize 后 ToTensor [0,1]；VAE 输入转为 [-1,1]，mode latent×0.18215；latent 4×16×64 |
| 扩散 | 1000 训练时间步，原 epsilon MSE；推理复用 KITTI DDIM，固定初始噪声；不修改 prediction type/scheduler |
| 历史接口 | UNet `configure_temporal_history`，最高分辨率 decoder 第一个 block 后；每个去噪步接收独立 history |
| 旧版本 | geometry/content 模式保留。实测点只替换一个深度候选，全局池化外观仍参与竞争；不视为确定性静态读出 |
| 数据 | `dataloader/KITTI_raw_sat_lidar.py`；原始 LiDAR、OXTS、相机标定、前帧 RGB 存在 |
| 几何 | `RawKittiGeometry` 使用完整 P_rect_02、rectification、Velodyne/camera/IMU 外参；relative_velo_pose 为 prev→cur |
| 图像坐标 | 新静态投影需使用 `(u + 0.5) * resize_scale - 0.5`，与 PIL resize 的像素中心一致；不改旧分支坐标行为 |
| 数据隔离 | 复用原 GPS 30m buffer 清单（train 17055 条、test 7542 条）。当前清单 drive 交集为空是观察结果，不是新增分割规则。同一帧对必须同 drive 且相邻，两帧均须属于同一原始 split；不同 split 可以包含同 drive 的地理分离区段 |
| 车辆关联缺口 | manifest 声明 tracklet_xml_path，但迁移后的 KITTI 根下未找到 XML；现有 parser 即使读到 XML 也不返回持久 track ID |
| 其他框文件 | 数据目录存在 bbox_2d/bbox_3d 检测结果，但来源/坐标格式/身份关联尚未核实，不自动作为可靠动态条件 |

不能从 manifest 的 `has_dynamic_xml` 直接认定当前文件可用。未验证稠密深度、车辆轨迹和精确实例 mask。

## P1：静态对齐

新增 `tools/temporal_static_geometry.py` 与 `tools/visualize_static_history.py`。
沿用传感器读取/标定工具，前帧 LiDAR 在源图取色，完整位姿投影到当前视角，输出稀疏 RGB、support、confidence、strict mask。
源与目标可见性均由包含全部测量点的 z-buffer 判断，再排除动态/非静态区域；不能先删前景再把后面的墙当可见。
不补洞、不膨胀，不把低置信候选当作已验证表面。当前 GT 只在可视化/评估代码读取。
深度容差首轮沿用已有 0.75m 做可比诊断，记录实际匹配误差；不宣称它是标定后的最优容差。

用户明确要求取消人工圈图范围。正式 P1/P2 使用整幅图像的传感器几何，不以 tracklet 或建筑标注为前置条件。
`strict_mask` 表示在自车运动假设下通过前后深度检查的对应，**不是语义静态标签**。
若 tracklet 存在，可额外排除其目标投影；缺少时记录动态排除不可用。运动车辆仍可能偶然通过深度检查，不能据此宣称获得车辆身份。
源和目标 RGB 均不参与几何匹配；源 RGB 仅为 LiDAR 样本取色。未被 LiDAR 支持的建筑高层不被伪装成可靠对应。
早期人工区域可视化保留为被放弃的诊断记录，人工区域 probe 在第一次参数更新前因组批问题结束，未产生模型权重；后续不使用其区域配置。

## P2：历史条件与冻结边界

新增 `ldm/modules/static_history.py`，在原注入位置提供 `mode=static`。
输入：对齐 RGB、有效支持、置信度。RGB 按有效样本归一化下采样，空洞的黑色不会参与颜色平均。
两层轻量卷积与零初始化输出投影产生局部残差；不做第二次 attention 搜索，不使用全局外观 fallback。
最终注入位置由支持及置信度控制，空 mask/关闭历史回到同一单帧路径。
支持同一 CFG 批次中的独立帧对；两个 CFG 分支均保留静态历史，沿用原卫星 CFG 语义。

`tools/train_static_history.py` 仅做有明确步数上限的小样本 probe：冻结整个原 UNet（包括 decoder）、VAE、卫星和 LiDAR 编码器，只训练新增 adapter。
标准目标保留 epsilon MSE，暂不增加静态保留 loss；正常允许梯度穿过冻结 decoder 到 adapter。
单帧底座及旧实验权重不覆盖。静态 adapter 单独保存，附基础权重哈希与完整设置。

## 验收层次

1. 几何合成单测、真实帧对对齐图及有效覆盖率。
2. 原单帧 vs 零初始化，关闭历史 vs 空 mask 的数值回退。
3. 新参数梯度非零、基础参数未更新、CFG 批次不串样本。
4. 固定帧对/噪声的训练前后完整采样，同时检查外观继承、目标质量与模糊。
5. 仅总 loss 下降、条件引起变化，不足以认定静态外观已经保持。

此次 P0–P2 不包含自动车辆关联、LiDAR 未覆盖建筑高层的完整对应、长序列递推或全量训练。

## 划分依据核验

原 `manifest_stats.json` 的 `split_strategy=remove_train_frames_within_gps_buffer_of_test_route`，`buffer_m=30.0`。
`tools/build_kitti_gps_buffer_split.py` 保留原 heldout 路线，去掉距离其小于 30m 的训练帧；从 17955 条训练记录移除 900 条。
时序代码直接复用输出 manifest，不重新按 drive 或随机帧对划分。每个样本的参考帧和目标帧都要留在原 split 内。

## 当前固定目标与实现差距

完整目标：输入前帧 GT RGB、前后帧 LiDAR、前后帧 Sat，生成后帧 RGB；当前几何服从后帧测量，并保持车辆和建筑外观。当前优先验证建筑/背景静态分支。

本次实现已使用前帧 GT、前后帧 LiDAR、位姿/标定和后帧 Sat。前帧 Sat 尚未显式接入历史关联路径；未覆盖建筑高层的对应和车辆身份关联尚未实现。因此不能把当前稀疏静态分支称为完整五输入方案。

## 2026-09-19 小样本验证结果

正式输出：`/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/p0p2_20260919/sensor_probe128`。
运行记录：`/mnt/shizhm/CS2S_run_control/temporal_static_p0p2_20260919/sensor_probe_command.json`。

- 使用原划分中的 6 个训练帧对、4 个留出帧对，无人工 ROI；GPU 4，batch 2，128 次更新，adapter LR 1e-4。
- 18:37:14 开始，18:40:38 正常结束；未启动全量训练。原全量任务保持暂停。
- 底座、decoder、VAE、Sat/LiDAR 编码器哈希均不变；关闭历史后的完整采样结果与训练前完全一致。
- 几何严格对应覆盖整幅图像约 8.9%–11.6%。特征网格的支持覆盖率因下采样而更高，不能当作原图可靠覆盖率。
- 训练时新参数梯度、残差非零，无 AMP 重试。adapter checkpoint 约 0.7 MB。

固定噪声、固定 t=500 的监控（仅各 2 对，不能代表完整验证集）：

| 帧对 | OFF MSE | 训练后 correct MSE | 判断 |
|---|---:|---:|---|
| train_01 | 0.05548366 | 0.05510294 | 改善 |
| train_05 | 0.05223204 | 0.05193035 | 改善 |
| heldout_00 | 0.15897095 | 0.15953827 | 退化 |
| heldout_01 | 0.14797546 | 0.14786056 | 微弱改善 |

50 步 DDIM 同噪声生成后，严格几何支持像素上的目标 RGB L1：

| 帧对 | OFF | 训练后 correct |
|---|---:|---:|
| train_01 | 0.09572385 | 0.09796063 |
| train_05 | 0.08670393 | 0.08590121 |
| heldout_00 | 0.19821012 | 0.19903968 |
| heldout_01 | 0.21080351 | 0.21061653 |

指标字段中的历史名称 `roi_rgb_l1` 在这次运行指传感器严格支持像素，不是人工建筑区域，也不是建筑身份指标。对照图没有显示稳定的建筑外观继承收益。

结论：P0/P1 接口与稀疏几何、P2 接入与学习通路已验证；**生成质量验收尚未通过，不能说建筑一致性已经解决**。128 步结果也不足以归因于训练时长、学习率或冻结范围。

验证记录：静态 adapter 9 项、probe 12 项、静态几何 16 项和原 persistent 分支 15 项测试通过；相关 Python 编译及 diff 检查通过。独立推理 CLI 已重载 adapter，完成 B2、CFG 7.5、4 步 DDIM 冒烟检查；输出 latent `[2,4,16,64]`、RGB `[2,3,128,512]` 均有限。这只验证重载与组批，不作为图像质量测试。

本地结果副本：`/media/shizhm/sda2/sat-street atlas/outputs/temporal_static_p0p2_20260919/probe`。
