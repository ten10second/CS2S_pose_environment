> 历史实验记录：以下配置、结论和命令对应当时版本，不是当前启动指南。
> 当前方案见 [时序说明](temporal_history.md) 和 [全量 B/C 协议](temporal_full_bc.md)。
> 部分旧实验脚本已退役，原源码保留于服务器 `CS2S_run_control/temporal_cleanup_commit_20260920/before_source.tar.gz`。

# 稠密静态历史条件：第二阶段小样本实验

## 固定目标

将已有稠密重投影的历史 RGB 与有效性、来源类型输入生成器，验证 decoder 是否继承建筑立面纹理，同时保持当前条件下的几何和清晰度。本轮不继续扩大几何实验，不启用额外一致性损失，不训练动态车辆分支或长序列。

## 数据与实验边界

- 使用上一轮固定的 6 个训练帧对和 4 个留出帧对，原 GPS30m 清单不变。
- 前帧 GT、前后 LiDAR 和位姿生成历史参考，复用 `dense_reprojection_20260919/run512/<name>/reference.npz`；目标 RGB 不参与参考构建。
- 后帧 Sat/LiDAR 继续走现有条件分支。前帧 Sat 尚未显式接入这一静态参考实验，不将本轮称为完整五输入模型。
- measured 是两帧深度检查通过，不是语义静态标签；缺少可靠车辆关联。估计支持只作条件，不施加强像素保留损失。

## 接入

新增独立 `static_dense` 模式，保留旧 `static` 和原单帧接口。输入 `dense_rgb`、`dense_valid`、`dense_measured`、`dense_estimated`；后两者互斥且并集等于 valid，hole 可以由 valid 取反得到。

完整 512×128 历史参考先经过三层 stride-2 轻量卷积，再通过零初始化输出投影注入现有最高分辨率 decoder 位置，每个去噪步均可读取。RGB 不先平均成 16×64 色块。关闭历史或整份参考为空时严格回退。

## 匹配消融

四组均从同一 200k 单帧底座、新初始化历史参数开始，使用相同网络参数形状、初始化种子、采样顺序、噪声和 Sat dropout：

| 组 | 历史 | 编码器可见的输入 |
|---|---|---|
| sparse_types | 稀疏 LiDAR 严格对应 | RGB + valid + measured/estimated |
| dense_rgb | 稠密参考 | RGB；mask 输入通道置零 |
| dense_valid | 稠密参考 | RGB + valid；类型输入通道置零 |
| dense_types | 稠密参考 | RGB + valid + measured/estimated |

四组相同的六通道网络结构。只保留统一的整样本空参考保护，不用隐含的逐位置输出 mask 给 RGB-only 组泄露额外有效性信息。无历史输出作为共享固定底座对照。

冻结整个原 UNet（包括 decoder）、VAE、Sat/LiDAR 编码器；只训练历史编码与注入投影。每组 1000 步、batch 2、LR 1e-4。使用原 epsilon 生成损失，无新增 loss。GPU 4/5 各运行一个独立实验，不使用 DDP。有限小样本训练完成后停止。

## 验收

- 零初始化/OFF/空参考回退、B2/CFG 顺序、输入 schema、梯度和 checkpoint 重载检查。
- 核对基础权重哈希不变、OFF 固定噪声完整采样不变。
- 固定三个噪声时间步监控训练与留出误差，关注趋势，不用总 loss 代替图像验收。
- 所有 10 对用相同 50 步 DDIM 和种子生成；展示前帧、目标 GT、warp 和各组输出，包含失败例。
- 统计固定支持区域的 RGB/边缘误差，并查看建筑窗户与立面局部。上半部指标仅作代理，不称为建筑语义指标；边缘强度增加也不单独代表清晰度改善。
- 针对最完整条件组检查历史干预，区分利用参考与固定残差修正。

结果为负也必须保留和报告；这一轮不自动升级成全量训练。
