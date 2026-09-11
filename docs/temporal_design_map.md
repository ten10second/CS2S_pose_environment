# 时序代码状态

更新：2026-09-11，分支 `temporal-model`。

旧 v3 / Stage F history adapter 已退役并删除。当前代码保留单帧生成路径，
没有启用中的 temporal attention，也没有 history 开关。讨论中的主干内时序方法
尚未实现；不能把本次清理后的模型当作已经具备时序能力。

## 当前保留的生成路径

`tools/train_kitti_raea.py` 训练单帧模型；`tools/raea_frame_sampling.py` 提供
`prepare_frame_inputs` / `sample_frame`，`tools/ar_dyn_utils.py` 提供噪声工具。
当前帧卫星条件与 LiDAR 条件经 `ldm/modules/KITTI_attention.py` 的
ray-posterior fusion 融合，再进入 feed-forward。没有上一帧 latent 注入。

本次只移除旧 history 接入，不改变单帧参数名称及卫星 / LiDAR 融合计算。

## 已删除的旧路径

- `HistoryLatentEncoder`、`GeometryHistoryAttention`、`HistoryEvidenceHub`。
- 动态注入、history payload、开关、残差统计及 host 参数存取。
- Stage F 专用训练、生成、launcher、checkpoint 加载及训练汇总。
- 依赖旧 adapter 的判决实验执行脚本及对应测试。
- 单帧 diffusion loss 中通过 `_history_appearance_x0` 注入的附加损失。

旧代码可从清理前提交 `c85d021` 恢复。该提交的精简 attention 与实际评估过的
transport checkpoint 不完全兼容；服务器判决实验在隔离 worktree 中使用了
`3c1d766` 的 `ldm/modules/temporal_history_attention.py`，见实验归档。

## 保留的独立工具

| 文件 | 用途 |
| --- | --- |
| `tools/temporal_history_geometry.py` | LiDAR / 标定 / 位姿对应、连续帧配对；不加载 adapter |
| `tools/audit_temporal_history_geometry.py` | 几何覆盖率审计；不运行生成模型 |
| `tools/summarize_history_necessity.py` | 只读取既有判决 JSON，便于复查历史结果 |
| `tests/test_temporal_history_geometry.py` | 几何工具测试 |
| `tests/test_history_necessity_summary.py` | 离线结果汇总测试 |

保留几何工具不表示新方法已经选定旧的硬掩码规则。
单帧 LiDAR range adapter 与第三方库中的 adapter 不属于本次退役范围。

## 后续设计边界

计划让当前 LiDAR 约束几何、上一帧 RGB latent 提供外观，时序模块与相关主干层
联合训练。可从相邻 GT 帧训练起步，再逐步引入生成历史；是否增加多帧训练由
rollout 评估决定。这些是待实现的设计，不是当前代码的运行行为。

历史实验见 [temporal_history.md](temporal_history.md)。
旧机制讨论见 [temporal_mechanism_options.md](temporal_mechanism_options.md)，
其中旧命令仅供追溯，不能在当前 checkout 执行。
