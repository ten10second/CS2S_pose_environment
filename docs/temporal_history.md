# 当前时序实验：centered A1 静态历史条件

更新：2026-09-20。分支 `temporal/v22-causal-appearance`，独立工作树
`/mnt/shizhm/CS2S_pose_environment_temporal_v22`。

## 当前目标与数据流

给定前帧 GT RGB、两帧 LiDAR/位姿及当前卫星条件，生成后一帧，检验是否继承建筑立面
外观，同时符合当前几何。车辆身份匹配未被证明具备；本轮没有新增车辆关联或实例损失。

```text
前帧 GT RGB → 稠密深度（前帧 LiDAR 校准）→ 自车运动重投影 → 当前 LiDAR 冲突检查
                                                        ↓
                                  RGB + valid + measured + estimated（六通道）
                                                        ↓
                                               history encoder → H
                                                        ↓
当前 GT latent 加噪 + 当前 sat/LiDAR → decoder h → h + F(h,H,t) − F(h,0,t) → ε
```

- 仅在 16×64 decoder 位置注入一次；两个 F 共享参数且都保留梯度。
- 参考是条件，不是硬拷贝或中途 latent 初始化；推理从噪声走完整 DDIM 时间表。
- 前后 RGB 使用同一组轻度颜色增强，再构造 warp 和重新编码目标 latent。
- 卫星、LiDAR、位姿不做颜色增强；后帧 RGB 不参与构建几何对应。
- `measured/estimated` 表示对应的来源和支持程度，不是建筑或车辆语义标签。
- 主目标仍为标准 epsilon MSE；本轮不新增时序一致性 loss 或 attention。
- 单帧路径仍默认不开启 history；先严格加载单帧底座，再显式配置时序分支。

## 当前入口

| 入口 | 用途 |
|---|---|
| `tools/prepare_full_centered_history.py` | 全量地理隔离帧对清单及紧凑重投影参考；计划、生产、自测 |
| `tools/prepare_centered_history_data.py` | 共享几何构建及固定子集准备 |
| `tools/cache_centered_conditions.py` | 固定评估子集的条件缓存 |
| `tools/train_centered_decoder_full.py` | B/C 一次完整 epoch，条件在线编码、固定评估及最终生成 |
| `tools/train_centered_decoder_probe.py` | 小样本 A/B/C 匹配实验及共享训练/评估函数 |
| `tools/summarize_centered_decoder_probe.py` | 原 ABC 实验汇总，不把它当成仅有 B/C 的自动汇总器 |

全量入口的 `--cache-root` **只用于固定评估样例**，训练数据来自 `--manifest` 的全量
`items`。两组顺序相同、不循环补 batch、不丢尾；最后一批可小于 batch size。

已有 `train_temporal_pairs.py`、`train_static_history.py`、`train_dense_static_history.py`、
`train_centered_static_history.py` 仍提供当前入口所需的公共函数，因此保留。它们不是
当前实验推荐的启动入口。旧 geometry/content/static 模式及相关 checkpoint 检查也仍
属于保留的依赖/兼容接口；本次清理没有改变模型结构或这些接口。

## 正在运行的全量 B/C

详见 [全量实验协议](temporal_full_bc.md)。同一个 200k 单帧底座，历史分支重新初始化：

| 组 | GPU | 更新参数 | 学习率 |
|---|---|---|---|
| B | 4 | history encoder/融合 + 后段 decoder | history 1e-4，decoder 1e-5 |
| C | 5 | 相同后段 decoder，关闭历史 | decoder 1e-5 |

每组 14,448 对相邻帧，B8，1,806 更新，**只跑 1 epoch**。原地理训练/测试隔离继续使用，
固定观察样例的帧不参与训练。decoder 范围是 output_blocks 9/10/11 和 out 的活跃参数。

- 训练前、903、1806 步：固定 8 个训练样例、8 个地理留出样例及原 10 个观察样例；
  固定 t100/500/900 和噪声，B 比较 correct/off/wrong，C 使用 off。
- 最后进行 50 DDIM、CFG7.5 的同种子生成；每组保留最近两个 probe checkpoint。
- probe checkpoint 仅含可训练权重和底座身份，**不包含 optimizer/scaler 续训状态**。
  它必须配合记录的底座使用；当前全量入口负责训练完成后的生成。
- `metrics.jsonl` 记录输入与随机性哈希，`status.json` 是实时进度；最终 `done.json`
  才表示完整训练和生成结束。不能仅凭进程退出或目录存在判断成功。

命令形状（输出必须是新目录，具体服务器清单和 settings 路径见实验协议）：

```bash
python tools/train_centered_decoder_full.py \
  --group B --manifest FULL_TRAIN_MANIFEST_JSON \
  --settings SETTINGS_JSON --selection FIXED_26_PAIR_SELECTION_JSON \
  --cache-root FIXED_EVAL_CACHE --out-dir NEW_B_OUTPUT \
  --batch-size 8 --num-workers 2 --device cuda:0
```

B/C 是两次独立实验，不是同一模型的 DDP。运行 C 时改为 `--group C`、新输出目录，并
使用相同数据/设置/seed。GPU 可通过 `CUDA_VISIBLE_DEVICES=4` 或 `5` 分配。参考生产器
写入原子 NPZ，训练可以等待尚未完成的参考；错误会显式报告，不会用空条件代替。

## 验收与历史证据

成功需要在留出场景上看到正确历史带来的真实颜色/纹理继承，并且不增加模糊或破坏几何。
仅有训练 loss 降低、梯度非零，或错误历史变差，都不构成成功证据。

之前 8 对/500 更新的 ABC 实验中，无历史 C 取得 B 训练误差降幅的约 99.17%；B 正确历史
对错误历史的训练误差优势仅约 0.051%。在有效区域给目标 GT 理想参考仍没有明显改善。
这些结果受小样本过拟合限制，不能据此断言全量训练无效，也不能声称当前方案已成功。

历史实验文档保留其数值和上下文，但标记为历史记录。已退役的实验脚本在仓库外源码快照
中保留；不再作为当前启动入口。清理范围见 [清理记录](temporal_cleanup_plan.md)。
