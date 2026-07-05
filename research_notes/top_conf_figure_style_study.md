# 开题报告三张研究内容图：顶会论文实现与画法抽取

整理时间：2026-06-26

目标：先学习相关论文的实现细节和插图布局，再把开题报告三项研究内容重画成更接近 AI 顶会论文的系统图，而不是简单流程框图。

## 结论先行

下一版三张图不应该继续走“线性模块 + 少量公式”的样式，而要采用顶会论文常见的三层表达：

1. 真实证据层：输入图像、BEV/occupancy/point cloud/map tile、mask、depth、Gaussian splat 等缩略图或仿真小视窗。
2. 方法模块层：用不同底色的大容器承载子模块，例如 encoder、memory、decoder、update、planner、heads，不把所有模块画成同一种矩形。
3. 状态演化层：用时间轴、old/new map、history/current/future、persistent/vanishing/emerging 等状态，表达“流式、累积、更新、预测”。

对应到你的三项研究内容：

- 研究内容一：借鉴 Skyfall-GS、ReconDrive、StreamSplat。重点画“双输入证据 + 双头 Gaussian 参数预测 + 坐标/传感器约束 + 可微渲染监督”。
- 研究内容二：借鉴 LT-Gaussian、GaussianWorld。重点画“地理地址化 tile memory、old/new map 对齐、结构变化检测、局部更新、动态轨迹生命周期”。
- 研究内容三：借鉴 GaussianWorld、Drive-OccWorld、OccWorld。重点画“历史状态对齐、当前观测注入、前馈流式更新、未来感知/占用增强”，底部必须有连续时间轴。

## 逐篇论文：实现细节与图形语法

### GaussianWorld

来源：https://arxiv.org/html/2412.10373v1

实现细节：

- 将自动驾驶 3D occupancy prediction 重述为由当前传感器输入条件化的 4D occupancy forecasting。
- 场景演化被拆成三类：ego-motion alignment、local dynamic object movement、newly observed scene completion。
- 历史 Gaussian 先对齐到当前坐标；新观测区域用随机/先验 Gaussian 补全；再经过 Gaussian World Layers 做 self-encoding、cross-attention、unified refinement。
- 输出分成 motion prediction 和 perception prediction，并有 GS-to-Occ 的转换。

画图风格：

- Fig.2 是一张全宽 overview，左侧是历史 Gaussian、随机先验、当前观测；中间是虚线大容器“3D Gaussian World Model”；右侧是 current occupancy / future occupancy。
- 每个 Gaussian world layer 内部有重复子块，形成“层级堆叠感”。
- 底部是视频帧时间轴，history/current/future 用颜色区分，强调 streaming sequence。

可借鉴到你的图：

- 研究内容二、三都应加入“history -> current -> future”的底部时间轴。
- 用“aligned historical Gaussians / completed Gaussians / refined current Gaussians”这种中间态，让评审看到你的系统不是黑盒。

### Drive-OccWorld

来源：https://arxiv.org/html/2408.14197v1

实现细节：

- 整体由 History Encoder、Memory Queue、World Decoder、Planner 组成。
- 使用 semantic/motion-conditional normalization 维护记忆队列。
- Decoder 在 action conditions 下生成 future occupancy 和 flow，Planner 基于 occupancy cost 做轨迹规划。
- 该模型强调 action-controllable generation 和 continuous forecasting/planning。

画图风格：

- Fig.2 是横向超宽架构图，用四个不同底色的大区域分隔 W_E、W_M、W_D、P。
- 左侧多帧观测压入 history encoder，中间 memory queue 是竖向堆叠 token/feature，右侧是两大输出：action-controllable generation 和 occupancy-based planning。
- 图里大量使用小型 BEV/occupancy 彩色网格，而不是只用文字描述。

可借鉴到你的图：

- 研究内容三适合采用“History Encoder / Streaming State / Feed-forward Decoder / Enhanced Perception”的四段式布局。
- 如果要画你的方法优于简单 temporal queue，可以把 queue 画成浅灰 baseline，再突出地理锚定状态和时序累积模块。

### OccWorld

来源：https://ar5iv.labs.arxiv.org/html/2311.16038

实现细节：

- 使用 VQVAE 将 occupancy token 化，再用 GPT-like spatial-temporal generative transformer 做自回归未来预测。
- 在 temporal causal self-attention 前后插入 spatial aggregation，分别建模空间结构和时间因果。
- 输入历史 occupancy，输出多步未来 occupancy。

画图风格：

- Fig.2 是按 t=0、t=1、...、T-1 排列的重复时间列。
- 中间一条横向“Temporal Causal Self-Attention”带贯穿所有时间步。
- 红色虚线箭头表达 autoregressive rollout，顶部/底部 occupancy cube 表达输入输出 token。

可借鉴到你的图：

- 研究内容三必须明确“时间因果方向”，适合使用横向 causal band。
- 你的方法若是前馈流式而非自回归 transformer，可用相似时间布局，但把核心带改成“Feed-forward State Update / Accumulative Perception”。

### Skyfall-GS

来源：https://arxiv.org/html/2510.15869v2

实现细节：

- 两阶段 pipeline：先从卫星视角进行 3DGS 重建，再通过 curriculum-style 数据集迭代把 sky-to-ground 视角逐步扩展到地面视角。
- satellite reconstruction 阶段使用 pseudo-camera depth supervision 约束几何。
- synthesis 阶段用 T2I diffusion/refinement 逐步生成更接近地面视角的数据。

画图风格：

- Fig.3 是两阶段大流程图，使用真实卫星/地面图像缩略图、相机锥体、渲染结果和优化回路。
- 阶段之间不是简单箭头，而是有“dataset update / curriculum / refinement”闭环。
- 视觉上图像素材占比很高，模块文字占比相对低。

可借鉴到你的图：

- 研究内容一如果涉及卫星/车载/地理坐标锚定，左侧应放“卫星地图 tile + 车载多相机/激光视窗 + 坐标系小图”，不要只放抽象 sensor icon。
- 可以加入“pseudo-depth / reprojection / geo-consistency loss”一条监督带，增强论文味。

### LT-Gaussian

来源：https://arxiv.org/html/2508.01704v1

实现细节：

- 三个模块：Multimodal 3D-GS、Structural Change Detection、Gaussian-Map Update。
- 先用多模态输入构建旧 Gaussian map，再把当前 LiDAR stream 与旧 Gaussian map 对齐。
- 检测 disappearing / emerging points，只对结构变化区域执行局部更新。
- 强调长期户外场景中地图的可维护性和更新效率。

画图风格：

- Fig.2 从 Map Segment-1 / Segment-2 开始，使用真实图像、point cloud、sky mask、depth map、Gaussian prior 等缩略图。
- 中间的 structural change detection 用 old/new LiDAR 和 Gaussian map 对齐展示变化。
- 右侧 update module 明确展示 Gaussian Map-1 到 Gaussian Map-2 的更新过程。

可借鉴到你的图：

- 研究内容二应直接采用 old map / current stream / change detection / local update / new map 的版式。
- 动态目标可用 emerging、persistent、vanishing 三色编码；静态地图用 tile address 和局部 dirty region 表达。

### StreamSplat

来源：https://arxiv.org/html/2506.08862v1

实现细节：

- 当前帧生成 canonical static 3D Gaussians，同时预测从当前到前一帧的 bidirectional deformation field。
- 使用 adaptive Gaussian fusion 将前一帧 Gaussian 与当前帧 Gaussian 融合。
- 使用 DINOv2 static encoder、dynamic decoder、spatial upsampler，并分为 static head 和 deformation head。
- 通过 RGB/depth/mask 等 2D supervision 训练。

画图风格：

- Fig.2 是紧凑横向 pipeline：Input Frames -> DINOv2 -> Static Encoder -> Dynamic Decoder -> Static Head / Deformation Head -> 3DGS Params / Deformation Field -> Rasterizer -> Rendered Frames。
- 右侧虚线框放 2D supervision，明确 RGB、depth、mask 和对应 loss。
- Fig.3 用三种曲线/小图解释 persistent、vanishing、emerging Gaussian 的 opacity deformation。

可借鉴到你的图：

- 研究内容一适合使用 dual-head 结构：Geometry/Center Head 与 Appearance/Attribute Head。
- 研究内容二的动态对象生命周期可以画成 persistent / emerging / vanishing 三条状态支路。

### ReconDrive

来源：https://arxiv.org/html/2603.07552v1

实现细节：

- 输入为多相机上下文帧，使用 VGGT-style backbone：DINOv2 tokenization + alternating attention transformer。
- 双头预测：Gaussian Center Prediction Head 用深度、相机内外参投影中心；Gaussian Parameter Prediction Head 融合图像和上采样特征，恢复外观/纹理属性。
- 使用 static-dynamic 4D Gaussian composition：静态中心不变，动态物体中心按局部线性速度随时间移动。
- 借助 SAM2 mask / dynamic object motion flow 区分动态区域，损失包括 perceptual、L2、projection、SSIM 等。

画图风格：

- 方法图强调“多相机图像 -> dense feature -> Gaussian center/parameter heads -> 4D scene composition”。
- 小图中会显式展示 image、mask、dense feature、Gaussian indices 的映射关系，让抽象索引/特征变得可视。

可借鉴到你的图：

- 研究内容一要画出“图像/深度/外参/坐标系 -> Gaussian center”和“特征/图像 -> Gaussian attributes”两条预测路径。
- 研究内容二可以借鉴 static/dynamic 分解：静态 Gaussian map 和动态 Gaussian tracks 分两个 memory bank。

## 三张 V3 图的具体设计

### 图 1：地理坐标锚定的传感器约束 4DGS 表征框架

建议版式：左证据、中编码、右 Gaussian 场景、下监督。

- 左侧：卫星/地图 tile、车载多相机帧、LiDAR/depth、小型 ENU/GPS 坐标轴。
- 中部：Sensor Calibration & Geo-normalization 大容器；内部画 camera-to-world、time sync、pose prior、uncertainty gate。
- 中右：Backbone + dual heads。一个 head 预测 Gaussian centers / scales / rotations，另一个 head 预测 color / opacity / semantics / dynamics。
- 右侧：Geo-anchored 4D Gaussian Field，画静态高斯、动态高斯、地理网格和可微渲染视图。
- 底部：loss strip，放 reprojection、depth consistency、geo consistency、temporal smoothness、sensor constraint。

风格关键词：ReconDrive 的双头，StreamSplat 的监督虚线框，Skyfall-GS 的真实图像缩略图。

### 图 2：地理地址化的有界 4DGS 流式记忆架构

建议版式：左输入流、中 old/new memory、右查询输出，底部生命周期。

- 左侧：streaming sensor windows，按 t-k 到 t 排列；每个窗口带 pose stamp 和 geo tile id。
- 中部：Bounded Geo-addressed Memory，用地图 tile 网格表达 key-value memory。
- 中部上支：Static Gaussian Map Bank，画 old tile、current evidence、structural change detection、local update。
- 中部下支：Dynamic Gaussian Track Bank，画 track birth、update、merge、retire。
- 右侧：memory readout，输出 local 4DGS、rendered novel view、occupancy/perception cue。
- 底部：cache policy / budget controller，画 prune、LOD、evict、compress，不要藏在文字里。

风格关键词：LT-Gaussian 的 old/new map 更新，GaussianWorld 的历史 Gaussian 对齐，StreamSplat 的 persistent/emerging/vanishing 状态。

### 图 3：前馈流式约束下的时序累积增强感知

建议版式：横向时间轴 + 中间状态更新带 + 上方多任务输出。

- 底部：t-3、t-2、t-1、t、t+1 的视频/BEV/occupancy 小缩略图。
- 中间：Feed-forward Accumulative State Update 横向大带，包含 ego-motion alignment、observation injection、uncertainty decay、state refinement。
- 上方：enhanced perception outputs，包括 occupancy、motion flow、semantic map、novel view 或 downstream detection cue。
- 左上角：轻量对比 baseline，用灰色 temporal queue / recurrent transformer 表示“重记忆/重计算”路径。
- 右侧：当前方法强调 bounded state、single-pass update、no full history replay。

风格关键词：OccWorld 的 temporal causal band，Drive-OccWorld 的 history/memory/decoder 区域分块，GaussianWorld 的 history-current-future 世界模型。

## 统一视觉规范

- 尺寸：论文单栏/双栏都可读，建议 SVG 原图 1800x1100 或 2200x1200，PNG 导出 2x。
- 结构：每张图 4-6 个一级区域，每个区域内部 2-5 个子块；避免超过 8 个同级大框。
- 色彩：用 4 套功能色，而不是同一蓝紫色系。建议：sensor 蓝、memory 绿、Gaussian 金、prediction 红、baseline 灰。
- 线型：实线表示数据流，虚线表示监督/约束，点线表示可选/历史状态，粗箭头表示状态更新。
- 缩略图：至少 30%-40% 画面面积应是图像化内容，例如 map tile、BEV、Gaussian cloud、occupancy grid、mask/depth。
- 公式：只放短公式或变量，不做大段推导。例如 `G_t = U(G_{t-1}, O_t, p_t)`、`key=(tile_id, level, time)`。
- 标签：使用顶会图常见的 `(a) Input Evidence`、`(b) Geo-normalization`、`(c) Memory Update` 分区标签。
- 叙事：每张图必须能从左到右读完，同时有一条底部时间/监督/预算线补充约束。

## 下一步重画方向

V2 图的主要问题不是信息量少，而是缺少“论文实现感”：没有足够的中间态、真实证据缩略图、训练/监督路径、状态生命周期。V3 应按上面的三张具体版式重画，优先把图二和图三画得更像 LT-Gaussian / GaussianWorld / OccWorld，因为这两张最能体现开题的创新点。
