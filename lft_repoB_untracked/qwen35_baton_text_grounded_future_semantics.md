# Qwen3.5–Baton 文本 Grounded 未来语义特征：机制、证据与验证方案

> 整理日期：2026-07-28

> 文档目的：回答“为什么用指令条件的 Qwen Planner 回归未来 SigLIP2
> patch 特征，有可能产生可靠的文本 grounded 未来语义特征”，并明确区分：
>
> 1. Baton 论文已经直接给出的证据；
> 2. 从模型结构与监督目标可以得到的合理推断；
> 3. 在当前 LIBERO/GE-Act 系统上仍然必须通过实验确认的部分。
>
> 当前结论：这条路线有明确的论文依据和合理的机制闭环，但不能仅凭训练
> MSE 或 attention heatmap 宣称已经实现可靠文本 grounding。可靠性需要同时
> 通过特征预测、文本因果敏感性、时空对应和下游控制四类实验。

## 1. 研究问题与术语

当前任务可以写为：

\[
\hat F_{1:K} = P_\theta(O,\ T),
\]

其中：

- \(O\)：当前机器人观察，包括主相机或腕部相机图像；
- \(T\)：自然语言任务指令；
- \(P_\theta\)：Qwen3.5 Planner、learnable-query alignment tower 和
  Sem-MLP；
- \(\hat F_{1:K}\)：预测的 \(K\) 个未来关键帧语义特征；
- 每个 \(\hat F_k\) 是一个保留空间布局的 SigLIP2 patch grid。

本文所说的三个关键词含义如下。

### 1.1 未来语义特征

它不是未来 RGB，也不是单个全局文本向量，而是：

\[
\hat F \in \mathbb{R}^{K\times P\times D_s},
\]

其中 \(K\) 是未来关键帧数，\(P\) 是每帧空间 patch 数，\(D_s\) 是
SigLIP2 特征维度。每个 token 同时拥有：

- 一个未来时间位置；
- 一个图像空间位置；
- 一个位于 SigLIP2 表征空间中的连续语义向量。

### 1.2 文本 grounded

“文本 grounded”至少应包含两层含义：

1. **条件性**：改变指令 \(T\) 时，预测结果应以符合任务含义的方式改变；
2. **空间与时间定位**：变化应集中在与指令相关的物体、区域和未来阶段，
   而不是整张图无区分地漂移。

只有“输出处于一个图文预训练模型的特征空间”还不够证明文本 grounding；
还必须证明 Planner 实际使用了当前输入的 Instruction。

### 1.3 可靠

这里的“可靠”不是指单个训练 loss 较低，而是要求以下证据链同时成立：

1. 预测特征接近真实未来帧特征；
2. 正确指令比错误/打乱指令产生更准确的预测；
3. token 的空间与时间结构没有被压成无序的全局特征；
4. GE-Act 使用预测特征时优于无语义条件，并接近真值特征条件的上界；
5. 上述提升在不同 LIBERO 任务、相机和随机种子上稳定存在。

## 2. Baton 原论文到底提出了什么

主要依据是 Tu et al. 的 *Baton: Explicit Semantic Blueprints for Joint
Video-Audio Generation*（2026，arXiv:2605.25195v2，预印本）。

Baton 将“语义推理”和“像素/音频生成”拆开：

```text
用户文本
   ↓
可训练 MLLM
   ↓
视频/音频 planning placeholder hidden states
   ↓
learnable-query semantic alignment towers
   ↓
连续 SigLIP2 / WavTokenizer planned tokens
   ↓
带时空位置对齐的 cross-attention
   ↓
视频/音频 DiT
```

论文对 planned token 的定义不是普通文本 embedding。每个视觉 planned
token 对应一个关键帧中的一个空间位置，用来表达该位置“发生什么、在哪里
发生、何时发生”。因此它本质上是一组关键帧级时空语义蓝图。

### 2.1 文本条件的 planning region

Baton 将视频和音频 planning placeholder 放在完整用户 Prompt 之后：

\[
T_{\text{user}} =
[T_{\text{sys}};\ T_v;\ T_a;\ T_v^{tag};\ T_a^{tag}].
\]

MLLM 在 placeholder 位置产生隐藏状态：

\[
H_v = \operatorname{MLLM}(T_{\text{user}})
[v_{\text{start}}+1:v_{\text{end}}].
\]

由于使用因果注意力：

- 每个视觉 planning state 都能读取此前完整的文本 Prompt；
- 后面的 planning state 能读取前面的 planning state；
- 因而它不是彼此独立的 patch 回归，而是具有隐式自回归依赖的规划序列。

这一步回答了“文本语义怎样进入未来 token”：文本不是训练后再附加，而是在
产生 planning hidden states 时已经进入条件计算图。

### 2.2 Learnable-query alignment tower

Baton 的视觉塔使用与目标 token 数量相同的 learnable queries：

\[
\hat H_v =
\operatorname{CAttn}_v(Q_v,H_v),
\]

然后用 Sem-MLP 将 MLLM hidden width 映射到 SigLIP2 width：

\[
H_v^{sem} =
\operatorname{SMLP}_v(\hat H_v).
\]

在原论文的视频—音频任务中，视觉和音频塔还会进行双向跨模态注意力；在
LIBERO 视觉-only 任务中没有音频分支，因此保留视觉塔即可。原论文用于
视频—音频跨模态时间对齐的 timestamp RoPE 不应被硬加进单模态视觉塔。

### 2.3 连续 SigLIP2 特征回归

Baton 从真实关键帧中提取冻结 SigLIP2 倒数第二层特征
\(F_v^{gt}\)，使用逐关键帧、逐空间 token 的 L2 监督：

\[
\mathcal L_{\text{plan}} =
\sum_{t=1}^{N}\sum_{i=1}^{n_v}
\left\|H_{v,(t,i)}^{sem}-F_{v,(t,i)}^{gt}\right\|_2^2.
\]

这一目标同时固定了三个东西：

1. **表征坐标系**：输出必须进入冻结 SigLIP2 的连续特征域；
2. **空间索引**：第 \(i\) 个预测 token 对齐第 \(i\) 个未来 patch；
3. **时间索引**：第 \(t\) 组 token 对齐第 \(t\) 个未来关键帧。

因此 Q-Former/Query Tower 本身不是语义对齐的唯一来源。真正的对齐来自：

```text
Prompt 条件的 MLLM hidden states
        +
learnable-query 信息提取
        +
冻结 SigLIP2 未来 patch target
        +
逐时间、逐空间连续特征回归
```

## 3. 为什么 SigLIP2 适合作为目标空间

Tschannen et al. 的 *SigLIP 2: Multilingual Vision-Language Encoders with
Improved Semantic Understanding, Localization, and Dense Features*
（2025，arXiv:2502.14786，预印本）报告：

- SigLIP2 延续图像—文本训练目标；
- 加入 captioning、自蒸馏和 masked prediction 等训练；
- 相比 SigLIP，在图文检索、零样本识别和 VLM 表征迁移上更强；
- 在 localization 和 dense prediction 上也有明显提升。

这些性质使 SigLIP2 patch token 同时具备两类信息：

1. **语义可读性**：特征空间受到图文训练约束，比纯重建特征更容易承载
   文本概念；
2. **空间可用性**：patch grid 保留局部结构，比一个 pooled global
   embedding 更适合告诉视频模型“相关内容在什么位置”。

需要保留一个重要边界：SigLIP2 的整体图文预训练不能自动保证“每个
penultimate patch token 都与某个单词严格一一对应”。它提供的是更适合
grounding 的表征基础，而不是像素级语义分割标注。

## 4. Baton 论文中的直接实验证据

以下结果来自 Baton 的 Sem100 消融。P-Acc 是 Prompt Following Accuracy，
越高表示生成内容越遵循文本 Prompt。

### 4.1 Planned tokens 本身是主要语义驱动

| 设置 | P-Acc ↑ | 含义 |
|---|---:|---|
| 无 VA-Planner | 0.51 | 只有传统粗粒度文本条件 |
| Prompt Enhancement | 0.62 | 用更强 Qwen 改写 Prompt 仍然有限 |
| Frozen LLM hidden states | 0.67 | 直接注入冻结 LLM hidden states 不够 |
| 只使用 Planned Tokens | 0.78 | 即使去掉全局文本 embedding，规划 token 仍很强 |
| 完整 Baton | **0.82** | 全局文本和细粒度规划互补 |

该结果支持两点：

1. 将 Prompt 写得更详细不能替代结构化未来语义规划；
2. planned tokens 不只是辅助噪声，而是下游 Prompt Following 的主要驱动。

### 4.2 Learnable queries 有独立贡献

| 设置 | P-Acc ↑ | Stage-1 MSE ↓ |
|---|---:|---:|
| 无 Learnable Query | 0.79 | 0.37 |
| 完整 Baton | **0.82** | 更低，具体完整模型值随 backbone 表列出 |

提升幅度不是最大的，但论文据此认为 query-based distillation 能提取更多
语义细节。这符合 Query Tower 作为“从长 planning hidden sequence 中为每个
目标位置提取相关信息”的作用。

### 4.3 连续、文本对齐的视觉目标优于替代目标

| 视觉/语义目标 | P-Acc ↑ | 论文解释 |
|---|---:|---|
| 离散 TA-Tok + WavTokenizer | 0.68 | 离散量化损失细粒度感知信息 |
| DINOv3 替代 SigLIP2 | 0.77 | 自监督视觉特征的文本 grounding 较弱 |
| 完整连续 SigLIP2 方案 | **0.82** | 保留文本对齐和连续细粒度结构 |

这组消融是“为什么回归连续 SigLIP2 特征能够提供文本 grounded 语义”的
最直接实验证据之一。论文明确将 SigLIP2 的优势归因于其 text-aligned
visual features。

### 4.4 时空位置编码不是可选装饰

生成器侧 RS-RoPE 消融：

| 设置 | P-Acc ↑ | 说明 |
|---|---:|---|
| 无 RS-RoPE | 0.46 | semantic tokens 被当成近似无序集合 |
| 只使用 Temporal RoPE | 0.73 | 恢复时间对应，但缺少空间对应 |
| 完整 3D RS-RoPE | **0.82** | 统一时间和二维空间坐标 |

无 RS-RoPE 甚至低于无 VA-Planner 的 0.51，说明“有语义 token”并不保证
有帮助。如果下游 latent 不知道应该关注哪个时间、哪个空间位置，错误的
semantic guidance 会比没有 guidance 更糟。

### 4.5 粗粒度文本与细粒度语义应级联注入

| 注入方式 | P-Acc ↑ |
|---|---:|
| 与文本 embedding 直接拼接 | 0.76 |
| 文本和 planned token 并行 cross-attention | 0.79 |
| 先文本、后 planned token 的级联方式 | **0.82** |

论文解释是：

1. 文本 cross-attention 先建立全局、粗粒度语义先验；
2. planned-token cross-attention 再补充未来关键帧级、位置相关细节。

这也是当前 GE-Act 中保持 text cross-attention 后再进行 semantic
cross-attention 的依据。

### 4.6 Planner 精度与下游质量显著相关，但 MSE 不是全部

Baton Table 9 给出：

| Planner | Stage-1 MSE ↓ | P-Acc ↑ |
|---|---:|---:|
| 无 Tower | 0.39 | 0.69 |
| 无 Tower RoPE | 0.46 | 0.44 |
| Qwen3-4B | 0.41 | 0.68 |
| Qwen3-32B | **0.27** | **0.85** |

论文观察到 Stage-1 预测误差和下游生成质量总体正相关：更准确的 planned
tokens 通常带来更好的 Prompt Following。

但论文同时明确指出，这个相关性不是严格单调的，因为逐 token MSE 无法
衡量跨 token 的整体时间一致性。因此我们的评测不能只报告 MSE，还需要
报告时序一致性和下游控制效果。

### 4.7 三阶段训练解决条件分布差异

| 设置 | P-Acc ↑ | 含义 |
|---|---:|---|
| Stage-2：真值 encoder features | 0.87 | 理想语义条件的上界 |
| 跳过 Stage-2 | 0.75 | 生成器没有先学会使用干净语义空间 |
| 完整 Stage-1/2/3 | **0.82** | 用 Stage-3 适配真实 Planner 噪声 |

这说明可靠性不仅取决于 Planner，还取决于 GE-Act 是否学会：

1. 使用干净 SigLIP2 语义特征；
2. 再适应实际 Planner 预测误差。

直接把一个尚不完美的 Planner 接到没有经过语义适配的生成器/动作模型上，
不能视为 Baton 的完整训练方式。

## 5. 其他工作提供的交叉支持

### 5.1 Plan-X

Huang et al. 的 *Plan-X: Instruct Video Generation via Semantic Planning*
（2025，arXiv:2511.17986，预印本）采用与机器人任务更相近的输入形式：

- 同时读取文本 Prompt 和视觉上下文/首帧；
- 自回归预测 text-grounded spatio-temporal semantic tokens；
- 用语义 token 表达未来关键帧中的“发生什么、在哪里、何时发生”；
- 再由视频 DiT 负责高保真渲染。

Plan-X 使用离散 TA-Tok，而当前方案选择 Baton 证明更有利于细粒度信息的
连续 SigLIP2 回归。两篇工作共同支持“MLLM 负责语义规划，生成器负责像素
或动作解码”的模块分工。

### 5.2 这些工作支持什么、不支持什么

它们共同支持：

- 从文本和当前视觉上下文预测未来视觉语义是可学习任务；
- 时空 semantic tokens 比单个 global text embedding 更适合长时规划；
- 文本对齐的视觉特征可以作为 Planner 与生成器之间的接口。

它们不能替代当前 LIBERO 实验：

- 视频生成 Prompt Following 不等价于机器人控制成功率；
- 大规模互联网视频训练不等价于约 2.1 万条 LIBERO 轨迹；
- 论文中的 8B/32B Planner 能力不等价于当前 2B Planner；
- 文本到视频的多样性与机器人单一成功轨迹分布不同。

## 6. 当前 Qwen3.5–Baton 实现与论文的对应关系

当前实现是 Baton **视觉 Stage-1 的 LIBERO 适配**。

### 6.1 当前数据与张量路径

```text
main current RGB + Instruction ─┐
                                ├─ 独立 Qwen3.5 rows
wrist current RGB + Instruction ┘
                 ↓
每路 4 × 256 个 <PLAN_PAD> hidden states
                 ↓
每路 1024 个 learned queries cross-attend 1024 个 plan states
                 ↓
Sem-MLP: 2048 → 2048 → 1024
                 ↓
每路 [4 future frames, 256 patches, 1024 dims]
                 ↓
冻结 SigLIP2 未来 patch targets
                 ↓
pointwise feature MSE
```

每个原始样本最终预测：

\[
[2\ \text{cameras},4\ \text{frames},256\ \text{patches},1024\ \text{dims}],
\]

总计 2048 个未来视觉语义 token。

### 6.2 与 Baton 完全对齐的核心

| Baton Stage-1 核心 | 当前实现 |
|---|---|
| 每个目标 perceptual token 对应一个 placeholder | 是 |
| 从因果 MLLM 的 placeholder 位置取 hidden states | 是 |
| 每个目标 token 对应一个 learnable query | 是，1024 个 |
| Query cross-attend planning hidden states | 是 |
| Sem-MLP 映射到冻结视觉 encoder 维度 | 是，输出 1024 维 |
| 冻结 teacher 的倒数第二层连续特征 | 是 |
| 只使用逐 token continuous-feature MSE | 是 |
| 整个 Planner 可训练，teacher 冻结 | 是 |

代码入口：

- Planning template：[`qwen35_baton/sequence.py`](../qwen35_baton/sequence.py)
- 双相机独立输入：[`qwen35_baton/data.py`](../qwen35_baton/data.py)
- Query Tower：[`qwen35_baton/query_tower.py`](../qwen35_baton/query_tower.py)
- Sem-MLP 与输出：[`qwen35_baton/model.py`](../qwen35_baton/model.py)
- Equation-8 MSE：[`qwen35_baton/losses.py`](../qwen35_baton/losses.py)
- 冻结 SigLIP2 teacher：[`qwen35_baton/teacher.py`](../qwen35_baton/teacher.py)

### 6.3 为 LIBERO 做的必要适配

| 原 Baton | 当前系统 | 原因 |
|---|---|---|
| Qwen3-8B | Qwen3.5-2B-VL | 需要直接读取当前机器人图像 |
| 文本生成整段视频计划 | 当前观察 + 指令预测未来 | 机器人是条件未来预测 |
| SigLIP2 So400m patch14、384 | SigLIP2 patch16、256 | 当前视觉接口和算力设计 |
| 视频与音频双塔 | 视觉单塔 | LIBERO 无音频 |
| 单视频视角 | main/wrist 独立 rows | 保留两路相机空间结构 |
| 互联网视频关键帧 | 4 个 LIBERO 未来关键帧 | 对齐当前控制时间范围 |

这些适配不改变“Prompt 条件 MLLM → query alignment → continuous SigLIP2
future targets”的核心，但意味着不能直接照搬 Baton 的绝对指标。

### 6.4 当前实现中的时序依赖

1024 个 learned queries 之间没有额外 self-attention；这与当前严格 Baton
视觉塔设计一致。但这不表示完全没有时序依赖：

- 1024 个 `<PLAN_PAD>` 先经过 Qwen 的因果 self-attention；
- 后面的 plan states 可以依赖前面的 plan states；
- learned queries 再从整段 causal plan states 中提取每个目标位置的信息。

因此显式时序推理主要发生在 Qwen planning sequence 中，而不是 Query
Tower 内部。

### 6.5 GE-Act 侧的 Baton 对齐点

当前 GE-Act 路径保留：

- 原文本 cross-attention；
- 随后的 semantic-plan cross-attention；
- semantic keyframe 时间与二维 patch 位置；
- latent query 和 semantic key 使用各自时空位置构造 RoPE；
- main/wrist semantic tokens 与对应相机 latent 对齐。

相关代码：

- Semantic 坐标构造：
  [`ge_act/models/ltx_models/semantic_conditioning.py`](../ge_act/models/ltx_models/semantic_conditioning.py)
- 级联 text/semantic cross-attention：
  [`ge_act/models/ltx_models/transformer_ltx_multiview.py`](../ge_act/models/ltx_models/transformer_ltx_multiview.py)

这对应 Baton 的“粗粒度文本先验 + 细粒度位置语义修正”和 RS-RoPE 核心
思想。完整方法仍要求实际执行 Stage-2 teacher condition adaptation 和
Stage-3 planner-prediction adaptation。

## 7. 为什么当前方案在机制上可能有效

下面给出完整因果链，而不是简单地说“因为 SigLIP2 有语义”。

### 7.1 当前观察解决场景歧义

同一句“拿起黑色碗”可能对应不同桌面布局。当前 RGB 提供：

- 黑色碗在哪里；
- 机械臂当前姿态；
- 障碍物和目标容器位置；
- main/wrist 两种互补视角。

因此 Planner 不需要从文本凭空生成空间布局，而是预测“在当前场景和目标
条件下，哪些区域将如何变化”。

### 7.2 Instruction 解决未来多解性

只给当前图像时，同一个状态可能对应：

- 拿黑碗；
- 拿盘子；
- 打开抽屉；
- 将手臂移动到安全位置。

Instruction 用于选择正确的未来分支。理论上：

\[
P(F_{\text{future}}\mid O,T)
\]

应比

\[
P(F_{\text{future}}\mid O)
\]

具有更低的任务歧义。

### 7.3 SigLIP2 target 把抽象任务投到空间视觉域

纯文本 embedding 能表达“拿起黑色碗”，但不直接表达：

- 当前图像中的哪个 patch 是黑色碗；
- 机械臂会移动到哪个空间区域；
- 第一个和第四个未来关键帧有何区别。

未来 SigLIP2 grid 将抽象任务转化为一组具有空间和时间索引的视觉目标，
成为 Qwen 与 GE-Act 之间可消费的中间接口。

### 7.4 连续回归避免不必要的信息瓶颈

连续 1024 维特征保留比离散 codebook 更多的：

- 物体身份和属性；
- 局部几何与上下文；
- 机械臂/物体相对关系；
- 不同未来阶段之间的细微变化。

Baton 的 TA-Tok 消融支持这一点。但连续回归也更容易被背景主导，因此必须
通过文本交换和 changed-region 分析确认有效语义没有被平均掉。

### 7.5 Query Tower 提供按目标位置的信息读取

learned query 的作用不是凭空创造语义，而是让每个未来 patch 位置从完整
Qwen planning sequence 中选择相关信息。它比直接线性投影每个 placeholder
更灵活，同时维持固定输出长度和固定时空布局。

### 7.6 分阶段训练让下游真正使用语义

即使 Planner 输出有意义，随机初始化的 semantic cross-attention 也可能：

- 忽略 semantic tokens；
- 被预测噪声误导；
- 破坏原 GE-Act baseline。

Stage-2 用真值 SigLIP2 教 GE-Act“怎样使用这个特征空间”，Stage-3 再用
Planner 输出适配预测误差，是把表示能力转化为控制收益的必要步骤。

## 8. 现有证据还不能证明什么

### 8.1 低 MSE 不等价于使用文本

在 LIBERO 中，一个当前观察往往只对应一个固定任务。Planner 可能仅凭当前
图像和轨迹统计预测一个平均未来，而忽略 Instruction，仍然获得不差的 MSE。

### 8.2 Attention heatmap 不等价于因果 grounding

Query Tower attention 只能说明输出从哪些 planning states 读取信息。它
不能单独证明：

- planning states 使用了哪些文本词；
- 换掉 Instruction 后输出是否改变；
- 高 attention 区域是否真的是任务目标。

Heatmap 是解释工具，不是充分证据。

### 8.3 SigLIP2 patch 不是语义分割 mask

倒数第二层 patch feature 兼有语义、纹理、上下文和空间信息。颜色化可视化
相似不代表精确物体边界，也不能用深度图的方式解读。

### 8.4 Baton 的下游结果不是 LIBERO 成功率保证

Baton 的 P-Acc 衡量视频—音频生成的 Prompt Following；当前最终目标是
机器人动作成功率。中间表示有效与否必须在 GE-Act policy 上重新验证。

### 8.5 当前 backbone 与原论文不同

Baton 的消融中 Qwen3-VL-8B 初始化弱于其 Qwen3-8B。论文认为视觉理解预训练
可能让 hidden distribution 更偏高层识别，不易迁移到 SigLIP2 的空间感知
域。当前必须使用 Qwen3.5-VL 读取机器人图像，因此这是合理但真实存在的
风险，需要靠训练曲线和 grounding 实验确认。

### 8.6 数据规模和文本多样性较小

Baton 使用约 150 万视频—音频片段；当前 LIBERO 轨迹数量约 2.1 万。重复
Instruction 和相似背景会增加记忆任务模板、忽略细粒度文本的风险。

## 9. 必须执行的可靠性验证

建议把验证分成五层。只有前四层同时通过，才能把输出称为“可靠的文本
grounded 未来语义特征”。

### 9.1 Level A：未来特征预测准确性

基础指标：

\[
\operatorname{MSE}(\hat F,F^{gt}),
\qquad
\operatorname{CosSim}(\hat F,F^{gt}).
\]

应分别报告：

- main / wrist；
- 4 个未来关键帧；
- 全部 patch；
- 高变化 patch 与低变化背景 patch；
- seen task 与 held-out episode。

MSE 下降只能证明“接近未来 teacher feature”，不能单独证明文本 grounding。

### 9.2 Level B：Instruction 因果敏感性

对同一个当前观察 \(O_i\)，构造：

- 正确指令 \(T_i\)；
- 同 suite 错误指令 \(T_j\)；
- 跨 suite 错误指令；
- 空指令或通用指令；
- 保留物体但交换动作；
- 保留动作但交换目标物体。

定义 grounding margin：

\[
\Delta_{\text{ground}} =
\operatorname{MSE}(P(O_i,T_j),F_i^{gt})
-
\operatorname{MSE}(P(O_i,T_i),F_i^{gt}).
\]

若模型使用正确文本，\(\Delta_{\text{ground}}\) 应显著大于零。

同时报告预测变化：

\[
S_{\text{text}} =
\left\|P(O_i,T_i)-P(O_i,T_j)\right\|_2.
\]

判断原则：

- 正确 Instruction 的未来 MSE 显著低于错误 Instruction；
- 交换目标物体时，变化集中在对应物体和机械臂未来区域；
- 交换无关修饰词时不应造成全局无规则漂移；
- 使用 paired bootstrap，置信区间应排除零。

这是当前最重要、也最缺失的一组证据。

### 9.3 Level C：空间 grounding

建议同时生成三类图，不只画 Query Tower attention：

1. 当前 RGB 与未来 RGB；
2. 未来 SigLIP2 target / prediction 的 PCA 或 probe 可视化；
3. Instruction swap 前后的 token 差异热图。

差异热图可定义为：

\[
D_{t,i} =
\left\|\hat F_{t,i}(O,T_{\text{correct}})
-\hat F_{t,i}(O,T_{\text{wrong}})\right\|_2.
\]

可靠 grounding 应表现为：

- pick/place 对象附近出现较高差异；
- 机械臂未来运动路径出现合理差异；
- 大面积静态墙面和桌面不应主导变化；
- main 与 wrist 各自对应其相机画面，不发生跨相机错位。

如果有目标框或可自动获得的仿真 segmentation，可进一步报告：

- pointing-game accuracy；
- target-region / background heat ratio；
- heatmap 与目标 mask 的 IoU 或 energy-in-mask。

没有 mask 时，应使用 instruction swap 差异和人工盲评，不能把彩色 PCA 图
当作定量 grounding 指标。

### 9.4 Level D：时间 grounding

验证四个关键帧是否表示不同未来阶段，而不是复制同一特征：

- 每帧单独 MSE/CosSim；
- 相邻帧预测变化与真值变化的相关性；
- 将四帧顺序打乱后，GE-Act 性能是否下降；
- 比较正确 keyframe time、全部同一 time、逆序 time；
- 检查早期 token 更关注接近当前的状态，后期 token 更关注任务结果。

可定义变化一致性：

\[
C_{\Delta} =
\operatorname{corr}
\left(
\|\hat F_{t+1}-\hat F_t\|_2,\,
\|F^{gt}_{t+1}-F^{gt}_t\|_2
\right).
\]

这用于弥补点对点 MSE 无法衡量全局时间结构的问题。

### 9.5 Level E：GE-Act 下游闭环

至少比较以下条件：

| 实验 | Semantic condition | 目的 |
|---|---|---|
| E0 | 无 semantic plan | 原 GE-Act baseline |
| E1 | 真值未来 SigLIP2 | 可利用语义信息的上界 |
| E2 | Planner + 正确 Instruction | 实际系统 |
| E3 | Planner + 错误/打乱 Instruction | 文本因果对照 |
| E4 | Planner tokens 时间打乱 | 时间结构对照 |
| E5 | Planner tokens 空间打乱 | 空间结构对照 |
| E6 | 仅 global pooled SigLIP2 | 验证 dense patch 的必要性 |

关键判据：

1. E1 明显优于 E0：证明 GE-Act 能利用这个语义接口；
2. E2 优于 E0：证明预测特征有实际价值；
3. E2 明显优于 E3：证明价值来自正确 Instruction，而非额外参数或噪声；
4. E4/E5 下降：证明时间/空间组织确实被下游使用；
5. E2 与 E1 的差距衡量 Planner 还有多少可提升空间。

LIBERO 应分别报告四个 suite、每任务成功率、总体成功率、随机种子均值和
置信区间。

## 10. 推荐的证据分级

最终论文或实验报告建议使用下列措辞，避免过度宣称。

### 10.1 已有直接证据

可以写：

- Baton 证明 planned tokens 能显著提升 Prompt Following；
- 连续 SigLIP2 目标优于 DINOv3 和离散 TA-Tok 替代；
- learnable queries、时空 RoPE 和三阶段训练都有消融支持；
- Planner 特征预测精度与下游质量总体正相关。

### 10.2 当前方法的合理推断

可以写：

- 当前观察提供空间场景，Instruction 选择任务相关未来分支；
- 回归冻结 SigLIP2 patch grid 有望将文本条件推理投射到未来空间语义域；
- 双相机独立预测减少拼接图导致的空间混淆；
- 级联 cross-attention 与 RS-RoPE 有望让 GE-Act 使用位置相关未来语义。

措辞应使用“有望”“机制上支持”“与 Baton/Plan-X 的发现一致”。

### 10.3 只有实验通过后才能写

以下表述需要 Level B–E 的结果：

- “预测特征可靠地依赖 Instruction”；
- “模型定位到了文本指定的目标物体”；
- “四个关键帧编码了正确的未来任务阶段”；
- “semantic planner 提升了 LIBERO 控制成功率”；
- “提升不是当前图像 shortcut、数据记忆或额外参数造成的”。

## 11. 主要失败模式与诊断

| 失败现象 | 可能原因 | 优先诊断 |
|---|---|---|
| 正确/错误指令输出几乎相同 | Planner 忽略文本，只做视觉外推 | Instruction swap |
| 全图背景预测很好、目标区域差 | MSE 被静态背景主导 | changed-region 分层指标 |
| 四帧输出非常相似 | 时序 planning 塌缩 | 相邻帧变化相关性、时间打乱 |
| Planner MSE 低但控制变差 | 下游未学会使用或被噪声误导 | E0/E1/E2 与 Stage-2/3 |
| Attention 集中但文本交换不变 | heatmap 只是结构注意力，不是语义因果 | swap 后差异热图 |
| main 好、wrist 差 | wrist 遮挡、尺度变化或预处理问题 | 分相机指标与可视化 |
| Teacher 条件有效、Planner 条件无效 | Stage-1 精度不足或 Stage-3 未适配 | E1/E2 差距 |
| 训练集好、验证集差 | 数据量小、指令模板记忆 | held-out task/episode |
| 语义 token 误导 LTX | 时空坐标或相机对应错误 | 时间/空间/相机打乱对照 |

## 12. 最终判断

从论文和机制上看，当前方案具备形成文本 grounded 未来语义特征所需的完整
表示链：

```text
Instruction 给出目标与动作意图
          +
当前 RGB 给出场景与对象位置
          ↓
因果 Qwen planning states 完成条件未来推理
          ↓
learnable queries 提取固定时空布局的计划
          ↓
连续 SigLIP2 回归把计划锚定到文本友好的视觉空间
          ↓
RS-RoPE semantic cross-attention 将计划对齐到 GE-Act latent
```

Baton 的消融表明 planned tokens、连续 SigLIP2、learnable queries、
时空 RoPE 和三阶段训练都对 Prompt Following 有实际贡献；SigLIP2 和
Plan-X 的结果也为“文本对齐的稠密视觉 token 可以作为未来规划接口”提供
交叉支持。

但对当前 LIBERO 系统，最关键的未决问题不是“预测 MSE 能否下降”，而是：

> 在同一个当前观察下，替换 Instruction 是否会以正确的空间和时间方式改变
> 未来特征，并进一步改变 GE-Act 的任务成功率？

只有 Instruction swap、空间/时间打乱和 GE-Act 闭环消融同时给出正结果，
才能把当前输出严谨地称为“可靠的文本 grounded 未来语义特征”。

## 参考文献

1. Shuyuan Tu, Qi Tian, Zihan Yang, Yue Wu, Xintong Han, Weijie Kong,
   Jiangfeng Xiong, Jian-Wei Zhang, Zhao Zhong, Liefeng Bo, Zuxuan Wu,
   Yu-Gang Jiang. **Baton: Explicit Semantic Blueprints for Joint
   Video-Audio Generation.** arXiv preprint arXiv:2605.25195v2, 2026.
   [本地 PDF](<../Baton- Explicit Semantic Blueprints for Joint Video-Audio Generation.pdf>) ·
   [arXiv](https://arxiv.org/abs/2605.25195)
2. Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem,
   Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer,
   Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner,
   Xiaohua Zhai. **SigLIP 2: Multilingual Vision-Language Encoders with
   Improved Semantic Understanding, Localization, and Dense Features.**
   arXiv preprint arXiv:2502.14786, 2025.
   [arXiv](https://arxiv.org/abs/2502.14786)
3. Lun Huang, You Xie, Hongyi Xu, Tianpei Gu, Chenxu Zhang, Guoxian Song,
   Zenan Li, Xiaochen Zhao, Linjie Luo, Guillermo Sapiro.
   **Plan-X: Instruct Video Generation via Semantic Planning.**
   arXiv preprint arXiv:2511.17986, 2025.
   [本地 PDF](<../Plan-X- Instruct Video Generation via Semantic Planning.pdf>) ·
   [arXiv](https://arxiv.org/abs/2511.17986)
