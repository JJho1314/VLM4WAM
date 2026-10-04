# RoboFollow 动作拟合与更新精度诊断

2026-10-04，Ola_2208。用户授权继续14D动作头的小批次拟合诊断。完成三组训练集诊断；尚未获得充分拟合，未产生新的RoboFollow评测或语言遵循能力提升结论。工作分支仍为`research/robofollow-autoresearch`，冻结complete-v3数据、统计和官方strict评分保持原样。

## 固定实验条件

使用train split的`scene1:pickup_yellow_block_1_left:episode0`，长度124。相同source权重与seed42初始化14D投影，batch1、lr3e-5、AdamW、clip1，每组400次更新；这400步属于train-only诊断，不修改R0/R1候选300步协议。视频、文本和当前状态通过现有trainer数据路径构建后缓存；训练使用6个固定动作噪声/时间对，另用3个未参与更新的噪声在sigma0.1/0.5/0.9测量。缓存后将不再使用的T5/VAE移至CPU，仅释放显存，不重算或修改条件。动作专家FP32对照使用BF16 autocast进行前向。

最初固定索引150超过124帧，loader按原契约裁剪到末帧，因此最初两次投影诊断是重复末帧的静止目标，不能称为运动轨迹拟合。后续明确改为索引50，未来54个目标均处于有效长度内；7个左臂/夹爪通道发生变化，右臂7通道静止。静止目标结果仅作辅助，主要判断使用运动片段。

以下MSE是normalized flow velocity误差。`held noise`是同一训练样本的新噪声，不是held-out episode、dev指标或仿真动作成功率。另记录sigma平方乘velocity MSE作为normalized x0误差代理，不能替代实际采样与执行结果。

## 运动片段：受控结果

| 模式 | 可训练参数 | 训练噪声MSE | 新噪声MSE | 新噪声x0误差代理 |
| --- | ---: | ---: | ---: | ---: |
| 初始化 | 0 | 2.570585 | 2.648916 | 0.992975 |
| 仅14D投影，BF16参数/状态 | 14,862 | 2.134868 | 2.188414 | 0.821608 |
| 仅14D投影，FP32参数/状态 | 14,862 | 1.210599 | 1.141095 | 0.414258 |
| 解冻动作专家，FP32投影+BF16其余动作参数 | 163,796,494 | 1.195955 | 1.139493 | 0.413691 |
| 解冻动作专家，动作参数/状态均FP32 | 163,796,494 | 1.090669 | 1.073823 | 0.401602 |

投影层FP32相对BF16的新噪声velocity MSE降低47.86%。同一FP32投影下，解冻BF16动作专家只降低0.14%；再将其余动作参数/状态改为FP32，降低5.76%。各比较都在相同样本、初始化、学习率与噪声上进行；动作专家的两个独立run初始噪声指标完全一致，已自动验证。

静止目标投影对照：新噪声MSE为BF16 1.357360、FP32 0.960417，降低29.24%。BF16输入投影weight仅2.96%元素在400步后发生数值变化，FP32为100%；梯度两者均存在。动作专家全FP32组后期输入投影梯度norm达到约0.014，BF16专家对照的相应记录仍约1e-4量级。

这些观察支持“低精度参数/优化器更新限制新动作投影的学习”这一局部判断。它们不能证明这是官方零分的唯一原因，也不能把约1的剩余velocity MSE称为已经收敛。400步、单个train episode、固定视频噪声与有限动作噪声不足以证明任务泛化或语言敏感性。

## 未完成的比较与保留的失败

首个脚本直接Python启动，没有初始化trainer所需distributed process group，训练前退出；随后改用正式`torchrun --standalone --nproc_per_node=1`入口。最初backbone对照在其他任务占用显存上升后，于首次optimizer更新中OOM，没有完整full模式结果。运动片段首次尝试又因GPU0被其他任务填满而在CUDA初始化阶段OOM；之后根据实时空闲显存选择GPU重跑成功。

后续完整backbone比较设置至少45GB空闲显存的准入，本次未满足而拒绝启动，没有GPU phase也没有结果。其他进程未被中止。动作专家FP32对照的队列最多等待300秒，等待后启动完成，队列已经结束。不得把这些失败或预检拒绝解释为模型效果。

## 费用、证据与后续决定

沿用原`runs/budget.json`，未新建root或重置费用。此次新增0.304530GPUh，累计7.121416/8GPUh，剩余0.878584GPUh，包含全部失败启动、OOM和重试。诊断进程已退出，没有后台等待或GPU工作。候选best仍为R0，R0/R1四任务开发集Intent/Execution/CR仍全0；未开展新的官方rollout。

证据根目录：`/data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_autoresearch/runs`。

| Run | 状态 | GPUh |
| --- | --- | ---: |
| 20261004_113852_action_fit_diagnostic | 启动方式失败 | 0.008870 |
| 20261004_114046_action_fit_diagnostic | 投影400步完成，full模式OOM | 0.049843 |
| 20261004_114541_projection_precision_diagnostic | 静止目标两组完成 | 0.074492 |
| 20261004_115111_motion_precision_diagnostic | CUDA初始化OOM | 0.010017 |
| 20261004_115251_motion_precision_diagnostic | 运动片段三组完成 | 0.103937 |
| 20261004_120057_action_master_precision_diagnostic | 动作专家全FP32完成 | 0.057372 |

每run保留源码快照、diagnostic.yaml、probe.py及hash、run.json、process.json/log和evidence。`runs/diagnostic_summary.json`保存可机器读取的均值、账本和核验结果；完成组的详细梯度与loss位于evidence/*.jsonl。核验确认所有指标有限、初始运动指标一致、所有GPU phase记账、无存活诊断进程，未修改原best和基线。

下一项有依据的实验是FP32动作参数/状态加足够训练步数的动作warm-up，先在多个train运动片段证明噪声去除与动作采样能学会，再回到完全相同的正确/打乱指令官方配对协议。只有实际Execution为正且语言配对增益达标才能认为instruction following有所改善。当前不直接推进R2/R3 planner扩展，也不因离线误差下降更新best。完整backbone比较仍缺失；显存与剩余GPU预算需按原账本准入。
