# 53k baseline 后的配对优化实验

目标：检验视觉语言预测 planner 是否改善 L1/L2，同时监控 L0 退化。当前 53k baseline 只有冻结 T5 条件，未启用 Qwen planner。历史 joint 实验存在 L2 Intent 正向证据，但不能替代本轮完整数据、32-action 设置的验证。

复用现有 RoboFollowTrainer / RoboFollowResearchPlanner，不新增架构。两组都从 53k text 的完整动作模型初始化：text 对照继续训练；joint 加入现有 Qwen3.5 预测 planner、可训练 planner head 和 semantic adapter。Qwen 主干冻结，predicted_probability=1，anchor_weight=0.1；teacher 仅提供训练辅助损失，部署只使用当前观测与指令。

两组均使用官方 3750 demonstrations、LR 1e-5、warmup 200、5000 optimizer steps、global batch 128（2/GPU × 8 GPU × accumulation 8）、action chunk/history stride 32。权重 warm-start，优化器重新初始化；不是严格恢复原 optimizer。两组数据采样从同一 seed/epoch 起点开始。新增层白名单排除 action projection，防止意外丢弃已学动作头。

先运行独立的 2-step joint smoke，检查权重迁移和非零有限 planner 梯度；失败则整作业停止。随后顺序训练 text、joint，最后自动评测 baseline、text、joint。不得修改正在运行的 baseline 评测目录。只申请 jhe724 / acd_u 一节点八卡，与原 baseline 评测合计最多两节点十六卡。

探索性评测：所有四个场景的 L0/L1/L2（75+86+173=334 tasks），每任务 3 rounds，固定 seed42、TOPP runtime、500-action budget。三组均使用同一评测协议。按 scene×level 报告 IS/ES/CR，使用任务配对差异，不能将未完成任务计为失败或从不平衡子集宣称优势。优先观察 L1/L2 Intent、Execution、Completion 是否一致改善；L0 CR 若下降超过 3 percentage points，不直接升级长训。

这是使用公开 benchmark 的开发实验，不是 untouched test；后续论文须披露调参范围。不得把 L1/L2/L3 的轨迹、目标标注或评测指令加入本轮训练。数据增强或新布局采集属于另一个数据设置，不能混称官方等数据结果。CFG 与 hold padding 本轮不改，避免同时改变多个变量。

输出：`/data/user/jhe724/outputs/robofollow_q35/optimization_20261012`。入口 `scripts/hpc3_robofollow_optimize.sh`。每次训练固定独立目录，存在目录则停止，避免覆盖；中断后需检查已有产物，不能盲目重新提交。运行至脚本结束只产生结果，不自动根据未核实分数晋级。

提交前验证：HPC3 上 planner/loading/policy/epoch/benchmark completion 共 27 项测试通过；meta-device 模型逐张量核对 53k checkpoint 的 1429 个已有参数，text 无新增参数，joint 模型新增 374 个白名单语义参数（planner head 在 trainer 内另行注册）。两组数据、seed、LR、steps、global batch、推理步数和评测预算一致；shell 语法通过。GPU smoke/效果增益须以实际作业日志为准。
