# RoboFollow 53k 诊断与下一轮设计（2026-10-18）

状态：已完成结果审计、数据覆盖核验及相关开源代码阅读；以下新方案尚未实施、未提交训练、无提分结论。用户已要求分析实验并结合近期代码优化；具体第一轮设计已发出确认。

## 已验证的实验

HPC3 完整结果：`/data/user/jhe724/outputs/robofollow_q35/eval/official3750_text_53k_ac32_bs128/full-709872`。
A6000 scene4 复测：`/data/LFT-W02_data/junfeng/robofollow_eval/scene4-53k-20261014`。
基线：53k optimizer steps，text-only T5 + GE Act，planner disabled，动作预测/执行均 32，10 次去噪，500 动作预算，最多 16 次 policy 调用。文件名 q35 不代表本次启用了 Qwen 视觉语言 planner。

解析时排除 `.failed-*`，每任务恰有 10 个 round；HPC3 共 1,108 个完整 task/mode 目录、11,080 trials，A6000 为 180 个目录、1,800 trials。A6000 复测与 HPC3 scene4 重叠，不能加入主结果扩大样本数。

| 正确指令 | trials | Intent 均分 | Execution 均分 | Completion |
|---|---:|---:|---:|---:|
| L0 | 750 | 0.8291 | 0.7815 | 84.13% |
| L1 | 860 | 0.1727 | 0.1666 | 7.56% |
| L2 | 1,730 | 0.2704 | 0.2302 | 10.00% |
| L3 | 2,200 | 0.2835 | 0.2402 | 10.18% |
| 全部 | 5,540 | 0.3360 | 0.2990 | 19.73% |

Intent/Execution 是分阶段归一化分数，不是任务成功率。Completion 使用 `legacy-final-layout-v1`，Intent/Execution 使用 strict gate。Shuffle 全集为 I=0.0690、E=0.0364、C=2.65%；说明依赖语言，但不足以证明组合指令泛化良好。

## 定位到的失败环节

- 对象选择：`stage1_source_select` Intent 满分，L0 为 670/670，L1 为 314/800（39.25%），L2 为 847/1450，L3 为 1123/2120。分母仅含该阶段的任务；这些是阶段满分比例，不是全任务成功率。布局变更后的视觉定位/对象选择是主要瓶颈，仍需图像和动作轨迹确认具体成因。
- 路径约束：scene3 L2 的 `stage2_route` 仅 9/640 满分，631 次直接未满足路线约束，无前序 gate；L0 为 160/160。L2 后续方向阶段 631/640 是被前序失败 gate，不能重复解释为 631 次独立朝向错误。
- L3 朝向：960 次中 644 次前序 gate；其余 316 次有 315 次报告 yaw error。存在区别于上游 gate 的朝向泛化问题。
- 结束稳定性：L0 `stage3_finish` 仅 5/590 满分，585 次未满；常见原因为回到 home 后再次离开、夹爪再次闭合。该现象与固定 rollout 预算内重复动作一致，但未看视频，尚不能把原因确定为动作 chunk 长或训练尾部不足。历史 `hold_pad=80` 已导致旧协议成绩下降，不重复照搬。

## 两平台一致性

按 mode/scene/task/seed/round 配对，1,800 条全部匹配，指令文本全部相同。全部 run 的 scorer hash、cuRobo source hash、记录的依赖版本、相机/机器人配置 hash、asset overrides、strict gate 和 stepper 均一致。

| scene4 正确指令（900 trials/平台） | HPC3 | A6000 | A6000−HPC3 |
|---|---:|---:|---:|
| Intent | 0.43467 | 0.43200 | −0.00267 |
| Execution | 0.35733 | 0.35389 | −0.00344 |
| Completion | 38.67% | 37.89% | −0.78 个百分点 |

逐 trial 的 I/E/C 分别有 20/30/27 条不同，不能称为逐位一致。该差异远小于 L0→L1 的下降；本次不是完整硬件非确定性研究，也未排除所有渲染数值差异。Shuffle C 分别 8.11%/8.44%。

## 数据与代码核验

权威 manifest：`/data/user/jhe724/outputs/robofollow_q35/data_official3750/manifest.json`。
75 tasks × 50 episodes = 3,750，全部 train，677,823 个时间起点；按任务计 229 个指令变体，全局去重 217（跨任务重复文本），不是漏下载 12 个变体。无内部未见 holdout；旧训练样本不得重新标成独立验证集。

读取分支 `experiment/robofollow-a6000-20261013` @ `4e8acba`：

1. `ge_act/experiments/robofollow_policy.py`：部署固定 `guidance_scale=1.0`；`cached_text_conditioning` 固定关闭 CFG。
2. `ge_act/models/pipeline/custom_pipeline.py`：已有动作 velocity CFG，公式 `v_uncond + scale * (v_cond - v_uncond)`，无需复制 OpenPI/JAX 或新建去噪器。
3. `ge_act/runner/ge_trainer.py`：已生成 `uncond_prompt_attention_mask`，训练 dropout 仅替换 `prompt_embeds`，没有同时替换 `prompt_attention_mask`。启用文本 dropout 后会出现空文本 embedding 配原指令 mask 的条件不一致。当前基线 dropout=0，所以该问题不解释现有基线低分。
4. `robofollow_policy.py` 执行长度与 history stride 当前绑定 32。直接把动作执行改为 8/16 会改变历史帧间隔，不能作为无副作用的一行调整。

## 近期代码筛选

### RSS：第一轮采用其语言残差引导部分

- [官方仓库](https://github.com/Doo-mon/RSS)，读取 commit `f51a0a3f7373b35a61b0690696496066bb5616b8`。
- [ACL 2026 论文](https://aclanthology.org/2026.acl-long.190/)。
- 具体实现：`src/openpi/models/pi0_cfg.py`，训练默认 cfg_dropout=0.1，同步切换 token 与 mask；推理默认 guidance_scale=1.5，保留相同视觉/状态条件，组合 conditional/unconditional velocity。
- 适配本项目：复用已有 T5、caption dropout、动作 CFG；补 mask 一致性及部署参数。第一轮只迁移这一机制，不声称完整复现 RSS（其语义扩展/图像 caption 路线尚未接入）。
- 当前 53k 权重没有训练空指令分支；不能只把 guidance 调大即宣称 RSS 成立。需要等预算续训对照。

### ACG：保留作第二轮候选

- [官方仓库](https://github.com/DAVIAN-Robotics/ACG)，读取 commit `efc9b49f933e5d77deccd816c53594d5c6d501e0`。
- [论文](https://arxiv.org/abs/2510.22201)，2025-10 预印本，ICRA 2026；不要误写成 2026-10 新发表。
- 具体实现：`libs/robomimic/robomimic/algo/guidance/acg.py`，通过部分 attention 层构造 incoherent 分支并进行引导，不等同于简单动作平滑。
- 当前实现针对 GR00T，GE Act 的视频/动作注意力耦合不同，移植需额外验证。它针对动作一致性，无法据此保证修复路径语义选择，故第一轮不混入以免失去归因。

## 第一轮可审阅设计（待确认后编码）

范围限定为已有训练/推理路径：

1. 修复 caption dropout 同步切换 embedding 和 attention mask，测试 p=0 不变、全丢弃、混合 batch、多视角保持同一条件。
2. 接通部署 `guidance_scale`（默认 1），缓存同时支持对应正/空文本条件；拒绝 NaN/非法值。使用项目既有 CFG，不改 scorer、奖励、测试指令或动作预算。
3. 检查三视角 CFG 的 batch 排列、state 复制和缓存；默认 scale=1 的输出应与现版一致，scale>1 必须完成真实 GPU smoke 后才训练/部署。
4. 从同一 53k 权重分别做普通续训和 dropout=0.1 续训；建议各 5k optimizer steps、相同 LR/数据/seed/global batch128，属于 warm start，不冒充恢复旧 optimizer。训练仅 HPC3 jhe724/acd_u，最多单节点 8 卡。
5. 对照：普通续训+scale1；dropout续训+scale1；同一 dropout 权重+scale1.5。只比较预先固定的两个 scale，记录成本与速度，避免无限扫测试集。
6. 先在预先固定、未用于训练的新仿真 seeds 上做开发诊断，基于官方训练任务；L1–L3 用于锁定方案后完整报告。已有 L1–L3 已参与本轮诊断，后续不能称它们为从未查看的验收集。
7. 报告 correct/shuffle 的 I/E/C 和差值、任务级 bootstrap 区间、分阶段分数、时延。只在预先确定的开发标准通过后进入全量评测；失败保留。

涉及文件：`ge_trainer.py`、`robofollow_policy.py`、对应单元测试和独立实验 YAML/launcher。仅若 GPU smoke 证明 pipeline CFG 存在问题才修改 `custom_pipeline.py`。配置不覆盖 baseline；不改现有排队或运行作业。

## 尚未验证

没有确认任何新方法能提分；没有修改模型代码、安装外部仓库依赖、提交新作业。尚未看逐帧视频，尚未做 RSS 在本项目三视角 pipeline 的 GPU smoke。新续训预算 5k 是可证伪的首轮对照预算，不是声称已足够收敛。

本地原始 JSON 和聚合审计：`.agent/robofollow-analysis-20261018/`（不入 Git），包含两平台结果、run metadata、manifest、analysis.json 及原始 tar 的 SHA256。
