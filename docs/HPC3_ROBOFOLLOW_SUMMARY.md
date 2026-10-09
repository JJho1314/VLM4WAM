# RoboFollow 最终研究总结（2026-10-09 汇总）

这里的“最终”指仓库与上下文汇总，不表示实验结束。来源为三个 Ola worktree 的实际文件、HPC3 部署和已保存结果；本次未重跑历史 benchmark。

## 研究目标

优先提升 RoboFollow 的语言遵循能力。Intent 衡量目标/对象/手臂选择，Execution 衡量执行，completion 衡量整任务完成，三者分别报告。正确指令与打乱指令需使用相同任务、round、seed 和评分目标。训练 loss、oracle teacher 特征收益、少量 smoke 的完成均不能替代真实 instruction-following 证据。

## 当前正式 baseline

| 参数 | 当前值 |
| --- | --- |
| 数据 | 官方 75 个训练任务，每任务 50 条，共 3,750 episodes；全部 train，无内部 holdout |
| HF 数据版本 | AutoLab-SJTU/robofollow-data @ bbb1e266ed585f1557773de5a3f1e3b4cd944f97 |
| 时间起点总数 | 677,823 |
| 模型 | text_p256，三相机，14 维双臂绝对关节动作 |
| 输入分辨率 | 原始 240×320 padding 到 256×320 |
| 步数 | 53,000 optimizer steps |
| Batch | 4/GPU × 16 GPU × accumulation 2 = 128 |
| Action chunk / history | 32 / 4 |
| 资源 | HPC3 jhe724，acd_u，2 nodes / 16 GPUs；Slurm 单次时限 7 天 |
| 作业 | 704659；2026-10-09 最后核验 pending，尚无新成绩 |

53,000 × 128 / 677,823 ≈ 10.01 是按全部时间起点折算的采样遍数，不能叫 10 次无重复完整遍历。Dataset 长度是 3,750 条轨迹，每次取样随机选择窗口，所以日志中的 loader epoch 与这个口径不同。较早 30k 配置保留用于追溯，当前启动器明确使用 53k text 配置；joint 的历史 30k 配置尚未启动。

## 已有证据（旧训练协议，不是 53k 新结果）

详见 [完整结果](robofollow_q35_results.md)，包括每个实验的原始协议、置信区间和限制。以下为 20k、global batch 128 的历史 text/joint 对照，554 tasks、seed 42、一轮：

| 层级 | Intent text → joint | Execution text → joint | Completion text → joint |
| --- | --- | --- | --- |
| L0 | 0.503 → 0.608 | 0.483 → 0.505 | 0.507 → 0.493 |
| L2 | 0.274 → 0.318 | 0.201 → 0.268 | 0.116 → 0.139 |
| 全部 554 tasks | 0.279 → 0.311 | 0.225 → 0.264 | 0.161 → 0.168 |

- 全部任务 Intent 增益的 paired 95% CI 为 [+0.009,+0.054]，Execution 为 [+0.017,+0.062]；Completion 为 [−0.018,+0.032]，尚不能宣称完成率显著提升。
- batch 32 的 L0/L2 三轮复测使部分单轮收益缩小，说明必须重复多轮、报告任务级不确定性。
- correct/shuffled 显示模型依赖语言，但已有 batch 32 对照未证明 joint 的语言差值显著优于 text。不要把动作能力提升自动写成更强的语言遵循。
- `hold_pad:80` 的历史 L0 completion 从 0.507 降到 0.413，未支持这一方案；记录失败，避免重复尝试而不改变假设。
- 早期 LIBERO teacher 可显著降低动作 MSE，但预测 planner 可能比 disabled 更差。oracle future information 的收益不代表部署收益。
- 与论文/官方基线比较前，必须匹配训练集、动作执行长度、仿真设置、评测轮数和指标；旧分数与本次正式训练不能混为一个实验。

## Autoresearch 的真实状态

已保留：候选控制器、registry、watchdog、paired evaluation、历史经验、指标表、执行账本、诊断和自进化设计。早期 `complete-v3` 使用内部 train/dev 划分；本轮正式 benchmark baseline 使用官方全量训练集，两种结果不可混用。

历史自进化设计文件明确写着草案。当前代码具有预设候选实验控制与 HPC3 自动提交能力；本次没有证据证明 HPC3 已运行自动生成新代码并持续评测的 coding agent。`sim` 与部分 `serve` 配置仍指向 Ola 仿真环境；汇总仓库不意味着仿真依赖已迁移。

## 接下来按此顺序推进

1. 完成官方全量 53k text baseline，检查 loss、checkpoint 完整性与动作输出。
2. 将 RoboFollow/RoboTwin 仿真依赖、assets、scorer 和运行版本迁入 HPC3，先验证 smoke 和完整评测协议。
3. 固定 text 基线及评测任务/seed 后，再训练等预算 joint；记录 correct/shuffled 的语言差值、Intent、Execution、Completion 和任务级 CI。
4. 自动迭代应使用固定开发协议；最终保留独立验收，避免反复根据测试集挑模型。以真实预测 planner 推理，不使用未来 teacher 信息。
5. 每次候选保留 Git commit、有效配置、数据/协议哈希、checkpoint、完整指标和采纳原因；失败也追加到经验，不清空历史账本。

## 工程经验

- 用 HPC3 登录域名；旧 10.120.48.26 指向的管理节点可能有错误的 Slurm 视图，曾只看到 mgt。不能据此判断整个 GPU 集群不可用。
- 域名密钥变化需从已认证的服务器会话核验；2026-10-09 已核验并更新本机记录。
- 数据直接在 HPC3 下载，锁定版本，续传后做 SHA256 验证；已完成全量下载，不要重下。
- Slurm 作业提交后独立于 Mac；tmux 仅在所在节点存活，登录节点重启或切换不会自动迁移 tmux。
- watcher 使用 jobid、SUBMITTING 和 flock 防重复。状态模糊时先查 Slurm，不能删除标记后盲目重提。
