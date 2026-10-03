# RoboFollow 优先的 autoresearch

目标是提高真实部署时的指令遵循能力。以官方严格 Intent 为主指标，Execution 和 task-macro CR 为非退化约束；正确/打乱指令使用相同任务、round、seed 和 scorer 目标，语言差值作为诊断。局部开发分数不能称为完整 benchmark 分数。

活动项目位于 Ola_2208：`/data/users/junjie/workspace/VLM4WAM_baton_robofollow_autoresearch`，分支 `research/robofollow-autoresearch`。原项目的 dirty/untracked 内容已带入隔离 worktree，未提交的用户修改保持原样。实验快照保存有效源码（包括 untracked research_provider、嵌套 gitlink 内容）、HEAD、dirty patch、配置及来源哈希。大型数据和权重放在研究输出目录。

## 数据和协议

- `frozen-v1`：首轮启动时冻结的 3,536 条有效 episode。当前 R0 与第一轮 R1/R2/R3 使用此版本。
- `complete-v2`：下载完成后核对 3,750 条原始 episode、75 个训练任务；4 条重复轨迹排除后 3,746 条有效数据，train 2,846 / dev 900，0 quarantine。下载 receipt 和各场景覆盖见 `configs/data_complete_v2.json`。
- 数据及统计：`/data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_autoresearch/{data,data_complete_v2}`。按 scene/task/instruction 组划分，统计量仅来自 train；绝对关节动作和 state 为 14 维，三相机 head/left_wrist/right_wrist。
- `configs/robofollow_eval*.json`：固定四个训练域 held-out L0 任务，每任务 2 round，每 round 5 个 50-action chunk，correct/shuffle 共 16 trials。采样历史 stride=50；simulation timestep 按现有默认 1/250、sim_steps15 配置，原始数据没有时间属性。
- `configs/robofollow_official*.json`：完整 554 任务、L0–L3、10 round、10 step，共 5,540 trials 的独立验收清单。首轮开发控制器只接受四任务配对协议；完整官方验收需单独安排预算和执行，不通过开发入口冒充完成。

## 启动和续跑

所有远程 Python 均通过 Pixi，复用已有训练及仿真解释器；主机路径和 Vulkan ICD 在 `runtime/pixi.toml` 中明确记录。

```bash
cd /data/users/junjie/workspace/VLM4WAM_baton_robofollow_autoresearch
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research -m autoresearch.controller status
# 顺序执行：prepare → train → paired evaluate → review；自动跳过已冻结 R0
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research -m autoresearch.controller loop --candidates R0 R1 R2 R3
```

也可分别执行，方便检查失败：

```bash
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research -m autoresearch.controller prepare --candidate R1
# 用上一条输出的 run-id 替换 ID
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research -m autoresearch.controller run --run-id ID
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research -m autoresearch.controller evaluate --run-id ID
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research -m autoresearch.controller review --run-id ID
# 崩溃后持锁核对 PID 创建身份、清理本 run 进程并保守计费
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research -m autoresearch.controller recover --run-id ID
```

使用完整数据版本时需要新的、明确分配预算的 registry，并重新建立 R0；不能复用 v1 的 best/阈值：

```bash
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research -m autoresearch.controller --root /data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_autoresearch/runs_complete_v2 --data-version complete-v2 loop --candidates R0 R1 R2 R3
```

首阶段累计上限 8 GPUh、最多 GPU0/1 两张、每候选从训练启动计总 wall time ≤2h、最多 300 optimizer steps。控制器持排他锁，按实际进程 wall time × 分配 GPU 数计费；评测含 policy server 和 simulator，按两 GPU 计。新 root 是新实验阶段，不能用来绕过本阶段累计预算。增大预算需要明确新的资源授权后修改 Registry 的上限，不能清空 budget.json。

候选模板固定在 `configs/robofollow_candidates.json`：R0文本、R1预测 planner、R2训练时 50% teacher mixing、R3联合 Query Tower/Sem MLP 加 .01 anchor。Qwen 冻结；推理只有当前三视角图像和指令，不接入 teacher future features。三视角是明确的旧双视角逐视角共享头迁移，尚需 RoboFollow 评测证明效果。

## 结果与失败

输出根：`/data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_autoresearch/runs`。每个 run 保存 train/evaluation 日志、checkpoint、代码快照、run.json、provenance.json 和 evidence.json；根目录保存 baseline.json、thresholds.json、best.json、budget.json。`trained` 表示训练完成但评测尚未完成。评测重跑使用 attempt 新目录，旧失败记录不会覆盖。

R0完整配对评测结束后先冻结阈值；R1/R2/R3 随后才允许 prepare。best 初始指向 baseline，仅表示对照起点。只有完整、有限、协议一致且 Intent 提升达到冻结阈值，Execution/CR 不退化的候选才可替换 best。无收益标为 inconclusive，不完整/超时/NaN 不选优。语言 uncertainty 按配对任务均值估算；仅四个任务，不能据此推断 L1–L3 泛化。

历史经验见 `EXPERIENCE.md`，首轮实际结果见 `REPORT.md`。300 步是有限预算探索，存在未收敛风险；GPU 单步 smoke 仅证明工程链路。

## 复现和验证

```bash
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml test tests/test_robofollow_data.py tests/test_robofollow_loading.py tests/test_robofollow_planner.py tests/test_robofollow_policy.py tests/test_robofollow_evaluation.py tests/test_autoresearch_controller.py -q
```

generic trainer 的两个 callback 接入 hunk 保存在 `patches/generic_condition_hook.patch`，当前 worktree 已应用。原 dirty baseline 归档在 `/tmp/robofollow-baseline.patch` 与 `/tmp/robofollow-untracked.tar`；运行快照包含有效 ge_trainer.py，单独 checkout 本分支还不能替代这些用户 baseline 内容。`provenance.json` 的环境白名单及 package 版本、checkpoint/data 哈希与 runtime 配置共同记录外部依赖；权重目录路径标识不等于把全部模型权重打包进 Git。
