# VLM4WAM — HPC3 RoboFollow 汇总仓库

当前主线：用官方完整 RoboFollow 训练集建立可复现基线，再验证预测 semantic planner 是否改善 instruction following。最终代码、历史经验、实验结论与 HPC3 启动入口集中在本仓库。

- **权威工作目录**：`/data/user/jhe724/workspace/VLM4WAM_robofollow_official3750`
- **登录**：`ssh jhe724@hpc3login.hpc.hkust-gz.edu.cn`，使用域名，不固定旧 IP。
- **当前配置**：3,750 episodes / 75 tasks × 50，53,000 optimizer steps，global batch 128（每卡 4 × 8 卡 × 累积 4），action chunk 32，1 node / 8 GPUs，`acd_u`。
- **2026-10-10 状态快照**：作业 `704659` 在 1,980/53,000 步因 DataLoader 句柄泄漏失败，尚无 checkpoint 或新 benchmark 结果。修复关闭每轮重建加载器中的持久 worker；最新作业号读取 `.agent/official3750/jobid`，实时状态以 `squeue` 为准。

## 从这里开始

| 内容 | 入口 |
| --- | --- |
| 最终研究结论、协议区别、下一步 | [研究总结](docs/HPC3_ROBOFOLLOW_SUMMARY.md) |
| 登录、环境、数据、训练和恢复 | [HPC3 操作说明](docs/HPC3_RUNBOOK.md) |
| 既有分层成绩与置信区间 | [完整结果记录](docs/robofollow_q35_results.md) |
| 历史研究经验与失败案例 | [autoresearch/EXPERIENCE.md](autoresearch/EXPERIENCE.md) |
| 旧 worktree 独有代码、设计和来源清单 | [归档说明](docs/archive/2026-10-09/README.md) |
| 原始模型项目说明 | [原 README](docs/archive/2026-10-09/entry-documents/README.md) |

## 核心目录

- `ge_act/`：RoboFollow dataset、policy、训练器与 text/joint 配置。
- `qwen35_baton/`、`qwen35_planx/`：语义 planner 及研究模块。
- `autoresearch/`：现有候选控制器、评测、经验、历史指标及 Pixi 入口。
- `.agent/official3750/`：已版本化的 HPC3 训练脚本、数据检查和提交保护；运行状态文件不入 Git。
- `.agent/robofollow_codec/`：训练读取 RGB 使用的 codec。
- `docs/archive/2026-10-09/`：旧目录中与当前主线不同的源码和文档，仅供追溯，不自动覆盖新版。

完整数据、模型权重、训练输出仍在仓库外。旧 autoresearch 文档中的 Ola 路径、8 GPUh/300-step 开发预算及内部划分是历史协议；不能代替当前 53k 全量训练配置。现有自动提交器不等于已运行的“自动改代码→训练→全量评测”闭环，HPC3 仿真环境和该闭环尚未完成端到端验收。
