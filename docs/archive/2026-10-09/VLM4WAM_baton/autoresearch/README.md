# RoboFollow autoresearch 活动入口

活动实现位于同仓库隔离 worktree：
`/data/users/junjie/workspace/VLM4WAM_baton_robofollow_autoresearch`
分支：`research/robofollow-autoresearch`。

这里保留原项目未提交修改。活动目录包含完整数据适配、三视角研究planner、14维绝对关节policy、固定语言配对评测、有预算控制器和研究记忆。请在活动目录操作，不要以当前目录旧文档作为运行状态。

- 使用/续跑：活动目录 `autoresearch/README.md`
- 经验：`autoresearch/EXPERIENCE.md`
- 实际状态与限制：`autoresearch/REPORT.md`
- agent迭代规则：`autoresearch/PROGRAM.md`
- 清洁完整数据版本：complete-v3，3746示范/75任务，train2896/dev850；轨迹别名连通分组，0跨划分轨迹重复。
- 输出与累计账本：`/data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_autoresearch/runs`

```bash
cd /data/users/junjie/workspace/VLM4WAM_baton_robofollow_autoresearch
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research -m autoresearch.controller status
```

首阶段8GPUh账本包含被数据审计取代的旧实验，未重置；旧v1/v2不允许新训练/选优。完整官方5540trials尚未执行，开发结果不能冒充完整benchmark成绩。

2026-10-03首阶段已完整结束：R0/R1各300训练步、各16个完整配对L0开发trials，Intent/Execution/CR全0，尚未证明instruction-following改善。控制器evidence_stop停止R2/R3，累计6.816886/8GPUh，研究服务和拥有进程已清理。报告、经验、PROGRAM、审查、动作诊断与执行账本的本轮副本已放在本目录。活动分支HEAD f1d886c；广泛项目测试未全绿，分支/worktree保留，未合并或推送。
