# RoboFollow 后台自动实验（2026-10-04 启动）

已在 Ola_2208 启动有界实验控制器，PID 1904524，源代码版本 54dc350。启动验证已看到第 51 步，loss=1.182299（首步 2.663147），训练进程 PID 1904717；这仅证明训练正常进行，不代表 held-noise 或 benchmark 改善。本文是启动记录；实时状态以远程 status.json 为准。

后台目录：
`/data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_autoresearch/background/fp32_warmup_20261004_2125`

- 控制器：controller.py；实验脚本模板：probe.py。
- 实时状态：status.json；控制器日志：controller.log。
- 首轮：runs/20261004_132415_auto_fp32_warmup_1（位于原 robofollow_autoresearch 输出根目录）。训练日志 train/process.log；进度 evidence/progress.json。
- 训练：单个 train motion episode、动作专家参数及 Adam 状态 FP32、BF16 autocast；最多 2000 步/候选。
- 先 lr=3e-5；若 held-noise velocity MSE 未小于 0.1，自动从相同初始模型比较 lr=3e-4。最多两个候选。
- 原累计预算 8 GPUh；启动前已用 7.121416 GPUh。每轮 GPU 进程上限 1100 秒；由独立 watchdog 执行。失败、超时、预算不足即停止；无可用 GPU 最多等待一小时。
- 每个成功完成的候选保存动作专家部分权重、配置、源码快照、指标和离线结论。部分权重须配合原模型和配置加载。

这是 RoboFollow 优化的训练拟合诊断阶段。held-noise 仍来自同一 train episode，不代表 benchmark 分数或指令泛化。拟合门槛通过后停止，下一步需采样动作、多 episode 验证，再做正式 correct/shuffled 指令配对 rollout。现有 baseline/best 和正式分数不更新。该控制器按预设候选自动实验与汇总，不会无限自行改写代码。

启动命令（已执行，请勿重复启动）：

```bash
cd /data/users/junjie/workspace/VLM4WAM_baton_robofollow_autoresearch
nohup /data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research /data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_autoresearch/background/fp32_warmup_20261004_2125/controller.py > /data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_autoresearch/background/fp32_warmup_20261004_2125/controller.log 2>&1 < /dev/null &
```
