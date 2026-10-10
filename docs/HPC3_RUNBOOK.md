# HPC3 操作说明

## 当前目录与环境

```bash
ssh jhe724@hpc3login.hpc.hkust-gz.edu.cn
cd /data/user/jhe724/workspace/VLM4WAM_robofollow_official3750
```

macOS 的 `HPC3_jhe724` 别名已使用该域名，并通过容器校园 VPN 的 SOCKS5 转发。不要在服务器上配置 Mac 的代理端口。

| 项目 | 位置 |
| --- | --- |
| 代码（独立 Git 仓库） | `/data/user/jhe724/workspace/VLM4WAM_robofollow_official3750` |
| Pixi | `/data/user/jhe724/.pixi/bin/pixi` |
| 已有 Python 环境 | `/data/user/jhe724/.conda/envs/qwen35` |
| 兼容 overlay | `/data/user/jhe724/envs/overlay_peft_tf5` |
| 全量原始训练数据 | `/data/user/jhe724/datasets/RoboFollow-official3750` |
| manifest / stats / DATA_READY | `/data/user/jhe724/outputs/robofollow_q35/data_official3750` |
| 53k 训练输出 | `/data/user/jhe724/outputs/robofollow_q35/official3750_text_53k_ac32_bs128` |

Pixi 复用以上既有 Python 与 overlay；仓库不是能在空白机器上一键安装完整环境的镜像。模型路径见 `ge_act/configs/ltx_model/robofollow/q35_text_p256_official3750_53k_hpc3.yaml`，由数据预检确认存在。

## 查询训练

```bash
squeue -u jhe724
JOB_ID=$(cat .agent/official3750/jobid)
scontrol show job "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,State,Elapsed,ExitCode
```

日志为输出目录下 `logs/slurm-704659.out` 和 `.err`，排队时文件可能尚未创建。实际 checkpoint 路径以训练日志为准；训练器含最终步保存，53,000 不必被周期保存间隔整除。

## 无 GPU 预检

```bash
sha256sum -c .agent/official3750/code.sha256
/data/user/jhe724/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research .agent/official3750/check_data.py
/data/user/jhe724/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research .agent/official3750/test_train_node.py
/data/user/jhe724/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research .agent/official3750/test_submit_guards.py
```

数据检查覆盖 3,750/75×50 清单、统计哈希、数据存在性、真实样本形状和 53k/global128/action32 配置。SHA 清单针对作业部署的关键文件，不是整个仓库的清单。完整来源追踪另见归档 `source-map.json`。

## 回归测试

```bash
/data/user/jhe724/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml test -q tests/test_robofollow_data.py tests/test_robofollow_policy.py tests/test_robofollow_epoch.py tests/test_robofollow_fps.py tests/test_robofollow_loading.py tests/test_robofollow_evaluation.py tests/test_robofollow_planner.py
```

根 pytest.ini 排除 docs/archive，避免将历史版本的测试误采集为当前测试。

## 提交与恢复

作业 704659 在 2026-10-10 11:56 启动，12:34 失败，停在 1,980 步，尚未到 2,500 步首次 checkpoint。**重提前先核验旧作业已终止及队列无重复 baseline**。已有 `.agent/official3750/jobid` 是运行状态，不提交到 Git；新 clone 没有它，不代表 Slurm 没有该作业。

失败原因：每轮 `skip_first_batches` 重建 DataLoader，持久 worker 与 pin_memory 触发 PyTorch atexit 引用保留，句柄随轮数增长。修复为 `persistent_workers=False`，保留 6 workers、pin_memory 和原训练配置。Linux CPU 回归保留真实进程、队列和锁页线程，仅替换张量 pin 操作，并验证 epoch 更新、恢复跳批和句柄稳定性：`test -q -s tests/test_robofollow_loader_fds.py`。重提时保留旧日志及 jobid 备份；无 checkpoint 则从初始权重重训。

修复验证：24 轮 CPU 回归从原版 FD 16→108、每轮 +4，变为固定 6；52 项 RoboFollow 测试通过。连同 BATON 扩展测试共 125 项通过、4 项失败：3 项引用不存在的旧工作站 Python 路径，1 项分布式 checkpoint 子进程超过 30 秒超时。官方数据预检再次通过（3,750 条、5 个真实样本）。这些检查不替代修复后 16 卡完整训练验证。

- `train.sbatch`：两节点，每节点 8 GPU / 96 CPU / 1920G，7 天。
- `train_node.sh`：使用 Slurm node rank 启动 torch distributed，每节点 8 进程。
- `submit_when_ready.py`：检查 DATA_READY、CODE_READY、哈希、真实数据、分区、账号和 `sbatch --test-only`，持 submit.lock 写 SUBMITTING/jobid。
- waiter 见到 jobid 后退出。不要将历史 tmux 日志当作训练进度。
- 如果存在 SUBMITTING 而无 jobid，先查 squeue/sacct 和 jobid.tmp，再人工判断；不要直接清理状态重提。
- 新实验应在新工作目录和输出目录内修改脚本、配置并重建代码哈希，不修改当前 baseline 的冻结文件。

本次仓库汇总保留原训练目录，因此不需要重提或取消作业。完整 Git 历史备份在仓库外 `/data/user/jhe724/workspace/VLM4WAM_consolidation_backup_20261009/`；GitHub 只发布汇总分支，不强制改写 main 或旧分支。

## 评测与自动迭代尚缺什么

现有 `autoresearch/runtime/pixi.toml` 的 `sim` 和部分 `serve` 路径依赖 Ola 的 RoboTwin。不要直接在 HPC3 运行这些任务并假设它们可用。需要迁移仿真环境、assets、官方 scorer 并实测，随后才可声明 HPC3 上的完整 train/evaluate 循环可用。

早期 autoresearch 文档保留原始预算、划分和启动命令用于追溯。当前 53k baseline 的资源授权不自动让旧控制器、旧 registry 或 coding agent 在后台启动；本次未启动新自进化循环。
