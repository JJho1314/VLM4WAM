# HPC3 训练后自动评测

评测在独立 worktree `/data/user/jhe724/workspace/VLM4WAM_robofollow_eval`，不修改训练作业 706451 使用的源码。启动入口 `scripts/hpc3_auto_eval.sh smoke|full`，Python 均经 `autoresearch/runtime/pixi.toml`。

- full 只读取训练 run `2026_10_10_21_42_18` 的 step_53000；无最终权重立即失败。
- Slurm full 依赖训练与 smoke 均成功（afterok），不需要 Mac 在线；无效依赖自动取消。
- smoke 使用 step_35000，各 scene 一个任务一轮，只验证环境，不作为 benchmark 分数。
- full: 554 tasks × 10 rounds × correct/shuffle，两种指令使用同一 base seed 42。
- 保持 fixed runtime、TOPP、sim_steps=15；action chunk=32，最多 16 次调用，策略按配置截断到总计 500 动作。
- 每任务 run.json 必须 complete，轮次唯一且全部完成才纳入汇总；保留失败尝试，缺失结果返回非零。
- 结果位于 `/data/user/jhe724/outputs/robofollow_q35/eval/official3750_text_53k_ac32_bs128/full-JOBID`，汇总 `summary.json` 分别记录 Intent、Execution、Completion 和 correct−shuffle。
- 运行环境在 `/data/user/jhe724/workspace/robofollow_eval_runtime`；迁移 Ola 既有 RoboFollow 源码及协议所需机器人/碗资源，归档哈希见 source.sha256。私有 venv 复用 qwen35 的 torch/sapien，只补齐仿真依赖，不修改训练环境。
- 单卡 smoke 编译 cuRobo 并验证四个场景；full 用 acd_u、jhe724、单节点 8 GPU / 96 CPU / 1920 GB，最长 7 天。提交记录在 `.agent/.superpowers/remote-jobs`。

旧 `acp/robofollow_eval.sh` 为 Ola 历史入口，本评测不使用。首次 GPU smoke 通过前，环境兼容性仍待验证。

## 2026-10-12 端口冲突恢复

709872 的固定 9101/9102 端口被已有服务占用，旧 TCP 连通性检查误判就绪。策略服务现使用 `--port 0`，由内核原子分配端口；从本次服务日志读取实际端口后，必须通过 RoboFollow reset 协议握手才能消费任务。服务退出后该 worker 停止，遗漏任务会使整体评测失败。

恢复时设置 `EVAL_RESUME_OUT` 指向已有 full 输出；核对 checkpoint、配置、种子、轮次与任务范围一致，保留旧 run.json 和 logs。完整任务跳过，不完整任务保留失败目录后重跑。进度仅计入完整任务。使用 Pixi research -m pytest 运行 tests/test_rf_bench_completion.py，包含真实 TCP 协议/HTTP 错误服务、进程退出和恢复兼容性检查。
