# VLM4WAM 散落文件归档 (2026-08-27)

孤儿分支 `archive/loose-files-20260827`。收集所有**不在任何 git 分支中**、
只存在于某台机器磁盘上的文件。整合来源与判定见 `docs/CONSOLIDATION_20260827.md`。

| 目录 | 来源 | 说明 |
|---|---|---|
| `hpc3_bigdir/` | HPC3 `workspace/VLM4WAM/{tools,scripts}` | 39 个代码文件在两个 LFT 仓库的任何分支中都不存在。what-where-softlogit / oracle 复现的分析与 sbatch 工具链。 |
| `qwen35_baton_strict/` | HPC3 `VLM4WAM_qwen35_baton_strict` | 该目录 181 个文件中仅这 3 个独有（其余已覆盖于 `wsB/qwen35-video-hindsight-grounding`）。 |
| `qwen35_worldarena/` | HPC3 `VLM4WAM_qwen35_worldarena` | 代码已 100% 覆盖于 `wsB/worldArena@c0812aa`；此处只留 35 组 runtime 训练记录 + agent review 笔记。 |
| `hdf5_pilot/` | HPC3 `VLM4WAM_hdf5_pilot_b607f0d` | 唯一独有文件：pilot sbatch。 |
| `lft_repoA_untracked/` | LFT `VLA_WM/VLM4WAM` 未跟踪 | 3 个 ge_act 数据加载器 + semantic_localization 代码（已剔除 4.9G 权重/视频）。 |
| `lft_repoB_untracked/` | LFT `workspace/VLM4WAM` 未跟踪 | 未提交的 plan/spec 文档与 7 个测试文件。 |
| `lft_snapshots/` | LFT 两个 snapshot 目录 | what-where-softlogit 方案说明、训练快照 README 与 docs。 |
