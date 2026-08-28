# HPC3 上剩余的独有代码

对 HPC3 `/data/user/jhe724/workspace/VLM4WAM` 下全部 347 个代码/配置文件逐一算
`git hash-object`，与合并后仓库的全部 blob 比对，这 31 个在仓库里找不到：

- `mg_eval/` —— matching-vs-noise / contrastive-ceiling / depth 三组 probe 及其 sbatch
- `semantic_localization/sg_improve/sg_qwen35_train.py` 与 4 个 oracle 复现 generator
- 8 个 HPC3 sbatch 启动器（planner 消融/评测/可视化、depth 全量微调、smoke）
- `read_gate.py`、`_hpc3_setup.sh`、`_copyenv.sh`、`scripts_tmp_convert_matchground.sh`

另有 160 个非代码文件（eval 输出目录里的 manifest/config json）未收录，它们与各自的产物同在。
