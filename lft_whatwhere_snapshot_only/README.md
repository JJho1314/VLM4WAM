# what-where-softlogit 快照独有文件

来自 `VLA_WM/VLM4WAM_what_where_softlogit_snapshot`，删除该快照目录前抢救。

对快照全部 1176 个文件逐一算 `git hash-object` 后比对：其中 1142 个的内容
能在 A 仓库的 git 对象库或其 `cosmos-predict2.5/` 工作目录里找到，
**这 34 个哪里都没有** —— 是 what-where-softlogit / match-ground-v3 / target-context
系列实验的 sbatch 启动器、可视化与分析脚本，外加 `target_attention_viz.py`。

方案说明见同一归档分支的 `lft_snapshots/README_what_where_softlogit.md`。
