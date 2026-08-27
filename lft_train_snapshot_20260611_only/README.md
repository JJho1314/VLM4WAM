# VLM4WAM_train_snapshot_20260611 独有文件

删除该快照目录前抢救。对其 1065 个非 results 文件逐一算 `git hash-object`
后与 A 仓库的 git 对象库及 `cosmos-predict2.5/` 工作目录比对（快照把 cosmos 放在
`third_party/` 下，比对时已做路径映射）：1059 个能找到同内容，**这 6 个哪里都没有**。

它们是 2026-06-11 那一版训练所用的、改动过的 Cosmos 核心文件 —— 包括
`minimal_v4_dit.py`（语义条件注入点）和 `experiments/base/robointer.py`。
快照自带的 `results/`（146M）是评测产物，未保留。
