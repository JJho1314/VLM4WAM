# RoboFollow 首阶段实验报告

状态：2026-10-03，清洁完整数据 complete-v3 上的基线与候选正在执行。最终指标以本目录后续更新和研究输出 evidence.json 为准。本报告不宣称已经提升指令遵循能力，不把历史诊断分数作为 held-out 能力结果。

## 数据与协议

完整下载3,750个原始episode、75个官方训练任务，保留3,746个有效示范、4个重复排除、0 quarantine。v3按task/instruction和相同joint-trajectory构建连通分量，保留语言别名并防止同轨迹跨划分：train2,896/dev850，0跨划分轨迹重复。manifest SHA256为84d6b935cc371a5c8213c7e06da5150e8655f3f68b3ed6de21131a54ed785871。统计仅来自训练部分。

固定开发协议四场景各一个held-out L0任务、每任务2round、每round5个50-action chunk，正确和打乱指令共16trials。新scene4任务是L0_s41_r_unstack_green_cylinder_to_slot6，其余任务清单见configs/robofollow_eval_complete_v3.json；协议hash a846b4e5a978fe8213aa2932d0ef5b49a838711e4bbaf98ceb63544f42bed61a。Intent/Execution使用官方strict门控，completion为task-macro；scorer目标不随policy收到的打乱指令改变。四任务的小样本语言差区间不能代表L1–L3泛化。

完整官方554任务、L0–L3、10round、10step的5,540trials尚未执行。开发协议比完整协议短，300optimizer steps仅作有限预算探索，不能保证动作头收敛。

## 已保留的诊断与失败

| 记录 | 结果 | 准入 |
| --- | --- | --- |
| 早期真实TCP四场景短smoke | Intent/Execution/CR均0 | 仅工程链路 |
| frozen-v1 R0 20261003_100622_R0 | 300步完成，16trials均0；语言差0 | 发现跨划分副本后失效，禁止作held-out证据 |
| frozen-v1 R1 20261003_113229_R1 | 数据审计后安全停止训练，无评测 | 不准入 |
| complete-v3 R0 20261003_114434_R0 | 配置根目录误替换，prepare拒绝 | 无GPU费用 |
| complete-v3 R0 20261003_114841_R0 | 新runner读取缺失semantic_plan字段，启动失败 | 不准入；已加回归修复 |
| complete-v3 R0 20261003_115552_R0 | 当前活动清洁基线 | 等待完整评测 |

v1有3组、完整v2有48组同图像/动作但不同指令的跨划分副本，验证了只用含instruction的content hash会漏检。旧best/thresholds/baseline已经归档到runs/superseded_frozen_v1；预算账本保留全部费用。此问题在得出性能结论前处理，旧零分不是清洁基线。

## 预算与复现

首阶段累计上限8GPUh，最多GPU0/1两张H800，最多300steps。清洁候选的总wall time从训练启动计75分钟；启动前按timeout×2GPU预留。正确/打乱指令使用各自服务和8999/9000端口，可并行运行，历史和prompt缓存互不混用；仍按两张GPU计费。并行worker任一个失败时停止另一个，partial结果不选优。

研究输出：/data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_autoresearch。状态、实际费用、代码/配置/权重hash、每run假设及next suggestion保存在runs目录。旧失败不覆盖；CPU prepare失败目录、所有中止及无收益结果保留。清洁阶段沿用原budget.json，没有以换root重置预算。

```bash
cd /data/users/junjie/workspace/VLM4WAM_baton_robofollow_autoresearch
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research -m autoresearch.controller status
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research -m autoresearch.controller --data-version complete-v3 loop --candidates R0 R1 R2 R3 --timeout-seconds 4500
```

自动跳过已经冻结的清洁R0，候选必须协议/runtime一致、结果完整且有限、Intent超过冻结阈值且Execution/CR不退化才更新best。无收益为inconclusive，预算不足保存budget_stop；Intent与Execution同时为零则停止扩展，先诊断14D投影、归一化和小样本拟合。

## 验证范围

预测/teacher/mix/joint单GPU步与joint梯度均实际验证，见configs/planner_validation.json。最初新增47项测试通过；加入别名连通划分后相关104项通过，路径修复及文本runner/并行worker又加入实际故障回归，最终验收仍在执行。

广泛项目tests曾1046通过、37失败、1跳过、10错误；其中3个静态失败在原baseline复现，其余有环境/共享状态与未分诊问题。不是整个项目测试通过。原始日志在outputs/robofollow_autoresearch/validation。原项目dirty代码保持未提交，generic trainer两个接入hunk单独保存，运行快照包括有效源码。
