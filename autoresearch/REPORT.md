# RoboFollow 首阶段实验报告

状态：2026-10-03，清洁完整数据 complete-v3 上的 R0 文本基线与 R1 预测 planner 对照已完整完成，首阶段有界实验已停止。最终指标以本目录后续更新和研究输出 evidence.json 为准。本报告不宣称已经提升指令遵循能力，不把历史诊断分数作为 held-out 能力结果。

## 数据与协议

完整下载3,750个原始episode、75个官方训练任务，本轮冻结保留3,746个示范、4条按关节/指令内容排除、0 quarantine。补充全图像审计发现，这4条的图像不同，不能称为完全重复；后续preflight已修复为关节/指令及全部可用图像共同去重。本轮保持原subset及统计，避免训练后更换数据。3746个保留episode的2031975张可用相机帧全部解码通过。v3按task/instruction和相同joint-trajectory构建连通分量，保留语言别名并防止同轨迹跨划分：train2,896/dev850，0跨划分轨迹重复。manifest SHA256为84d6b935cc371a5c8213c7e06da5150e8655f3f68b3ed6de21131a54ed785871。统计仅来自训练部分。

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
| complete-v3 R0 20261003_115552_R0 | 300步完成，训练302.722秒；第一次配对启动因显存竞争失败；第二次16trials完整，所有指标0 | 清洁校准基线，失败费用保留 |
| complete-v3 R1 20261003_130256_R1 | prepare来源校验顺序错误，CPU阶段失败；回归已修复 | 无GPU费用，目录保留 |
| complete-v3 R1 20261003_131008_R1 | 300步完成，16trials全部完整 | inconclusive；见下表 |

v1有3组、完整v2有48组同图像/动作但不同指令的跨划分副本，验证了只用含instruction的content hash会漏检。旧best/thresholds/baseline已经归档到runs/superseded_frozen_v1；预算账本保留全部费用。此问题在得出性能结论前处理，旧零分不是清洁基线。

## 清洁基线实测

R0 `20261003_115552_R0`，文本条件，300训练步。correct/shuffle分别8个episode，四场景的官方输出全部complete。所有结果均为L0开发子集。

| 场景/等级 | correct Intent | correct Execution | correct CR | shuffle Intent | shuffle Execution | shuffle CR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scene1/L0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene2/L0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene3/L0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene4/L0 | 0 | 0 | 0 | 0 | 0 | 0 |

语言差ΔIntent/ΔExecution/ΔCR全部0；四任务配对均值近似normal区间为[0,0]，standard error0。这是零分样本的描述，不是语言不重要或具有泛化能力的证据。冻结Intent最低增益0.01、Execution/CR容忍退化0.02。best指向校准R0，不表示改善。

R0在环境隔离修复前已完成训练，原provenance未捕获BATON_RESEARCH开关；后验审计contract不能恢复未记录的历史环境。R1使用显式sanitized环境，因此R0环境一致性有这一已知限制，不能隐去或宣称原记录已补全。

训练最后batch loss约1.77，源GE模型迁移新初始化14D动作投影的四个key。58步训练actions取最后54步，推理生成54步并执行前50步，实际代码检查没有发现该切片错位。官方scene1日志出现wrong-arm和source/target门控失败，下一步需训练集小批过拟合及动作头/主干学习率对照；尚不能把失败归因于语言或某一归一化bug。详见`ACTION_DIAGNOSTIC.md`。

## R1 完整对照与停止理由

R1 `20261003_131008_R1`，仅增加predicted planner条件，训练300步。最后batch loss1.97，训练phase含模型加载542.738秒/0.150761GPUh，峰值allocated37.805GB。冻结Qwen、SigLIP及planner的content hash记录在provenance与checkpoint contract。

| 场景/等级 | correct Intent | correct Execution | correct CR | shuffle Intent | shuffle Execution | shuffle CR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scene1/L0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene2/L0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene3/L0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene4/L0 | 0 | 0 | 0 | 0 | 0 | 0 |

准入：`inconclusive`，Intent增益未达到冻结阈值。language_delta：`{"mean_intent_score": 0.0, "mean_exec_score": 0.0, "completion_rate": 0}`；language uncertainty：`{"mean": 0.0, "standard_error": 0.0, "ci95": [0.0, 0.0], "tasks": 4, "episodes": 8, "method": "approximate normal interval over task means; small fixed development subset, no benchmark generalization"}`。

停止：`evidence_stop`，inspect action/state scaling, 14D projection initialization and small-batch fitting before attributing failure to language。R2/R3本轮未执行，不作为失败或已有收益报告。best仍指向`20261003_115552_R0`，是基线校准指针。R0/R1合计32个完整配对开发trials，未证明instruction-following改善。下一轮先做训练集小批拟合和动作投影warmup/LR单因素验证，再决定语言切换和mixing/joint；不能放宽scorer或使用部署teacher future features。

## 预算与复现

首阶段累计已记账6.816886/8GPUh（其中pre-controller smoke按1GPUh保守估计，其余由phase wall×GPU记录；包含失效数据划分、中止、OOM及重试），最多GPU0/1两张H800，最多300steps。清洁候选的总wall time从训练启动计75分钟；启动前按timeout×2GPU预留。正确/打乱指令使用各自服务和8999/9000端口，可并行运行，历史和prompt缓存互不混用；仍按两张GPU计费。并行worker任一个失败时停止另一个，partial结果不选优。

研究输出：/data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_autoresearch。状态、实际费用、代码/配置/权重hash、每run假设及next suggestion保存在runs目录。旧失败不覆盖；CPU prepare失败目录、所有中止及无收益结果保留。清洁阶段沿用原budget.json，没有以换root重置预算。

```bash
cd /data/users/junjie/workspace/VLM4WAM_baton_robofollow_autoresearch
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research -m autoresearch.controller status
/data/users/junjie/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research -m autoresearch.controller --data-version complete-v3 loop --candidates R2 R3 --timeout-seconds 4500
```

上述续跑R2/R3命令仅在诊断证据支持且预算另行明确后使用；当前零分停止规则和剩余预算不支持直接扩展，按原账本执行会被预算准入拒绝。不得新建root隐藏费用。自动跳过已经冻结的清洁R0，候选必须协议/runtime一致、结果完整且有限、Intent超过冻结阈值且Execution/CR不退化才更新best。无收益为inconclusive，预算不足保存budget_stop；Intent与Execution同时为零则停止扩展，先诊断14D投影、归一化和小样本拟合。

## 验证范围

预测/teacher/mix/joint单GPU步与joint梯度均实际验证，见configs/planner_validation.json。最初新增47项测试通过；加入别名连通划分后相关104项通过，路径修复及文本runner/并行worker又加入实际故障回归，最后独立审查四项Important均修复，回归122项通过（34.96秒）。包括相同shape的错误checkpoint语义拒绝、全部可用帧校验、继承研究环境开关隔离和控制器SIGTERM/SIGKILL后清理/计费/新任务阻拦。见REVIEW.md及validation日志。

广泛项目tests曾1046通过、37失败、1跳过、10错误；其中3个静态失败在原baseline复现，其余有环境/共享状态与未分诊问题。不是整个项目测试通过。原始日志在outputs/robofollow_autoresearch/validation。原项目dirty代码保持未提交，generic trainer两个接入hunk单独保存，运行快照包括有效源码。
