# Baton autoresearch 实验上下文和经验

截至 2026 年 10 月 3 日，从 Ola_2208 上的现有代码、评估汇总和日志整理。本文保留历史研究记忆，并追加本轮已实施的 RoboFollow 控制器与工程验证；不包含先前未保存的对话。历史表格不替代本轮真实 RoboFollow 评测，当前结果见 REPORT.md。

当前用户首要目标是提升 RoboFollow 的语言遵循能力，并用受控实验体现模型确实依赖指令。预测 planner 的动作效果与 LIBERO 误差保留为辅助诊断，不再作为 autoresearch 首要排序指标。RoboFollow Intent、Execution 和任务完成率分别记录。

## 项目和现有入口

项目：`/data/users/junjie/workspace/VLM4WAM_baton`。读取时分支 `qwen35-video-hindsight-grounding`，HEAD `4a58318`，工作区包含大量未提交修改。复现实验必须保存 dirty diff 和新增文件内容，单独记录 HEAD 不够。

- `qwen35_baton/model.py`：Qwen3.5、Query Tower 和 Sem MLP，输出 `[B,2,4,256,1024]`。
- `qwen35_baton/losses.py`：基础特征 MSE，研究扩展支持 DA3 多层 WSA。
- `ge_act/runner/ge_trainer.py`：teacher conditioning、预测 planner mixing、可训练 research planner head。
- `acp/stage2_additive.sh`：现有 additive 训练入口。
- `qwen35_baton/scripts/research/research_eval_stage2_modes.py`：相同窗口、相同噪声的 teacher、disabled、planner 和视角消融评估。
- 历史输出：`/data/users/junjie/workspace/hpc3_jhe724/outputs/research_q35`。

根 README 主要介绍 Cosmos；当前 Baton LTX 路线应以代码、配置与实际启动环境为准。

## 实测结果

以下均为保存的评估汇总，未重新运行。动作列是 `action_all` MSE，越低越好。disabled 指同一个 checkpoint 关闭语义条件，不是单独训练的无语义模型。

| 实验 | 样本 | Teacher | Disabled | Planner |
| --- | ---: | ---: | ---: | ---: |
| additive 双视角 20k | 210 | 0.003057 | 0.016475 | 未评估于该汇总 |
| additive 主视角 20k | 210 | 0.006350 | 0.016231 | 0.113386 |
| control 20k | 210 | 0.004975 | 0.006398 | 未评估于该汇总 |
| Stage 3 p100 3k | 210 | 0.007025 | 0.011913 | 0.027457 |
| additive 双视角 5k | 210 | 0.106510 | 0.194876 | 0.207387 |

双视角 20k 的 Teacher 相对 Disabled 动作误差下降约 81.4%；这证明该模型能够利用 oracle future features，不能据此宣称部署时预测 planner 同样有效。主视角 20k 的 Planner 为 Disabled 的约 6.99 倍，paired difference 为 +0.097155，SE 0.013774，动作获胜率仅 2.38%。Stage 3 p100 3k 的 Planner 仍为 Disabled 的约 2.30 倍，paired difference +0.015544，SE 0.004007。

不同实验的训练初始化、预算和条件并不完全一致；跨实验的绝对 MSE 只作为描述，不构成受控因果比较。4 个样本的 smoke 只证明流程可运行，不用于宣告效果。

## 可以继承的研究判断

1. **已经观察到：oracle 有效不等于预测 planner 有效。** 后续候选必须评估真实 planner 输入，而不能只按 teacher 指标保留。
2. **待验证假设：teacher 与 planner 特征分布或语义质量差距限制部署效果。** 现有混合训练和 head 联合训练提供验证入口；尚未排除初始化、归一化、视角和加载错误等替代解释。
3. **视角作用与训练配置相关。** 双视角 20k 中 wrist masked 的动作 MSE 为 0.020313，main masked 为 0.009812，完整 teacher 为 0.003057。主视角路线 wrist masked 与 teacher 完全相同，不能据此推断 wrist 普遍无用。
4. **语义依赖增强可能伴随关闭语义时退化。** additive 双视角 20k 的 disabled 0.016475 高于 control 20k 的 0.006398；需要受控实验确认，不能只追求 teacher 与 disabled 的相对差距。
5. **视频平均误差对动作差距不敏感。** 主视角 20k teacher/planner 的 video MSE 约为 0.100397/0.101993，而 action MSE 差距很大。动作与语言指标应单独保留。

## 工程经验及其证据级别

- **日志实测**：loader w4/prefetch2 与 w8/prefetch4 的 60 步最终进度分别约 10.24 和 10.22 秒每步，峰值 allocated 均约 28.774 GB。这两个短测试不支持明显吞吐提升；需排除 warmup、checkpoint 和其他作业干扰后复测。
- **启动脚本说明**：Qwen research plain DDP 的 fp32 权重、梯度、AdamW 状态约占 37 GB 每 GPU，4 samples/GPU 会超过 80 GB；脚本默认降为 2 samples/GPU 并开启 gradient checkpointing。这是既有注释经验，当前整理未复现 OOM。
- **代码实现经验**：warm start 保留同名同形状参数的 AdamW moments，避免新 AdamW 的首步扰动；新模块使用单独学习率。依据 `research_train.py` 注释，尚无本次独立对照。
- **存储经验**：ACP 脚本在只读 HDF5 数据访问上设置 `HDF5_USE_FILE_LOCKING=FALSE`，注释记录 AFS/quarkfs FUSE 偶发 errno 9；不要把该设置推广到并发写入。
- **检查点约定**：production provider 依赖可信 topology、元数据和完整 checkpoint。研究 variant 另有契约；使用前检查实际版本，文档的早期 v2 描述与研究代码的 v3 描述不能混用。
- **配置记录缺口**：许多 `BATON_RESEARCH_*` 开关由环境变量读取，保存的 config.json 不足以证明有效配置。新实验必须保存实际环境白名单和加载日志，包括 planner probability、joint、anchor、gate、additive、初始化路径。
- **运行状态局限**：日志末尾显示 s2planner20k_v2、s2mix050_20k_v2、s3planA_p050_v2 尚未达到 20k；仅凭日志不能判定仍在运行。s3joint_smoke 日志只有启动输出，不能视为成功完成。

## RoboFollow 接入状态

已有代码位于 `/data/users/junjie/workspace/robofollow/RoboTwin/robofollow`，快捷路径 `/data/users/junjie/workspace/robofollow/RoboFollow` 指向它。读取的版本为 `a867718`。已有 env、assets 和 HoldPolicy smoke 结果；smoke 的 Intent/Execution 为零，只说明该测试执行完毕，不是 GE Act 或 Baton 的能力基线。

数据完成状态已更新：`/data/users/junjie/workspace/datasets/RoboFollow` 共3,750条原始轨迹，覆盖全部75个训练任务；排除4条内容重复后3,746条有效轨迹，train2,846/dev900，0 quarantine。下载 receipt 确认15158个文件与LFS SHA256/Git blobs验证完成。完整下载先冻结为complete-v2；追加轨迹审计发现v1有3组、v2有48组同图像/动作不同指令的跨划分副本。旧基线不可准入，旧R1停止；完整数据重划分为complete-v3，train2,896/dev850，保留全部75任务和3,746示范，跨划分轨迹重复为0。相同task/instruction组和轨迹别名按连通分量划分，统计量重新计算。旧清单3,536个文件的内容哈希均与完整下载一致；详细路径、划分和版本见 README 与 configs/data_complete_v2.json。

官方协议使用 head、left wrist、right wrist 三相机和 14 维双臂绝对关节目标。当前 LIBERO Baton 配置是双相机、7 维 EEF 动作。GE Act 的基础 transformer、通用 semantic adapter 和 policy_model.yaml 已有三视角或 14 维动作支持，可以复用；Baton 的 strict geometry/provider 仍固定双相机。接入必须明确数据转换、动作/state 归一化、三视角语义布局和权重兼容性，不能填零或直接复用 LIBERO 统计量宣称适配完成。初始读取时没有现成 RoboFollow policy adapter；本轮新增 ge_act/experiments/robofollow_policy.py，实际 TCP rollout 已验证。

RoboFollow policy 需要 `reset()`、`set_instruction(text)`、`predict(observation)`，输出有限的 float32 `[T,14]` 绝对关节目标。训练环境与仿真环境可通过官方 TCP policy server 隔离。

语言遵循按场景与 L0 到 L3 分层记录官方 `mean_intent_score`、`intent_full_rate`；物理执行记录 `mean_exec_score`、`exec_full_rate`；同时记录 task macro `completion_rate`。CR 不等同于 Intent 或 Execution，不用单个 CR 覆盖全部语言结论。正式结果必须满足 run.json 的 complete 和试验数一致；smoke、局部筛选、完整协议分别标记。

官方参考： https://github.com/AutoLab-SAI-SJTU/RoboFollow

## 后续实验候选

以下是待设计与验证的候选，不是已获得收益的方案。

1. 先补齐代表性 checkpoints 的真实 planner 评估与加载证据，再决定是否继续已有混合路线。
2. 固定初始化、训练预算、窗口、噪声和视角，对比 teacher/planner mixing 概率；记录真实选中样本及 planner teacher 特征差异。
3. 比较冻结 planner 与仅训练 Query Tower/Sem MLP 的 joint head；加入 anchor 权重对照，监测无语义回退效果。
4. RoboFollow 完成 schema 检查与 adapter 后，同场景替换指令，检查 Intent 随指令改变；语言打乱或为空的实验标为独立诊断，不能混入正式 benchmark 分数。
5. 用固定开发子集筛选，用未参与优化的保留集确认；最终报告 L0 到 L3，避免反复按同一 benchmark 测试结果调参。

## 证据索引

`historical_metrics.csv` 保存所有汇总的每模式每指标，包括 paired difference、SE、win rate 和源路径。`historical_summaries.json` 保存原始汇总 JSON 的快照与路径。来源是历史文件，不是新执行结果。

优先核对的历史目录：`e3_s2add_main20k_cci_r2_020000`、`e3p_s2add_mainview20k_r2_020000`、`e3_s2control20k_020000`、`e3_s3plan_p100_003000`、`e3p_s2add_main20k_cci_r2_005000`。

后续每条经验记录：假设、有效配置、代码快照、数据划分、资源预算、评估路径、结果、保留或放弃原因、结论置信度。失败实验也保留。

## 本轮追加：工程验证与实验约束

- 三相机 head/left_wrist/right_wrist、14维绝对关节动作与train-only统计已接通；history按官方每次执行50 action对齐。旧Baton双视角头逐视角共享迁移到三视角，这只是明确的研究迁移，不代表原权重已适配该benchmark。
- 真实GPU单步：文本loss2.32、预测planner2.05、teacher1.93、mix1.93、joint2.09。该数值证明可以反向传播，不能拿来判定RoboFollow收益。joint四组头梯度非零且有限，Qwen冻结；optimizer和保存权重核对新增25,177,088个head参数，见configs/planner_validation.json。
- 启动失败揭示现有host Vulkan ICD在/etc；TCP真实请求揭示state需要[B,1,14]；真实planner推理接口是predict(...).tokens。失败日志保留，不能仅依赖mock单测宣告链路成功。
- 控制器执行prepare/train/paired evaluate/review，冻结完整基线阈值，保存每run假设和next suggestion，有限预算按证据停止。它是实验编排器；下一次源码假设仍由研究agent阅读PROGRAM.md并提出，不能宣称脚本会自动创造有效模型改进。
- 超时使用monotonic时钟，恢复核对boot/start_ticks；失败launcher退出后的同session进程也清理。故障测试同时检查NaN/partial不选优、PID复用不误杀、prelaunch失败重跑不覆盖旧目录。
- 固定开发协议为四场景各一个held-out L0任务、2轮、5step；正确/打乱指令共16trials。第一次10step基线因仿真耗时校准中止并完整保留，候选前才冻结5step。完整官方协议仍是554任务、L0–L3、10轮10step的5540trials，尚未执行，不能报告为完整benchmark分数。
- 正确/打乱指令必须同任务、round、seed和官方scorer目标；打乱指令来自train池，policy的随机种子固定。语言差值采用任务均值配对近似区间，四任务不足以证明泛化。
- 当Intent和Execution同时为零，优先检查14D投影初始化、归一化和小样本拟合；零语言差不能区分语言条件无效与动作能力未建立。不能用teacher未来特征作为部署输入或放宽scorer来取得表面提升。
- 本轮初版新增研究测试47通过；修正轨迹别名划分后再执行验收，相关provider/training/pipeline57通过。项目级tests曾1046通过、37失败、1跳过、10错误，其中3项静态检查在原baseline复现失败，其余尚有环境/共享状态及未分诊问题；不宣称整个项目测试通过。验证原始日志保存在研究输出validation目录。

旧结果只作接入诊断，不能称为held-out结果；旧best/thresholds/baseline归档，累计8GPUh账本不重置。新v3基线和候选使用同一清洁数据与新协议重新比较，运行上限由2h收紧为75min以保留重跑预算。
