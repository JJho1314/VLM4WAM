# RoboFollow 优先的 autoresearch 实验设计

状态：用户于 2026-10-03 确认按推荐设计推进；实现计划已保存，尚未实施自动控制器或启动新训练。

## 目标和成功条件

用户要求将 `/data/users/junjie/workspace/VLM4WAM_baton` 构建为能积累经验并迭代实验的项目，现将 RoboFollow 语言遵循设为首要目标。研究要展示当前 Qwen3.5 Baton 语义计划结合 GE Act 世界动作模型的指令依赖能力。提升预测 planner 的动作效果作为辅助任务。

成功必须同时有实际 RoboFollow policy rollout、官方 Intent 和 Execution 分数、任务宏平均 completion rate，以及匹配场景和种子的语言消融证据。teacher future 特征只能作为诊断上界，不能作为部署 policy 输入。动作 MSE、特征 cosine、指令改变引起动作改变都不能单独作为语言遵循成功。

## 当前事实和复用边界

- GE Act transformer 和通用 semantic adapter 已支持多视角，policy_model.yaml 存在 14 维动作配置；不是从零实现双臂网络。
- `BatonGeometry`、provider 和 Baton trainer 条件契约固定双相机；直接把三相机传入会违反形状校验。
- `research_provider.py` 已支持固定 Qwen backbone 加可训练 Query Tower 和 Sem MLP。现有 trainer 已有 planner mixing、joint head 和 anchor 实验开关。
- RoboFollow 位于 `/data/users/junjie/workspace/robofollow/RoboTwin/robofollow`。仿真 env 已有 HoldPolicy smoke。未找到 GE Act adapter，也未取得 Baton 的 RoboFollow 分数。
- `/data/users/junjie/workspace/datasets/RoboFollow` 有场景和 HDF5 文件，但完整性尚未核验；另一个 robofollow-data 路径为空并出现下载重试。
- 项目 HEAD 为 `4a58318`，存在大量 dirty 修改。新实验要捕获完整有效代码快照、环境开关和依赖版本，不对现有工作区做 reset 或自动合并。

## 方案选择

推荐 Agent 读取经验、提出候选，确定性控制器负责运行、评估和记录。第一版复用已有训练与评估入口，控制器不自行生成并执行任意 shell 文本。Agent 可以提出代码改进，但每个候选在独立快照中验证，胜出结果先记录为研究候选。

纯参数网格实现快，但不足以诊断语言路径与表示问题；完全开放的自主代码修改灵活，但早期难以区分接口错误、评估变更和真实收益，因此不作为首版默认模式。

## 数据适配

新增 RoboFollow 专用 HDF5 loader 和 manifest，使用原始数据，不要求先转成体积更大的视频副本。借鉴上游 episodes.py 的 raw 和 paired schema 解析。raw 数据用观测和 state 在 t 时刻配对 t+1 的绝对关节目标；paired 数据不得再次移位。相机顺序固定 head、left wrist、right wrist；动作顺序固定左臂 6、左夹爪 1、右臂 6、右夹爪 1。

从训练数据计算独立 state/action 统计量，保存动作约定、有效维度、夹爪范围和时间采样。用 episode 和 task/instruction 身份确定训练与开发划分，记录重复文件和身份；官方 evaluation 指令和 L1 到 L3 任务不进入训练。jpeg 解码使用上游实际约定，验证 RGB 通道和相机次序。

首个 preflight 检查每场景一份样本、全局文件清单、序列长度和指令来源。统计阶段遍历全部纳入训练的 episode。缺损数据标记并报告；不会把缺失 episode 当作成功训练输入。

## 模型和推理适配

新增 RoboFollow 研究配置：三相机、14 维 joint 动作、独立归一化与状态输入。复用 transformer 多视角和 generic semantic adapter。按可用 checkpoint 的真实 topology 加载主干；若动作输出层不匹配，明确初始化新层并记录其列表，不静默跳过任意 missing keys。

先建立使用相同数据和初始化的 GE Act 文本条件基线；随后接入基于当前观察与指令预测的 Baton semantic plan。三视角研究 provider 按 batch/camera 展平调用共享 Qwen 和 query head，再恢复三视角布局。使用明确的研究元数据版本，避免修改旧双相机生产检查点含义。若测试发现 planner 真正依赖双视角绑定，则升级为独立三视角训练，不假定 reshape 足以迁移能力。

policy adapter 实现 reset、set_instruction、predict，返回 float32 `[T,14]` 绝对 joint targets。reset 清除缓存、历史帧与动作队列，set_instruction 更新文本编码和 planner 条件。首次历史不足时采用已记录的 repeat-current 策略；动作块执行和 history 时间点跟实际 simulator 一致。推理只读取实时观测、state 和指令。

通过 RoboFollow 官方 TCP 接口分离模型与仿真环境。服务启动、端口占用、连接失败和超时均记录；资源清理由本次创建的进程标识限定。

## 评估与候选取舍

固定官方协议版本、任务清单、种子、动作预算、图像变换和 checkpoint。分别输出四场景与 L0 到 L3 的 mean_intent_score、intent_full_rate、mean_exec_score、exec_full_rate、task macro completion_rate，不将它们压缩为不透明的单分数。

开发迭代使用训练域的独立场景种子和指令开发子集。候选排序以开发集 Intent 为主，但 Execution 和 CR 不得相对基线显著退化。首轮基线采样后，根据 episode/task 的波动固定阈值，再开始筛选，不能看完候选结果后追改阈值。保留集只用于阶段确认和最终报告；正式 L1 到 L3 不参与每轮选择。

语言诊断包括：同场景与种子下的不同正确指令；正确指令对比打乱指令或空指令；无预测语义计划的文本基线；必要时 teacher 条件离线诊断。正确/打乱指令用相同目标任务 scorer 评估，两者分开标记。报告 Intent 改善是否同时反映到执行，避免把模型不动或输出改变误当作理解。

短 smoke 使用四场景至少各一个任务，证明环境、协议和输入输出可用，不用于宣布模型强。固定开发集用于筛选；最终完整协议按官方设置执行，注明是否覆盖全部试验。只有 run.json complete 且 completed_trials 与 expected_trials 一致的结果可进入候选比较。

## 自进化实验闭环

1. 读取 EXPERIENCE.md、历史结果和候选状态，形成一个明确假设。
2. 保存候选代码快照、diff、配置、实际 BATON_RESEARCH 环境白名单、数据与权重标识。
3. 执行数据/schema 和模型加载 preflight、微型 train/inference smoke。
4. 在确定的预算内训练并做固定开发评估。
5. 根据既定规则记录 keep、reject、inconclusive 或 infrastructure_failure。
6. 写回假设、结果、失败原因和下一条建议，更新研究 best 指针，不覆盖基线权重。

实验记录采用每 run 一个不可混写目录、JSON 状态文件和追加日志；运行锁防止重复启动。状态区分 prepared、running、evaluating、completed、failed、interrupted。恢复时核对已有输出和进程身份；RoboFollow 不支持 evaluate resume，重跑使用新目录。缺失或 NaN 指标、未完成 rollout、空任务集不作为高分候选。

## 首轮候选顺序

- R0：纯文本 GE Act 适配基线及语言打乱对照，确认可执行性与现有语言依赖。
- R1：预测 Baton 语义条件；固定其它设置，比较真实 planner 与关闭 planner。
- R2：teacher/planner 混合训练，验证部署特征差距。
- R3：固定 Qwen，仅联合训练 Query Tower/Sem MLP；anchor 作为独立对照。
- R4：若 R1 到 R3 显示指令依赖仍弱，再试同场景指令对比训练。损失和负例由训练 task 元数据构建，不使用评估目标标签。

先一次一项改动，不同时更换 backbone、训练预算、语义注入、数据配比并宣称归因。现有 LIBERO 模型表现用于迁移初始化和诊断，不作为 RoboFollow 训练完成的证据。

## 资源和停止条件

已看到该 SSH 容器两张 H800；读数时空闲，但启动前重新检查进程和 GPU 使用。其它 ACP 工作可能共享存储，控制器不会取消它们。首版一次一个候选，用资源白名单和 wall time 限制运行，不默认启动无限循环或提交额外 GPU。

建议首轮资源配置为最多两张 H800、每候选最多两小时、总预算八 GPU 小时。该预算已随推荐设计获用户确认，完整 RoboFollow 训练与评估很可能超出；阶段时间不足则保留可恢复状态，不伪造完成。Agent 能力用当前 Codex 会话执行提案与回顾，控制器本身不需要新增 API key。

## 实施文件边界和验证

计划新增 `ge_act/data/robofollow_hdf5_dataset.py`、`ge_act/configs/ltx_model/robofollow/`、RoboFollow policy adapter、三视角研究 provider，以及 `autoresearch/` 下的控制器配置、实验账本和程序。仅在现有 generic trainer 的数据与条件分发处做必要扩展，不放宽旧 strict Baton checkpoint 验证。

测试覆盖原始/paired 时序映射、三相机与动作顺序、归一化往返、指令更新与 reset、三视角 shape、严格权重加载及控制器预算/锁/失败恢复。随后运行实际单 batch forward/backward、policy TCP smoke 和四场景短 rollout。没有真实 GPU 与仿真验证不能宣称已具备 benchmark 优化能力。

## 证据与限制

历史证据在 autoresearch/EXPERIENCE.md、historical_metrics.csv 和 historical_summaries.json。官方 RoboFollow 来源： https://github.com/AutoLab-SAI-SJTU/RoboFollow 。此设计不保证性能提升；实验失败或无收益也是闭环必须保存的结果。
