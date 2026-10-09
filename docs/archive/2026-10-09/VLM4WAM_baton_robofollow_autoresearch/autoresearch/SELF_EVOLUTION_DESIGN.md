# RoboFollow 无人值守自进化循环设计（待审阅）

日期：2026-10-05。状态：设计草案，尚未实现或启动新循环。

## 目标和事实

用户要求在 Ola_2208 的 VLM4WAM_baton 项目持续自动修改代码、训练、评测，优化 RoboFollow 指令遵循能力。成功标准是固定开发协议下的官方 Intent 改善、Execution/CR 不退化，并有 correct/shuffled 指令差异证据；训练 loss 仅作诊断。

现有研究工作树为 `/data/users/junjie/workspace/VLM4WAM_baton_robofollow_autoresearch`，当前研究提交 54dc350。它带有大量原有未提交修改，不允许重置、清理或覆盖。原 registry 位于 `/data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_autoresearch/runs`。累计消耗 7.520815675542172 / 8 GPUh，剩余 0.479184324457828 GPUh。

已存在快照、训练、官方 paired evaluation、结果完整性验证、预算账本与独立 watchdog。现有控制器只接受 R0–R3 预设候选，不能生成新代码。Ola_2208 的 PATH 未发现 codex、claude、node；Mac 有 codex-cli 0.160.0。服务器上的 agent 运行时和登录尚需配置和验证，不能声称后台代码 agent 已可用。

## 方案选择

推荐服务器上的 headless coding agent + 独立可信实验控制器。服务器不依赖 Mac 或聊天会话存活；代价是需建立服务器端 agent 运行时、认证和独立费用上限。

备选是 Mac 上 agent 提案、服务器实验执行，改动较少，但要求 Mac 持续在线，不作为默认无人值守方案。另一备选是预设参数搜索，容易实现，但不能满足自动修改代码的目标。

## 控制边界

控制器保存在独立、agent 不可写的运行目录，以冻结的源码哈希启动。每轮从当前被保留的方法快照创建候选工作区，agent 仅收到训练代码及历史证据的副本，输出一个方法假设、有限范围补丁、配置变化和预期验证。agent 不获取 GPU 作业管理权限、评测程序写权限、原始数据或凭据文件。文件系统隔离必须是实际执行边界，不能仅靠提示词约定；实现前验证服务器可用的 sandbox/container 机制，隔离不可用则暂停提案，不用不受限 shell 替代。

修改范围限于 RoboFollow 相关训练器、模型、policy 推理和候选训练配置；禁止修改 scorer、protocol、数据划分、归一化统计、评测数据、预算、控制器、watchdog、tests 和 Pixi 运行时。各白名单路径需以允许的准确文件或子目录登记；通用 ge_trainer 和模型文件的变化必须额外运行 RoboFollow 接口回归检查。禁止符号链接、路径穿越、二进制补丁、依赖安装、删除原始文件、任意 agent shell 直接提交作业。

文件白名单不能代替评测可信边界。候选训练程序只可读 train 清单中的 episode；评测进程只提供观测、状态与指令输入，scorer 和指标产物由控制器拥有。候选代码不得写 results、读取评测成功条件或使用 future/teacher 特征推理。评测容器/目录权限边界验证失败则暂停，不把分数作为可采纳证据。

冻结原始脏树的完整来源快照，候选与原仓库隔离。控制器保存 base、patch、有效配置、checkpoint 和各输入来源哈希。保留只改变研究循环的 best 指针；不自动合并、push 或覆盖原工作树。

## 每轮状态机

1. `preflight`：获取 registry 独占锁，核验运行时、冻结输入、预算、GPU 可用性和历史进程身份；确认没有未清算的本循环 GPU 任务。
2. `propose`：读取经验和之前各轮结构化证据，要求 agent 提出一个可证伪因素的修改。输出 JSON 包括 hypothesis、changed_paths、patch、training_config_delta、expected_effect。训练配置允许改变学习率、FP32 master 参数、训练长度和模型方法，禁止改变训练/开发划分或输入来源。变更因素和训练计算量须逐轮披露，不把额外训练量自动归因于 planner。
3. `validate`：控制器审查路径、补丁和哈希，在候选副本中应用，运行被冻结的测试与配置/forward 契约检查。失败日志反馈给下一次提案；不得由 agent 改测试来通过。
4. `diagnose`：优先解决已观察到的动作拟合/未见噪声差距。多噪声与多个 train episode 的拟合、采样动作检查只是低成本筛选，不能更新 benchmark best。
5. `train`：通过 Pixi 固定 launcher 和 run_process 运行候选，记录实际参数、来源、完整 checkpoint 与语义契约。不能把此前 625MB 动作部分权重直接冒充全模型 checkpoint。
6. `evaluate`：在冻结 official evaluator 上运行现有 complete-v3 correct/shuffle 配对协议，固定 task、round、seed、max_steps、assets 和 runtime 指纹。两种指令模式均完成才进入 review。当前短开发协议只覆盖四个 L0 task、每 task 两轮，不能宣称完整 RoboFollow 分数。
7. `review`：复用完整性检查和 frozen thresholds。候选 Intent 对 baseline 的提升达到门槛，Execution/CR 无退化，并优于当前 best，才更新 benchmark best。另记录 language_delta 及任务级置信区间；correct 提升但 shuffle 同样提升只记动作能力改善，不记指令遵循改善。
8. `confirm`：首次满足开发提升门槛后，使用预先冻结且未提供给提案 agent 的独立 task/seed 复核。其协议和基线在候选运行前登记；不能按候选结果挑选任务。保留 validation 与 dev 结果的区别。完整官方验收单独记录覆盖与费用。
9. `record`：每轮写 evidence.json、decision.json、经验摘要与全量历史索引，然后在被保留的研究状态上继续；未提升候选归档，下一轮从 best 重新开始。

## 预算、恢复和停止

继续使用原 registry 账本；新增预算必须有显式授权并记录 authorization.json，新的累计上限等于 8 + 授权新增 GPUh。没有授权则继续保留 8 GPUh，剩余不足完整候选预留时暂停。watchdog、控制器和账本必须读取同一个冻结授权上限，禁止使用新 registry 绕过已有费用。GPU 预算不含 coding agent 的额度/费用，agent 调用另设轮数、调用超时和可核验额度上限；超限或账户限流时暂停。

默认每次 agent 提案上限 20 分钟、每轮训练与评测合计墙钟上限 75 分钟、最多两张 GPU。启动前保守预留完整训练+评测费用，再按真实运行时结算；没有 GPU 只等待，最多一小时后暂停。阶段完成更新状态，GPU 训练进度使用原 progress.json；外部监视器不得通过陈旧 PID 杀进程，需 boot_id/start_ticks 匹配。

后台 supervisor 在异常退出后恢复控制器，恢复时先检查进程身份和已有阶段产物，避免重复训练或重跑已完整评测。无法确认存活/清算的 GPU 阶段暂停并报告，不按零成本跳过。三轮连续基础设施故障暂停；无收益结果由 agent 改假设继续，但不能超过预算。用户 stop 标志在阶段边界停止，硬超时由 watchdog 清理本循环进程。不会停止其他用户进程。

## 首轮研究方向

已有证据：lr=3e-4 的 FP32 动作专家在单个 train episode 上 train-noise MSE=0.0033659，held-noise MSE=0.8679214；lr=3e-5 对应 0.8124985/1.0573063。首轮应检验增加噪声覆盖、多个训练窗口和 FP32 训练状态能否降低训练域噪声泛化误差，再进行真实采样动作和 paired rollout。不得继续把固定六组噪声上的拟合当作 instruction following 提升。

## 实现接口和验收

新增 evolution_controller、agent_proposer 和 proposal_schema；复用 registry/run_process/watchdog、evaluation.compare_candidate、run_eval.collect_pair。保留原 R0–R3 命令兼容；新循环接受动态 E0001... 候选，但训练/评测命令由可信代码构造，不执行提案中的任意 shell。动态候选的配置和完整 checkpoint 绑定单独实现，不能强行冒充原注册 R1。

验收测试覆盖路径穿越/越权补丁拒绝、预算累计与 watchdog 上限一致、mock agent 自动产生下一候选、真实方法变更进入隔离源码、阶段崩溃恢复不重复计费、缺失/NaN/指纹错配指标不采纳、candidate best 与语言证据分开、停止标志及自有进程清理。随后真实运行至少一轮 agent 提案→改代码→小训练→固定 paired evaluation→decision，若无预算则只交付经过测试的代码，明确尚未做 GPU 端验收。

当前需要审阅：上述架构设计，以及新增 GPU 预算。服务器端 agent 认证仅使用部署机器上授权的账户，不能未经确认复制 Mac 的登录凭据；部署时验证现有认证或给出用户登录步骤。
