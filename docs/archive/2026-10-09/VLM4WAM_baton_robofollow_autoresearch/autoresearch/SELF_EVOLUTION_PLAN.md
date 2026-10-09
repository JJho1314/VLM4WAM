# RoboFollow Self-Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Ola_2208 实现能按实验结果自动修改代码、训练、固定评测、保留/拒绝并继续的无人值守研究循环。

**Architecture:** 独立可信控制器管理候选、冻结评测和预算；隔离 coding agent 只产生方法补丁。沿用原 registry 和 GPU watchdog，每次训练与评测保存可重现快照。

**Tech Stack:** Python 标准库、现有 Pixi/pytest/PyTorch、Linux namespace、Codex CLI。

**Spec:** autoresearch/SELF_EVOLUTION_DESIGN.md（本地镜像 SELF_EVOLUTION_DESIGN.md）。

## Global Constraints

- 原 registry 路径不变；新增授权 24 GPUh，累计上限 32 GPUh，之前 7.520815675542172 GPUh 继续计入。
- 数据 complete-v3、scorer/protocol/统计/runtime 指纹冻结；开发集与独立复核结果分开。
- 每个 agent 提案 20 分钟；每轮训练与评测墙钟上限 75 分钟；最多两张 GPU。
- 保护原脏工作树；不得复制 Mac 凭据、改测试以通过、推送/合并候选或停止他人进程。
- 先测试失败再实现；远程 Python 均通过 Pixi。
- 原有 R0–R3 命令兼容；动态候选使用 E0001... 独立接口。

## Review Focus

- agent 越权、符号链接或路径穿越：拒绝候选且保护评测文件。
- watchdog 上限与控制器授权不同步：拒绝启动。
- 中途掉电或 PID 复用：不重复已完成阶段，不误杀其他进程。
- 部分/NaN/指纹错配指标：不得更新 best。
- 登录失效、GPU 缺席或连续故障：明确暂停状态，不无限消耗资源。

### Task 1: 授权预算与可信快照

**Files:** autoresearch/evolution_policy.py、autoresearch/registry.py、autoresearch/watchdog.py；tests/test_evolution_policy.py。

**Interfaces:** `load_authorized_budget(root: Path) -> float`；`capture_trusted_inputs(root: Path, output: Path) -> dict`。

- [ ] 写测试：无授权时上限 8；授权新增 24 时上限 32；原账本 7.520815675542172 仍保留；负数、NaN、篡改授权拒绝；controller/watchdog 同源读取授权。
- [ ] 通过 Pixi 跑新测试，确认因缺少接口失败。
- [ ] 实现授权校验及冻结输入/文件哈希；原接口默认值兼容；授权部署文件禁止 agent 写。
- [ ] 跑新测试与原预算/锁/timeout 测试，确认通过；仅提交本任务文件。

### Task 2: 隔离 agent 与补丁验证

**Files:** autoresearch/agent_proposer.py、autoresearch/proposal_schema.json、autoresearch/agent_sandbox.py；tests/test_agent_proposer.py。

**Interfaces:** `propose(context: dict, workspace: Path, timeout: int) -> dict`；`validate_proposal(proposal: dict, workspace: Path) -> list[str]`；`apply_proposal(proposal: dict, workspace: Path) -> dict`。

- [ ] 写测试：拒绝 scorer/tests/registry/protocol 改动、绝对路径、../、符号链接、任意 shell、空假设、多余 JSON 字段和大补丁；允许一个白名单方法修改且保留补丁哈希。
- [ ] 跑测试确认接口缺失；实现精确文件白名单与 JSON schema 检查，用 git apply --check 后应用，禁止执行提案命令。
- [ ] 写隔离集成测试：agent 能写候选方法文件，不能读取真实数据/账本/认证文件，不能改 trusted input；namespace 失败应暂停。
- [ ] 实现 user/mount/pid namespace 沙箱；网络与认证只给外层 Codex，模型工具进程只见允许的代码副本；验证这一边界，不用提示词代替隔离。
- [ ] 安装固定版本 CLI 到专用运行目录，记录版本/二进制哈希；验证官方 headless 登录。用户自行登录或提供授权凭据来源，不复制 Mac 登录文件。
- [ ] 跑 fake proposer 和真实一次方法提案测试；保存提案及事件日志，限制 20 分钟和调用次数。若认证未建立，保留明确 auth_required 状态。

### Task 3: 动态训练、固定评测与采纳

**Files:** autoresearch/evolution_runner.py、autoresearch/evolution_review.py；tests/test_evolution_runner.py、tests/test_evolution_review.py。

**Interfaces:** `prepare_candidate(proposal: dict, parent: Path, run: Path) -> dict`；`execute_candidate(run: Path, registry: Registry) -> dict`；`review_candidate(run: Path, registry: Registry) -> dict`。

- [ ] 写测试：动态 E0001 配置有效；变更数据来源被拒绝；缺少完整 checkpoint 或合同不能评测；候选源码进入训练和推理，冻结评测代码不变。
- [ ] 确认失败后实现动态配置与来源绑定，复用 run_process、Pixi train 和现有 run_eval；不复用仅支持 R0–R3 的候选校验。
- [ ] 写测试：只完成 correct、不配对 seed、NaN、协议/运行时不匹配不能采纳；相同 benchmark 分数不替换 best；动作收益与 language_delta 分别记录。
- [ ] 实现复用 collect_pair/compare_candidate 的 review；旧 baseline/best 保留，新的研究 best 必须关联完成的官方 evidence。
- [ ] 在首个候选前冻结独立复核任务/seed 及基线。开发提升后自动触发复核，记录独立结果，不把四个 L0 开发任务称作完整 benchmark。
- [ ] 跑契约和评测测试，保存结果；仅提交本任务文件。

### Task 4: 无人值守控制器、恢复与部署

**Files:** autoresearch/evolution_controller.py、autoresearch/evolution_supervisor.sh、autoresearch/SELF_EVOLUTION_RUNBOOK.md；tests/test_evolution_controller.py。

**Interfaces:** `run_loop(root: Path, agent: AgentProposer, max_rounds: int) -> dict`；CLI `start/status/stop/resume`。生产默认 max_rounds=100，超出后暂停并保留状态，budget 仍为主停止条件。

- [ ] 写端到端 fake agent 测试：前一轮失败或拒绝后自动提出不同代码并进入下一轮；仅 keep 改 parent；stop 不开始下一阶段；连续三次基础设施失败暂停。
- [ ] 写恢复测试：训练完成评测未开始从评测恢复；已完整评测不重复；未清算进程暂停；身份匹配才清理并计费，重复恢复不重复 charge。
- [ ] 确认失败后实现持久阶段状态、exclusive lock、failures/evidence 索引、停止标志与 budget 预留，冻结 agent 费用/调用限制。
- [ ] 实现 nohup supervisor；重启前核验控制器进程身份和 registry；账户限流/认证失效不自动重试到无穷。
- [ ] 跑所有新增测试及 RoboFollow/controller 回归测试；再跑项目 pytest，逐项报告已有失败，不能宣称整项目全绿。
- [ ] 独立审查白名单、沙箱、预算、恢复、训练 checkpoint、评分来源；确认关键问题已处理。
- [ ] 真正执行一轮 agent 改代码→训练→paired evaluation→decision，确认日志与指标；然后验证下一轮已开始或正确暂停，再交给后台。
- [ ] 同步运行手册、PID/状态/日志路径到项目与本地镜像；报告实际能力和未完成项，不把 mock 测试当成真实刷分。

## 当前预检结果

2026-10-05：Ola_2208 的 `unshare --user --map-root-user --mount --pid --fork /bin/true` 成功；具备 namespace 基础能力，但尚未完成完整沙箱验证。未发现 Codex 可执行文件，两个约定位置未发现 auth.json。服务器端真实 agent 调用仍需部署和认证。

## 设计与计划审查结果

已对应设计中的动态提案、补丁限制、训练/评测绑定、独立复核、累计预算、暂停/恢复和后台部署。认证不可自动冒用；沙箱无法隔离评测输入则暂停。计划选择由当前 agent 原生执行，不派生任务 agent；实现前请用户审阅本计划。
