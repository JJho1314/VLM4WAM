# RoboFollow Autoresearch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Ola_2208 的 VLM4WAM_baton 中建立可复现实验闭环，优先提升 RoboFollow Intent，并用语言消融、Execution 和 completion rate 验证实际指令遵循。

**Architecture:** 复用 GE Act 的三视角、14 维 action expert 和 CustomPipeline，新增 RoboFollow 数据、policy 和独立三视角研究 planner 适配。Agent 提出单因素候选；确定性控制器管理固定协议、预算、进程、完整性验证和结果归档。先完成 R0 文本基线，再依次进行 R1/R2/R3，R4 由证据触发。

**Tech Stack:** Python、PyTorch、HDF5、现有 GE Act/Qwen3.5/SigLIP2、RoboFollow TCP；stdlib JSON、hashlib、fcntl、subprocess；远端 Pixi 编排现有训练和仿真环境。

**Spec:** `docs/superpowers/specs/2026-10-03-robofollow-autoresearch-design.md`（用户已于 2026-10-03 回复“按照你的建议来”）。

## Global Constraints

- 项目：`/data/users/junjie/workspace/VLM4WAM_baton`，SSH：`Ola_2208`。保留原工作区全部 dirty 和 untracked 文件；独立实验快照捕获有效代码，禁止 reset/stash 自动清理。
- 相机顺序固定 head、left wrist、right wrist；动作顺序固定左臂 6、左夹爪 1、右臂 6、右夹爪 1。
- raw 数据 t 状态对应 t+1 绝对 joint target；paired 数据不再次移位。统计只用训练 split；官方 L1–L3 不用于训练或反复选择。
- 部署仅用当前观测、state 和指令；teacher future 仅训练/诊断。旧双相机 Baton 元数据与校验保持原含义。
- 每次一个候选、最多两张 H800、每候选最多两小时、首阶段最多八 GPU 小时。训练、服务和评测共享预算；超时记录 interrupted，不能声称完成。
- Intent 为首要排序项；Execution/CR 非退化阈值在基线校准后、查看候选前冻结。未完成、空集合、NaN 或不匹配协议不能进入比较。
- 每 run 独立目录；官方评测不能 resume，重跑建立新 attempt 目录。只清理由本次启动的进程，禁止终止其它 ACP 工作。
- 所有远端 Python 从 `/data/users/junjie/.pixi/bin/pixi` 的任务执行；复用现有环境，不默认重装 CUDA/PyTorch。单元测试和真实 GPU/仿真检查分开记录。

## Review Focus

- 同一 episode 的副本或不同指令版本跨 split：按内容身份和任务/指令组检测泄漏，测试归 Task 1。
- HDF5 损坏、空/短轨迹或缺相机：明确 quarantine 和统计原因；不进入有效样本，测试归 Task 1。
- checkpoint 拥有兼容形状但错误动作/相机语义：必须同时检查元数据，测试归 Task 2/3。
- episode 中途换指令或 simulator 重置：不得复用旧 prompt/planner/action 缓存，测试归 Task 4。
- 服务/评测退出或 SSH 断开留下子进程：身份核验、锁和预算恢复不得误杀或重复计费，测试归 Task 6。

---

### Task 1: 数据清单、统计量与 HDF5 loader

**Files:** Create `ge_act/data/robofollow_schema.py`, `ge_act/data/robofollow_hdf5_dataset.py`, `autoresearch/preflight.py`, `autoresearch/runtime/pixi.toml`; Test `tests/test_robofollow_data.py`.

**Interfaces:**
- `read_episode(path: Path) -> Episode`: images uint8 `[N,3,H,W,3]`、states/actions float32 `[N,14]`、instructions、scene/task/episode identity；raw 对齐一次，paired 保持已有对齐。
- `build_manifest(root: Path, output: Path, seed: int = 42) -> dict`：包含 content hash、schema、长度、身份、训练/开发 split、quarantine 理由和固定采样索引；同身份不得跨 split。
- `compute_statistics(manifest: Path, output: Path) -> dict`：train-only state/action mean/std、有效维度、夹爪范围、绝对 joint 和相机约定；常量维 std 用 1，所有统计 finite。
- `RoboFollowHDF5Dataset(...)`：参数、返回 batch 键/形状与已有 `libero_fastwam_hdf5_dataset.py` 的 generic trainer 路径对齐，数据配置明确 `action_type=absolute`。记录 source FPS 和动作/视频时间采样，不能套用 LIBERO gripper 变换。

- [ ] 写测试：raw 的 states 为 `[0,1]`、actions 为 `[1,2]`；paired 的 actions 原样保留；三相机像素标记顺序；14 维 joint 顺序；normalize/denormalize 往返；副本不跨 split；损坏/短轨迹 quarantine；开发样本极值不改变 train stats。
- [ ] 建立只编排已有 interpreter 的 Pixi test/train/sim tasks，记录版本；运行 `pixi run --manifest-path autoresearch/runtime/pixi.toml test tests/test_robofollow_data.py -q`，确认缺实现失败。
- [ ] 实现上述接口，借鉴上游 `robofollow/episodes.py`，HDF5 每 worker 延迟打开，不共享父进程文件句柄；持久化 manifest/stats 到研究输出目录。
- [ ] 同命令通过；对真实四场景各一份 episode 做 decode/shape preflight，生成全局清单并报告覆盖率、缺失和泄漏。不得把文件存在等同数据完整。
- [ ] 在独立研究分支只提交本任务文件：`feat: add RoboFollow HDF5 data contract`。

### Task 2: 三视角 14 维文本基线与权重迁移

**Files:** Create `ge_act/configs/ltx_model/robofollow/action_model_text.yaml`, `ge_act/experiments/robofollow_loading.py`; Modify `ge_act/runner/ge_trainer.py`（仅必要 generic 数据/加载分发）；Test `tests/test_robofollow_loading.py`.

**Interfaces:**
- `load_robofollow_weights(model: torch.nn.Module, checkpoint: Path, allowed_new_keys: set[str]) -> dict`：报告 loaded/new/rejected keys 和 topology；未明确允许的 missing/unexpected/shape mismatch 抛错。
- 配置引用 Task 1 manifest/stats、三视角、14 维 absolute joints、state 输入和现有 action_full 流程；使用实际可用 checkpoint topology，不照抄未验证的 28 层配置。

- [ ] 写测试：action_dim=14、views=3、stats 语义匹配；白名单新动作层允许初始化，主干 missing、错 action metadata、错相机顺序均失败；文本 baseline 无 future teacher 输入。
- [ ] 运行 Pixi `test tests/test_robofollow_loading.py -q`，确认失败。
- [ ] 审计实际 safetensors/config topology 后实现加载报告和配置；选择兼容主干，精确记录新初始化层及种子，复用现有 T5/VAE 路径。
- [ ] 单测通过；通过 Pixi 训练入口调用现有 `ge_act/main.py --config_file ... --max_train_steps 1`，验证真实 GPU forward/backward、finite loss、checkpoint 和显存。禁止用随机 stub 代替此验收。
- [ ] 只提交本任务文件：`feat: add RoboFollow text action baseline`。

### Task 3: 独立三视角研究 planner 与训练条件

**Files:** Create `qwen35_baton/robofollow_provider.py`, `ge_act/configs/ltx_model/robofollow/action_model_planner.yaml`; Modify `ge_act/runner/ge_trainer.py`（generic semantic 路径）；Test `tests/test_robofollow_planner.py`.

**Interfaces:**
- `RoboFollowResearchPlanner.predict_tokens(images: Tensor, instructions: list[str]) -> Tensor`：输入 `[B,3,H,W,3]`，输出 `[B,3,4,256,1024]`。
- 三视角 metadata 独立 kind `qwen35_baton_robofollow_research`、version=1；校验 source checkpoint、camera order、keyframes `(0,3,5,8)`、feature_dim=1024。显式迁移旧权重，不能把旧检查点声明为已有三视角能力。
- 共享 Qwen 和 research query head；按 view 展平，必要的旧双视角输入采用该 view 的重复成对输入，取对应输出并记录迁移策略。先检验模型真实耦合：不满足独立共享头契约时改为独立三视角 head，禁止仅 reshape 伪造输出。
- mixing/joint/anchor 复用已有 optimizer 与 checkpoint 规则；teacher 使用三视角 SigLIP2，推理拒绝 teacher source。

- [ ] 写测试：输出 shape/view 标记不串位、指令逐 view 一致、梯度到 query/MLP 且冻结 Qwen、错误 metadata 拒绝、关闭 planner 的文本路径一致；旧双视角契约仍拒绝三视角。
- [ ] Pixi `test tests/test_robofollow_planner.py -q` 确认失败。
- [ ] 实现 provider 与明确配置分发，保留旧 strict Baton 路径；添加混合概率、joint 和 anchor 的独立候选字段。
- [ ] 单测及相关现有 Baton/generic semantic 回归通过；真实三视角 batch 的 teacher/predicted/joint 各运行一个 forward/backward，记录 finite 梯度和正确参数组。
- [ ] 只提交本任务文件：`feat: add three-view RoboFollow research planner`。

### Task 4: 官方 policy adapter 与 TCP smoke

**Files:** Create `ge_act/experiments/robofollow_policy.py`; Test `tests/test_robofollow_policy.py`.

**Interfaces:**
- `make_policy(config_path: str, checkpoint_path: str, stats_path: str, planner_mode: str = 'disabled') -> RoboFollowPolicy`。
- Policy 的 `reset() -> None`, `set_instruction(text: str) -> None`, `predict(observation: dict) -> np.ndarray`；输出非空 finite float32 `[T,14]` 绝对 joint targets。
- 用现有 `CustomPipeline.infer`，`n_view=3, action_dim=14, return_action=True`；history 不足 repeat-current，采样时序与 Task 1 同契约；inverse stats 不做 LIBERO gripper flip。

- [ ] 写测试：三相机 observation 与 joint_action/vector 的映射；正常输出 dtype/shape；NaN/维数错误拒绝；reset 清空 history/queues；换指令清空 prompt、planner 和 queued actions；缺相机拒绝。
- [ ] Pixi `test tests/test_robofollow_policy.py -q` 确认失败。
- [ ] 实现 adapter，仅用公开实时输入，明确 history/action chunk 时间点；通过官方 serve factory 加载。
- [ ] 单测通过；真实模型 TCP request/response 和四场景各一任务短 rollout，通过官方 ValidatedPolicy 校验；保留 run.json 和 logs，smoke 分数不当模型性能结论。
- [ ] 只提交本任务文件：`feat: serve GE Act policy for RoboFollow`。

### Task 5: 固定协议、语言配对消融与指标准入

**Files:** Create `autoresearch/evaluation.py`, `autoresearch/configs/robofollow_eval.json`; Test `tests/test_robofollow_evaluation.py`.

**Interfaces:**
- `read_complete_metrics(output_dir: Path, protocol_hash: str) -> dict`：只接受官方 run.json complete、completed_trials==expected_trials>0、finite 指标及匹配 protocol。
- `compare_candidate(baseline: dict, candidate: dict, thresholds: dict) -> dict`：返回 keep/reject/inconclusive；Intent 主排序并检查 Execution/CR 非退化；阈值缺失拒绝选择。
- `calibrate_thresholds(baseline_trials: list[dict], output: Path) -> dict`：只依据 baseline 的 task/episode 波动，输出固定规则、版本/hash；候选评测前冻结。

- [ ] 写测试：partial/empty/NaN/protocol mismatch 拒绝；Intent 增且 Execution/CR 越界拒绝；未冻结阈值不选优；正确/打乱输入共用 task scorer、seed 和预算；四场景/L0–L3 输出保留且 task macro CR 不替换 Intent。
- [ ] Pixi `test tests/test_robofollow_evaluation.py -q` 确认失败。
- [ ] 实现解析与比较；开发协议使用训练域独立 split，官方 retained 指令不用来选候选；语言 shuffle/empty 仅改变 policy 文本，不能改变 scorer 目标。
- [ ] 单测通过；用真实官方 smoke 输出验证解析，同时确认不完整输出被拒；冻结开发 task 清单、baseline 配对种子及预定校准算法。最终协议单独使用官方 all scenes/L0–L3/10 rounds/base seed42/max steps10/actions per step50。
- [ ] 只提交本任务文件：`feat: validate RoboFollow metrics and language ablations`。

### Task 6: 有预算的实验控制器与经验回写

**Files:** Create `autoresearch/controller.py`, `autoresearch/registry.py`, `autoresearch/configs/robofollow_candidates.json`, `autoresearch/README.md`; Test `tests/test_autoresearch_controller.py`.

**Interfaces:**
- CLI `prepare --candidate R0`, `run --run-id ID`, `status`, `recover --run-id ID`, `review --run-id ID`；候选 JSON 只能引用已注册 train/serve/evaluate 参数模板，不执行任意 shell 文本。
- `prepare_run(candidate: dict, root: Path) -> Path`：生成独立 code snapshot（HEAD、dirty patch 和 untracked 内容）、config、来源hash、环境白名单和权重/data标识；显式列出排除的大型数据/产物，不能漏掉 untracked research_provider。
- Registry 状态 prepared/running/evaluating/completed/failed/interrupted；原子 JSON、fcntl 运行锁、追加日志；记录 wall time/GPU hours/进程启动身份。
- `review_run(run_dir: Path) -> dict` 复用 Task 5；保存假设、证据、失败原因、next suggestion；best 指针仅更新到通过准入的候选，不覆盖 baseline checkpoint。

- [ ] 写测试：锁冲突；两小时 timeout；累计八 GPUh 拒绝启动；崩溃恢复的状态与计费；PID 复用不误杀；未授权 argv 拒绝；评测重跑新目录；dirty/untracked 快照可复现；NaN/partial 不更新 best。
- [ ] Pixi `test tests/test_autoresearch_controller.py -q` 确认失败。
- [ ] 实现 stdlib 控制器，进程组与创建身份限定清理，训练/服务/评测均计预算；命令日志保留可复现 argv，不记录密钥；经验采用每 run evidence 文件及可重建 summary，避免并发覆盖。
- [ ] 单测通过；故障注入验证 train failure、服务超时和 partial eval 各被正确记录，再实际串接一个 bounded smoke run。README 给出启动、恢复、预算调整及结果解释命令。
- [ ] 只提交本任务文件：`feat: add bounded RoboFollow autoresearch loop`。

### Task 7: 首轮真实实验与证据报告

**Files:** Create per-run `autoresearch/runs/<id>/` records in研究输出目录；Update `autoresearch/EXPERIENCE.md`、`autoresearch/README.md`；Test 不新增镜像实现的测试。

**Interfaces:** R0/R1/R2/R3 使用 Task 6；每候选只改文本基线、predicted semantic、mixing、joint/anchor 中一个明确因素；R4 仅有 evidence 后开启。

- [ ] 用 preflight 和所有相关单测验收运行环境；重新检查两张 H800 占用，不能只依据规划时的空闲读数。
- [ ] 在首阶段八 GPUh 内先训练 R0，完成固定开发正确/打乱指令配对评测；记录未收敛风险，依据基线冻结非退化阈值。
- [ ] 预算允许时执行 R1，再按证据执行 R2、R3；保留所有失败和无收益 run。剩余预算不足则停止并列出精确续跑命令，不把 smoke 当性能改善。
- [ ] 输出各场景与等级 Intent/Execution/CR、语言消融及 uncertainty；best 需通过完整性与预算准入。完整官方 5540 trials 不一定能在八 GPUh 内完成，报告实际覆盖并明确剩余工作。
- [ ] 更新经验与复现命令，运行相关回归和整体验收；提交本次文档/配置，保持大权重与 raw run outputs 在研究目录，给用户实际分数、状态和已用 GPUh。

## Self-review

数据、三视角模型、policy、固定评测、控制器、实验及报告分别由 Tasks 1–7 覆盖；五项 Review Focus 已分配到具体测试。Task 3 需要先审计旧 planner 的相机耦合，不能假定旧模型迁移后有效；Task 5 阈值以 baseline 实测冻结；Task 7 将预算不足与真实完成区分。各接口返回形状、状态枚举与元数据在生产者和消费者之间一致。实现期间若 topology 或数据覆盖迫使改变设计，应记录证据并修改相应任务，不能放宽评分协议。
