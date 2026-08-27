# VLM4WAM 代码整合报告 (2026-08-27)

整合前，VLM4WAM 的代码散落在 **2 台机器、11 个目录、2 个互相分叉的 git 仓库**中，
其中 13 个分支从未推送到任何 remote。本次整合把所有代码收敛到单一 GitHub 仓库
`JJho1314/VLM4WAM`，磁盘上的副本目录全部变为可删。

---

## 1. 整合前的实际状况

### 两个分叉的主仓库（都指向同一个 GitHub origin）

| | 路径 | 工作区 | 分支线 | 最后提交 |
|---|---|---|---|---|
| **A** | `LFT:/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM` | 213G | ge_act / joint-VLM / LIBERO 评测 | 2026-07-29 |
| **B** | `LFT:/data/LFT-W02_data/junjie/workspace/VLM4WAM` | 8.4G | qwen35 planner / WorldArena | 2026-08-05 |

两者在 `main` 上同源（`1503cd5`，2026-06-07），**从 2026-07-17 起分叉**，此后各自
独立演进，互不包含。`qwen35_baton/` 和 `qwen35_planx/` 在 A 的任何分支中都不存在。

`semantic-guidance` 是**三方分叉**：`origin`(a4ddf2c) / A(8df12bf) / B(3630392)
互不为祖先，共同基点是 `bb66aab`(2026-07-16)。三份都已独立保存，未做任何合并或覆盖。

### HPC3 的 8 个目录

| 目录 | 大小 | 判定 | 依据 |
|---|---|---|---|
| `VLM4WAM` | 351G | 部分保留 | `tools/`+`scripts/` 中 39 个代码文件在两个 LFT 仓库的任何分支都不存在 |
| `VLM4WAM_joint_geact_02b89af` | 354M | **独有** | 6 个未推送 commit + 63 个未提交改动 |
| `VLM4WAM_tgt_sdd_20260724` | 189M | **独有** | 8 个未推送 commit（LIBERO target-text 预处理链） |
| `VLM4WAM_qwen35_worldarena` | 90M | 部分保留 | git worktree 已损坏；2072 个代码文件 100% 覆盖于 B@`c0812aa`，只有 35 组 runtime 记录独有 |
| `VLM4WAM_qwen35_baton_strict` | 8.7M | 部分保留 | 181 个文件中仅 3 个独有 |
| `VLM4WAM_geact_semantic_d0ec565` | 132M | 冗余 | clean @ `a200628`，A 中完整存在 |
| `VLM4WAM_hdf5_benchmark_95b1992` | 165M | 冗余 | clean @ `95b1992`，A 中完整存在 |
| `VLM4WAM_hdf5_pilot_b607f0d` | 196K | 冗余 | `b607f0d` 在 A 中；仅 1 个 sbatch 独有 |

判定方法：对每个目录逐文件计算 `git hash-object`，与两个 LFT 仓库全部分支的
blob 集合做精确比对，而非按文件名或时间戳推断。

---

## 2. 代码功能地图

两条线在 **2026-07-17** 分叉，用的是**不同的世界模型**，不是同一套代码的两个版本。

### A 线 — Joint VLM + GE-Act 动作模型（世界模型 = LTX-Video）

把语义 planner 和动作模型**联合训练**，输出直接跑 LIBERO rollout。

| 模块 | 作用 |
|---|---|
| `ge_act/models/ltx_models/joint_vlm_geact.py` | Qwen planner + GE-Act LTX 合成单一模块，联合训练 |
| `ge_act/models/ltx_models/vlm_semantic_planner.py` | 冻结双视角 Qwen planner，作为 GE-Act 的语义条件 |
| `ge_act/models/ltx_models/semantic_conditioning.py` | SigLIP2 语义条件注入 LTX 视频模型 |
| `ge_act/experiments/eval_libero_joint.py` | joint 双相机导出模型的 LIBERO rollout 评测 |
| `ge_act/experiments/joint_libero_eval_contract.py` | 评测的 fail-closed 契约校验 |

分支：`ge-act-dual-camera-planner`(主干) · `joint-vlm-geact-libero-eval` ·
`libero-episode-feature-export` · `frame80-action-attention` ·
`hpc3/joint-geact-attention` · `hpc3/tgt-sdd-libero-target-text`

### B 线 — Qwen3.5 语义 planner（世界模型 = Cosmos Predict 2.5）

planner 产出 `semantic_plan` → Cosmos DiT cross-attention → 视频预测。planner 迭代了三代：

| 代 | 包 | 方案 | 规模 |
|---|---|---|---|
| 1 | `qwen3_vl_semantic_planner/` | Qwen3-VL 三条子线：CoVT·SigLIP·2B(baseline) / tasktoken 富KV头 / lingbot-DINO·4B | — |
| 2 | `qwen35_planx/` | Plan-X **离散** TA-Tok grounded planner，带 video-hindsight 缓存 | 22 模块 |
| 3 | `qwen35_baton/` | Baton **连续** SigLIP2 网格回归（Equation-8 MSE），WorldArena Stage-1 是最终形态 | 25 模块 |

`qwen35_baton` 的 WorldArena 契约：Qwen3.5-2B 全量可训，冻结 teacher =
SigLIP2-large-patch16-256 penultimate；target `[1,4,256,1024]`；5000 步 / global batch 128 / 8 GPU；
每 500 步跑 44 个验证 episode 的三路对比（正确指令 / 打乱指令 / 当前帧持久化基线），
带明确的 checkpoint 准入门槛。

分支：`worldArena`(最新) · `semantic-guidance-ws-20260805` ·
`qwen35-video-hindsight-grounding` · `qwen35-planx-implementation` · `lingbot-zero2-q64-k1`

### 早期共同线（2026-06，两条线的共同祖先）

InstructSAM target-feature guidance → what-where softlogit 定位方案。
分支：`main` · `prev-dense-spatial-target-20260611` · `improvement-20260609` ·
`what-where-softlogit` · `planner/baton-continuous` · `planner/planx-discrete-ce`

---

## 3. 本次整合做了什么

1. 在 A 中 `git remote add wsB <B路径>`，把 B 的 11 个分支全部 fetch 进来（新增 ~1MB，
   两仓库共享 cosmos vendored 树，delta 压缩）。
2. HPC3 `joint_geact` 的 63 个未提交改动提交为 `9a57f58`，与 `tgt_sdd` 一起打 bundle
   传回 A，落为 `hpc3/joint-geact-attention` 和 `hpc3/tgt-sdd-libero-target-text`。
3. 所有"不在任何 git 分支中"的散落文件收进孤儿分支 `archive/loose-files-20260827`
   （287 个文件，7.6MB），provenance 见该分支 README。
4. 20 个 ref 推送到 GitHub。**全程未使用 `--force`**；`semantic-guidance` 的三方分叉
   用独立分支名保存，origin 上原有分支未被覆盖。

### GitHub 上的最终分支

推送新增：`archive/loose-files-20260827` · `frame80-action-attention` ·
`hpc3/joint-geact-attention` · `hpc3/tgt-sdd-libero-target-text` · `improvement-20260609` ·
`joint-vlm-geact-libero-eval` · `libero-episode-feature-export` · `planner/baton-continuous` ·
`planner/planx-discrete-ce` · `qwen35-planx-implementation` · `semantic-guidance-a6000-20260717` ·
`semantic-guidance-ws-20260805` · `backup-dual-camera-probe-docs-20260715` ·
`backup-morgbd-minidpt-with-docs-20260715` · `backup/lingbot-zero2-q64-k1-before-code-only-push` ·
`backup/semantic-guidance-local-20260716`

快进更新：`main`(+1) · `ge-act-dual-camera-planner`(+8) · `worldArena`(+35) · `lingbot-zero2-q64-k1`(+3)

仓库内无模型权重，最大文件是 29MB 的 Baton 论文 PDF，整个 pack 167MB。

---

## 4. 磁盘占用与清理

**代码本身两个仓库加起来不到 20MB**，其余全是权重、输出和数据集。

### HPC3 `/data/user/jhe724/workspace`

| 对象 | 大小 | 说明 |
|---|---|---|
| `VLM4WAM/outputs/cosmos_semantic_plan` | 190G | B 线 Cosmos WM stage-2 训练输出 |
| `VLM4WAM/data/qwen35_train_mw` | 47G | qwen35 训练数据（predecoded） |
| `VLM4WAM/outputs/qwen3vl_semantic_planner` | 37G | 第 1 代 Qwen3-VL planner 输出 |
| `VLM4WAM/outputs/qwen3vl4b_lingbot_dino_*` ×3 | 25.6G | lingbot-DINO 4B 线 |
| `VLM4WAM/outputs/smoke_lat128` | 8.6G | **smoke 测试** |
| `VLM4WAM/outputs/smoke_depth_hpc3` | 8.4G | **smoke 测试** |
| `VLM4WAM/third_party` | 6.6G | vendored 依赖，可从上游重建 |
| `VLM4WAM/outputs/qwen35_ftsmoke` | 5.5G | **smoke 测试** |
| `VLM4WAM/outputs/qwen35_ft_mw4` | 5.5G | qwen35 微调 |
| `VLM4WAM/mg_eval` | 5.0G | 评测输出 |
| `VLM4WAM/models` | 4.9G | InstructSAM stage2 merged |
| `VLM4WAM/data/qwen35_train` | 3.6G | 旧版训练数据（已被 `_mw` 取代） |
| `VLM4WAM/attention_vis` | 2.3G | 注意力可视化，其中 3 个 `*_feature_tokens` 目录各 757M |
| 7 个 VLM4WAM_* 副本目录 | 940M | **代码已全部入库，可整体删除** |

smoke 测试输出合计约 **23G**，是最没有争议的清理目标。

### LFT `VLA_WM/VLM4WAM` (213G)

`cosmos-predict2.5` 83G · `outputs` 60G · `checkpoints` 57G · `third_party` 9.8G ·
`semantic_localization` 4.9G（其中代码部分已归档，其余是 .pt/.npz/.mp4）。

### LFT `workspace/VLM4WAM` (8.4G)

`artifacts` 3.4G（未跟踪）· `.git` 4.7G（含大量不可达对象，`git gc --prune=now` 后
应降到 ~170MB）。

---

## 5. 清理命令（已验证内容安全，执行前请自行确认）

```bash
# HPC3：7 个副本目录，代码 100% 已入 GitHub
cd /data/user/jhe724/workspace
rm -rf VLM4WAM_geact_semantic_d0ec565 VLM4WAM_hdf5_benchmark_95b1992 \
       VLM4WAM_hdf5_pilot_b607f0d VLM4WAM_qwen35_worldarena \
       VLM4WAM_qwen35_baton_strict VLM4WAM_joint_geact_02b89af \
       VLM4WAM_tgt_sdd_20260724                                    # ~940M

# HPC3：smoke 测试输出
rm -rf VLM4WAM/outputs/smoke_lat128 VLM4WAM/outputs/smoke_depth_hpc3 \
       VLM4WAM/outputs/qwen35_ftsmoke VLM4WAM/outputs/qwen35_smoke  # ~23G

# LFT：回收 RepoB 的不可达 git 对象
git -C /data/LFT-W02_data/junjie/workspace/VLM4WAM gc --prune=now --aggressive  # ~4.5G

# LFT：两个 snapshot 目录（README 与 docs 已归档到 archive/loose-files-20260827）
#   VLM4WAM_train_snapshot_20260611 (198M，其中 146M 是 eval 结果，删前确认是否还需要)
#   VLM4WAM_what_where_softlogit_snapshot (31M，代码等同 what-where-softlogit 分支)
```

需要各自判断、本报告不做建议的：`outputs/cosmos_semantic_plan` (190G)、
`checkpoints` (57G)、`data/qwen35_train_mw` (47G) —— 取决于哪些实验还要复现。
`data/qwen35_train` (3.6G) 已被 `_mw` 版本取代。

---

## 6. 追加：退役 planner 线的磁盘产物清单 (2026-08-27)

代码侧已在分支 `baton-only-20260827` 删除 `qwen3_vl_semantic_planner/` 和 `qwen35_planx/`
（113 个文件，-38,238 行）。以下是 HPC3 上对应的磁盘产物。**本节列出清单；执行记录见第 7 节。**

### 明确属于已退役的 qwen3vl 线 —— 约 79.6G

| 目录 | 大小 | 归属依据 |
|---|---|---|
| `outputs/qwen3vl_semantic_planner` | 37G | 子目录名全为 `qwen3vl2b_baton_siglip2_*`（此处 `baton` 是 head 类型，非 `qwen35_baton` 包） |
| `outputs/qwen3vl4b_lingbot_dino_depth_uniform_k5_fullft` | 8.4G | lingbot-DINO·4B 子线 |
| `outputs/qwen3vl4b_lingbot_dino_depth_uniform_k5_fullft_gbs64` | 8.6G | 同上 |
| `outputs/qwen3vl4b_lingbot_dino_depth_uniform_k5_fullft_gbs64_lat128s128o` | 8.6G | 同上 |
| `outputs/smoke_lat128` | 8.6G | `planner_meta.json` 含 `<\|sem_plan\|>` token，为 qwen3vl planner smoke |
| `outputs/smoke_depth_hpc3` | 8.4G | 同上 |
| `outputs/eval_lingbot_dino_gbs64` | 21M | lingbot-DINO 评测 |

### 明确属于已退役的 planx 线 —— 约 834M

| 目录 | 大小 | 归属依据 |
|---|---|---|
| `outputs/qwen35_discrete_plan` | 417M | "discrete plan" 即 Plan-X 的离散 TA-Tok 形式 |
| `outputs/qwen35_discrete_plan_mw2` | 417M | 同上 |

### 格式上不属于 Baton，但归属无法确证 —— 约 62G

`qwen35_baton/configs/worldarena_stage1.json` 声明 `dataset_type: worldarena_hdf5`，
数据经 `hdf5_manifest_path` 读取。下列产物是 `.npy`（`*_kf.npy` / `*_prompt.npy` / `*_rgb.npy`）
或 2026-07-24 的早期 qwen35 实验（早于 planx 07-25 与 Baton 07-27 的 SDD），
**不是 Baton 数据格式**，但也没有足够元数据把它们确证归到某一条已退役线上。删除前请自行确认：

`data/qwen35_train_mw` 47G · `data/qwen35_train` 3.6G · `outputs/qwen35_ft_mw4` 5.5G ·
`outputs/qwen35_ftsmoke` 5.5G · `outputs/qwen35_smoke` 417M · `outputs/qwen35_ftmini` 4K

### 不属于本次退役范围

`outputs/cosmos_semantic_plan` (190G) 是 **Cosmos 世界模型自身**的训练输出，
其中 `2b_semplan_gt_online_*` 系列直接用 GT SigLIP2 plan 训练，与用哪个 planner 无关。
虽然体积最大，但不应按"退役 qwen3vl planner"的理由删除。

### Baton 线需要保留的

`qwen35_baton/` 与 `ge_act/` 双向依赖，两者都保留：`ge_act/runner/ge_trainer.py:81-82`
顶层 `from qwen35_baton.hashing/teacher import ...`，而 `qwen35_baton/cli` 反向 import
`ge_act.data.libero_fastwam_hdf5_*`。GE-Act LTX 就是 Baton planner 的 Stage-2 消费方
（`ge_act/models/ltx_models/baton_semantic_planner.py`）。

### 代码删除的验证方式

无法运行测试（LFT 登录节点无 torch，`python3` 为 3.8）。改用 AST 静态验证：解析全部 118 个
存活 `.py` 文件，检查模块级 import 节点 —— **对已删模块的顶层 import 为 0**。
残留的 17 处引用全部是 `vlm_planner` / `qwen35_grounded` 两条退役分派路线里的函数内延迟 import，
选中这两条路线时才会 `ImportError`。`ge_trainer.py` 的分派代码**刻意未重构**，
因为无法在本机验证一次约 400 行的 trainer 改写。

> 附带发现：`ge_act/scripts/benchmark_libero_fastwam_hdf5.py:1445` 用了 PEP 617 括号化 `with`
> （需 Python 3.10+），在 3.8 下无法解析。这是预先存在的，与本次删除无关。

---

## 7. 清理执行记录 (2026-08-27)

第 6 节的清单已执行。执行前用 `git ls-remote` 直接向 GitHub 核对，确认
`hpc3/joint-geact-attention`(9a57f58)、`hpc3/tgt-sdd-libero-target-text`(5211cc3)、
`archive/loose-files-20260827`、`baton-only-20260827` 均已在远端，且三个 clean 副本的
HEAD（`a200628` / `95b1992` / `b607f0d`）都被远端分支包含。

### 已删除 —— HPC3 `/data/user/jhe724/workspace`

7 个副本目录（940M）：`VLM4WAM_geact_semantic_d0ec565` · `VLM4WAM_hdf5_benchmark_95b1992` ·
`VLM4WAM_hdf5_pilot_b607f0d` · `VLM4WAM_qwen35_worldarena` · `VLM4WAM_qwen35_baton_strict` ·
`VLM4WAM_joint_geact_02b89af` · `VLM4WAM_tgt_sdd_20260724`

11 个产物目录（86G）：`outputs/qwen3vl_semantic_planner` 37G ·
`outputs/qwen3vl4b_lingbot_dino_depth_uniform_k5_fullft{,_gbs64,_gbs64_lat128s128o}` 25.4G ·
`outputs/smoke_lat128` 8.5G · `outputs/smoke_depth_hpc3` 8.4G · `outputs/qwen35_ftsmoke` 5.5G ·
`outputs/qwen35_discrete_plan` 417M · `outputs/qwen35_discrete_plan_mw2` 417M ·
`outputs/qwen35_smoke` 417M · `outputs/eval_lingbot_dino_gbs64` 21M

HPC3 `VLM4WAM`：**351G → 265G**。

### 已删除 —— LFT `workspace/VLM4WAM`

`git gc --prune=now` 只回收 0.1G —— 第 4 节"gc 后应降到 ~170MB"的估计是错的。
实际 `.git` 的 4.6G 是 **Git LFS 对象缓存**，git objects 本身只有 92M。

扫描全仓库所有 blob 收集 LFS 指针后确认：本地 214 个 LFS 对象中**只有 2 个**被任何 commit
引用（`third_party/siglip2-so400m-patch14-384/tokenizer.{json,model}`），其余 **212 个是孤儿，
共 4.47 GB**，其中单个 4.2G 的对象在任何分支的任何指针里都查不到。`git lfs prune` 删除这 212 个。

LFT `workspace/VLM4WAM`：**8.4G → 3.9G**（`.git` 4.7G → 94M）。
删除后校验：23 个 ref 不变，`git fsck` 无输出，剩余 LFS 对象数为 2。

### 保留未动

| 对象 | 大小 | 原因 |
|---|---|---|
| `outputs/cosmos_semantic_plan` | 190G | Cosmos 世界模型自身训练，与退役 planner 无关 |
| `data/qwen35_train_mw` | 47G | 归属存疑，非 Baton 数据格式但无法确证属于哪条退役线 |
| `data/qwen35_train` | 3.6G | 同上 |
| `outputs/qwen35_ft_mw4` | 5.5G | 同上 |
| `third_party` / `models` / `mg_eval` / `attention_vis` | 18.8G | 不在本次范围 |
| LFT `VLA_WM/VLM4WAM` (A 线) | 213G | A 线保留 |
| LFT 两个 snapshot 目录 | 229M | 未确认是否还需要其中的 eval 结果 |

**合计释放约 91.5G**（HPC3 87G + LFT 4.5G）。

---

## 8. 仓库合并与第二轮清理 (2026-08-27)

### 8.1 归属确认：62G 全部属于退役的 Plan-X 线

第 6 节标为"归属存疑"的 62G 已查实，**与 Baton 无关**：

HPC3 `logs/q35-train-*.out` 显示 2026-07-24 那批 6 个作业（job 435419–435886）用的是同一套配置：
`codes=2048` + `codebook k-means initialized (frozen)`，损失是 **`code-CE`**（离散码本交叉熵），
`qwen35_ft_mw4/step_5000.pt` 里带 `code_head` 权重。而 Baton 是**连续** MSE
（`qwen35_baton/losses.py`：*"Baton Equation-8 continuous visual-feature regression"*），
全包 grep `code_head|codebook|kmeans|cross_entropy` 零命中。A 线有个分支就叫
`planner/planx-discrete-ce`，正对应这个目标函数。

precompute 日志确认 `data/qwen35_train`(1712 窗口) 与 `data/qwen35_train_mw`(22278 窗口)
是这批 run 的唯一数据源（各 run 日志的 `data N=` 与之对应），内容是 LIBERO 四套件
（object/spatial/goal/10）的 SigLIP2 特征。且 `qwen35_baton/teacher.py` 是
*"**Online**, frozen SigLIP2 patch-feature teacher"* —— 训练时在线编码，包里没有任何
`.npy`/cache 读取路径，所以 Baton 用不上这份预计算。

已删除：`outputs/qwen35_ft_mw4` 5.5G · `outputs/qwen35_ftmini` · `data/qwen35_train_mw` 47G ·
`data/qwen35_train` 3.6G。HPC3 `VLM4WAM`：**265G → 209G**（`data/` 已空，
`outputs/` 只剩 `cosmos_semantic_plan`）。

### 8.2 两个仓库合并为一

LFT 上原有两个 checkout 指向同一个 GitHub 仓库。现已合并为单一仓库
`/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM`，B 线作为 worktree 共存：

```
VLA_WM/VLM4WAM                                    ge-act-dual-camera-planner   (A 线主干)
VLA_WM/VLM4WAM/.worktrees/baton-only              baton-only-20260827          (B 线)
VLA_WM/VLM4WAM/.worktrees/frame80-action-attention
VLA_WM/VLM4WAM/.worktrees/joint-vlm-geact-libero-eval
VLA_WM/VLM4WAM/.worktrees/libero-episode-feature-export
```

B 的未跟踪数据（`artifacts/` 3.4G、`outputs/` 23M 含 `worldarena_eval_step015000_20260805`、
`logs/`）已移入 `.worktrees/baton-only/`。`workspace/VLM4WAM` 已整个删除；
删除前逐一验证其 11 个分支都能从某个 `origin/*` 引用到。A 里指向它的 `wsB` remote 已移除。

### 8.3 删除前的四批抢救

每个待删目录都做了逐文件 `git hash-object` 比对，只要内容在别处找不到就先归档：

| 批次 | 文件数 | 内容 |
|---|---|---|
| `lft_planx_tatok_untracked/` | 4 | planx worktree 里从未提交的 TA-Tok 训练器（`ta_tok_trainer.py`、`cli/train_ta_tok.py`、`cli/preflight.py` 及测试），`qwen35-planx-implementation` 分支里也没有 |
| `lft_repoB_sdd_history/` | 82 | B 线 `.superpowers/sdd/` —— Baton 从 07-27 到 08-05 的逐任务简报、完成报告、review diff、阶段快照。被 gitignore，任何分支都没有 |
| `lft_whatwhere_snapshot_only/` | 34 | what-where 快照里独有的文件：`target_attention_viz.py` + 33 个 sbatch 启动器/可视化/分析脚本（比对 1176 个文件，1142 个能在别处找到） |
| `lft_train_snapshot_20260611_only/` | 6 | 2026-06-11 训练版改动过的 Cosmos 核心文件，含 `minimal_v4_dit.py` 与 `experiments/base/robointer.py`（比对 1065 个文件，1059 个能在别处找到） |

### 8.4 已删除

| 对象 | 大小 |
|---|---|
| `VLA_WM/VLM4WAM/outputs/qwen3vl_semantic_planner` | 59G（退役 qwen3vl 线在 LFT 上的另一份） |
| `workspace/VLM4WAM`（整个 checkout，含 3 个 worktree 与 132M `.git`） | 459M |
| `VLA_WM/VLM4WAM_train_snapshot_20260611` | 198M |
| `VLA_WM/VLM4WAM_what_where_softlogit_snapshot` | 31M |

LFT `VLM4WAM`：**213G + 3.9G + 229M → 158G**（含并入的 3.8G worktree）。

### 8.5 当前总账

| 位置 | 清理前 | 清理后 |
|---|---|---|
| HPC3 `workspace/VLM4WAM` + 7 个副本目录 | 352G | 209G |
| LFT `VLA_WM/VLM4WAM` + `workspace/VLM4WAM` + 2 个快照 | 217G | 158G |
| **合计** | **569G** | **367G** |

累计释放约 **202G**，代码全部收敛到单一仓库与单一 GitHub 远端（25 个分支）。

剩余占用的大头：LFT `cosmos-predict2.5/outputs` 83G、`checkpoints` 57G、
HPC3 `outputs/cosmos_semantic_plan` 190G —— 都是仍在用的世界模型训练产物，不在清理范围内。
