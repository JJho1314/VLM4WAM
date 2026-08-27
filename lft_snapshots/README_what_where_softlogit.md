# what-where-softlogit 方案快照

本目录是 **what_where_softlogit** 方案的独立代码副本（仅源码，已排除 `outputs/`、可视化、权重、媒体）。
对应主仓库 git 分支：**`what-where-softlogit`**（origin: `JJho1314/VLM4WAM`，从 `planner/baton-continuous` 切出，只含本方案代码，不含 baton/qwen planner 改动）。

- 主仓库：`/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM`
- 本副本：`/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM_what_where_softlogit_snapshot/cosmos-predict2.5`
- 生成时间：2026-06-26

---

## 1. 目标

让 InstructSAM(VLM) 的目标特征**真正引导 Cosmos 视频生成**"操作哪个/哪里的物体"，与文本**互补**（不是 text-free），
解决纯文本无法消歧的场景。关键改动是把"where"从不可分的 query-平均 dense 换成**判别性的 `[SEG]·dense` 软定位**，
并在训练上逼模型非用特征不可。

---

## 2. 数据流（含维度变化）

```
InstructSAM(.pt)              Loader                 DiT (28 blocks, ctx_dim=1024, latent_dim=4096)
──────────────────────  ─────────────────  ────────────────────────────────────────────────────
what  [SEG]        2048 ─► [16,2048] ──┐
                                       ├─► WhatWhereAdapter ─► ctx tokens [B,1024,1024] ─► 拼到文本前 ─► cross-attn
where target_dense_weighted [1024,256]─┘                                                       ▲
 (= decoder_dense × where_prob)         └─► PriorHead ─► [32,32] ─►插值[T,H,W]─► sigmoid ─► gate ─┘ (逐block乘target分支)
```

- **what（身份）**：`target_features_rawseg_ft`，`[SEG]` 隐状态 `[≤16, 2048]`（`target_feature_dim=2048, max_tokens=16`）。
- **where（定位，本方案关键）**：`target_features_where_softlogit_stage2_lora` 里的
  `target_dense_weighted [1024,256]` = `decoder_dense[32×32,256] × where_prob[32×32]`
  （`where_prob` 来自 InstructSAM `pred_masks` 选中目标 slot 的软 mask-logit）。旧版用 query-平均 dense（跨目标 cosine 0.994、不可分）。
- **注入两路**：① `TargetWhatWhereContextAdapter` 把 what(平均成身份向量) + where(每空间格一 token) 融成 1024 个 1024-d
  context token，拼到 T5 文本 token 之前进 cross-attn；② `DenseTargetWherePriorHead` 从同一对特征解出 `[32,32]` 空间 prior，
  插值到潜空间 `[T,H,W]` → sigmoid 得 spatial gate，在每个 block 把 target cross-attn 分支按位置乘进 4096-d 视频潜空间。

---

## 3. 方案核心代码文件（= 分支提交内容）

| 文件 | 作用 |
|---|---|
| `cosmos_predict2/_src/predict2/datasets/local_datasets/dataset_video.py` | 读 `target_dense_weighted` 作 "where"；`caption_dropout_prob` → 通用 `[TGT]` caption；`tgt_token_text` 永远设置(collate 修复) |
| `cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py` | `TargetWhatWhereContextAdapter` + `DenseTargetWherePriorHead`（空间门控注入） |
| `cosmos_predict2/_src/predict2/models/video2world_model_rectified_flow.py` | 注意力对齐损失 = `-log(mass_inside)`（取代失效的 min-max MSE） |
| `cosmos_predict2/experiments/base/robointer.py` | 实验 `predict2_video2world_training_2b_droid_success_v21_what_where_softlogit` |
| `cosmos_predict2/config.py` | `instructsam_feature_mode` / `route_decoder_dense_to_target_dense_feature` / `pass_instructsam_mask_to_cosmos` |
| `cosmos_predict2/inference.py` | `_get_target_condition`（推理侧 target_dense_feature 装载） |
| `cosmos_predict2/_src/predict2/callbacks/iter_speed.py`, `wandb_log.py` | `target_where_prior_*` 指标 |
| `scripts/generate_tavid_mask_samples.py` | 评测/生成，认 `target_dense_weighted` key |
| `scripts/smoke_test_textfree_multisource.py` | adapter 冒烟测试 |
| `scripts/analyze_latent_grounding_features.py` | latent grounding 诊断（mask 形状兼容） |

> ⚠️ **注意**：`local_datasets/`（含核心的 `dataset_video.py`、`dataset_utils.py`、`__init__.py`）在主仓库被
> `.gitignore` 第 47 行 `datasets/` 规则**误伤忽略**，常规 `git add` 加不进去。分支里已用 `git add -f` 强制纳入；
> 本副本由 rsync 直接拷文件系统，也已保留。**改这些文件后别忘了 `-f` 重新提交。**

---

## 4. 关键超参 / 训练配置

- 实验名：`predict2_video2world_training_2b_droid_success_v21_what_where_softlogit`
- 分辨率/帧数：320×576，49 帧，frame_stride ∈ {1,2,3}
- `caption_dropout_prob=0.4`（40% 换成保留 `[TGT]` 的通用句，逼模型从特征读身份）
- `target_mask_dropout_prob=0.1`（CFG null）
- `target_attention_loss_weight=0.05`
- **从 base 重训 3000 iter**（非 fine-tune），bs4×accum4，gbs128

---

## 5. HPC3 侧资产（不在本副本，需在 HPC3 上）

- 数据集：`/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half`
  - `what`：`target_features_rawseg_ft/`（2048-d）
  - `where`：`target_features_where_softlogit_stage2_lora/`（18,803 个 .pt，9.5GB）
    每个 .pt：`where_logit[32,32]`、`where_prob[32,32]`、`target_dense_weighted[1024,256] fp16`、`target_proto[256]`
- 预计算脚本：`/data/user/jhe724/workspace/VLM4WAM/tools/precompute_where_softlogit.py`（+ `sbatch_where_full.sh`，8-GPU torchrun，instructsam env）
- checkpoint（iter3000）：
  `/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_v21_what_where_softlogit_320x576_49f_s123_vlm4wam/cosmos_predict_v2p5/video2world/2b_what_where_softlogit_iou50_320x576_49f_s123_bs4accum4_gbs128_3000/checkpoints/iter_000003000`
- InstructSAM：model `/data/user/jhe724/workspace/InstructSAM/work_dirs/instructsam_stage2_complete_lora`，
  source `/data/user/jhe724/workspace/InstructSAM`，env `/data/user/jhe724/.conda/envs/instructsam`
- 推理/训练需：`export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights`（+ `HF_HUB_OFFLINE=1`）；
  torchrun 不在 PATH，用 `$VENV/bin/python -m torch.distributed.run`

---

## 6. 验证结论（2026-06-24）

新模型确实**用上了**特征：where_prior 内/外比 4–6（旧 ~1.6）、注意力 inside/outside 比从钉死的 1.0 升到 30–275（震荡，权重或偏高）；
carrot/banana 特征互换使生成变化 +70%（10.2→17.3），diff 图从弥散变结构化。12 样本干净生成评测画质连贯。
TODO：更多目标/场景验证、收紧空间定位、稳定注意力损失。
