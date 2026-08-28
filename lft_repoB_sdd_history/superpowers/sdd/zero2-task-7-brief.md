### Task 7: Deploy an isolated POD smoke run and validate the checkpoint

**Files:**
- Remote copy: `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713`
- Remote output: `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/outputs/smoke_zero2_b8a2`

**Interfaces:**
- Consumes the verified local implementation and existing POD weights/data.
- Produces a two-step ZeRO-2 checkpoint compatible with the FastWAM provider.

- [ ] **Step 1: Invoke the run-experiment skill before remote mutation**

Read and follow `/home/LFT-W02/.codex/skills/run-experiment/SKILL.md`. Confirm
the active long-running PID is not targeted by any command.

- [ ] **Step 2: Create an isolated remote code copy and sync only scoped files**

Run from the local repository root:

```bash
ssh -p 30282 root@182.242.159.145 \
  'test ! -e /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713 && cp -a /root/nas/junjie/code/VLM4WAM_k1_fastwam_20260712 /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713'
rsync -avR -e 'ssh -p 30282' \
  ./scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py \
  ./scripts/qwen3_vl_semantic_planner/README.md \
  ./scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py \
  ./scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py \
  ./scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh \
  ./scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh \
  ./scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch \
  ./tests/test_lingbot_zero2_runtime.py \
  ./tests/test_lingbot_dino_depth_contract.py \
  ./tests/test_lingbot_k1_current_future.py \
  root@182.242.159.145:/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/
ssh -p 30282 root@182.242.159.145 \
  'rm -f /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1.sh /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1_pod30274.sh /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1_hpc3.sbatch /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4.sh /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4_hpc3.sbatch /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_pod30274.sh /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_hpc3.sbatch'
```

These commands do not sync outputs, logs, caches, `.git`, or unrelated dirty
files.

- [ ] **Step 3: Launch the preferred b8/a2 smoke**

From the isolated remote repository, run:

```bash
RUN_KIND=smoke \
REPO_ROOT=/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713 \
OUTPUT_DIR=/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/outputs/smoke_zero2_b8a2 \
MAX_STEPS=2 SAVE_STEPS=2 BATCH_SIZE=8 GRAD_ACCUM=2 \
bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh
```

Capture stdout/stderr in a new log and record only the new wrapper/rank PIDs.

- [ ] **Step 4: Monitor runtime, memory, and completion**

Verify the log reports:

```text
distributed_type=DEEPSPEED
world_size=8
batch_size_per_gpu=8
gradient_accumulation_steps=2
global_batch_size=128
zero_stage=2
gradient_checkpointing=false
```

Also verify all eight ranks are alive during training, GPU memory stays below
the device limit, and the log contains no OOM, traceback, NCCL error, NaN, or
accumulation mismatch.

- [ ] **Step 5: Use the documented memory fallback only if b8/a2 OOMs**

If and only if the smoke fails with CUDA OOM, stop only the smoke PIDs and
repeat with:

```bash
BATCH_SIZE=4 GRAD_ACCUM=4 OUTPUT_DIR=/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/outputs/smoke_zero2_b4a4
```

Do not change global batch, query geometry, loss weights, checkpointing, or the
active pre-existing training process.

- [ ] **Step 6: Validate the step-2 export**

Verify `step_000002` contains the model, processor, all four head files,
`plan_token_embedding.pt`, and `planner_meta.json`. Parse metadata and assert:

```text
num_keyframes = 1
num_task_tokens = 64
latent_len = 256
total_unique_latent_per_keyframe = 256
target_tokens = 256
independent_modality_task_tokens = true
keyframe_offsets = [8]
```

Load the metadata through `dino_depth_plan_provider.validate_planner_metadata`
to prove the production consumer accepts the export. Run:

```bash
ssh -p 30282 root@182.242.159.145 \
  'cd /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713 && /opt/conda/envs/vlm4wam/bin/python -c "import json,sys; from pathlib import Path; root=Path(\"outputs/smoke_zero2_b8a2/step_000002\"); required=(\"qwen3vl_lora_or_model\",\"processor\",\"plan_head.pt\",\"depth_head.pt\",\"current_plan_head.pt\",\"current_depth_head.pt\",\"plan_token_embedding.pt\",\"planner_meta.json\"); missing=[name for name in required if not (root/name).exists()]; assert not missing, missing; meta=json.loads((root/\"planner_meta.json\").read_text()); assert meta[\"num_keyframes\"] == 1; assert meta[\"num_task_tokens\"] == 64; assert meta[\"latent_len\"] == 256; assert meta[\"total_unique_latent_per_keyframe\"] == 256; assert meta[\"target_tokens\"] == 256; assert meta[\"independent_modality_task_tokens\"] is True; assert meta[\"keyframe_offsets\"] == [8]; sys.path.insert(0, str(Path(\"scripts/qwen3_vl_semantic_planner/lingbot_dino_4b\").resolve())); from dino_depth_plan_provider import validate_planner_metadata; contract=validate_planner_metadata(meta); assert contract.num_task_tokens == 64; print(\"step-2 export valid\")"'
```

Expected: the remote command prints `step-2 export valid`.
