### Task 5: Remove superseded wrappers and launcher-only tests

**Files:**
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1.sh`
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1_pod30274.sh`
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1_hpc3.sbatch`
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4.sh`
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4_hpc3.sbatch`
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_pod30274.sh`
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_hpc3.sbatch`
- Modify: `tests/test_lingbot_k1_current_future.py`
- Modify: `tests/test_lingbot_dino_depth_contract.py`
- Modify: `scripts/qwen3_vl_semantic_planner/README.md`

**Interfaces:**
- Preserves all behavioral query/teacher/loss/export/provider tests.
- Removes only tests whose sole observable is literal content in a superseded wrapper.

- [ ] **Step 1: Remove obsolete launcher-only assertions**

Delete these test functions:

```text
tests/test_lingbot_k1_current_future.py::test_independent_query_pod_launcher_pins_fair_ablation_contract
tests/test_lingbot_k1_current_future.py::test_independent_query_hpc3_launcher_pins_64_tokens_per_feature
tests/test_lingbot_dino_depth_contract.py::test_fastwam_launcher_pins_nine_frame_dual_branch_contract
tests/test_lingbot_dino_depth_contract.py::test_hpc3_launcher_defaults_to_recommended_12k_budget
```

Keep the generic-launcher cache/path tests because the generic launcher remains
production code. Keep all depth-probe and planner-evaluation files and tests.

- [ ] **Step 2: Delete the seven superseded wrappers**

Use an `apply_patch` delete operation for each exact file listed under Task 5.

- [ ] **Step 3: Update the README to the canonical contract**

Document:

```text
Qwen3-VL 4B LingBot FastWAM current configuration:
- frames: current 0 and future 8 from a 9-frame sample
- VLM queries: 4 independent groups × 64 = 256 tokens
- outputs: current/future DINO and current/future depth, each 256 × 1024
- distributed runtime: Accelerate + DeepSpeed ZeRO-2
- preferred batch: 8 GPUs × 8/GPU × accumulation 2 = global 128
- generic launcher: lingbot_dino_4b/train_lingbot_dino_4b.sh
- POD profile: lingbot_dino_4b/train_lingbot_fastwam_pod.sh
- HPC3 profile: lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch
```

- [ ] **Step 4: Prove no live references remain**

Run:

```bash
rg -n "train_lingbot_(current_future_fastwam_k1|dino_depth_fastwam_k4|independent_queries_fastwam_k1)" scripts tests docs -S
```

Expected: matches may remain only in historical design/plan documents; no match may remain in executable scripts, tests, or the current README.

---

