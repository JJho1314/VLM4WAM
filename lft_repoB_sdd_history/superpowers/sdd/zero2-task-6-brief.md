### Task 6: Run the full local regression and static verification

**Files:**
- Verify all implementation and test files from Tasks 1–5.

**Interfaces:**
- Produces a locally verified candidate for the remote two-step smoke run.

- [ ] **Step 1: Run focused planner and FastWAM tests**

Run:

```bash
pytest -q \
  tests/test_lingbot_zero2_runtime.py \
  tests/test_lingbot_k1_current_future.py \
  tests/test_lingbot_dino_depth_contract.py \
  tests/test_dino_depth_plan_provider.py \
  tests/test_fastwam_online_semantic_planner.py \
  tests/test_fastwam_semantic_timing_routing.py \
  tests/test_fastwam_cosmos_semantic_plan.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Compile Python and validate shell files**

Run:

```bash
python -m py_compile \
  scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch
git diff --check
```

Expected: every command exits zero and produces no error output.

- [ ] **Step 3: Review the scoped diff without staging user changes**

Run:

```bash
git status --short
git diff -- scripts/qwen3_vl_semantic_planner tests scripts/qwen3_vl_semantic_planner/README.md
```

Expected: the diff contains the approved runtime, launcher, test cleanup, and
README changes and preserves all unrelated dirty-worktree files.

---

