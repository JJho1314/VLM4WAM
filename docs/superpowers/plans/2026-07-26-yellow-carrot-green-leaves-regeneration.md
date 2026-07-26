# Yellow-Carrot Green-Leaves Regeneration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the iter-3000 yellow-carrot oracle video with positive and negative prompt constraints that keep the green leafy top attached to the carrot.

**Architecture:** Reuse the existing HPC3 Slurm reproduction launcher and all original oracle artifacts. Build a new staged inference JSON that changes only the positive and negative prompts, submit it to a separate output directory, then verify both the technical video contract and the visible carrot-leaf connection.

**Tech Stack:** Bash, Slurm, `jq`, Cosmos-Predict2.5 inference, PyAV, Pillow, SSH/rsync

## Global Constraints

- Reuse `semantic_localization/oracle_repro/reproduce_oracle_yc.sh`.
- Keep the original `yc74616` first frame and yellow-carrot GT-future SigLIP2 oracle plan.
- Keep the iter-3000 SG-WAM checkpoint, seed `0`, guidance `7`, `35` denoising steps, and `49` output frames.
- Keep five semantic-plan keyframes on the native spatial grid.
- Write only to the new input and result directories; do not overwrite earlier reproductions.
- Do not change repository source code during experiment execution.

---

### Task 1: Stage the Prompt-Only Inference Input

**Files:**
- Read: `semantic_localization/oracle_repro/reproduce_oracle_yc.sh`
- Create remotely: `/data/user/jhe724/workspace/VLM4WAM/eval_inputs_oracle_iter3000_yellowcarrot_greenleaves_REPRO/yc74616_s0.json`
- Reference remotely: `/data/user/jhe724/workspace/VLM4WAM/eval_results_oracle_iter3000_yellowcarrot_20260703_170049/`
- Reference remotely: `/data/user/jhe724/workspace/VLM4WAM/eval_results_oracle_iter3000_yellowcarrot_REPRO/yc74616_s0_oracle.json`

**Interfaces:**
- Consumes: the original first frame, carrot oracle plan, and the previously emitted sample JSON containing Cosmos's standard negative prompt.
- Produces: a staged `yc74616_s0.json` accepted by the unchanged reproduction launcher.

- [ ] **Step 1: Confirm target paths are unused and source artifacts exist**

Run:

```bash
ssh HPC3_jhe724 '
set -e
ROOT=/data/user/jhe724/workspace/VLM4WAM
INPUT=$ROOT/eval_inputs_oracle_iter3000_yellowcarrot_greenleaves_REPRO
OUTPUT=$ROOT/eval_results_oracle_iter3000_yellowcarrot_greenleaves_REPRO
SOURCE=$ROOT/eval_results_oracle_iter3000_yellowcarrot_20260703_170049
BASE_JSON=$ROOT/eval_results_oracle_iter3000_yellowcarrot_REPRO/yc74616_s0_oracle.json
test ! -e "$INPUT"
test ! -e "$OUTPUT"
test -s "$SOURCE/first_frames/yc74616_f0.png"
test -s "$SOURCE/plans/yc74616_s0_oracle.pt"
test -s "$BASE_JSON"
echo PREFLIGHT_PATHS_OK
'
```

Expected: `PREFLIGHT_PATHS_OK`.

- [ ] **Step 2: Check Slurm GPU availability before deployment**

Run:

```bash
ssh HPC3_jhe724 '
sinfo -p acd_u -o "%P %a %l %D %t %G"
squeue -u jhe724 -o "%.18i %.12P %.30j %.8T %.10M %.6D %R"
'
```

Expected: partition `acd_u` is available and the scheduler reports cluster state. HPC3's login node has no `nvidia-smi`; the Slurm GRES allocation is the authoritative GPU check.

- [ ] **Step 3: Create staged links and the modified JSON**

Run:

```bash
ssh HPC3_jhe724 '
set -euo pipefail
ROOT=/data/user/jhe724/workspace/VLM4WAM
INPUT=$ROOT/eval_inputs_oracle_iter3000_yellowcarrot_greenleaves_REPRO
SOURCE=$ROOT/eval_results_oracle_iter3000_yellowcarrot_20260703_170049
BASE_JSON=$ROOT/eval_results_oracle_iter3000_yellowcarrot_REPRO/yc74616_s0_oracle.json
POSITIVE="A Franka robotic arm with a parallel-jaw gripper carefully grasp only the [TGT] whole yellow carrot with its fresh green leafy top firmly attached, located in the sink basin, and place the intact carrot into the black pot next to the banana, without moving the banana. The green leafy top remains continuously attached to the carrot and moves together with it as one intact object throughout the entire video."
NEGATIVE_SUFFIX=" Detached green leaves, separated leafy top, broken carrot leaves, floating leaves, disconnected carrot and leaves."
mkdir -p "$INPUT/plans" "$INPUT/first_frames"
ln -s "$SOURCE/plans/yc74616_s0_oracle.pt" "$INPUT/plans/yc74616_s0_oracle.pt"
ln -s "$SOURCE/first_frames/yc74616_f0.png" "$INPUT/first_frames/yc74616_f0.png"
jq --arg name "yc74616_s0_oracle_greenleaves" \
   --arg prompt "$POSITIVE" \
   --arg suffix "$NEGATIVE_SUFFIX" \
   ".name=\$name | .prompt=\$prompt | .negative_prompt=(.negative_prompt + \$suffix)" \
   "$BASE_JSON" > "$INPUT/yc74616_s0.json"
jq -e . "$INPUT/yc74616_s0.json" >/dev/null
'
```

Expected: command exits `0` and creates one JSON plus two artifact symlinks.

- [ ] **Step 4: Validate the staged contract**

Run:

```bash
ssh HPC3_jhe724 '
set -e
SPEC=/data/user/jhe724/workspace/VLM4WAM/eval_inputs_oracle_iter3000_yellowcarrot_greenleaves_REPRO/yc74616_s0.json
jq -e "
  .name == \"yc74616_s0_oracle_greenleaves\" and
  .seed == 0 and
  .guidance == 7 and
  .num_steps == 35 and
  .num_output_frames == 49 and
  (.prompt | contains(\"green leafy top firmly attached\")) and
  (.prompt | contains(\"moves together with it as one intact object\")) and
  (.negative_prompt | contains(\"Detached green leaves\")) and
  (.negative_prompt | contains(\"disconnected carrot and leaves\"))
" "$SPEC" >/dev/null
echo STAGED_SPEC_OK
'
```

Expected: `STAGED_SPEC_OK`.

### Task 2: Submit and Monitor the HPC3 Generation

**Files:**
- Sync: `semantic_localization/oracle_repro/reproduce_oracle_yc.sh`
- Create remotely: `/data/user/jhe724/workspace/VLM4WAM/eval_results_oracle_iter3000_yellowcarrot_greenleaves_REPRO/`
- Create remotely: `/data/user/jhe724/workspace/VLM4WAM/logs/oracle-greenleaves-%j.out`
- Create remotely: `/data/user/jhe724/workspace/VLM4WAM/logs/oracle-greenleaves-%j.err`

**Interfaces:**
- Consumes: Task 1's staged input directory.
- Produces: a completed Slurm job and `yc74616_s0_oracle_greenleaves.mp4`.

- [ ] **Step 1: Sync the exact existing launcher**

Run:

```bash
ssh HPC3_jhe724 'mkdir -p /data/user/jhe724/workspace/VLM4WAM/semantic_localization/oracle_repro'
rsync -av \
  semantic_localization/oracle_repro/reproduce_oracle_yc.sh \
  HPC3_jhe724:/data/user/jhe724/workspace/VLM4WAM/semantic_localization/oracle_repro/reproduce_oracle_yc.sh
sha256sum semantic_localization/oracle_repro/reproduce_oracle_yc.sh
ssh HPC3_jhe724 'sha256sum /data/user/jhe724/workspace/VLM4WAM/semantic_localization/oracle_repro/reproduce_oracle_yc.sh'
```

Expected: local and remote SHA-256 values are identical.

- [ ] **Step 2: Submit one-GPU Slurm job**

Run:

```bash
ssh HPC3_jhe724 '
cd /data/user/jhe724/workspace/VLM4WAM
JOB_ID=$(sbatch --parsable \
  --job-name=cosmos-carrot-greenleaves \
  --output=/data/user/jhe724/workspace/VLM4WAM/logs/oracle-greenleaves-%j.out \
  --error=/data/user/jhe724/workspace/VLM4WAM/logs/oracle-greenleaves-%j.err \
  --export=ALL,ORIG=/data/user/jhe724/workspace/VLM4WAM/eval_inputs_oracle_iter3000_yellowcarrot_greenleaves_REPRO,OUT=/data/user/jhe724/workspace/VLM4WAM/eval_results_oracle_iter3000_yellowcarrot_greenleaves_REPRO \
  semantic_localization/oracle_repro/reproduce_oracle_yc.sh)
printf "%s\n" "$JOB_ID" > /data/user/jhe724/workspace/VLM4WAM/logs/oracle-greenleaves.jobid
echo "$JOB_ID"
'
```

Expected: one numeric Slurm job ID, also persisted in `logs/oracle-greenleaves.jobid`.

- [ ] **Step 3: Verify allocation and launch**

Run:

```bash
ssh HPC3_jhe724 '
JOB_ID=$(cat /data/user/jhe724/workspace/VLM4WAM/logs/oracle-greenleaves.jobid)
squeue -j "$JOB_ID" -o "%.18i %.12P %.30j %.8T %.10M %.6D %R"
scontrol show job "$JOB_ID" | grep -E "JobState=|Reason=|NodeList=|TresPerNode="
'
```

Expected: job is `PENDING` or `RUNNING`; once running, `TresPerNode=gres/gpu:1`.

- [ ] **Step 4: Monitor until terminal state**

Run periodically:

```bash
ssh HPC3_jhe724 '
JOB_ID=$(cat /data/user/jhe724/workspace/VLM4WAM/logs/oracle-greenleaves.jobid)
squeue -j "$JOB_ID" -o "%.18i %.8T %.10M %R"
tail -20 "/data/user/jhe724/workspace/VLM4WAM/logs/oracle-greenleaves-${JOB_ID}.out" 2>/dev/null || true
'
```

Expected: generation reaches `36/36`, then the job leaves `squeue`.

### Task 3: Verify the Video and Inspect Leaf Attachment

**Files:**
- Read remotely: `/data/user/jhe724/workspace/VLM4WAM/eval_results_oracle_iter3000_yellowcarrot_greenleaves_REPRO/yc74616_s0_oracle_greenleaves.mp4`
- Create remotely: `/data/user/jhe724/workspace/VLM4WAM/eval_results_oracle_iter3000_yellowcarrot_greenleaves_REPRO/keyframes/frame_{000,024,048}.png`
- Copy locally: `/tmp/yellowcarrot_greenleaves_verify/`

**Interfaces:**
- Consumes: Task 2's Slurm job ID and generated result directory.
- Produces: technical verification evidence plus three frames for qualitative review.

- [ ] **Step 1: Verify job, prompt, config, logs, and full video decode**

Run:

```bash
ssh HPC3_jhe724 '
set -euo pipefail
JOB_ID=$(cat /data/user/jhe724/workspace/VLM4WAM/logs/oracle-greenleaves.jobid)
ROOT=/data/user/jhe724/workspace/VLM4WAM
OUT=$ROOT/eval_results_oracle_iter3000_yellowcarrot_greenleaves_REPRO
VIDEO=$OUT/yc74616_s0_oracle_greenleaves.mp4
SPEC=$OUT/yc74616_s0_oracle_greenleaves.json
PY=/data/user/jhe724/workspace/cosmos-predict2.5/.venv/bin/python
STATE=$(sacct -j "$JOB_ID" --format=State,ExitCode -n -X -P | head -1)
[ "$STATE" = "COMPLETED|0:0" ]
test -s "$VIDEO"
jq -e "
  (.prompt | contains(\"green leafy top firmly attached\")) and
  (.negative_prompt | contains(\"Detached green leaves\")) and
  .seed == 0 and .guidance == 7 and .num_steps == 35 and .num_output_frames == 49
" "$SPEC" >/dev/null
grep -q "semantic_plan_num_keyframes: .5." "$OUT/config.yaml"
grep -q "semantic_plan_spatial_grid: .0." "$OUT/config.yaml"
test ! -s "$ROOT/logs/oracle-greenleaves-${JOB_ID}.err"
grep -q "Saved video to $VIDEO" "$OUT/inference.log"
"$PY" -c "import av,sys; c=av.open(sys.argv[1]); fs=list(c.decode(video=0)); shapes={tuple(f.to_ndarray(format=\"rgb24\").shape) for f in fs}; print({\"frames\":len(fs),\"shapes\":sorted(shapes)}); assert len(fs)==49; assert len(shapes)==1" "$VIDEO"
sha256sum "$VIDEO"
echo VIDEO_CONTRACT_OK
'
```

Expected: PyAV reports exactly `49` frames and the command ends with `VIDEO_CONTRACT_OK`.

- [ ] **Step 2: Extract beginning, middle, and ending frames**

Run:

```bash
ssh HPC3_jhe724 '
set -e
OUT=/data/user/jhe724/workspace/VLM4WAM/eval_results_oracle_iter3000_yellowcarrot_greenleaves_REPRO
VIDEO=$OUT/yc74616_s0_oracle_greenleaves.mp4
PY=/data/user/jhe724/workspace/cosmos-predict2.5/.venv/bin/python
mkdir -p "$OUT/keyframes"
"$PY" -c "import av,sys; from pathlib import Path; from PIL import Image; video,out=sys.argv[1],Path(sys.argv[2]); fs=list(av.open(video).decode(video=0)); picks=(0,24,48); [Image.fromarray(fs[i].to_ndarray(format=\"rgb24\")).save(out/f\"frame_{i:03d}.png\") for i in picks]" "$VIDEO" "$OUT/keyframes"
'
mkdir -p /tmp/yellowcarrot_greenleaves_verify
scp HPC3_jhe724:/data/user/jhe724/workspace/VLM4WAM/eval_results_oracle_iter3000_yellowcarrot_greenleaves_REPRO/keyframes/frame_000.png /tmp/yellowcarrot_greenleaves_verify/
scp HPC3_jhe724:/data/user/jhe724/workspace/VLM4WAM/eval_results_oracle_iter3000_yellowcarrot_greenleaves_REPRO/keyframes/frame_024.png /tmp/yellowcarrot_greenleaves_verify/
scp HPC3_jhe724:/data/user/jhe724/workspace/VLM4WAM/eval_results_oracle_iter3000_yellowcarrot_greenleaves_REPRO/keyframes/frame_048.png /tmp/yellowcarrot_greenleaves_verify/
```

Expected: three non-empty PNG files.

- [ ] **Step 3: Perform qualitative review**

Open all three images with `view_image` and inspect:

1. The target carrot includes a visible green leafy top.
2. The leafy top touches the carrot body without a visible gap.
3. The carrot and leafy top move as one object in the middle and ending frames.

Report the exact observation. Do not call the topology fixed if any inspected frame shows detachment, floating leaves, or a disconnected duplicate.

- [ ] **Step 4: Report the artifact and provenance**

Report:

```text
/data/user/jhe724/workspace/VLM4WAM/eval_results_oracle_iter3000_yellowcarrot_greenleaves_REPRO/yc74616_s0_oracle_greenleaves.mp4
```

Include the Slurm job ID, SHA-256, technical decode result, and the qualitative leaf-attachment assessment.
