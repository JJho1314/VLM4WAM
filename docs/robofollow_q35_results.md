# RoboFollow results (robofollow-q35)

Status: 2026-10-06. Four GE-Act models trained for 20k steps at global batch 32 (4 per GPU × 8 GPUs), lr 3e-5, DeepSpeed ZeRO-2 bf16. Each was evaluated on all 554 official tasks (L0 75, L1 86, L2 173, L3 220), 1 round, seed 42, `--max-steps 10 --actions-per-step 50 --runtime fixed --sim-steps 15`. Every task ran twice, once with the correct instruction and once with a shuffled training instruction.

- `text`: GE-Act conditioned on the T5 instruction only.
- `joint`: also uses predicted semantic plan tokens from the frozen Qwen3.5 planner (LIBERO-trained). Its head is trained on RoboFollow with a 0.1 anchor to SigLIP2 teacher features.
- `p256`: native 240×320 frames padded to 256×320. `r192`: frames resized to 192×256.

## Per-level scores (intent / exec / completion)

| Model | Instruction | L0 | L1 | L2 | L3 |
|---|---|---|---|---|---|
| text_p256 | correct | 0.516 / 0.403 / 0.373 | 0.248 / 0.193 / 0.116 | 0.240 / 0.184 / 0.098 | 0.196 / 0.139 / 0.059 |
| | shuffled | 0.072 / 0.056 / 0.013 | 0.066 / 0.037 / 0.012 | 0.112 / 0.076 / 0.058 | 0.088 / 0.061 / 0.045 |
| text_r192 | correct | 0.547 / 0.392 / 0.387 | 0.227 / 0.153 / 0.105 | 0.234 / 0.177 / 0.087 | 0.208 / 0.143 / 0.059 |
| | shuffled | 0.092 / 0.052 / 0.013 | 0.071 / 0.017 / 0.000 | 0.091 / 0.060 / 0.029 | 0.076 / 0.035 / 0.018 |
| joint_p256 | correct | 0.500 / 0.409 / 0.440 | 0.214 / 0.142 / 0.116 | 0.290 / 0.224 / 0.139 | 0.199 / 0.138 / 0.068 |
| | shuffled | 0.076 / 0.033 / 0.013 | 0.048 / 0.031 / 0.012 | 0.122 / 0.088 / 0.092 | 0.113 / 0.059 / 0.032 |
| joint_r192 | correct | 0.605 / 0.480 / 0.440 | 0.256 / 0.177 / 0.128 | 0.278 / 0.210 / 0.116 | 0.248 / 0.182 / 0.059 |
| | shuffled | 0.047 / 0.015 / 0.013 | 0.047 / 0.026 / 0.000 | 0.092 / 0.061 / 0.046 | 0.065 / 0.043 / 0.023 |

## Joint minus text, paired by task (correct instruction, 95% task-bootstrap CI)

| Resolution | Level | n | Completion diff | Intent diff |
|---|---|---|---|---|
| p256 | L0 | 75 | +0.067 [-0.027, +0.160] | -0.016 [-0.065, +0.032] |
| p256 | L2 | 173 | +0.040 [-0.006, +0.087] | **+0.050 [+0.019, +0.082]** |
| p256 | L1–L3 | 479 | +0.019 [-0.006, +0.044] | +0.013 [-0.007, +0.033] |
| r192 | L0 | 75 | +0.053 [-0.067, +0.173] | +0.059 [-0.017, +0.135] |
| r192 | L2 | 173 | +0.029 [-0.023, +0.081] | **+0.044 [+0.004, +0.086]** |
| r192 | L3 | 220 | +0.000 [-0.032, +0.032] | **+0.040 [+0.007, +0.074]** |
| r192 | L1–L3 | 479 | +0.015 [-0.013, +0.042] | **+0.039 [+0.015, +0.063]** |

Language dependence was measured as correct minus shuffled completion. Joint does not differ from text on it anywhere (largest gap +0.023 for p256 L3, CI crosses 0).

**Reading:**
- Semantic guidance raises intent, i.e. picking the right object, on semantic recombination (L2) and on L3. These gains are significant at 192×256, and on L2 at 256×320.
- Completion gains are positive but not significant with one round. Execution (grasp/place), not target choice, limits completion.
- The two resolutions are equivalent for the text baseline. 256×320 was kept for the bs128 runs because joint's completion gain looked larger there.

### Three rounds per task (p256, L0 and L2)

The seed-42 round was extended with two more rounds per task (seed 1042) for joint_p256 and text_p256 on L0 and L2. Each task is averaged over its 3 rounds before pairing.

| Level | n | intent joint / text (diff) | exec joint / text (diff) | completion joint / text (diff) |
|---|---|---|---|---|
| L0 | 75 | 0.484 / 0.507 (−0.023 [−0.063, +0.016]) | 0.414 / 0.434 (−0.020 [−0.069, +0.028]) | 0.431 / 0.413 (+0.018 [−0.049, +0.084]) |
| L2 | 173 | 0.268 / 0.240 (+0.028 [+0.002, +0.055]) | 0.203 / 0.186 (+0.017 [−0.008, +0.044]) | 0.115 / 0.096 (+0.018 [−0.016, +0.054]) |

The seed-42 round alone overstated the batch-32 effect: the L2 intent gain shrinks from +0.050 to +0.028 (borderline), and the L0 completion gain from +0.067 to +0.018. The seed-1042 rounds alone give L2 intent +0.015 [−0.016, +0.047]. At batch 32 the semantic planner gives at most a small L2 intent gain; the clearer evidence is at batch 128 (below).

## Comparison with the official Intent Score (IS, %)

| Policy | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| π0.5 | 92.1 | 56.0 | 38.5 | 36.1 |
| π0 | 93.6 | 43.5 | 26.7 | 29.8 |
| FAST-WAM | 79.0 | 25.4 | 22.4 | 30.4 |
| Motus | 67.7 | 29.7 | 24.1 | 25.8 |
| ours text_p256 | 51.6 | 24.8 | 24.0 | 19.6 |
| ours joint_p256 | 50.0 | 21.4 | 29.0 | 19.9 |
| ours joint_r192 | 60.5 | 25.6 | 27.8 | 24.8 |

L1–L3 are on par with the other world-action models (FAST-WAM, Motus). The gap is in-distribution (L0).

## Global batch 128 (text_p256_bs128 vs text_p256, correct instruction)

`rf_text_p256_bs128`: same as `text_p256` but global batch 128 (4 per GPU × 16 GPUs × grad-accum 2, 2 nodes), lr 3e-5 unchanged, 20k optimizer steps (4× the samples). Same eval protocol (seed 42, 1 round, correct instruction only). Paired by task, 95% task-bootstrap CI on bs128 − bs32.

| Level | n | intent bs128 / bs32 (diff) | exec bs128 / bs32 (diff) | completion bs128 / bs32 (diff) |
|---|---|---|---|---|
| L0 | 75 | 0.503 / 0.516 (−0.013 [−0.079, +0.049]) | 0.483 / 0.403 (**+0.080** [+0.001, +0.160]) | 0.507 / 0.373 (**+0.133** [+0.027, +0.240]) |
| L1 | 86 | 0.234 / 0.248 (−0.014 [−0.076, +0.049]) | 0.160 / 0.193 (−0.033 [−0.085, +0.016]) | 0.128 / 0.116 (+0.012 [−0.047, +0.070]) |
| L2 | 173 | 0.274 / 0.240 (+0.034 [−0.009, +0.076]) | 0.201 / 0.184 (+0.017 [−0.023, +0.057]) | 0.116 / 0.098 (+0.017 [−0.029, +0.064]) |
| L3 | 220 | 0.225 / 0.196 (**+0.029** [+0.001, +0.057]) | 0.181 / 0.139 (**+0.043** [+0.014, +0.071]) | 0.091 / 0.059 (+0.032 [+0.000, +0.064]) |
| L1–L3 | 479 | 0.244 / 0.221 (**+0.023** [+0.001, +0.046]) | 0.185 / 0.165 (+0.020 [−0.001, +0.041]) | 0.106 / 0.084 (+0.023 [−0.002, +0.048]) |
| all | 554 | 0.279 / 0.261 (+0.018 [−0.003, +0.039]) | 0.225 / 0.197 (**+0.028** [+0.006, +0.049]) | 0.161 / 0.123 (**+0.038** [+0.011, +0.065]) |

Larger batch mostly improves execution and completion; intent barely moves. Part of the L0 gain may be eval noise: a second text_p256 run on L0 (seed 1042, 2 rounds) reached completion 0.433 instead of 0.373. L0 failures at bs128 are still dominated by "arm left home again" (26/75 first intent failures).

## Joint vs text at global batch 128 (L0–L3)

`rf_joint_p256_bs128`: `joint_p256` trained like `rf_text_p256_bs128` (global batch 128 on 2 nodes, lr 3e-5, 20k steps). All 554 tasks, correct instruction, seed 42, 1 round. Paired by task, 95% task-bootstrap CI on joint − text (both at batch 128).

| Level | n | intent joint / text (diff) | exec joint / text (diff) | completion joint / text (diff) |
|---|---|---|---|---|
| L0 | 75 | 0.608 / 0.503 (**+0.105** [+0.029, +0.184]) | 0.505 / 0.483 (+0.023 [−0.068, +0.109]) | 0.493 / 0.507 (−0.013 [−0.133, +0.093]) |
| L1 | 86 | 0.249 / 0.234 (+0.015 [−0.045, +0.073]) | 0.199 / 0.160 (+0.038 [−0.014, +0.090]) | 0.151 / 0.128 (+0.023 [−0.035, +0.081]) |
| L2 | 173 | 0.318 / 0.274 (**+0.045** [+0.004, +0.087]) | 0.268 / 0.201 (**+0.067** [+0.027, +0.108]) | 0.139 / 0.116 (+0.023 [−0.023, +0.069]) |
| L3 | 220 | 0.228 / 0.225 (+0.003 [−0.030, +0.036]) | 0.204 / 0.181 (+0.022 [−0.010, +0.055]) | 0.086 / 0.091 (−0.005 [−0.036, +0.027]) |
| L1–L3 | 479 | 0.265 / 0.244 (+0.020 [−0.003, +0.044]) | 0.226 / 0.185 (**+0.041** [+0.019, +0.064]) | 0.117 / 0.106 (+0.010 [−0.013, +0.035]) |
| all | 554 | 0.311 / 0.279 (**+0.032** [+0.009, +0.054]) | 0.264 / 0.225 (**+0.039** [+0.017, +0.062]) | 0.168 / 0.161 (+0.007 [−0.018, +0.032]) |

Against joint at batch 32 (all 554 tasks): intent +0.041 [+0.017, +0.065], exec +0.062 [+0.039, +0.084], completion +0.020 [−0.007, +0.047].

At batch 128 the semantic planner improves intent (which object, which arm, which target) and execution on L0 and L2, and over all tasks. Completion is unchanged: the extra correct intents do not turn into more fully finished tasks. Partial results during the run showed a significant L3 intent drop (−0.071 with 76 L3 tasks); it disappeared once all 220 L3 tasks were in (+0.003).

L0 failure structure (75 trials each):
- Wrong object at first close (first exec failure): joint 8, text 14. This matches the intent gain.
- Wrong-arm participation: 3 for both. It was 19 for joint at batch 32.
- Most remaining joint failures come after the task is done: gripper re-closed after completion (12) and an arm leaving home again (13). Text has 26 "left home again".

## End-of-episode hold samples (text_p256_bs128_hold vs text_p256_bs128, L0)

`rf_text_p256_bs128_hold` is `rf_text_p256_bs128` plus `hold_pad: 80`: training start frames are drawn up to 80 frames past the episode end, where frames and actions repeat the final state (about 35% of samples). It targets the "arm leaves home again" failures.

| L0, n=75 | intent | exec | completion |
|---|---|---|---|
| correct: hold / no hold (diff) | 0.512 / 0.503 (+0.009 [−0.065, +0.089]) | 0.457 / 0.483 (−0.025 [−0.104, +0.059]) | 0.413 / 0.507 (−0.093 [−0.200, +0.027]) |
| shuffled: hold / bs32 text_p256 | 0.060 / 0.072 | 0.035 / 0.056 | 0.027 / 0.013 |

First intent failures: "left/right arm left home again" 22 with hold vs 26 without. Wrong-arm participation rose from 3 to 12. Hold padding does not fix the restart failures and does not help L0. L1–L3 hold evals are still running.

## Why L0 is low: first failing stage on text_p256 (75 trials)

- 17 (23%): the arm leaves home again after finishing (`stage3_finish: left arm left home again; gripper re-closed`). The evaluator always runs 10 calls × 50 actions = 500 steps, while training episodes have median length 190, so the policy restarts after completing.
- 11: the source object is not lifted (`source lift < 1 cm`).
- 10: the wrong object at first close.
- 7 wrong-arm participation (joint_p256: 19).

Training budget is also small: 640k samples, an AgiBot-pretrained base and a freshly initialized 14-D action head.

## Follow-ups running

- Shuffled-instruction L0 and L2 evals for text_p256_bs128 and joint_p256_bs128, to compare their language dependence.
- Hold-sample L2 and L3 eval (paused at 208/393).
- Speed: joint steps cost 1.03 s per microstep at 2 samples/GPU. Qwen planner forward takes 38%, DiT backward 41%, DiT forward 13%, SigLIP2 teacher 4%, VAE + T5 3%. Packing the three views into two Qwen rows (bitwise-identical outputs) removed a third of the Qwen rows.

## Official full training protocol (2026-10-08)

The next baseline uses all 75 official L0 training tasks, 50 demonstrations per task (3,750 episodes). The old internal development set is now included in training; L1–L3 benchmark evaluation episodes remain evaluation-only. The previous 2,896-train / 850-dev manifest remains unchanged for old-run reproducibility.

Four scene4 yellow-cylinder episodes excluded by an earlier action-only duplicate check were restored: their actions match another task's episodes but their images differ. Full manifests preserve all official episode identities and fail if any task has fewer than 50 valid demonstrations. All retained files match the previously completed all-frame audit, and all frames of the four restored files were independently decoded. Normalization statistics were recomputed from all 3,750 episodes (677,823 joint state/action pairs).

Source manifest: `/data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_q35/data_official3750/manifest.json`, SHA256 `a52284ae05690089b680aad57226f3ab50051dc2fd94bebd961598f4321b4a8a`. Coverage and image-audit provenance are saved alongside it.

The Accelerate skipped-loader epoch reset was reproduced and fixed in the shared trainer: the newly created epoch loader now receives the current epoch before iteration. Related data, loading and epoch tests passed (27 tests).

Planned baseline: `q35_text_p256_official3750_30k_hpc3.yaml`, fresh base initialization, 30,000 optimizer updates, 1 node / 8 GPUs, 4 samples/GPU and accumulation 4 (global batch 128), LR 3e-5. HPC3 snapshot: `/data/user/jhe724/workspace/VLM4WAM_robofollow_official3750`. Full data destination: `/data/user/jhe724/datasets/RoboFollow-official3750`; hashes are verified for every HDF5 and instruction JSON file before publishing `DATA_READY`.

At preparation time both HPC3 Slurm controllers were DOWN, so no new training job or result exists yet. Background data transfer and a guarded submission waiter run on the Mac as LaunchAgents; after data/config verification and scheduler recovery the waiter resolves jhe724's live Slurm default account and submits exactly one baseline. Live state is in the snapshot's `.agent/official3750/jobid` and the Mac logs under `~/.local/state/robofollow-official3750/`.

Update during preparation: the primary controller returned UP, but live `sinfo` exposes only the CPU management partition `mgt`. GPU partition `acd_u` remains absent, so training is still blocked. The submission waiter explicitly requires an UP GPU partition plus successful `sbatch --test-only`; it never submits GPU training to management nodes. The first migrated task (50 HDF5 + 50 instruction JSON files) passed target SHA256 verification. HPC3 CPU runtime imports and the vendored RGB codec passed verification.

Additional source-data smoke check passed: five real padded batches, one from each scene plus restored yellow-cylinder episode2, have video shape `[3,3,13,256,320]`, actions `[58,14]`, state `[1,14]`, finite values and nonempty captions. Offline submission guard checks passed for missing data, scheduler outage, missing GPU partition, unavailable account, rejected resources and existing job ID. Transfer/submission logs must be checked for current completion; setup does not mean training has started.

## Two-node / 32-action baseline revision (2026-10-08)

The user requested two nodes and a shorter action chunk. Both new full-data configs now predict and execute 32 absolute-joint actions per observation, with four history frames/actions spaced by 32 actions. The loader yields 36 action rows; only the last 32 are future targets. Nine future visual samples are retained for the existing six-latent-frame VAE/planner contract and selected at action offsets 3, 7, 10, 14, 17, 21, 24, 28, 31. Effective sampled-video FPS is 4.6875 from the 250/15 source control rate. Action statistics and the 3,750 raw episodes are unchanged.

The fresh text baseline uses 2 nodes / 16 GPUs, per-device batch 4, accumulation 2, global batch 128 and 30,000 optimizer steps. Each Slurm node launches eight workers with its own node rank and a shared master endpoint. Output: `/data/user/jhe724/outputs/robofollow_q35/official3750_text_30k_ac32_bs128`. The joint configuration receives the same timing/batch changes but is not submitted as an additional concurrent job.

For this campaign use the official evaluator with `--max-steps 16 --actions-per-step 32`; the policy's configured 500-action budget returns 32 actions for the first 15 calls and 20 on the final call. The evaluator already accepts variable-length action sequences; scoring code is unchanged. Ten 32-action calls would shorten the episode budget to 320 and must not be compared as an equivalent run. Old 54-action checkpoints have a different history-stride contract and cannot be silently used with this configuration. The existing complete-v3 autoresearch ledger/evaluation manifest is unchanged; these full-data baselines are a separate campaign.

No GPU training or benchmark gain is claimed by this configuration change. Live job/data status remains authoritative in `.agent/official3750/jobid` and the transfer/submission logs.

Verification for this revision: 36 data/policy/checkpoint/epoch/FPS tests passed on both Ola and the HPC3 snapshot. Two valid node ranks, three invalid allocation guards and six guarded submission paths passed without starting workers or submitting a job. Full 16-GPU distributed training and benchmark rollout remain unverified until GPU resources return.

Five real 32-action padded samples also passed: one from each scene and restored scene4 yellow-cylinder episode2; video `[3,3,13,256,320]`, actions `[36,14]`, state `[1,14]`, finite values and nonempty captions.

## Direct HPC3 data download (2026-10-08)

The user requested direct downloading instead of relaying all HDF5 data through the Mac/VPN. HPC3 cannot reach huggingface.co reliably and hf-mirror file downloads timed out. Official Hugging Face file CDN `us.aws.cdn.hf.co` is reachable: eight pinned-revision files totaling 105,137,222 bytes downloaded directly on HPC3 in 82.1 seconds (1.28 MB/s aggregate), with all SHA256 values matching the independently audited source manifest. This small probe does not establish the sustained full-dataset speed. HPC3 curl 7.61.1 rejected `--retry-all-errors`; removing that unsupported flag fixed the network probe.

The background transfer LaunchAgent now runs `transfer_direct.py`, resolving temporary CDN URLs for revision `bbb1e266ed585f1557773de5a3f1e3b4cd944f97`. HPC3's `.agent/official3750/download_task.py` downloads eight HDF5 files concurrently, reuses valid existing files, resumes partial downloads, checks SHA256 and atomically publishes each verified file. Only small instruction JSON files and control messages pass through the Mac. All 3,750 HDF5 and instruction files must pass verification before the existing `DATA_READY` guard can enable submission. Three downloader checks passed (reuse, corruption/atomic replacement, and path/URL boundary).

The old relay job was stopped and its script/log retained. Current download log: `~/.local/state/robofollow-official3750/transfer-direct.log`. The Mac must still stay online for link resolution, tiny annotation transfers and the existing guarded submitter; this is not a fully Mac-independent downloader. Training code fingerprints remain unchanged. Live GPU partition recovery is still required before the two-node/16-GPU baseline can start.

## HPC3 download concurrency tuning (2026-10-08)

In response to the request for faster download, the live download worker was atomically updated from 8 to 24 connections; the Mac coordinator and training code were not restarted. Recent 8-worker complete batches measured 3.79, 5.68 and 3.99 MB/s. The first 24-worker batch measured 592,131,835 bytes in 302.6 seconds (1.96 MB/s): three slow connections dominated completion time, so concurrency alone has not established a speedup. An initial 64KB/s / 15-second cutoff caused curl timeout on a few files; the existing task retry retained 47 verified files and recovered the remaining files.

The final worker uses separate curl invocations for up to five attempts, recalculating the resume offset each time, with a 60-second request limit and a 16KB/s / 20-second slow-connection threshold. This avoids curl internal retries discarding new partial progress. SHA256 and atomic publication remain required. A regression test reproducing timeout followed by resumed completion failed before the fix and passed afterward; all four downloader checks pass on Mac and HPC3. The final retry policy takes effect at the next worker invocation. Sustained throughput improvement is still unverified. Background transfer continues; 31 tasks / 1,550 episodes were fully verified at recording time.

## Official3750 baseline extended to 53,000 steps (2026-10-09)

The user requested 53,000 optimizer steps, approximately ten equivalent passes over 677,823 time-start positions at nominal global batch 128. The active text baseline now uses `q35_text_p256_official3750_53k_hpc3.yaml` and output `/data/user/jhe724/outputs/robofollow_q35/official3750_text_53k_ac32_bs128`. The old 30k configurations remain historical; the joint configuration was not changed. The launcher, Slurm job/log names, data preflight, submission message and code fingerprints were updated together. Two nodes / 16 GPUs, action chunk 32 and checkpoint interval 2,500 remain configured; the trainer also saves the final step 53,000 checkpoint.

The existing submission service was stopped during the update after confirming no job ID or submission intent. All prior code fingerprints matched. Launcher and six submission-guard checks passed; HPC3 verified new fingerprints, full manifest/statistics and five real training samples with the 53k configuration. The Mac automatic submission service was restarted and repeats preflight before waiting for GPU availability. The dataset is fully verified; GPU execution is not yet validated.

## HPC3-native automatic submission (2026-10-09)

The submission watcher now runs entirely on HPC3 / jhe724 in `/data/user/jhe724/workspace/VLM4WAM_robofollow_official3750`, using tmux session `rf3750-submit` on ACD-Manage-2. It invokes local Slurm commands, not SSH through Mac. The Mac LaunchAgent `com.junjie.robofollow.official3750.submit` was removed. The watcher is protected by `watcher.lock`; existing `submit.lock`, durable SUBMITTING intent and jobid guards prevent duplicate submission.

Log: `.agent/official3750/submit-hpc3.log`. The watcher checks every 300 seconds for partition `acd_u`; after code/data preflight and resource validation it submits the 53,000-step baseline using two nodes / 16 GPUs, global batch 128 and action chunk 32. Both local and HPC3 runs passed all six submission-guard checks. Detached tmux process ancestry was verified; it does not require the Mac, VPN or SSH session to stay alive. A login-host restart would terminate tmux and require restarting the watcher; a successfully submitted Slurm job is managed by Slurm independently.
