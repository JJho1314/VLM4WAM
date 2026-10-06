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

## Why L0 is low: first failing stage on text_p256 (75 trials)

- 17 (23%): the arm leaves home again after finishing (`stage3_finish: left arm left home again; gripper re-closed`). The evaluator always runs 10 calls × 50 actions = 500 steps, while training episodes have median length 190, so the policy restarts after completing.
- 11: the source object is not lifted (`source lift < 1 cm`).
- 10: the wrong object at first close.
- 7 wrong-arm participation (joint_p256: 19).

Training budget is also small: 640k samples, an AgiBot-pretrained base and a freshly initialized 14-D action head.

## Follow-ups running

- `rf_text_p256_bs128` and `rf_joint_p256_bs128`: global batch 128 on 2 nodes, lr 3e-5, 20k steps.
- `rf_text_p256_bs128_hold`: identical to `rf_text_p256_bs128` except end-of-episode hold samples (`hold_pad: 80`). This targets the "leaves home again" failures.
- Extra rounds for text_p256 and joint_p256 on L0 and L2 (seed 1042, 2 rounds, correct instruction) to tighten the CIs.
- Speed: joint steps cost 1.03 s per microstep at 2 samples/GPU. Qwen planner forward takes 38%, DiT backward 41%, DiT forward 13%, SigLIP2 teacher 4%, VAE + T5 3%. Packing the three views into two Qwen rows (bitwise-identical outputs) removed a third of the Qwen rows.
