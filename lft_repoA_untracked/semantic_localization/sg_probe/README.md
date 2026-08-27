# SG-WAM feature probe: baseline vs semantic-guided (local, ge-act env)

Cleaner + quantitative alternative to raw action->video attention. Trains a lightweight FiLM+conv
loc-head probe to localize the target noun FROM the SG-WAM's video features (mid block PROBE_BLOCK=16,
8x8 latent, 2048-d), separately on BASELINE (no semantic plan) vs SG (with plan) features. Probe output
upsampled to 32x32; distills CLIPSeg pseudo-GT. Metric: peak-hit (CLIPSeg mask value at probe argmax)
+ pixel correlation (robust to the coarse 8x8 feature grid; hard soft-IoU is ~0 at 8x8).

Result (40 scenes, 80/20 train/test): **peak-hit baseline=0.23 vs SG=0.40 (+78%)**, corr 0.45 vs 0.51.
=> semantic guidance makes the world-model representation more target-localizable. Figure ../figs/sg_probe.png
(top=baseline-feature probe, bottom=SG-feature probe). Model: joint_vlm_geact_action_k4_50k/step_40000/ltx.
Run: ge-act conda env, GPU with headroom. This is the strongest SG>baseline evidence in this line.
