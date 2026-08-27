# SG-WAM feature probe — RIGOROUS result (honest, negative)

Setup: 160 scenes (40 tasks x 4 episodes), 5-fold CV (every scene held out once), LINEAR probe
(bilinear feature.text, exposes feature quality; a strong conv probe fits either and hides the gap),
peak-hit metric (CLIPSeg mask value at probe argmax), paired t-test + Wilcoxon.

Result: NO significant SG>baseline difference in target-localizability of the WAM video features.
  - current frame (LATF=0): baseline 0.422 vs SG 0.434, d=+0.012, paired t p=0.30, Wilcoxon p=0.98
  - future frame  (LATF=5): baseline 0.419 vs SG 0.435, d=+0.015, paired t p=0.13, Wilcoxon p=0.14
Sanity: baseline-vs-SG feature rel-diff = 0.14 -> the semantic plan IS applied (not a bug); it changes
the features by 14% but that change does NOT make the target more linearly localizable.

The earlier eye-catching 0.23 vs 0.40 (sg_probe first run) was a small-sample fluke: single 8-scene test
split + a high-capacity FiLM+conv probe overfitting. It did not survive cross-validation.

Interpretation: at step 40000, semantic guidance does not measurably improve the *static target-
localizability* of the video representation (linear, 8x8). Its benefit, if any, may lie elsewhere
(action/trajectory prediction, dynamics) or need a higher-step checkpoint. Do NOT claim SG>baseline
target-awareness from this probe.

## Prediction-error test (also inconclusive / likely confounded)
sg_predict.py: noise the FUTURE latents, predict WITH plan (SG) vs WITHOUT (baseline), measure
flow-matching x0-recon MSE on future latent frames, paired across scenes/noise levels.
Preliminary (n=51): baseline 0.338 vs SG 0.371 -> SG WORSE, 0% win-rate, p=5e-27. This is almost
certainly a CONFOUND, not a real "SG hurts": the plan I inject is GT-future SigLIP (oracle info that
should HELP prediction), so SG being uniformly worse points to an inference-time distribution/format
mismatch (model trained with planner-predicted plans + i2v conditioning; my test uses ad-hoc GT-SigLIP
injection + uniform noising over all frames). Do NOT report this as a finding.

## Bottom line for the paper
Ad-hoc representation probes / recon tests here do NOT cleanly show the semantic-guidance benefit
(probe: no sig diff; prediction: confounded). The semantic guidance's benefit should be measured with
the model's OWN eval pipeline under the correct inference setup: LIBERO task success and/or action-MSE
with the PLANNER providing the plan (planner mode) vs a no-plan baseline, using proper i2v conditioning
and multi-step denoising -- not the ad-hoc probes in this folder.
