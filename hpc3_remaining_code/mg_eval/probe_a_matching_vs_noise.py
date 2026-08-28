"""Probe A: does the TRAINED match-ground head localize the target in the cosmos
latent, and how does that degrade with diffusion noise sigma?

Teacher-forcing sweep: encode a real video -> clean latent x0; for a grid of t
(=> sigma), build xt, run the net forward (matching logits are a side effect),
compare logits to the GT mask on the latent token grid. Controls: real (3)
query vs a SHUFFLED (3) from another sample vs ZERO (3) -> isolates whether the
head's localization is feature-specific, not reading position from elsewhere.
"""

import os
import sys

os.environ.setdefault(
    "DROID_SUCCESS_V21_TAVID_DIR",
    "/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half",
)
os.environ.setdefault("DROID_SUCCESS_V21_TAVID_VAL_DIR", os.environ["DROID_SUCCESS_V21_TAVID_DIR"])

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_predict2._src.predict2.conditioner import DataType

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_v21_match_ground_2000_vlm4wam/"
    "cosmos_predict_v2p5/video2world/2b_droid_success_v21_match_ground_cap200_49f_bs2accum8_gbs128_2000/"
    "checkpoints/iter_000002000/model_ema_bf16.pt"
)
EXPERIMENT = "predict2_video2world_training_2b_droid_success_v21_match_ground"
CONFIG = "cosmos_predict2/_src/predict2/configs/video2world/config.py"
N = int(os.environ.get("PROBE_N", "24"))
T_GRID = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97]


def auc_inside_outside(logits_thw, mask_thw):
    """Rank-AUC P(logit_inside > logit_outside) over frames with both classes."""
    aucs, p_in, p_out = [], [], []
    T = logits_thw.shape[0]
    for t in range(T):
        m = mask_thw[t].reshape(-1) > 0.5
        lg = logits_thw[t].reshape(-1).float()
        pos, neg = lg[m], lg[~m]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        # AUC via rank: fraction of (pos,neg) pairs with pos>neg (subsample neg for speed)
        if neg.numel() > 2000:
            idx = torch.randperm(neg.numel(), device=neg.device)[:2000]
            neg = neg[idx]
        gt = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean()
        tie = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().mean()
        aucs.append((gt + 0.5 * tie).item())
        p_in.append(torch.sigmoid(pos).mean().item())
        p_out.append(torch.sigmoid(neg).mean().item())
    if not aucs:
        return None
    return float(np.mean(aucs)), float(np.mean(p_in)), float(np.mean(p_out))


def main():
    print(f"ckpt={CKPT}\nexperiment={EXPERIMENT}\nN={N}", flush=True)
    # Force the math SDPA backend GLOBALLY (persistent toggles propagate through
    # the blocks' activation-checkpoint + dynamo-compiled regions, unlike the
    # sdp_kernel context manager). Fixes "No available kernel" on fp32/bf16 attn.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pipe = Video2WorldInference(
        experiment_name=EXPERIMENT, ckpt_path=CKPT, s3_credential_path="",
        config_file=CONFIG, offload_text_encoder=True, offload_tokenizer=False,
    )
    model = pipe.model
    model.net.eval()
    rf = model.rectified_flow
    tkw = model.tensor_kwargs
    tkw32 = model.tensor_kwargs_fp32
    net_dtype = next(model.net.parameters()).dtype
    print(f"net_dtype={net_dtype} | tensor_kwargs={model.tensor_kwargs}", flush=True)

    net = model.net

    def matching_logits(xt_bf16, cond, dense_in):
        """Compute ONLY the match-ground WHERE logits: patch-embed the noised
        latent (prepare_embedded_sequence, no attention) then the trained
        match_k(x).match_q(query) dot product. This is exactly the localization
        signal the head produces; it runs BEFORE the DiT blocks, so we skip the
        whole block/attention/checkpoint machinery (and the WHAT-branch MHA,
        which does not affect the logits). num_conditional_frames=0 => the LVG
        mask channel is zeros, uniform sigma across all frames."""
        B, C, T, H, W = xt_bf16.shape
        mask_ch = torch.zeros(B, 1, T, H, W, device=xt_bf16.device, dtype=xt_bf16.dtype)
        x_in = torch.cat([xt_bf16, mask_ch], dim=1)
        fps = cond.fps.to(device="cuda", dtype=xt_bf16.dtype) if cond.fps is not None else None
        pad = cond.padding_mask.to(device="cuda", dtype=xt_bf16.dtype) if cond.padding_mask is not None else None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x_BTHWD, _, _ = net.prepare_embedded_sequence(
                x_in, fps=fps, padding_mask=pad, target_mask_B_C_T_H_W=cond.target_mask_B_C_T_H_W
            )
            mod = net.target_match_ground_module
            Bx, Tx, Hx, Wx, D = x_BTHWD.shape
            pdt = mod.match_q.weight.dtype
            query, q_valid = mod._pool_valid_rows(dense_in, Bx)
            has_q = (q_valid.sum(dim=1) > 0).float().view(Bx, 1)
            q = mod.match_q(mod.query_norm(query.to(device="cuda", dtype=pdt)))           # [B,M]
            k = mod.match_k(x_BTHWD.reshape(Bx, -1, D).to(pdt))                            # [B,N,M]
            logits = (k @ q.unsqueeze(-1)).squeeze(-1) * mod.match_scale                  # [B,N]
        return logits.view(Bx, Tx, Hx, Wx).float()

    ds = lazy_instantiate(pipe.config.dataloader_train.dataset)
    print(f"dataset size={len(ds)}", flush=True)

    # gather N samples' raw tensors
    samples = []
    for i in range(0, len(ds), max(1, len(ds) // (N * 3))):
        if len(samples) >= N:
            break
        try:
            s = ds[i]
        except Exception:
            continue
        if s.get("target_feature") is None or s.get("target_dense_feature") is None or s.get("target_mask") is None:
            continue
        if float(s["target_mask"].float().sum()) <= 0:
            continue
        samples.append(s)
    print(f"usable samples={len(samples)}", flush=True)

    # accumulators: per-t-grid, per-condition
    acc = {c: {t: [] for t in T_GRID} for c in ("real", "shuffled", "zero")}
    pin = {t: [] for t in T_GRID}
    pout = {t: [] for t in T_GRID}

    gen = torch.Generator(device="cuda").manual_seed(0)
    for si, s in enumerate(samples):
        video = s["video"]
        if video.dtype != torch.uint8:
            video = (video.clamp(0, 1) * 255).to(torch.uint8) if video.max() <= 1.5 else video.to(torch.uint8)
        db = {
            model.input_data_key: video.unsqueeze(0).cuda(),
            "target_mask": s["target_mask"].unsqueeze(0).cuda(),
            "target_feature": s["target_feature"].unsqueeze(0).cuda(),
            "target_dense_feature": s["target_dense_feature"].unsqueeze(0).cuda(),
            "fps": torch.tensor([s.get("fps", 16)]).cuda(),
            "padding_mask": s["padding_mask"].unsqueeze(0).cuda(),
            "num_conditional_frames": 0,
            # zero text (matching head is text-independent). crossattn_proj is
            # Linear(100352,1024): one context token of dim 49*2048=100352.
            "t5_text_embeddings": torch.zeros(1, 1, 100352, **tkw),
            "t5_text_mask": torch.ones(1, 1, dtype=torch.long).cuda(),
        }
        with torch.no_grad():
            x0, latent, condition = model.get_data_and_condition(db)
        mask_lat = condition.target_mask_B_C_T_H_W.float().clamp(0, 1)[0, 0]  # [Tl,Hl,Wl]
        feat = condition.target_feature_B_L_D
        dense = condition.target_dense_feature_B_L_D
        # shuffled control: dense (3) query from a different sample
        other = samples[(si + 1) % len(samples)]
        dense_shuf = other["target_dense_feature"].unsqueeze(0).cuda().to(dense.dtype)
        dense_zero = torch.zeros_like(dense)

        eps = torch.randn(latent.shape, generator=gen, **tkw32)
        for t in T_GRID:
            t_B = torch.full((1, 1), t, **tkw32)
            timesteps = rf.get_discrete_timestamp(t_B, tkw32)
            sigmas = rf.get_sigmas(timesteps, tkw32)
            timesteps = timesteps.reshape(1, 1)
            sigmas = sigmas.reshape(1, 1)
            xt, _ = rf.get_interpolation(eps, latent, sigmas)

            xt_bf16 = xt.to(device="cuda", dtype=net_dtype)
            for cond_name, dense_in in (("real", dense), ("shuffled", dense_shuf), ("zero", dense_zero)):
                lg = matching_logits(xt_bf16, condition, dense_in)
                if lg is None:
                    continue
                lg = lg.unsqueeze(0) if lg.dim() == 3 else lg
                # logits live on the PATCHED token grid (H/2,W/2); downsample the
                # latent-grid GT mask to match (same as the net's patched mask).
                Tx, Hx, Wx = lg.shape[1], lg.shape[2], lg.shape[3]
                mask_p = F.interpolate(
                    mask_lat.unsqueeze(0).unsqueeze(0), size=(Tx, Hx, Wx), mode="nearest"
                )[0, 0]
                res = auc_inside_outside(lg[0], mask_p)
                if res is None:
                    continue
                acc[cond_name][t].append(res[0])
                if cond_name == "real":
                    pin[t].append(res[1])
                    pout[t].append(res[2])
        if (si + 1) % 5 == 0:
            print(f"  processed {si+1}/{len(samples)}", flush=True)

    print("\n=== Probe A: matching localization AUC vs noise t (0=clean,1=pure noise) ===", flush=True)
    print(f"{'t':>5} | {'AUC_real':>9} {'AUC_shuf':>9} {'AUC_zero':>9} | {'p_in':>6} {'p_out':>6} {'n':>4}", flush=True)
    for t in T_GRID:
        ar = np.mean(acc['real'][t]) if acc['real'][t] else float('nan')
        ash = np.mean(acc['shuffled'][t]) if acc['shuffled'][t] else float('nan')
        az = np.mean(acc['zero'][t]) if acc['zero'][t] else float('nan')
        pi = np.mean(pin[t]) if pin[t] else float('nan')
        po = np.mean(pout[t]) if pout[t] else float('nan')
        print(f"{t:5.2f} | {ar:9.3f} {ash:9.3f} {az:9.3f} | {pi:6.3f} {po:6.3f} {len(acc['real'][t]):4d}", flush=True)
    print("\nInterpretation: AUC_real>>0.5 at low t => head DOES localize on clean "
          "latent (failure was noise/gate/redundancy, fixable). AUC_real~=AUC_shuf "
          "=> not feature-specific. AUC_real~0.5 even at low t => alignment never "
          "formed (needs architecture change).", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
