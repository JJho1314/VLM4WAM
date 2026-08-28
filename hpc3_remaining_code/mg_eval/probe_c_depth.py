"""Probe C: query-specific localization gap vs DiT block DEPTH.

Hooks blocks [0,4,8,12,16,20], runs the trained net forward at t=0.1 over N
samples (math SDPA forced so the cosmos block attention runs), caches each
depth's middle-frame tokens, then trains an MLP matching head per depth and
reports AUC_own vs AUC_shuf gap. The depth with the largest gap is where v2
should place the matching point.
"""

import os
import contextlib

os.environ.setdefault(
    "DROID_SUCCESS_V21_TAVID_DIR",
    "/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half",
)
os.environ.setdefault("DROID_SUCCESS_V21_TAVID_VAL_DIR", os.environ["DROID_SUCCESS_V21_TAVID_DIR"])

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Force math SDPA for the cosmos block attention: neutralise its local
# sdpa_kernel(backends=[FLASH/CUDNN/EFFICIENT]) context so the global math
# backend applies (those backends reject our inputs -> "No available kernel").
import cosmos_predict2._src.predict2.networks.attention as _attn_mod
_attn_mod.sdpa_kernel = lambda *a, **k: contextlib.nullcontext()
# mem-efficient SDPA: O(S) memory (math would OOM on the ~10^5-token sequence),
# supports bf16. Neutralising the module's local sdpa_kernel context lets this
# global choice apply.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate

CKPT = (
    "/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_v21_match_ground_2000_vlm4wam/"
    "cosmos_predict_v2p5/video2world/2b_droid_success_v21_match_ground_cap200_49f_bs2accum8_gbs128_2000/"
    "checkpoints/iter_000002000/model_ema_bf16.pt"
)
EXPERIMENT = "predict2_video2world_training_2b_droid_success_v21_match_ground"
CONFIG = "cosmos_predict2/_src/predict2/configs/video2world/config.py"
N = int(os.environ.get("PROBE_N", "200"))
T_FIX = float(os.environ.get("PROBE_T", "0.1"))
DEPTHS = [0, 4, 8, 12, 16, 20]


def pool_query(dense_B_L_D):
    feat = torch.nan_to_num(dense_B_L_D.float())
    valid = feat.abs().sum(dim=-1, keepdim=True) > 0
    denom = valid.sum(dim=1).clamp_min(1).float()
    return (feat * valid).sum(dim=1) / denom


def auc_pairs(logit_P, mask_P):
    m = mask_P > 0.5
    pos, neg = logit_P[m], logit_P[~m]
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    if neg.numel() > 4000:
        neg = neg[torch.randperm(neg.numel())[:4000]]
    gt = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean()
    tie = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().mean()
    return (gt + 0.5 * tie).item()


def train_eval(X, Q, M):
    torch.manual_seed(0)
    N0, P, D = X.shape
    ntr = int(N0 * 0.8)
    Xtr, Qtr, Mtr = X[:ntr].cuda().float(), Q[:ntr].cuda().float(), M[:ntr].cuda().float()
    Xte, Qte, Mte = X[ntr:].cuda().float(), Q[ntr:].cuda().float(), M[ntr:].cuda().float()
    d, h = 128, 512
    mk = nn.Sequential(nn.Linear(D, h), nn.GELU(), nn.Linear(h, d)).cuda()
    mq = nn.Sequential(nn.Linear(256, h), nn.GELU(), nn.Linear(h, d)).cuda()
    opt = torch.optim.Adam(list(mk.parameters()) + list(mq.parameters()), lr=1e-3)
    scale = d ** -0.5
    for ep in range(300):
        perm = torch.randperm(ntr, device="cuda")
        for s0 in range(0, ntr, 64):
            bi = perm[s0:s0 + 64]
            xb, qb, mb = Xtr[bi], Qtr[bi], Mtr[bi]
            k = mk(xb)
            qv = mq(qb)
            pos = (k @ qv.unsqueeze(-1)).squeeze(-1) * scale
            loss = F.binary_cross_entropy_with_logits(pos, mb)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        k = mk(Xte); qv = mq(Qte)
        own = (k @ qv.unsqueeze(-1)).squeeze(-1) * scale
        shuf = (k @ torch.roll(qv, 1, 0).unsqueeze(-1)).squeeze(-1) * scale
    ao = [auc_pairs(own[i], Mte[i]) for i in range(Xte.shape[0])]
    as_ = [auc_pairs(shuf[i], Mte[i]) for i in range(Xte.shape[0])]
    ao = [a for a in ao if a is not None]; as_ = [a for a in as_ if a is not None]
    return float(np.mean(ao)), float(np.mean(as_))


def main():
    print(f"N={N} t={T_FIX} depths={DEPTHS}", flush=True)
    pipe = Video2WorldInference(
        experiment_name=EXPERIMENT, ckpt_path=CKPT, s3_credential_path="",
        config_file=CONFIG, offload_text_encoder=True, offload_tokenizer=False,
    )
    model = pipe.model
    net = model.net
    net.eval()
    net_dtype = next(net.parameters()).dtype
    rf = model.rectified_flow
    tkw, tkw32 = model.tensor_kwargs, model.tensor_kwargs_fp32
    model.tensor_kwargs = {**model.tensor_kwargs, "dtype": net_dtype}

    captured = {}
    def mk_hook(di):
        def hook(mod, inp, out):
            captured[di] = out.detach()
        return hook
    for di in DEPTHS:
        net.blocks[di].register_forward_hook(mk_hook(di))

    ds = lazy_instantiate(pipe.config.dataloader_train.dataset)
    print(f"dataset size={len(ds)}", flush=True)

    feats = {di: [] for di in DEPTHS}
    Qs, Ms = [], []
    gen = torch.Generator(device="cuda").manual_seed(0)
    step = max(1, len(ds) // (N * 3))
    i = 0
    while len(Qs) < N and i < len(ds):
        idx = i; i += step
        try:
            s = ds[idx]
        except Exception:
            continue
        if s.get("target_feature") is None or s.get("target_dense_feature") is None or s.get("target_mask") is None:
            continue
        if float(s["target_mask"].float().sum()) <= 0:
            continue
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
            "t5_text_embeddings": torch.zeros(1, 1, 100352, **tkw),
            "t5_text_mask": torch.ones(1, 1, dtype=torch.long).cuda(),
        }
        try:
            with torch.no_grad():
                x0, latent, condition = model.get_data_and_condition(db)
                mask_lat = condition.target_mask_B_C_T_H_W.float().clamp(0, 1)[0, 0]
                q = pool_query(condition.target_dense_feature_B_L_D)[0]
                t_B = torch.full((1, 1), T_FIX, **tkw32)
                timesteps = rf.get_discrete_timestamp(t_B, tkw32)
                sigmas = rf.get_sigmas(timesteps, tkw32).reshape(1, 1)
                timesteps = timesteps.reshape(1, 1)
                eps = torch.randn(latent.shape, generator=gen, **tkw32)
                xt, _ = rf.get_interpolation(eps, latent, sigmas)
                captured.clear()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model.denoise(noise=eps, xt_B_C_T_H_W=xt, timesteps_B_T=timesteps.to(net_dtype), condition=condition)
        except Exception as e:
            if not globals().get("_pe"):
                import traceback; traceback.print_exc(); globals()["_pe"] = True
                print("FIRST_EXC:", repr(e), flush=True)
            continue
        if any(di not in captured for di in DEPTHS):
            continue
        any_d = captured[DEPTHS[0]]
        Bx, Tx, Hx, Wx, D = any_d.shape
        tf = Tx // 2
        mask_p = F.interpolate(mask_lat[None, None], size=(Tx, Hx, Wx), mode="nearest")[0, 0]
        Ms.append((mask_p[tf].reshape(-1) > 0.5).float().cpu())
        Qs.append(q.cpu())
        for di in DEPTHS:
            feats[di].append(captured[di][0, tf].reshape(Hx * Wx, -1).half().cpu())
        if len(Qs) % 25 == 0:
            print(f"  cached {len(Qs)}/{N}", flush=True)

    Q = torch.stack(Qs); M = torch.stack(Ms)
    print(f"\n=== Probe C: localization gap vs depth (MLP head, t={T_FIX}, N={len(Qs)}) ===", flush=True)
    print(f"{'depth':>6} | {'AUC_own':>8} {'AUC_shuf':>9} {'gap':>7}", flush=True)
    for di in DEPTHS:
        X = torch.stack(feats[di])
        own, shuf = train_eval(X, Q, M)
        print(f"{di:6d} | {own:8.3f} {shuf:9.3f} {own-shuf:+7.3f}", flush=True)
    print("\nLargest gap depth => best matching point for v2.", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
