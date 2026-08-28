"""Probe B: can a FRESH matching head, trained with negative-query (contrastive)
supervision, achieve QUERY-SPECIFIC localization on the cosmos latent? If yes,
the match-ground architecture is fine and v1 failed only on the objective
(per-sample BCE shortcut). If shuffled-AUC stays ~= own-AUC even after
contrastive training, the ③ feature space lacks discriminative power.

Pipeline: cache (clean-ish, t=0.1) patched latent tokens x[N,Np,D] + pooled ③
query[N,256] + patched GT mask[N,Np] for many samples (reusing the trained net's
patch-embedding, NO blocks). Then train a small head from scratch:
  logit(i,j) = <match_k(x_i), match_q(q_j)>      (token-wise)
  positive: BCE(logit(i,i), mask_i)   negative: BCE(logit(i,j!=i), 0)  in-batch
Eval held-out: AUC_own (q_i on x_i) vs AUC_shuf (q_j on x_i).
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
import torch.nn as nn
import torch.nn.functional as F

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate

CKPT = (
    "/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_v21_match_ground_2000_vlm4wam/"
    "cosmos_predict_v2p5/video2world/2b_droid_success_v21_match_ground_cap200_49f_bs2accum8_gbs128_2000/"
    "checkpoints/iter_000002000/model_ema_bf16.pt"
)
EXPERIMENT = "predict2_video2world_training_2b_droid_success_v21_match_ground"
CONFIG = "cosmos_predict2/_src/predict2/configs/video2world/config.py"
N = int(os.environ.get("PROBE_N", "400"))
T_FIX = float(os.environ.get("PROBE_T", "0.1"))  # low noise: test the ceiling
CACHE = "/data/user/jhe724/workspace/VLM4WAM/mg_eval/probe_b_cache.pt"


def pool_query(dense_B_L_D):
    feat = torch.nan_to_num(dense_B_L_D.float())
    valid = feat.abs().sum(dim=-1, keepdim=True) > 0
    denom = valid.sum(dim=1).clamp_min(1).float()
    return (feat * valid).sum(dim=1) / denom  # [B,256]


def build_cache():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pipe = Video2WorldInference(
        experiment_name=EXPERIMENT, ckpt_path=CKPT, s3_credential_path="",
        config_file=CONFIG, offload_text_encoder=True, offload_tokenizer=False,
    )
    model = pipe.model
    net = model.net
    net.eval()
    net_dtype = next(net.parameters()).dtype
    rf = model.rectified_flow
    tkw = model.tensor_kwargs
    tkw32 = model.tensor_kwargs_fp32
    ds = lazy_instantiate(pipe.config.dataloader_train.dataset)
    print(f"dataset size={len(ds)} caching N={N} at t={T_FIX}", flush=True)

    Xs, Qs, Ms = [], [], []
    gen = torch.Generator(device="cuda").manual_seed(0)
    step = max(1, len(ds) // (N * 3))
    i = 0
    while len(Xs) < N and i < len(ds):
        idx = i
        i += step
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
        with torch.no_grad():
            x0, latent, condition = model.get_data_and_condition(db)
            mask_lat = condition.target_mask_B_C_T_H_W.float().clamp(0, 1)[0, 0]
            q = pool_query(condition.target_dense_feature_B_L_D)[0]  # [256]
            t_B = torch.full((1, 1), T_FIX, **tkw32)
            timesteps = rf.get_discrete_timestamp(t_B, tkw32)
            sigmas = rf.get_sigmas(timesteps, tkw32).reshape(1, 1)
            eps = torch.randn(latent.shape, generator=gen, **tkw32)
            xt, _ = rf.get_interpolation(eps, latent, sigmas)
            B, C, T, H, W = xt.shape
            mask_ch = torch.zeros(B, 1, T, H, W, device="cuda", dtype=net_dtype)
            x_in = torch.cat([xt.to(net_dtype), mask_ch], dim=1)
            fps = condition.fps.to(device="cuda", dtype=net_dtype) if condition.fps is not None else None
            pad = condition.padding_mask.to(device="cuda", dtype=net_dtype) if condition.padding_mask is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x_BTHWD, _, _ = net.prepare_embedded_sequence(
                    x_in, fps=fps, padding_mask=pad, target_mask_B_C_T_H_W=condition.target_mask_B_C_T_H_W
                )
            Tx, Hx, Wx, D = x_BTHWD.shape[1], x_BTHWD.shape[2], x_BTHWD.shape[3], x_BTHWD.shape[4]
            mask_p = F.interpolate(mask_lat[None, None], size=(Tx, Hx, Wx), mode="nearest")[0, 0]
            # use the MIDDLE frame's tokens (target present, single frame keeps it small)
            tf = Tx // 2
            Xs.append(x_BTHWD[0, tf].reshape(-1, D).float().cpu())  # [Hx*Wx, D]
            Ms.append((mask_p[tf].reshape(-1) > 0.5).float().cpu())  # [Hx*Wx]
            Qs.append(q.cpu())
        if len(Xs) % 50 == 0:
            print(f"  cached {len(Xs)}/{N}", flush=True)
    X = torch.stack(Xs)  # [N, P, D]
    Q = torch.stack(Qs)  # [N, 256]
    M = torch.stack(Ms)  # [N, P]
    torch.save({"X": X, "Q": Q, "M": M}, CACHE)
    print(f"cached X={tuple(X.shape)} Q={tuple(Q.shape)} M={tuple(M.shape)} -> {CACHE}", flush=True)
    return X, Q, M


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


def train_eval(X, Q, M, contrastive, center=False):
    torch.manual_seed(0)
    N0, P, D = X.shape
    ntr = int(N0 * 0.8)
    if center:
        # cross-sample centering: subtract the TRAIN-set mean query (removes the
        # dominant common-mode direction, exposing the object-specific residual)
        qmean = Q[:ntr].mean(0, keepdim=True)
        Q = Q - qmean
    Xtr, Qtr, Mtr = X[:ntr].cuda(), Q[:ntr].cuda(), M[:ntr].cuda()
    Xte, Qte, Mte = X[ntr:].cuda(), Q[ntr:].cuda(), M[ntr:].cuda()
    d = 128
    h = 512
    mk = nn.Sequential(nn.Linear(D, h), nn.GELU(), nn.Linear(h, d)).cuda()
    mq = nn.Sequential(nn.Linear(256, h), nn.GELU(), nn.Linear(h, d)).cuda()
    qn = nn.LayerNorm(256, elementwise_affine=False).cuda()
    opt = torch.optim.Adam(list(mk.parameters()) + list(mq.parameters()), lr=1e-3)
    scale = d ** -0.5
    for ep in range(300):
        perm = torch.randperm(ntr, device="cuda")
        b = 64
        for s0 in range(0, ntr, b):
            bi = perm[s0:s0 + b]
            xb, qb, mb = Xtr[bi], Qtr[bi], Mtr[bi]  # [B,P,D],[B,256],[B,P]
            k = mk(xb)                                # [B,P,d]
            qv = mq(qn(qb))                           # [B,d]
            pos_logit = (k @ qv.unsqueeze(-1)).squeeze(-1) * scale  # [B,P] own query
            loss = F.binary_cross_entropy_with_logits(pos_logit, mb)
            if contrastive:
                # in-batch negative: a rolled (wrong) query should NOT light up mb
                qroll = torch.roll(qv, 1, dims=0)
                neg_logit = (k @ qroll.unsqueeze(-1)).squeeze(-1) * scale
                loss = loss + F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(mb))
            opt.zero_grad()
            loss.backward()
            opt.step()
    # eval held-out
    with torch.no_grad():
        k = mk(Xte)
        qv = mq(qn(Qte))
        own = (k @ qv.unsqueeze(-1)).squeeze(-1) * scale
        qshuf = torch.roll(qv, 1, dims=0)
        shuf = (k @ qshuf.unsqueeze(-1)).squeeze(-1) * scale
    a_own = [auc_pairs(own[i], Mte[i]) for i in range(Xte.shape[0])]
    a_shuf = [auc_pairs(shuf[i], Mte[i]) for i in range(Xte.shape[0])]
    a_own = [a for a in a_own if a is not None]
    a_shuf = [a for a in a_shuf if a is not None]
    return float(np.mean(a_own)), float(np.mean(a_shuf))


def main():
    if os.path.exists(CACHE):
        print(f"loading cache {CACHE}", flush=True)
        d = torch.load(CACHE, map_location="cpu")
        X, Q, M = d["X"], d["Q"], d["M"]
    else:
        X, Q, M = build_cache()
    print(f"\n=== Probe B: fresh-head query-specific localization ceiling (t={T_FIX}) ===", flush=True)
    own0, shuf0 = train_eval(X, Q, M, contrastive=False)
    print(f"BCE-only            : AUC_own={own0:.3f}  AUC_shuf={shuf0:.3f}  gap={own0-shuf0:+.3f}", flush=True)
    own1, shuf1 = train_eval(X, Q, M, contrastive=True)
    print(f"+contrastive        : AUC_own={own1:.3f}  AUC_shuf={shuf1:.3f}  gap={own1-shuf1:+.3f}", flush=True)
    own2, shuf2 = train_eval(X, Q, M, contrastive=False, center=True)
    print(f"+center             : AUC_own={own2:.3f}  AUC_shuf={shuf2:.3f}  gap={own2-shuf2:+.3f}", flush=True)
    own3, shuf3 = train_eval(X, Q, M, contrastive=True, center=True)
    print(f"+center+contrastive : AUC_own={own3:.3f}  AUC_shuf={shuf3:.3f}  gap={own3-shuf3:+.3f}", flush=True)
    print("\nRead: contrastive AUC_own high AND AUC_shuf->0.5 (big gap) => architecture "
          "CAN do query-specific localization; v1 failed on objective => v2 contrastive loss. "
          "If gap stays ~0 => ③ feature lacks discriminative power => change features.", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
