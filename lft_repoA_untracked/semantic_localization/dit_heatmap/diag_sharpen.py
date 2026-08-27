"""Why is the plan cross-attention uniform? Two very different diseases, two different cures.

  A) the logits themselves carry no contrast (std of q.k is tiny) -> softmax CANNOT sharpen;
     this is a scale/temperature or representation-collapse problem.
  B) the logits do vary but point nowhere in particular -> no spatial correspondence between video
     positions and plan positions; a RoPE-alignment / supervision problem.

Measures, from the real q,k inside SemanticRopeCrossAttention:
  * logit std, top1-minus-mean gap (contrast)
  * entropy at several temperatures (how sharp COULD it get)
  * spatial correspondence: does video position (h,w) attend to the matching plan position?
    -> correlation between a query's attention centroid on the 27x27 plan grid and its own
       normalized (h,w). Zero correlation = no spatial routing at all.
"""
import os, sys, math
from pathlib import Path
import numpy as np, torch

COSMOS = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/cosmos-predict2.5"
sys.path.insert(0, COSMOS)
W = "/data/LFT-W02_data/junjie/weights"
os.environ.setdefault("COSMOS_HF_LOCAL_DIRS", W)
os.environ.setdefault("COSMOS_LOCAL_MODEL_DIR", f"{W}/Cosmos-Predict2.5-2B")
os.environ.setdefault("SEMANTIC_PLAN_ONLINE_ENCODER_PATH",
                      "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/siglip2-so400m-patch14-384")
os.environ.setdefault("SEMANTIC_PLAN_NUM_KEYFRAMES", "5")     # must match iter3000 training
os.environ.setdefault("SEMANTIC_PLAN_SPATIAL_GRID", "0")
HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization")
FRAME = str(HERE / "oracle_repro/yc74616_f0.png")
PLAN = str(HERE / "oracle_repro/yc74616_s0_oracle.pt")
CKPT = f"{W}/cosmos_semantic_plan_iter3000/model_ema_bf16.pt"
OUT = HERE / "dit_heatmap"
PROMPT = ("A Franka robotic arm with a parallel-jaw gripper carefully grasp only the [TGT] yellow "
          "carrot in the sink basin and place it into the black pot next to the banana, without moving the banana.")
BLOCKS = [20, 27]
H, W_, TEARLY = 20, 36, 3
NKF, G = 5, 27
TEMPS = [1.0, 0.5, 0.2, 0.1, 0.05]


def main():
    from cosmos_predict2.config import SetupArguments, DEFAULT_MODEL_KEY
    from cosmos_predict2.inference import Inference
    setup = SetupArguments(
        experiment="predict2_video2world_training_2b_droid_semantic_plan_320x576_93f",
        checkpoint_path=CKPT, config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        output_dir=OUT / "gen", context_parallel_size=1,
        offload_diffusion_model=False, offload_text_encoder=True, offload_tokenizer=True,
        disable_guardrails=True, model=DEFAULT_MODEL_KEY.name)
    inf = Inference(setup)
    net = inf.pipe.model.net
    res = {b: None for b in BLOCKS}

    def wrap(mod, bi):
        orig = mod.compute_qkv
        def patched(x, context=None, query_rope_emb=None, key_rope_emb=None):
            q, k, v = orig(x, context, query_rope_emb=query_rope_emb, key_rope_emb=key_rope_emb)
            with torch.no_grad():
                hh, dd = q.shape[-2], q.shape[-1]
                qf = q.reshape(1, -1, hh, dd).float()[0]      # (Sq,h,d)
                kf = k.reshape(1, -1, hh, dd).float()[0]      # (L,h,d)
                Sq, L = qf.shape[0], kf.shape[0]
                T = Sq // (H * W_)
                sel = np.arange(TEARLY * H * W_)              # early frames only, keeps it light
                scale = 1.0 / math.sqrt(dd)
                # plan grid coords for the centroid test
                kf_idx = torch.arange(L, device=qf.device)
                gy = ((kf_idx % (G * G)) // G).float() / (G - 1)
                gx = ((kf_idx % (G * G)) % G).float() / (G - 1)
                stats = dict(std=0.0, gap=0.0, ent={t: 0.0 for t in TEMPS})
                cy = torch.zeros(len(sel), device=qf.device); cx = torch.zeros_like(cy)
                for h in range(hh):
                    lg = qf[sel, h] @ kf[:, h].T * scale       # (S,L) raw logits
                    stats["std"] += float(lg.std())
                    stats["gap"] += float((lg.max(-1).values - lg.mean(-1)).mean())
                    for t in TEMPS:
                        p = torch.softmax(lg / t, dim=-1)
                        stats["ent"][t] += float((-(p * (p + 1e-9).log()).sum(-1)).mean())
                    p1 = torch.softmax(lg, dim=-1)
                    cy += p1 @ gy; cx += p1 @ gx
                for kk in ("std", "gap"): stats[kk] /= hh
                for t in TEMPS: stats["ent"][t] /= hh
                cy /= hh; cx /= hh
                qy = (torch.arange(len(sel), device=qf.device) % (H * W_)) // W_
                qx = (torch.arange(len(sel), device=qf.device) % (H * W_)) % W_
                qy = qy.float() / (H - 1); qx = qx.float() / (W_ - 1)
                def corr(a, b):
                    a = a - a.mean(); b = b - b.mean()
                    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
                stats.update(L=L, Sq=Sq, T=T, heads=hh, dim=dd,
                             corr_y=corr(cy, qy), corr_x=corr(cx, qx),
                             cy_std=float(cy.std()), cx_std=float(cx.std()),
                             lnL=float(np.log(L)))
                res[bi] = stats
            return q, k, v
        mod.compute_qkv = patched

    for bi in BLOCKS:
        for _, sub in net.blocks[bi].named_modules():
            if type(sub).__name__ == "SemanticRopeCrossAttention":
                wrap(sub, bi)
    inf.pipe.generate_vid2world(prompt=PROMPT, input_path=FRAME, guidance=7, num_video_frames=49,
                                resolution="320,576", seed=0, num_steps=4, semantic_plan_path=PLAN)

    lines = []
    for bi in BLOCKS:
        s = res[bi]
        if s is None: lines.append(f"block{bi}: no capture"); continue
        lines.append(f"block{bi}  L={s['L']} heads={s['heads']} dim={s['dim']}  ln(L)={s['lnL']:.3f}")
        lines.append(f"  logit std={s['std']:.4f}   top1-minus-mean gap={s['gap']:.4f}")
        lines.append("  entropy vs temperature: " + "  ".join(f"T={t}:{s['ent'][t]:.3f}" for t in TEMPS))
        lines.append(f"  spatial correspondence: corr_y={s['corr_y']:+.4f} corr_x={s['corr_x']:+.4f} "
                     f"(centroid spread y={s['cy_std']:.4f} x={s['cx_std']:.4f})")
    (OUT / "sharpen_diag.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True); print("SHARPEN-DIAG-DONE", flush=True)


if __name__ == "__main__":
    main()
