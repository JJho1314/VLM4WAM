"""Capture the REAL attention weights inside SemanticRopeCrossAttention (plan-ON).

Why not just "attention mass per video token": cross-attention softmax normalizes over the KEY (plan
token) axis, so every video token's attention sums to 1 by construction. The measurable quantities are:
  * entropy per video position  -- low entropy = that position sharply locks onto specific plan tokens
  * attention received per plan token -- WHICH region of the future keyframes the WM actually reads
Both are recorded per forward call so the CFG conditional/unconditional branches can be told apart.
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
# MUST match how iter3000 was trained (native k5, 27x27 = 3645 plan tokens). This local cosmos repo
# defaults to 6 keyframes / grid 9 (=486 tokens); leaving those defaults silently resamples the plan
# into a layout the checkpoint never saw, which collapses the plan attention to uniform.
os.environ.setdefault("SEMANTIC_PLAN_NUM_KEYFRAMES", "5")
os.environ.setdefault("SEMANTIC_PLAN_SPATIAL_GRID", "0")
HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization")
FRAME = str(HERE / "oracle_repro/yc74616_f0.png")
PLAN = str(HERE / "oracle_repro/yc74616_s0_oracle.pt")
CKPT = f"{W}/cosmos_semantic_plan_iter3000/model_ema_bf16.pt"
OUT = HERE / "dit_heatmap"
PROMPT = ("A Franka robotic arm with a parallel-jaw gripper carefully grasp only the [TGT] yellow "
          "carrot in the sink basin and place it into the black pot next to the banana, without moving the banana.")
BLOCKS = [20, 27]
NUM_STEPS = int(os.environ.get("NUM_STEPS", 35))
NFRAMES = int(os.environ.get("NFRAMES", 49))
RES = os.environ.get("RES", "320,576")
TEARLY = int(os.environ.get("TEARLY", 3))       # query frames aligned with the initial frame


def main():
    from cosmos_predict2.config import SetupArguments, DEFAULT_MODEL_KEY
    from cosmos_predict2.inference import Inference
    setup = SetupArguments(
        experiment="predict2_video2world_training_2b_droid_semantic_plan_320x576_93f",
        checkpoint_path=CKPT, config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        output_dir=OUT / "gen", context_parallel_size=1,
        offload_diffusion_model=False, offload_text_encoder=True, offload_tokenizer=True,
        disable_guardrails=True, model=DEFAULT_MODEL_KEY.name,
    )
    inf = Inference(setup)
    net = inf.pipe.model.net
    rec = {b: [] for b in BLOCKS}                    # per call: (entropy_map, plan_received, shapes)

    def wrap(mod, bi):
        orig = mod.compute_qkv
        def patched(x, context=None, query_rope_emb=None, key_rope_emb=None):
            q, k, v = orig(x, context, query_rope_emb=query_rope_emb, key_rope_emb=key_rope_emb)
            try:
                with torch.no_grad():
                    qq = q[:, :TEARLY] if q.dim() >= 5 else q      # (B,T,H,W,h,d) -> early frames
                    hh, dd = qq.shape[-2], qq.shape[-1]
                    qf = qq.reshape(1, -1, hh, dd).float()          # (1,Sq,h,d)
                    kf = k.reshape(1, -1, hh, dd).float()           # (1,L,h,d)
                    Sq, L = qf.shape[1], kf.shape[1]
                    scale = 1.0 / math.sqrt(dd)
                    ent = torch.zeros(Sq, device=qf.device)
                    recv = torch.zeros(L, device=qf.device)
                    for h in range(hh):                             # per head: keeps memory tiny
                        p = torch.softmax(qf[0, :, h] @ kf[0, :, h].T * scale, dim=-1)   # (Sq,L)
                        ent += -(p * (p + 1e-9).log()).sum(-1)
                        recv += p.sum(0)
                    rec[bi].append((ent.div(hh).cpu().numpy().astype(np.float32),
                                    recv.div(hh * Sq).cpu().numpy().astype(np.float32),
                                    (int(Sq), int(L), int(hh))))
            except Exception as e:
                rec[bi].append(("ERR", str(e), ()))
            return q, k, v
        mod.compute_qkv = patched

    n = 0
    for bi in BLOCKS:
        for name, sub in net.blocks[bi].named_modules():
            if type(sub).__name__ == "SemanticRopeCrossAttention":
                wrap(sub, bi); n += 1
    print(f"patched {n} SemanticRopeCrossAttention modules on blocks {BLOCKS}", flush=True)

    inf.pipe.generate_vid2world(prompt=PROMPT, input_path=FRAME, guidance=7, num_video_frames=NFRAMES,
                                resolution=RES, seed=0, num_steps=NUM_STEPS, semantic_plan_path=PLAN)

    save = {}
    for bi in BLOCKS:
        good = [r for r in rec[bi] if not isinstance(r[0], str)]
        print(f"block{bi}: {len(rec[bi])} calls, {len(good)} ok, shapes={good[0][2] if good else None}", flush=True)
        if not good:
            print(f"  errors: {rec[bi][:1]}", flush=True); continue
        save[f"ent_b{bi}"] = np.stack([g[0] for g in good])          # (calls, Sq)
        save[f"recv_b{bi}"] = np.stack([g[1] for g in good])         # (calls, L)
        save[f"shape_b{bi}"] = np.array(good[0][2])
    save["res"] = np.array([int(x) for x in RES.split(",")]); save["tearly"] = np.array([TEARLY])
    np.savez(OUT / "plan_attn.npz", **save)
    print("ATTN-CAPTURE-DONE", flush=True)


if __name__ == "__main__":
    main()
