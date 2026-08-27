"""Decisive test: is the plan-attention logit RANKING signal or noise?

The diagnostic showed healthy logit contrast (std ~0.9, top1-mean gap ~3.0) and that temperature can
collapse the entropy (T=0.2 -> 3.0 from 7.8). So sharpening is mechanically available. The open
question is what it sharpens ONTO.

Generate the same clip at several attention temperatures by scaling the plan cross-attention logits.
  * video stays coherent and still grasps the carrot  -> the ranking carries signal, an entropy
    penalty during training is the right fix.
  * video degrades / grabs the wrong object           -> the ranking is noise; sharpening cannot help
    and the spatial routing itself must be fixed.
Implemented by scaling q inside compute_qkv (softmax(q.k/sqrt(d)/T) == softmax((q/T).k/sqrt(d))).
"""
import os, sys
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
OUT = HERE / "dit_heatmap/temp_out"; OUT.mkdir(parents=True, exist_ok=True)
PROMPT = ("A Franka robotic arm with a parallel-jaw gripper carefully grasp only the [TGT] yellow "
          "carrot in the sink basin and place it into the black pot next to the banana, without moving the banana.")
TEMPS = [float(t) for t in os.environ.get("TEMPS", "1.0,0.3,0.1").split(",")]
STATE = {"T": 1.0}


def main():
    import imageio.v2 as imageio
    from cosmos_predict2.config import SetupArguments, DEFAULT_MODEL_KEY
    from cosmos_predict2.inference import Inference
    setup = SetupArguments(
        experiment="predict2_video2world_training_2b_droid_semantic_plan_320x576_93f",
        checkpoint_path=CKPT, config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        output_dir=OUT, context_parallel_size=1, offload_diffusion_model=False,
        offload_text_encoder=True, offload_tokenizer=True, disable_guardrails=True,
        model=DEFAULT_MODEL_KEY.name)
    inf = Inference(setup)
    net = inf.pipe.model.net
    n = 0
    for blk in net.blocks:
        for _, sub in blk.named_modules():
            if type(sub).__name__ == "SemanticRopeCrossAttention":
                orig = sub.compute_qkv
                def patched(x, context=None, query_rope_emb=None, key_rope_emb=None, _o=orig):
                    q, k, v = _o(x, context, query_rope_emb=query_rope_emb, key_rope_emb=key_rope_emb)
                    T = STATE["T"]
                    return (q if T == 1.0 else q / T), k, v
                sub.compute_qkv = patched; n += 1
    print(f"patched {n} plan cross-attn modules (all blocks)", flush=True)

    for T in TEMPS:
        STATE["T"] = T
        vid = inf.pipe.generate_vid2world(prompt=PROMPT, input_path=FRAME, guidance=7,
                                          num_video_frames=49, resolution="320,576", seed=0,
                                          num_steps=35, semantic_plan_path=PLAN)
        v = vid[0] if hasattr(vid, "ndim") and vid.ndim == 5 else vid
        v = v.float().clamp(-1, 1).add(1).div(2).mul(255).byte().cpu().numpy()   # (C,T,H,W)
        v = np.transpose(v, (1, 2, 3, 0))
        p = OUT / f"temp_{T}.mp4"
        imageio.mimwrite(p, list(v), fps=10, quality=8)
        print(f"T={T} -> {p} frames={v.shape}", flush=True)
    print("TEMP-SHARPEN-DONE", flush=True)


if __name__ == "__main__":
    main()
