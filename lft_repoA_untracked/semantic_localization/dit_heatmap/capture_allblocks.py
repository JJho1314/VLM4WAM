"""Scan EVERY DiT block for target focus, in the format of the mask-guided reference visualisation.

Why all blocks: the reference run (2b_mgv3_target_context, mask-supervised) is sharply focused ONLY in
blocks 12 and 16 -- inside/outside ratio 37.7 and 46.0 there, but ~1.0 (uniform) at blocks 0/4/8/20/24/27.
An earlier pass here probed only blocks 20 and 27 and concluded the plan attention was uniform; those
are exactly the blocks that read as uniform even in a model that demonstrably does focus. So this
sweeps all 28 blocks before drawing any conclusion.

Captures two attention paths per block:
  text : video tokens -> caption tokens (comparable to the reference figure, whose caption carries [TGT])
  plan : video tokens -> semantic-plan tokens (our method's own mechanism; plan-ON only)
Metrics per block match the reference pack: attention mass inside the target mask, inside/outside
ratio, normalised entropy, peak location.
"""
import os, sys, math, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

COSMOS = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/cosmos-predict2.5"
sys.path.insert(0, COSMOS)
W = "/data/LFT-W02_data/junjie/weights"
os.environ.setdefault("COSMOS_HF_LOCAL_DIRS", W)
os.environ.setdefault("COSMOS_LOCAL_MODEL_DIR", f"{W}/Cosmos-Predict2.5-2B")
os.environ.setdefault("SEMANTIC_PLAN_ONLINE_ENCODER_PATH",
                      "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/siglip2-so400m-patch14-384")
os.environ.setdefault("COSMOS_NUM_FRAMES", "49")        # state_t=13, as the checkpoint was trained
os.environ.setdefault("SEMANTIC_PLAN_NUM_KEYFRAMES", "5")
os.environ.setdefault("SEMANTIC_PLAN_SPATIAL_GRID", "0")

HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization")
REPRO = HERE / "oracle_repro"
FRAME = str(REPRO / "yc74616_f0.png")
PLAN = str(REPRO / "yc74616_s0_oracle.pt")
CKPT = f"{W}/cosmos_semantic_plan_iter3000/model_ema_bf16.pt"
OUT = HERE / "dit_heatmap/allblocks"; OUT.mkdir(parents=True, exist_ok=True)
PROMPT = ("A Franka robotic arm with a parallel-jaw gripper carefully grasp only the [TGT] yellow "
          "carrot in the sink basin and place it into the black pot next to the banana, without moving the banana.")
PLAN_ON = int(os.environ.get("PLAN_ON", 1))


def main():
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
    nb = len(net.blocks)
    store = {"text": {}, "plan": {}, "thw": None}

    def hook_attn(mod, bi, kind):
        """Wrap compute_qkv, keeping the last call's mean-over-heads attention as (S_video, L_ctx).
        Signature-agnostic pass-through: the plan and text attention classes take different keyword
        arguments (query_rope_emb/key_rope_emb vs rope_emb), so forward everything untouched."""
        orig = mod.compute_qkv
        def patched(*a, **kw):
            out = orig(*a, **kw)
            if not isinstance(out, (tuple, list)) or len(out) != 3:
                return out                                       # not the (q,k,v) form -- leave alone
            q, k, v = out
            ctx = kw.get("context", a[1] if len(a) > 1 else None)
            if ctx is not None and torch.is_tensor(q) and torch.is_tensor(k):
                try:
                    with torch.no_grad():
                        hh, dd = q.shape[-2], q.shape[-1]
                        qf = q.reshape(-1, hh, dd).float(); kf = k.reshape(-1, hh, dd).float()
                        scale = 1.0 / math.sqrt(dd)
                        acc = None
                        for h in range(hh):
                            p = torch.softmax(qf[:, h] @ kf[:, h].T * scale, dim=-1)
                            acc = p if acc is None else acc + p
                        store[kind][bi] = (acc / hh).cpu()        # (S_video, L_ctx)
                except Exception as e:
                    store.setdefault("err", {})[f"{kind}_b{bi}"] = repr(e)[:120]
            return out
        mod.compute_qkv = patched

    def pre(m, a, kw):
        x = kw.get("x", a[0] if a else None)
        if torch.is_tensor(x) and x.dim() == 5:
            store["thw"] = tuple(int(v) for v in x.shape[2:])
    net.register_forward_pre_hook(pre, with_kwargs=True)

    hooked = {"text": [], "plan": []}
    for bi, blk in enumerate(net.blocks):
        for name, sub in blk.named_modules():
            if not hasattr(sub, "compute_qkv"): continue
            t = type(sub).__name__
            if t == "SemanticRopeCrossAttention":
                hook_attn(sub, bi, "plan"); hooked["plan"].append((bi, name, t))
            elif "cross" in name.lower():
                hook_attn(sub, bi, "text"); hooked["text"].append((bi, name, t))
    print(f"blocks={nb}  plan-hooks={len(hooked['plan'])}  text-hooks={len(hooked['text'])}", flush=True)
    for kind in ("plan", "text"):
        if hooked[kind]: print(f"  {kind} example: block{hooked[kind][0][0]} '{hooked[kind][0][1]}' ({hooked[kind][0][2]})", flush=True)

    inf.pipe.generate_vid2world(prompt=PROMPT, input_path=FRAME, guidance=7, num_video_frames=49,
                                resolution="320,576", seed=0, num_steps=2,
                                semantic_plan_path=(PLAN if PLAN_ON else None))

    # The net pre-hook does not always see a 5-D x, so derive the grid from the captured attention:
    # 320x576 pixels / (8x VAE * 2x DiT patch) = 20x36 spatial, and T = S_video / (H*W).
    any_attn = next((a for d in (store["plan"], store["text"]) for a in d.values()), None)
    if any_attn is None:
        raise RuntimeError(f"no attention captured; errors={store.get('err')}")
    if store["thw"] is not None:
        T, H, Wd = store["thw"]
    else:
        H, Wd = 320 // 16, 576 // 16        # training resolution -> 20x36 latent grid
        T = int(any_attn.shape[0]) // (H * Wd)
        assert T * H * Wd == int(any_attn.shape[0]), (
            f"grid mismatch: S={int(any_attn.shape[0])} not divisible by {H}x{Wd}; "
            "the run probably used a different resolution than training")
    print(f"latent grid T={T} H={H} W={Wd}  (S_video={int(any_attn.shape[0])}, "
          f"L_ctx plan/text={ {k: int(next(iter(v.values())).shape[1]) for k, v in (('plan', store['plan']), ('text', store['text'])) if v} })", flush=True)
    save = {"thw": np.array([T, H, Wd])}
    for kind in ("text", "plan"):
        for bi, a in store[kind].items():
            g = a.mean(-1)                                     # mean over context tokens -> per video token
            n = g.numel()
            if n != T * H * Wd:                                 # tolerate a flattened layout
                g = g[-T * H * Wd:]
            save[f"{kind}_b{bi}"] = g.reshape(T, H, Wd).numpy().astype(np.float32)
    np.savez(OUT / f"attn_{'on' if PLAN_ON else 'off'}.npz", **save)
    print(f"saved {OUT}/attn_{'on' if PLAN_ON else 'off'}.npz  "
          f"text_blocks={sorted(store['text'])} plan_blocks={sorted(store['plan'])}", flush=True)
    print("ALLBLOCKS-DONE", flush=True)


if __name__ == "__main__":
    main()
