"""Plan-token ablation: does the world model actually USE the spatial content of the plan?

Temperature sharpening changed the output only chaotically, suggesting the plan is consumed as a
global average rather than a spatial map. This test decides it. We keep only a subset of the plan's
3645 tokens (5 keyframes x 27x27) and zero the rest:

  full     : all tokens (reference)
  target   : only tokens on the yellow carrot (CLIPSeg on the real future keyframes)
  distract : only tokens on the banana -- the object the prompt says to LEAVE ALONE
  none     : all plan tokens zeroed (plan-OFF-ish, upper bound on how much the plan matters at all)

Readout: pixel distance of each variant from `full`, compared with the plan-matters scale (full vs none).
  target ~ full  and  distract far from full  -> the model reads the target region: spatial content is used
  everything ~ full                            -> only the global average matters; spatial focus is moot
"""
import os, sys
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
os.environ.setdefault("SEMANTIC_PLAN_NUM_KEYFRAMES", "5")
os.environ.setdefault("SEMANTIC_PLAN_SPATIAL_GRID", "0")
HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization")
FRAME = str(HERE / "oracle_repro/yc74616_f0.png")
PLAN = str(HERE / "oracle_repro/yc74616_s0_oracle.pt")
GT = HERE / "oracle_repro/gt_clips/yc74616_s0_gt.mp4"
CKPT = f"{W}/cosmos_semantic_plan_iter3000/model_ema_bf16.pt"
OUT = HERE / "dit_heatmap/ablate_out"; OUT.mkdir(parents=True, exist_ok=True)
PROMPT = ("A Franka robotic arm with a parallel-jaw gripper carefully grasp only the [TGT] yellow "
          "carrot in the sink basin and place it into the black pot next to the banana, without moving the banana.")
NKF, G = 5, 27
STATE = {"mask": None}


def keyframe_masks(noun):
    """CLIPSeg mask of `noun` on each of the 5 plan keyframes of the real future clip -> (5,27,27)."""
    import av
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    c = av.open(str(GT)); frames = [np.asarray(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)]; c.close()
    idx = np.linspace(0, len(frames) - 1, NKF).round().astype(int)
    pr = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    sg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval()
    out = []
    for i in idx:
        inp = pr(text=[noun], images=[Image.fromarray(frames[i]).resize((352, 352))],
                 return_tensors="pt", padding=True)
        with torch.no_grad():
            lg = sg(**inp).logits
        if lg.ndim == 2: lg = lg[None]
        m = F.interpolate(torch.sigmoid(lg[0])[None, None], (G, G), mode="bilinear")[0, 0].numpy()
        out.append((m - m.min()) / (m.max() - m.min() + 1e-6))
    return np.stack(out)


def main():
    import imageio.v2 as imageio
    from cosmos_predict2.config import SetupArguments, DEFAULT_MODEL_KEY
    from cosmos_predict2.inference import Inference

    m_car, m_ban = keyframe_masks("yellow carrot"), keyframe_masks("banana")
    keep_car = torch.from_numpy((m_car > 0.5).reshape(-1).astype(np.float32))
    keep_ban = torch.from_numpy((m_ban > 0.5).reshape(-1).astype(np.float32))
    print(f"plan tokens kept: carrot={int(keep_car.sum())}/{NKF*G*G}  banana={int(keep_ban.sum())}", flush=True)

    setup = SetupArguments(
        experiment="predict2_video2world_training_2b_droid_semantic_plan_320x576_93f",
        checkpoint_path=CKPT, config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        output_dir=OUT, context_parallel_size=1, offload_diffusion_model=False,
        offload_text_encoder=True, offload_tokenizer=True, disable_guardrails=True,
        model=DEFAULT_MODEL_KEY.name)
    inf = Inference(setup)
    n = 0
    for blk in inf.pipe.model.net.blocks:
        for _, sub in blk.named_modules():
            if type(sub).__name__ == "SemanticRopeCrossAttention":
                orig = sub.compute_qkv
                def patched(x, context=None, query_rope_emb=None, key_rope_emb=None, _o=orig):
                    mk = STATE["mask"]
                    if mk is not None and context is not None and context.shape[-2] == mk.numel():
                        context = context * mk.to(context.device, context.dtype).view(
                            *([1] * (context.dim() - 2)), -1, 1)
                    return _o(x, context, query_rope_emb=query_rope_emb, key_rope_emb=key_rope_emb)
                sub.compute_qkv = patched; n += 1
    print(f"patched {n} plan cross-attn modules", flush=True)

    vids = {}
    for tag, mk in (("full", None), ("target", keep_car), ("distract", keep_ban),
                    ("none", torch.zeros(NKF * G * G))):
        STATE["mask"] = mk
        v = inf.pipe.generate_vid2world(prompt=PROMPT, input_path=FRAME, guidance=7,
                                        num_video_frames=49, resolution="320,576", seed=0,
                                        num_steps=35, semantic_plan_path=PLAN)
        v = v[0] if v.ndim == 5 else v
        arr = v.float().clamp(-1, 1).add(1).div(2).mul(255).byte().cpu().numpy()
        arr = np.transpose(arr, (1, 2, 3, 0))
        imageio.mimwrite(OUT / f"{tag}.mp4", list(arr), fps=10, quality=8)
        vids[tag] = arr.astype(np.float32)
        print(f"generated {tag}", flush=True)

    lines = [f"plan tokens kept: carrot={int(keep_car.sum())}/{NKF*G*G} banana={int(keep_ban.sum())}"]
    ref = vids["full"]
    for tag in ("target", "distract", "none"):
        d = np.abs(ref - vids[tag]).mean()
        lines.append(f"|{tag} - full| mean = {d:.3f}/255")
    lines.append(f"(scale reference) within-video frame-to-frame = {np.abs(ref[1:]-ref[:-1]).mean():.3f}/255")
    (HERE / "dit_heatmap/ablate_result.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True); print("ABLATE-DONE", flush=True)


if __name__ == "__main__":
    main()
