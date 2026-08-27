"""Capture DiT evidence that the semantic plan focuses the WM on the target object (iter3000 SG-WAM).

Two signals, same first frame / seed / steps:
  (a) block output features, plan-OFF vs plan-ON      -> generic saliency comparison
  (b) SemanticRopeCrossAttention output per block     -> the plan's DIRECT additive contribution to
      each video token; its per-position norm is literally "how much the semantic plan moved this
      spatial location", the cleanest answer to "does the plan attend to the target object?".

Verified execution path (global-hook diagnostic): net=MinimalV1LVGDiT, blocks fire, and
SemanticRopeCrossAttention fires when a plan is supplied.
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
# MUST match how iter3000 was trained (native k5, 27x27 = 3645 plan tokens); this repo's defaults
# (6 keyframes / grid 9) silently resample the plan into a layout the checkpoint never saw.
os.environ.setdefault("SEMANTIC_PLAN_NUM_KEYFRAMES", "5")
os.environ.setdefault("SEMANTIC_PLAN_SPATIAL_GRID", "0")
HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization")
FRAME = str(HERE / "oracle_repro/yc74616_f0.png")
PLAN = str(HERE / "oracle_repro/yc74616_s0_oracle.pt")
CKPT = f"{W}/cosmos_semantic_plan_iter3000/model_ema_bf16.pt"
OUTDIR = HERE / "dit_heatmap"
PROMPT = ("A Franka robotic arm with a parallel-jaw gripper carefully grasp only the [TGT] yellow "
          "carrot in the sink basin and place it into the black pot next to the banana, without moving the banana.")
BLOCKS = [14, 20, 27]
NUM_STEPS = int(os.environ.get("NUM_STEPS", 35))
NFRAMES = int(os.environ.get("NFRAMES", 49))
RES = os.environ.get("RES", "320,576")


def main():
    from cosmos_predict2.config import SetupArguments, DEFAULT_MODEL_KEY
    from cosmos_predict2.inference import Inference
    setup = SetupArguments(
        experiment="predict2_video2world_training_2b_droid_semantic_plan_320x576_93f",
        checkpoint_path=CKPT, config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        output_dir=OUTDIR / "gen", context_parallel_size=1,
        offload_diffusion_model=False, offload_text_encoder=True, offload_tokenizer=True,
        disable_guardrails=True, model=DEFAULT_MODEL_KEY.name,
    )
    inf = Inference(setup)
    net = inf.pipe.model.net
    BLK, PLN = {}, {}          # NOTE: never rebind these -- hooks close over the object identity

    def keep(store, bi):
        def h(m, i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if torch.is_tensor(t):                       # last call of the run wins = final denoise step
                store[bi] = t.detach().float().cpu().numpy()
        return h

    n_plan_mod = 0
    for bi in BLOCKS:
        blk = net.blocks[bi]
        blk.register_forward_hook(keep(BLK, bi))
        for name, sub in blk.named_modules():
            if type(sub).__name__ == "SemanticRopeCrossAttention":
                sub.register_forward_hook(keep(PLN, bi)); n_plan_mod += 1
    print(f"hooks: {len(BLOCKS)} blocks, semantic-plan xattn modules found={n_plan_mod}", flush=True)

    save = {}
    for tag, plan in (("plan_off", None), ("plan_on", PLAN)):
        BLK.clear(); PLN.clear()                        # clear in place, do NOT rebind
        inf.pipe.generate_vid2world(
            prompt=PROMPT, input_path=FRAME, guidance=7, num_video_frames=NFRAMES,
            resolution=RES, seed=0, num_steps=NUM_STEPS, semantic_plan_path=plan)
        for bi, v in BLK.items():
            save[f"{tag}_b{bi}"] = v
        for bi, v in PLN.items():
            save[f"{tag}_planattn_b{bi}"] = v
        shp = {bi: tuple(v.shape) for bi, v in BLK.items()}
        np.savez(OUTDIR / "dit_feats.npz", res=np.array([int(x) for x in RES.split(",")]),
                 nframes=np.array([NFRAMES]), **save)     # save after each run so partials survive
        print(f"captured {tag}: blk={sorted(BLK)} shapes={shp} planattn={sorted(PLN)}", flush=True)

    with open(OUTDIR / "DONE.txt", "w") as fh:
        fh.write(f"keys={sorted(save.keys())}\nres={RES} nframes={NFRAMES}\n")
    print("CAPTURE-DONE", flush=True)


if __name__ == "__main__":
    main()
