"""Oracle plan ON vs OFF on the iter3000 SG-WAM -- the clean ablation.

Both clips come from ONE session: same weights, same first frame, same prompt, same seed, same 35
denoise steps. The only difference is whether `semantic_plan_path` is supplied, so any difference is
attributable to the plan itself (unlike comparing against a separately-trained RGB-only model, where
domain fine-tuning would be confounded with semantic guidance).

  plan_on  : oracle plan = SigLIP2 of the REAL future frames (upper bound, not a planner prediction)
  plan_off : semantic_plan_path=None -- the world model runs on RGB + text alone

Writes plan_on.mp4 / plan_off.mp4 and reports how far apart they are relative to the clip's own
frame-to-frame motion, which is the scale that makes the number interpretable.
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
# iter3000 was trained native k5 / 27x27 = 3645 plan tokens; this repo defaults to 6 keyframes /
# grid 9 (486), which silently resamples the plan into a layout the checkpoint never saw.
os.environ.setdefault("SEMANTIC_PLAN_NUM_KEYFRAMES", "5")
os.environ.setdefault("SEMANTIC_PLAN_SPATIAL_GRID", "0")

HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization/oracle_repro")
FRAME = str(HERE / "yc74616_f0.png")
PLAN = str(HERE / "yc74616_s0_oracle.pt")
CKPT = f"{W}/cosmos_semantic_plan_iter3000/model_ema_bf16.pt"
OUT = HERE / "plan_on_off"; OUT.mkdir(parents=True, exist_ok=True)
PROMPT = ("A Franka robotic arm with a parallel-jaw gripper carefully grasp only the [TGT] yellow "
          "carrot in the sink basin and place it into the black pot next to the banana, without moving the banana.")
STEPS, NFRAMES, RES, SEED, GUID = 35, 49, "320,576", 0, 7


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
    vids = {}
    for tag, plan in (("plan_on", PLAN), ("plan_off", None)):
        v = inf.pipe.generate_vid2world(prompt=PROMPT, input_path=FRAME, guidance=GUID,
                                        num_video_frames=NFRAMES, resolution=RES, seed=SEED,
                                        num_steps=STEPS, semantic_plan_path=plan)
        v = v[0] if v.ndim == 5 else v
        a = np.transpose(v.float().clamp(-1, 1).add(1).div(2).mul(255).byte().cpu().numpy(), (1, 2, 3, 0))
        imageio.mimwrite(OUT / f"{tag}.mp4", list(a), fps=10, quality=8)
        vids[tag] = a.astype(np.float32)
        print(f"generated {tag}: {a.shape} -> {OUT}/{tag}.mp4", flush=True)

    on, off = vids["plan_on"], vids["plan_off"]
    d = float(np.abs(on - off).mean())
    ref = float(np.abs(on[1:] - on[:-1]).mean())
    psnr = float(10 * np.log10(255 ** 2 / ((on - off) ** 2).mean()))
    lines = [f"|plan_on - plan_off| mean = {d:.3f}/255   PSNR = {psnr:.1f} dB",
             f"(scale reference) within-clip frame-to-frame motion = {ref:.3f}/255",
             f"ratio = {d/ref:.2f}x  -> the plan moves the output {d/ref:.0%} as much as one frame of motion"]
    (OUT / "result.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True); print("GEN-DONE", flush=True)


if __name__ == "__main__":
    main()
