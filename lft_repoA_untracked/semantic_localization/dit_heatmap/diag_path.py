"""Diagnostic: which modules actually run during generate_vid2world? Uses a GLOBAL module forward
hook (fires for every nn.Module), so it cannot miss the real execution path. Tiny generation."""
import os, sys
from pathlib import Path
from collections import Counter
import torch, torch.nn as nn

COSMOS = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/cosmos-predict2.5"
sys.path.insert(0, COSMOS)
W = "/data/LFT-W02_data/junjie/weights"
os.environ.setdefault("COSMOS_HF_LOCAL_DIRS", W)
os.environ.setdefault("COSMOS_LOCAL_MODEL_DIR", f"{W}/Cosmos-Predict2.5-2B")
os.environ.setdefault("SEMANTIC_PLAN_ONLINE_ENCODER_PATH",
                      "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/siglip2-so400m-patch14-384")
HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization")
FRAME = str(HERE / "oracle_repro/yc74616_f0.png")
PLAN = str(HERE / "oracle_repro/yc74616_s0_oracle.pt")
CKPT = f"{W}/cosmos_semantic_plan_iter3000/model_ema_bf16.pt"
OUT = HERE / "dit_heatmap/diag_result.txt"


def main():
    from cosmos_predict2.config import SetupArguments, DEFAULT_MODEL_KEY
    from cosmos_predict2.inference import Inference
    setup = SetupArguments(
        experiment="predict2_video2world_training_2b_droid_semantic_plan_320x576_93f",
        checkpoint_path=CKPT, config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        output_dir=HERE / "dit_heatmap/gen", context_parallel_size=1,
        offload_diffusion_model=False, offload_text_encoder=True, offload_tokenizer=True,
        disable_guardrails=True, model=DEFAULT_MODEL_KEY.name,
    )
    inf = Inference(setup)
    model = inf.pipe.model
    net = model.net
    lines = [f"id(pipe.model.net)={id(net)} type={type(net).__name__}",
             f"has net_ema={hasattr(model,'net_ema')}"]
    if hasattr(model, "net_ema") and model.net_ema is not None:
        lines.append(f"id(net_ema)={id(model.net_ema)}")

    fired = Counter(); ids = {}
    def gh(mod, inp, out):
        fired[type(mod).__name__] += 1
        ids.setdefault(type(mod).__name__, id(mod))
    h = nn.modules.module.register_module_forward_hook(gh)
    try:
        inf.pipe.generate_vid2world(prompt="a robot arm grasps the yellow carrot", input_path=FRAME,
                                    guidance=7, num_video_frames=13, resolution="320,576", seed=0,
                                    num_steps=2, semantic_plan_path=PLAN)
        lines.append("GEN-OK")
    except Exception as e:
        lines.append(f"GEN-ERROR {type(e).__name__}: {e}")
    h.remove()
    lines.append(f"total distinct module types fired: {len(fired)}")
    for name, c in fired.most_common(18):
        lines.append(f"  {name}: {c}")
    for key in ("MiniTrainDIT", "Block", "Attention", "VideoAttn", "DiT"):
        hit = [n for n in fired if key.lower() in n.lower()]
        if hit: lines.append(f"MATCH {key}: {hit}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("DIAG-DONE", flush=True)


if __name__ == "__main__":
    main()
