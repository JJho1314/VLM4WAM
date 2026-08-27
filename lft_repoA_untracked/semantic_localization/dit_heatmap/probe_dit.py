"""Probe: build local iter3000 Cosmos semantic-plan WM, hook DiT blocks, run a short plan-ON
inference, dump each block's video-token feature shape + latent (T,H,W). Run in cosmos venv."""
import os, sys
from pathlib import Path
import torch, torch.nn as nn

COSMOS = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/cosmos-predict2.5"
sys.path.insert(0, COSMOS)
W = "/data/LFT-W02_data/junjie/weights"
os.environ.setdefault("COSMOS_HF_LOCAL_DIRS", W)
os.environ.setdefault("COSMOS_LOCAL_MODEL_DIR", f"{W}/Cosmos-Predict2.5-2B")
os.environ.setdefault("SEMANTIC_PLAN_ONLINE_ENCODER_PATH",
                      "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/siglip2-so400m-patch14-384")
HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization")
FRAME = HERE / "oracle_repro/yc74616_f0.png"
PLAN = HERE / "oracle_repro/yc74616_s0_oracle.pt"
CKPT = f"{W}/cosmos_semantic_plan_iter3000/model_ema_bf16.pt"
PROMPT = ("A Franka robotic arm with a parallel-jaw gripper carefully grasp only the [TGT] yellow "
          "carrot in the sink basin and place it into the black pot next to the banana, without moving the banana.")


def main():
    from cosmos_predict2.config import SetupArguments
    from cosmos_predict2.inference import Inference
    setup = SetupArguments(
        experiment="predict2_video2world_training_2b_droid_semantic_plan_320x576_93f",
        checkpoint_path=CKPT, config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        output_dir=HERE / "dit_heatmap/probe_out", context_parallel_size=1,
        offload_diffusion_model=True, offload_text_encoder=True, offload_tokenizer=True,
        disable_guardrails=True, disable_prompt_refiner=True, num_steps=3, guidance=7, seed=0,
    )
    inf = Inference(setup); pipe = inf.pipe
    net = None
    for a in ("model", "dit", "net", "denoiser"):
        m = getattr(pipe, a, None)
        if m is None: continue
        for n, sub in m.named_modules():
            if n.endswith("blocks") and isinstance(sub, nn.ModuleList):
                net = m; print(f"FOUND blocks at pipe.{a}.{n} ({len(sub)} blocks)", flush=True); break
        if net is not None: break
    thw = {}
    def pre(mod, args, kwargs):
        x = kwargs.get("x", args[0] if args else None)
        if torch.is_tensor(x) and x.dim() == 5: thw["v"] = tuple(x.shape)
    shapes = {}
    def mk(bi):
        def h(m, i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if torch.is_tensor(t): shapes[bi] = tuple(t.shape)
        return h
    net.register_forward_pre_hook(pre, with_kwargs=True)
    for bi in range(len(net.blocks)): net.blocks[bi].register_forward_hook(mk(bi))
    print("RUN plan-ON num_steps=3", flush=True)
    inf.pipe.generate_vid2world(prompt=PROMPT, input_path=str(FRAME), num_output_frames=49,
                                num_steps=3, guidance=7, seed=0, semantic_plan_path=str(PLAN))
    print("NET_INPUT_5D", thw.get("v"), flush=True)
    for bi in sorted(shapes)[:6]: print(f"  block{bi}: {shapes[bi]}", flush=True)
    print("NBLOCKS", len(shapes)); print("PROBE-DONE", flush=True)


if __name__ == "__main__":
    main()
