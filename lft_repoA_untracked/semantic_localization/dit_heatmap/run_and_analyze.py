"""Self-contained DiT-feature target-focus experiment (routes all results to files).
Same iter3000 SG-WAM, same first frame / seed. Two runs: plan-OFF (no plan) vs plan-ON (oracle SigLIP
plan). Hook late DiT blocks' video-token features (last denoise step of last chunk), map saliency to
the 20x36 latent grid, and measure how much of the top saliency lands inside the CLIPSeg 'yellow
carrot' mask. plan-ON > plan-OFF  =>  our method concentrates on the target.
Outputs: figs/dit_heatmap_target_focus.png + dit_heatmap/focus_result.txt"""
import os, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

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
FIGS = HERE / "figs"; RES = HERE / "dit_heatmap/focus_result.txt"
PROMPT = ("A Franka robotic arm with a parallel-jaw gripper carefully grasp only the [TGT] yellow "
          "carrot in the sink basin and place it into the black pot next to the banana, without moving the banana.")
BLOCKS = [14, 20, 27]; NUM_STEPS = int(os.environ.get("NUM_STEPS", 35)); CHUNK = int(os.environ.get("CHUNK", 13))
dev = "cuda"


def clipseg_mask(img_rgb, noun, hw):
    from transformers.models.clipseg.modeling_clipseg import CLIPSegForImageSegmentation
    from transformers.models.clipseg.processing_clipseg import CLIPSegProcessor
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    seg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval().to(dev)
    x = Image.fromarray(img_rgb).resize((352, 352))
    inp = proc(text=[noun], images=[x], return_tensors="pt", padding=True)
    inp = {k: v.to(dev) for k, v in inp.items()}
    with torch.no_grad():
        lg = seg(**inp).logits
    if lg.ndim == 2: lg = lg[None]
    return F.interpolate(torch.sigmoid(lg[0])[None, None], hw, mode="bilinear")[0, 0].cpu().numpy()


def main():
    from cosmos_predict2.config import SetupArguments
    from cosmos_predict2.inference import Inference
    setup = SetupArguments(
        experiment="predict2_video2world_training_2b_droid_semantic_plan_320x576_93f",
        checkpoint_path=CKPT, config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        output_dir=HERE / "dit_heatmap/gen", context_parallel_size=1,
        offload_diffusion_model=True, offload_text_encoder=True, offload_tokenizer=True,
        disable_guardrails=True, disable_prompt_refiner=True, num_steps=NUM_STEPS, guidance=7, seed=0,
    )
    inf = Inference(setup); net = inf.pipe.model.net
    st = {"thw": None, "feat": {}}
    def pre(m, a, k):
        x = k.get("x", a[0] if a else None)
        if torch.is_tensor(x) and x.dim() == 5: st["thw"] = tuple(x.shape[2:])
    net.register_forward_pre_hook(pre, with_kwargs=True)
    def mk(bi):
        def h(m, i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if torch.is_tensor(t): st["feat"][bi] = t.detach().float().cpu().numpy()
        return h
    for bi in BLOCKS: net.blocks[bi].register_forward_hook(mk(bi))

    runs = {}
    for tag, plan in (("plan_off", None), ("plan_on", str(PLAN))):
        st["feat"] = {}
        inf.pipe.generate_vid2world(prompt=PROMPT, input_path=str(FRAME), num_output_frames=49,
                                    num_steps=NUM_STEPS, guidance=7, seed=0, semantic_plan_path=plan,
                                    chunk_size=CHUNK, chunk_overlap=1)
        runs[tag] = {bi: st["feat"][bi] for bi in BLOCKS if bi in st["feat"]}

    T, H, Wl = st["thw"]
    img = np.asarray(Image.open(FRAME).convert("RGB"))
    mask = clipseg_mask(img, "yellow carrot", (H, Wl))

    def saliency(feat):
        f = torch.from_numpy(feat).float()[0]           # [S,C]
        f = f[-(T * H * Wl):]
        g = f.reshape(T, H, Wl, -1).norm(dim=-1).mean(0)  # [H,W]
        return ((g - g.amin()) / (g.amax() - g.amin() + 1e-6)).numpy()

    def focus(sal):
        s = sal.flatten(); m = (mask.flatten() > 0.4).astype(np.float32)
        k = max(1, int(0.20 * len(s))); top = np.argsort(-s)[:k]
        return float(m[top].mean())

    res, sals = {}, {}
    for tag in ("plan_off", "plan_on"):
        sal = np.mean([saliency(runs[tag][b]) for b in BLOCKS if b in runs[tag]], axis=0)
        sals[tag] = sal; res[tag] = focus(sal)
    gain = res["plan_on"] - res["plan_off"]

    base = np.asarray(Image.fromarray(img).resize((Wl * 16, H * 16))).astype(float) / 255.
    TURBO = cm.get_cmap("turbo")
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.2))
    for ci, tag in enumerate(["plan_off", "plan_on"]):
        up = F.interpolate(torch.from_numpy(sals[tag])[None, None].float(), base.shape[:2], mode="bilinear")[0, 0].numpy()
        up = (up - up.min()) / (up.max() - up.min() + 1e-6)
        ax[ci].imshow(base * 0.5 + TURBO(up)[..., :3] * 0.5)
        ax[ci].set_title(f"{tag}  target-focus={res[tag]:.3f}", fontsize=11); ax[ci].axis("off")
    mup = F.interpolate(torch.from_numpy(mask)[None, None].float(), base.shape[:2], mode="bilinear")[0, 0].numpy()
    ax[2].imshow(base * 0.5 + TURBO(mup)[..., :3] * 0.5); ax[2].set_title("CLIPSeg: yellow carrot", fontsize=11); ax[2].axis("off")
    fig.suptitle(f"DiT-feature target-focus | plan-OFF {res['plan_off']:.3f} vs plan-ON {res['plan_on']:.3f} "
                 f"(delta={gain:+.3f}) iter3000 SG-WAM", fontsize=12)
    FIGS.mkdir(exist_ok=True); fig.tight_layout()
    fig.savefig(FIGS / "dit_heatmap_target_focus.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    np.savez(HERE / "dit_heatmap/dit_feats.npz", thw=np.array(st["thw"]),
             **{f"{t}_b{b}": runs[t][b] for t in runs for b in runs[t]})
    with open(RES, "w") as fh:
        fh.write(f"grid T={T} H={H} W={Wl}\nplan_off_focus={res['plan_off']:.4f}\n")
        fh.write(f"plan_on_focus={res['plan_on']:.4f}\ngain={gain:+.4f}\n")
    print("ALL-DONE", flush=True)


if __name__ == "__main__":
    main()
