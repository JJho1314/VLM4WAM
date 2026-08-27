"""Decisive test: do video positions at the TARGET attend to the semantic plan more sharply?

Cross-attention softmax is normalized over plan tokens, so "attention mass per video token" is 1 by
construction and carries no signal. The assumption-free quantity is SHARPNESS: entropy of each video
position's distribution over the 486 plan tokens. Low entropy = that position locks onto specific plan
content. If the plan encodes the target, positions on the target should be sharper than elsewhere --
and sharper than on the distractor (banana), which the prompt says to leave alone.

Also plots which plan tokens receive the most attention (plan is resampled to 6 x 9 x 9).
"""
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization")
NPZ = HERE / "dit_heatmap/plan_attn.npz"
FRAME = HERE / "oracle_repro/yc74616_f0.png"
FIGS = HERE / "figs"
RESULT = HERE / "dit_heatmap/attn_result.txt"
H, W = 20, 36
TEARLY = 3


def clipseg(img, noun, hw):
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    pr = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    sg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval()
    inp = pr(text=[noun], images=[Image.fromarray(img).resize((352, 352))], return_tensors="pt", padding=True)
    with torch.no_grad():
        lg = sg(**inp).logits
    if lg.ndim == 2: lg = lg[None]
    m = F.interpolate(torch.sigmoid(lg[0])[None, None], hw, mode="bilinear")[0, 0].numpy()
    return (m - m.min()) / (m.max() - m.min() + 1e-6)


def main():
    z = np.load(NPZ)
    blocks = sorted({int(k.split("_b")[1]) for k in z.files if k.startswith("ent_b")})
    img = np.asarray(Image.open(FRAME).convert("RGB"))
    m_car, m_ban = clipseg(img, "yellow carrot", (H, W)), clipseg(img, "banana", (H, W))
    lines = [f"blocks={blocks}  entropy is over L plan tokens; max possible = ln(L)"]

    def ratio(s, m):
        s = s - s.min()
        mi = (s * m).sum() / (m.sum() + 1e-6); mo = (s * (1 - m)).sum() / ((1 - m).sum() + 1e-6)
        return float(mi / (mo + 1e-6))

    sharp_maps, recv_maps = {}, {}
    for b in blocks:
        ent = z[f"ent_b{b}"]            # (calls, Sq)
        Sq, L, nh = [int(v) for v in z[f"shape_b{b}"]]
        T = Sq // (H * W)
        e_last = ent[-1].reshape(T, H, W)
        e_early = e_last[:TEARLY].mean(0)
        sharp = -e_early                                     # higher = sharper = more locked on
        sharp_maps[b] = sharp
        lines.append(f"block{b}: calls={ent.shape[0]} T={T} L={L} heads={nh} "
                     f"entropy mean={e_last.mean():.3f} min={e_last.min():.3f} max={e_last.max():.3f} (ln L={np.log(L):.3f})")
        lines.append(f"  sharpness focus ratio  carrot={ratio(sharp, m_car):.3f}  banana={ratio(sharp, m_ban):.3f}")
        recv = z[f"recv_b{b}"][-1]
        recv_maps[b] = recv
        top = np.argsort(-recv)[:5]
        lines.append(f"  plan-token attention: uniform={1.0/L:.5f} max={recv.max():.5f} "
                     f"(x{recv.max()*L:.2f} uniform)  top-idx={top.tolist()}")
        lines.append(f"  fraction of plan mass in top-10% tokens = {np.sort(recv)[::-1][:max(1,L//10)].sum():.3f}")

    # evolution over denoising steps (block with most blocks[-1])
    b = blocks[-1]
    ent_all = z[f"ent_b{b}"]
    Sq, L, _ = [int(v) for v in z[f"shape_b{b}"]]
    T = Sq // (H * W)
    rr = []
    for ci in range(ent_all.shape[0]):
        s = -ent_all[ci].reshape(T, H, W)[:TEARLY].mean(0)
        rr.append((ratio(s, m_car), ratio(s, m_ban)))
    lines.append(f"\nblock{b} sharpness ratio across {len(rr)} denoise calls:")
    lines.append("  carrot: " + " ".join(f"{a:.2f}" for a, _ in rr[::6]))
    lines.append("  banana: " + " ".join(f"{c:.2f}" for _, c in rr[::6]))
    RESULT.write_text("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

    base = np.asarray(Image.fromarray(img).resize((W * 20, H * 20))).astype(float) / 255.
    def ov(s):
        u = F.interpolate(torch.from_numpy(s)[None, None].float(), base.shape[:2], mode="bilinear")[0, 0].numpy()
        u = (u - u.min()) / (u.max() - u.min() + 1e-6)
        return base * 0.5 + cm.get_cmap("turbo")(u)[..., :3] * 0.5
    P = [(base, "initial frame"), (ov(m_car), "target: yellow carrot"), (ov(m_ban), "distractor: banana")]
    for b in blocks:
        P.append((ov(sharp_maps[b]), f"block{b} attention SHARPNESS\ncarrot {ratio(sharp_maps[b], m_car):.2f} / banana {ratio(sharp_maps[b], m_ban):.2f}"))
    fig, ax = plt.subplots(1, len(P), figsize=(4.3 * len(P), 3.4))
    for a, (im, t) in zip(np.atleast_1d(ax), P):
        a.imshow(im); a.set_title(t, fontsize=9); a.axis("off")
    fig.suptitle("Where does the world model attend to the semantic plan most sharply? (iter3000 SG-WAM, oracle plan)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "plan_attn_sharpness.png", dpi=115, bbox_inches="tight"); plt.close(fig)

    # which plan tokens are read: plan is N keyframes x g x g (native k5 -> 5 x 27 x 27 = 3645)
    b = blocks[-1]; recv = recv_maps[b]; L = recv.size
    N = 5
    g = int(round((L / N) ** 0.5))
    if N * g * g == L:
        gm = recv.reshape(N, g, g)
        fig, ax = plt.subplots(1, N, figsize=(2.7 * N, 3.0))
        for i in range(N):
            ax[i].imshow(gm[i], cmap="turbo")
            ax[i].set_title(f"plan keyframe {i}\nmax {gm[i].max()*L:.2f}x uniform", fontsize=9); ax[i].axis("off")
        fig.suptitle(f"Attention received by each plan token (block{b}, plan = {N} keyframes x {g}x{g})", fontsize=11)
        fig.tight_layout(); fig.savefig(FIGS / "plan_token_attention.png", dpi=115, bbox_inches="tight"); plt.close(fig)
        np.save(HERE / "dit_heatmap/plan_recv_grid.npy", gm)
        print(f"SAVED figs/plan_token_attention.png (grid {N}x{g}x{g})", flush=True)
    print("SAVED figs/plan_attn_sharpness.png", flush=True); print("ANALYZE-ATTN-DONE", flush=True)


if __name__ == "__main__":
    main()
