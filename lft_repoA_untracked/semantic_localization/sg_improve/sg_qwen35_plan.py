"""Option 6 on Qwen3.5-2B: the Qwen3.5-2B VLM planner reads (current frame + instruction) and, via a
lightweight learnable-query module, PARALLEL-predicts discrete SigLIP codes (no autoregression).
codes -> codebook 1024-d vectors -> fed to the WAM unchanged. Discrete + text-grounded, Plan-X flavor
without the autoregressive generator. Runs in the `qwen35` env (transformers 5.14.1 + Qwen3_5)."""
import os, sys, torch
import torch.nn as nn, torch.nn.functional as F
from PIL import Image
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sg_discrete_plan import SigLIPVQ, ParallelCodePlanHead, discrete_plan_loss, predict_plan_vectors
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
dev = "cuda"; dt = torch.bfloat16
QWEN = "/data/LFT-W02_data/junjie/weights/Qwen3.5-2B"
K, P, NUM_CODES, PLAN_DIM = 4, 256, 2048, 1024   # keyframes, tokens/keyframe, codebook size, SigLIP dim
def log(m): print(m, flush=True)


class PlanQueryModule(nn.Module):
    """K*P learnable query tokens cross-attend to Qwen3.5 hidden states -> per-token query features.
    (Stand-in for the lingbot query; swap in the real lingbot block for the full planner.)"""
    def __init__(self, qwen_dim: int, n_query: int, heads: int = 8, depth: int = 2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_query, qwen_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(qwen_dim, heads, batch_first=True),
                "ln_q": nn.LayerNorm(qwen_dim), "ln_kv": nn.LayerNorm(qwen_dim),
                "ff": nn.Sequential(nn.Linear(qwen_dim, qwen_dim * 4), nn.GELU(), nn.Linear(qwen_dim * 4, qwen_dim)),
                "ln_ff": nn.LayerNorm(qwen_dim),
            }) for _ in range(depth)
        ])

    def forward(self, hidden: torch.Tensor, kv_mask: torch.Tensor | None = None) -> torch.Tensor:
        # hidden [B, L, Dq] ; returns [B, n_query, Dq]
        B = hidden.shape[0]
        x = self.q[None].expand(B, -1, -1).to(hidden.dtype)
        kpm = (~kv_mask.bool()) if kv_mask is not None else None
        for lyr in self.layers:
            a, _ = lyr["attn"](lyr["ln_q"](x), lyr["ln_kv"](hidden), hidden, key_padding_mask=kpm)
            x = x + a
            x = x + lyr["ff"](lyr["ln_ff"](x))
        return x


class Qwen35DiscretePlanner(nn.Module):
    def __init__(self, qwen, qwen_dim: int):
        super().__init__()
        self.qwen = qwen                                   # freeze (or LoRA) in real training
        self.query = PlanQueryModule(qwen_dim, K * P)
        self.code_head = ParallelCodePlanHead(qwen_dim, NUM_CODES)
        self.vq = SigLIPVQ(NUM_CODES, PLAN_DIM)

    def forward(self, qwen_inputs, gt_siglip=None):
        with torch.no_grad():                              # backbone frozen for this demo
            out = self.qwen(**qwen_inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1].float()             # [B, L, Dq]
        mask = qwen_inputs.get("attention_mask")
        qfeat = self.query(hidden, mask)                   # [B, K*P, Dq]
        logits = self.code_head(qfeat).reshape(hidden.shape[0], 1, K, P, NUM_CODES)  # [B,V=1,K,P,C]
        if gt_siglip is not None:
            loss, codes = discrete_plan_loss(logits, gt_siglip, self.vq)
            return logits, loss
        return logits, None

    @torch.no_grad()
    def plan_vectors(self, logits):
        return predict_plan_vectors(logits, self.vq)       # [B,V,K,P,1024] -> WAM


def main():
    proc = AutoProcessor.from_pretrained(QWEN)
    qwen = Qwen3_5ForConditionalGeneration.from_pretrained(QWEN, dtype=dt, device_map=dev).eval()
    qdim = qwen.config.text_config.hidden_size
    log(f"Qwen3.5-2B loaded, hidden={qdim}")
    model = Qwen35DiscretePlanner(qwen, qdim).to(dev)
    model.query = model.query.float(); model.code_head = model.code_head.float()  # heads run fp32 (hidden is fp32)

    # one (frame, instruction) sample
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "pick up the black bowl"}]}]
    txt = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[txt], images=[img], return_tensors="pt").to(dev)

    gt = torch.randn(1, 1, K, P, PLAN_DIM, device=dev)     # GT SigLIP plan (dummy; use real keyframe SigLIP in training)
    for _ in range(50):                                     # 1) warm the codebook on GT, then freeze targets
        model.vq.ema_update(gt, model.vq.quantize(gt))
    opt = torch.optim.AdamW(list(model.query.parameters()) + list(model.code_head.parameters()), lr=1e-3)
    for step in range(60):                                  # 2) fit the heads to the (now fixed) GT codes
        logits, loss = model(inp, gt.to(dt))               # ema=True inside, but codebook is warm -> stable
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0: log(f"  step {step}: code-CE loss={loss.item():.3f}")
    logits, _ = model(inp)
    plan = model.plan_vectors(logits)
    log(f"predicted plan for WAM: {tuple(plan.shape)} (1024-d codebook vectors, WAM unchanged)")
    log(f"code usage: {int(logits.argmax(-1).unique().numel())}/{NUM_CODES} unique codes")
    log("QWEN35-OPTION6-OK"); log("ALLDONE")


if __name__ == "__main__":
    main()
