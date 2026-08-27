"""Qwen3.5-2B discrete-plan planner, trained on Ola (8xH100) against PRECOMPUTED codes.

Difference from the HPC3 trainer: the VQ codebook is fitted offline and the targets are stored as
int16 code indices, so no features are loaded and no quantization happens in the training loop --
the dataset is ~3.5 GB instead of ~47 GB.

Qwen3.5-2B reads (current frame + instruction); K*P learnable queries cross-attend to its hidden
states; a per-token classifier predicts one of NUM_CODES codes IN PARALLEL (no autoregression).
Inference maps the argmax codes back through the codebook to a 1024-d plan the WAM consumes unchanged.

FULL_FT=1 fine-tunes the backbone (warmup then cosine). Set
GRADIENT_CHECKPOINTING=0 to trade activation memory for higher throughput.
Run: torchrun --nproc_per_node=8 ola_train_codes.py
"""
import os, sys, math, glob, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sg_qwen35_plan import PlanQueryModule
from sg_discrete_plan import ParallelCodePlanHead
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

HOME = os.path.expanduser("~")
QWEN = os.environ.get("QWEN_DIR", f"{HOME}/workspace/VLM4WAM/weights/Qwen3.5-2B")
DATA = os.environ.get("DATA_DIR", f"{HOME}/workspace/VLM4WAM/data/qwen35_codes")
OUT = os.environ.get("OUT_DIR", f"{HOME}/workspace/VLM4WAM/runs/qwen35_discrete_ola")
K, P, D = 4, 256, 1024
STEPS = int(os.environ.get("MAX_STEPS", 15000))
BS = int(os.environ.get("BATCH_SIZE", 4))
LR = float(os.environ.get("LR", 1e-5))            # backbone lr when full-FT
HEAD_LR = float(os.environ.get("HEAD_LR", 1e-4))
WARMUP = int(os.environ.get("WARMUP_STEPS", 300))
SAVE = int(os.environ.get("SAVE_STEPS", 5000))    # full-FT ckpts carry the 2B backbone (~5GB each)
EVAL = int(os.environ.get("EVAL_STEPS", 1000))    # eval is cheap -- keep visibility between saves
FULL_FT = int(os.environ.get("FULL_FT", 1))
GRADIENT_CHECKPOINTING = int(os.environ.get("GRADIENT_CHECKPOINTING", 1))
if GRADIENT_CHECKPOINTING not in (0, 1):
    raise ValueError("GRADIENT_CHECKPOINTING must be 0 or 1")
dt = torch.bfloat16


def is_main(): return int(os.environ.get("RANK", 0)) == 0
def log(m):
    if is_main(): print(m, flush=True)


class Planner(nn.Module):
    def __init__(self, qwen, qdim, num_codes, full_ft, gradient_checkpointing):
        super().__init__()
        self.qwen, self.full_ft = qwen, full_ft
        if not full_ft:
            for p in qwen.parameters(): p.requires_grad_(False)
        else:
            if gradient_checkpointing:
                qwen.gradient_checkpointing_enable()
            elif hasattr(qwen, "gradient_checkpointing_disable"):
                qwen.gradient_checkpointing_disable()
            qwen.config.use_cache = False
            if hasattr(qwen, "lm_head"):
                for p in qwen.lm_head.parameters(): p.requires_grad_(False)   # unused -> keeps DDP quiet
        self.query = PlanQueryModule(qdim, K * P).float()
        self.code_head = ParallelCodePlanHead(qdim, num_codes).float()
        self.num_codes = num_codes

    def forward(self, inp, codes=None):
        ctx = torch.enable_grad() if self.full_ft else torch.no_grad()
        with ctx:
            hs = self.qwen(**inp, output_hidden_states=True).hidden_states[-1].float()
        q = self.query(hs, inp.get("attention_mask"))
        logits = self.code_head(q).reshape(hs.shape[0], K, P, self.num_codes)
        if codes is None: return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, self.num_codes), codes.reshape(-1))


def load_shards():
    rgb, codes, pr = [], [], []
    for f in sorted(glob.glob(f"{DATA}/*_rgb.npy")):
        s = f[:-8]
        rgb.append(np.load(f, mmap_mode="r"))
        codes.append(np.load(f"{s}_codes.npy"))
        pr.append(np.load(f"{s}_prompt.npy", allow_pickle=True))
    return (np.concatenate([np.asarray(r) for r in rgb]), np.concatenate(codes), np.concatenate(pr))


def make_inputs(proc, rgb_b, prompts, dev):
    msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}] for p in prompts]
    txt = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    return proc(text=txt, images=[Image.fromarray(np.asarray(r)) for r in rgb_b],
                return_tensors="pt", padding=True).to(dev)


@torch.no_grad()
def evaluate(core, proc, rgb, codes, pr, va, cb, dev, nb=8):
    core.query.eval(); core.code_head.eval()
    acc, ret, n = 0.0, 0.0, 0
    for i in range(0, min(len(va), nb * 8), 8):
        bi = va[i:i + 8]
        logits, _ = core(make_inputs(proc, rgb[bi], [str(x) for x in pr[bi]], dev))
        gt = torch.from_numpy(codes[bi].astype(np.int64)).to(dev)
        pred = logits.argmax(-1)
        acc += float((pred == gt).float().mean()); n += 1
        pv = F.normalize(F.embedding(pred.reshape(-1, P), cb), dim=-1)      # (B*K,P,D)
        gv = F.normalize(F.embedding(gt.reshape(-1, P), cb), dim=-1)
        sim = torch.bmm(pv, gv.transpose(1, 2))
        ret += float((sim.argmax(-1) == torch.arange(P, device=dev)).float().mean())
    log(f"  [val] code-acc={acc/max(n,1):.4f} retrieval@1={ret/max(n,1):.4f}")
    core.query.train(); core.code_head.train()


def main():
    ws = int(os.environ.get("WORLD_SIZE", 1)); rk = int(os.environ.get("RANK", 0))
    lrk = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lrk)
    if ws > 1: dist.init_process_group("nccl")
    dev = f"cuda:{lrk}"; os.makedirs(OUT, exist_ok=True)

    cb = torch.from_numpy(np.load(f"{DATA}/codebook.npy")).float().to(dev)
    num_codes = cb.shape[0]
    proc = AutoProcessor.from_pretrained(QWEN)
    qwen = Qwen3_5ForConditionalGeneration.from_pretrained(QWEN, dtype=dt).to(dev)
    qwen.train() if FULL_FT else qwen.eval()
    qdim = qwen.config.text_config.hidden_size
    model = Planner(
        qwen, qdim, num_codes, bool(FULL_FT), bool(GRADIENT_CHECKPOINTING)
    ).to(dev)
    model.query = model.query.to(dev).float(); model.code_head = model.code_head.to(dev).float()

    rgb, codes, pr = load_shards()
    n = len(rgb); rng = np.random.default_rng(0); perm = rng.permutation(n)
    ntr = int(n * 0.95); tr, va = perm[:ntr], perm[ntr:]
    log(
        f"data N={n} train={len(tr)} val={len(va)} codes={num_codes} world={ws} "
        f"full_ft={FULL_FT} batch={BS} grad_ckpt={GRADIENT_CHECKPOINTING}"
    )

    core = model
    if FULL_FT:
        if ws > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[lrk], find_unused_parameters=True)
            core = model.module
        groups = [{"params": [p for p in core.qwen.parameters() if p.requires_grad], "lr": LR},
                  {"params": list(core.query.parameters()) + list(core.code_head.parameters()), "lr": HEAD_LR}]
    else:
        if ws > 1:
            core.query = nn.parallel.DistributedDataParallel(core.query, device_ids=[lrk])
            core.code_head = nn.parallel.DistributedDataParallel(core.code_head, device_ids=[lrk])
        groups = [{"params": list(core.query.parameters()) + list(core.code_head.parameters()), "lr": HEAD_LR}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    params = [p for g in groups for p in g["params"]]

    def lr_lambda(s):
        if s < WARMUP: return (s + 1) / max(1, WARMUP)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - WARMUP) / max(1, STEPS - WARMUP))))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    grng = np.random.default_rng(1000 + rk)

    for step in range(STEPS):
        bi = tr[grng.choice(len(tr), BS, replace=False)]
        inp = make_inputs(proc, rgb[bi], [str(x) for x in pr[bi]], dev)
        gt = torch.from_numpy(codes[bi].astype(np.int64)).to(dev)
        _, loss = (model if FULL_FT else core)(inp, gt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
        if step % 50 == 0: log(f"step {step}/{STEPS} code-CE={loss.item():.4f} lr={sch.get_last_lr()[-1]:.2e}")
        last = step + 1 == STEPS
        if (step + 1) % EVAL == 0 or last:
            if is_main():
                log(f"[eval @ step {step+1}]")
                evaluate(core, proc, rgb, codes, pr, va, cb, dev)
            if ws > 1: dist.barrier()
        if (step + 1) % SAVE == 0 or last:
            if is_main():
                sd = {"query": core.query.state_dict(), "code_head": core.code_head.state_dict(),
                      "codebook": cb.cpu(), "step": step + 1,
                      "cfg": {"K": K, "P": P, "D": D, "num_codes": num_codes,
                              "full_ft": FULL_FT, "batch_size": BS,
                              "gradient_checkpointing": GRADIENT_CHECKPOINTING}}
                if FULL_FT: sd["qwen"] = {k: v.cpu() for k, v in core.qwen.state_dict().items()}
                torch.save(sd, f"{OUT}/step_{step+1}.pt"); log(f"  saved {OUT}/step_{step+1}.pt")
            if ws > 1: dist.barrier()
    log("ALLDONE")


if __name__ == "__main__":
    main()
