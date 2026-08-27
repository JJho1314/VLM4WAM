"""Plan-regression objectives for the SG-WAM planner: MSE (current) / cosine / InfoNCE.

Why: the plan loss is where mean-collapse is born. `argmin E||y_hat - y||^2 = E[y]`, so MSE returns the
conditional MEAN over all valid futures -- a blurry, non-discriminative vector. Cosine only swaps that
for the mean DIRECTION (still an average), but it does fix norm-domination by SigLIP's high-norm
register tokens and it matches the cosine metric used by SigLIPVQ / cross-attention.

InfoNCE is the cheap objective that attacks what our probes actually measured (`../sg_probe/
RESULT_honest.md`: the predicted plan is not target-localizable). A regression loss never asks a token
to differ from any other token, so a planner can collapse all tokens toward each other and still score
well. InfoNCE with IN-IMAGE SPATIAL NEGATIVES demands exactly the missing property: token p's
prediction must match GT token p better than GT tokens at every other spatial position.

Drop-in: replace the plan MSE in the planner's training step. WAM side unchanged.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def mse_plan_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Current objective. Optimum = conditional mean -> blur. Also norm-dominated."""
    return F.mse_loss(pred.float(), gt.float())


def cosine_plan_loss(pred: torch.Tensor, gt: torch.Tensor, norm_weight: float = 0.0) -> torch.Tensor:
    """1 - cos. Kills norm-domination and matches the VQ/attention metric, but the optimum is still an
    average (the mean direction) -- it does NOT fix mean collapse. `norm_weight` > 0 re-adds a mild
    magnitude term, since SigLIP token norm correlates with objectness."""
    p, g = pred.float(), gt.float()
    loss = (1.0 - F.cosine_similarity(p, g, dim=-1)).mean()
    if norm_weight > 0.0:
        loss = loss + norm_weight * F.mse_loss(p.norm(dim=-1), g.norm(dim=-1))
    return loss


def infonce_plan_loss(
    pred: torch.Tensor,              # [B, V, K, P, D] predicted plan tokens
    gt: torch.Tensor,                # [B, V, K, P, D] GT SigLIP tokens
    *,
    temperature: float = 0.07,
    batch_negatives: int = 256,      # extra hard negatives drawn from OTHER (b,v,k) groups; 0 = spatial only
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Contrastive plan objective. For each keyframe image (group g = one (b,v,k)), token p must pick
    its own GT token out of all P spatial positions -- i.e. the plan is trained to be SPATIALLY
    DISCRIMINATIVE, which is the property a regression loss never requests. Optional batch negatives
    add cross-sample/instance discrimination. Returns a scalar."""
    b, v, k, p_n, d = pred.shape
    p = F.normalize(pred.reshape(-1, p_n, d).float(), dim=-1)     # [G, P, D]
    g = F.normalize(gt.reshape(-1, p_n, d).float(), dim=-1)
    n_g = p.shape[0]
    logits = torch.bmm(p, g.transpose(1, 2)) / temperature         # [G, P, P] in-image spatial negatives

    if batch_negatives > 0 and n_g > 1:
        flat = g.reshape(n_g * p_n, d)
        idx = torch.randperm(flat.shape[0], device=flat.device, generator=generator)[:batch_negatives]
        extra = torch.einsum("gpd,nd->gpn", p, flat[idx]) / temperature       # [G, P, N]
        own = (idx // p_n)[None, :] == torch.arange(n_g, device=flat.device)[:, None]   # [G, N] same group
        extra = extra.masked_fill(own[:, None, :], float("-inf"))            # never use own-image tokens twice
        logits = torch.cat([logits, extra], dim=-1)

    labels = torch.arange(p_n, device=logits.device).expand(n_g, p_n)
    return F.cross_entropy(logits.reshape(n_g * p_n, -1), labels.reshape(-1))


def plan_loss(pred: torch.Tensor, gt: torch.Tensor, kind: str = "infonce", **kw) -> torch.Tensor:
    """Dispatcher so the trainer can switch objectives from config: 'mse' | 'cosine' | 'infonce' |
    'infonce+cosine' (contrastive for discriminability + a light cosine anchor to stay in SigLIP space,
    which matters because the WAM's semantic_adapter was fit to raw SigLIP statistics)."""
    if kind == "mse":
        return mse_plan_loss(pred, gt)
    if kind == "cosine":
        return cosine_plan_loss(pred, gt, **kw)
    if kind == "infonce":
        return infonce_plan_loss(pred, gt, **kw)
    if kind == "infonce+cosine":
        w = kw.pop("cosine_weight", 0.5)
        return infonce_plan_loss(pred, gt, **kw) + w * cosine_plan_loss(pred, gt)
    raise ValueError(f"unknown plan loss kind: {kind}")


@torch.no_grad()
def plan_diagnostics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """Metrics that expose collapse, which a falling loss value hides.
      retrieval@1 : fraction of tokens whose prediction is nearest its OWN GT token among all P
                    positions of the same image -> spatial discriminability (the probed property).
      self_sim    : mean pairwise cosine BETWEEN a prediction's own tokens. Compare to gt_self_sim:
                    pred >> gt means the tokens have collapsed toward one another.
      cos_gt      : plain per-token cosine to GT (fidelity)."""
    b, v, k, p_n, d = pred.shape
    p = F.normalize(pred.reshape(-1, p_n, d).float(), dim=-1)
    g = F.normalize(gt.reshape(-1, p_n, d).float(), dim=-1)
    sim = torch.bmm(p, g.transpose(1, 2))                                    # [G, P, P]
    ret1 = (sim.argmax(dim=-1) == torch.arange(p_n, device=p.device)).float().mean()
    off = ~torch.eye(p_n, dtype=torch.bool, device=p.device)
    return {
        "retrieval@1": float(ret1),
        "self_sim": float(torch.bmm(p, p.transpose(1, 2))[:, off].mean()),
        "gt_self_sim": float(torch.bmm(g, g.transpose(1, 2))[:, off].mean()),
        "cos_gt": float((p * g).sum(-1).mean()),
    }
