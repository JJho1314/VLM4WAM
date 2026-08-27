# LaWAM-Aligned Qwen Vision Joint Training Design

## Goal

Retrain the joint VLM + GE-Act model with the Qwen3-VL vision encoder
trainable, using the official LaWAM LIBERO SFT recipe as the source of truth
for Qwen freeze policy and optimizer settings. The effective global batch size
must be 256 on eight GPUs.

Reference repository: `RLinf/LaWAM` at commit
`1add20a376126eacab02f19a62d726072a322cae`.

## Configuration Contract

The joint training recipe will use:

- `batch_size: 4`
- `gradient_accumulation_steps: 8`
- eight distributed workers, producing global batch `4 * 8 * 8 = 256`
- `train_steps: 25000`
- `lr_warmup_steps: 1500`
- cosine decay with minimum learning rate `5e-7`
- AdamW betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `1e-8`
- gradient clipping at `1.0`
- seed `2026`
- checkpoints at steps 5000, 10000, 15000, 20000, and 25000

Micro-batch 4 is intentionally retained because this model also trains LTX
and the action expert. Matching LaWAM's per-device batch 32 would not be a
memory-equivalent configuration; the effective global batch is the invariant.

## Qwen Freeze and Learning-Rate Policy

The Qwen policy will match LaWAM's LIBERO recipe:

- train the complete Qwen vision backbone, including its merger modules
- freeze the first 16 Qwen language-model transformer layers
- train the remaining Qwen language-model layers
- freeze the token embeddings
- freeze the LM head
- use learning rate `1e-4` for every trainable Qwen/VLM parameter
- keep Qwen gradient checkpointing disabled, matching the reference code

SigLIP2 and DA3 remain frozen target encoders. They are not part of Qwen and
LaWAM's Qwen fine-tuning policy does not imply training these teachers.

## GE-Act-Specific Parameters

The existing GE-Act architecture, semantic-token layout, losses, LTX/action
parameter groups, pretrained initialization, and dual-camera data path remain
unchanged. Their learning rates remain architecture-specific. Only optimizer
settings shared by the whole AdamW instance and the global scheduler adopt the
LaWAM values.

This avoids treating LaWAM's LAM decoder as if it were structurally equivalent
to the much larger LTX video/action model.

## Implementation

1. Extend the joint planner trainability policy to freeze Qwen embeddings and
   the first configured number of language layers while leaving vision
   trainable.
2. Split Qwen vision and Qwen language parameters into explicit optimizer
   groups so their membership is testable; both use `1e-4` in this recipe.
3. Add minimum-LR cosine scheduling support to the joint trainer.
4. Update the HPC3 recipe and preflight checks to enforce the new contract.
5. Update unit tests first, verify they fail under the old behavior, then make
   the implementation pass.

## Verification

Automated checks must prove:

- Qwen vision parameters require gradients and appear in the optimizer
- Qwen embeddings, LM head, and first 16 language layers do not require
  gradients
- later language layers remain trainable
- Qwen vision and language groups both use `1e-4`
- optimizer groups are disjoint and cover every trainable parameter
- effective global batch is 256
- checkpoints are saved every 5000 steps through step 25000
- the recipe uses 25k steps, 1500 warmup steps, cosine minimum LR `5e-7`,
  AdamW `(0.9, 0.95)`, epsilon `1e-8`, and weight decay `1e-8`
- preflight accepts the aligned recipe and rejects a re-frozen vision encoder

Before a full run, a one-step distributed smoke test must also confirm that
vision gradients are finite and optimizer setup fits GPU memory.
