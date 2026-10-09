# HPC3 Repository Consolidation Implementation Plan

> Implement inline using superpowers:executing-plans; preserve the running job.

**Goal:** Consolidate the three Ola worktrees into the existing HPC3 training deployment as an independent Git repository, and publish to the existing GitHub repository on a new branch.

**Architecture:** Q35 is the active code base; verified HPC3 training files are authoritative for the 53k baseline. Historical worktree differences remain archived with provenance rather than overriding newer code.

**Tech Stack:** Git, tar, SSH, Pixi, Slurm.

**Spec:** User request dated 2026-10-09 in this session.

## Constraints and review focus
- Preserve original worktrees and queued job 704659.
- Use only HPC3 jhe724, at most 2 nodes / 16 GPUs.
- Do not publish credentials, datasets, weights, cache directories .
- Preserve distinct uncommitted source and experiment records; report exclusions.
- Verify final repository has its own .git and is independent of Ola.

## Tasks
- [x] Capture Git history and source inventories from all three Ola worktrees and HPC3.
- [x] Assemble active code, archive divergent old source, summarize results and operational lessons.
- [x] Restore independent repository on HPC3; run relevant tests and compare deployed training files.
- [x] Review publication contents, commit and push a dedicated GitHub branch; verify remote commit.

## Execution evidence

- Preserved 6,351 source-version mappings with matching SHA256; all mapped files staged.
- Kept active training and runtime files identical to the HPC3 frozen code.sha256.
- HPC3: data/config preflight passed, two node ranks + three allocation guards passed, six submission guards passed.
- HPC3: 51 RoboFollow tests passed in 14.04s after updating the stale planner probe to the existing two-pair contract from commit 88e0c1a. Original test archived.
- Root pytest.ini excludes docs/archive to prevent collecting old duplicate tests.
- Common credential/private-key pattern scan clean; archival whitespace preserved byte-for-byte.
- Reviewer verified source recovery, frozen hashes and honest result/status documentation; archive test collection finding fixed.
- Existing tracked reference PDFs remain in their existing history; no new datasets, weights, runtime environments or secrets added.

- HPC3 independent .git installed; clean tracked working tree, frozen deployment hashes and 6,351 source mappings verified remotely. GitHub consolidation branch published successfully. Job 704659 remains pending (Priority).
