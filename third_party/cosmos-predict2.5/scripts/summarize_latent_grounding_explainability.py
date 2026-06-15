#!/usr/bin/env python3
"""Summarize latent-grounding explainability artifacts into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-probe", type=Path)
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--generation-metrics", type=Path)
    parser.add_argument("--eval-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    feature = load_json(args.feature_probe)
    gate = load_json(args.gate_json)
    gen = load_json(args.generation_metrics)
    eval_summary = load_json(args.eval_summary)

    lines: list[str] = []
    lines.append("# Latent Grounding Explainability Report")
    lines.append("")
    lines.append("## Method Path")
    lines.append("")
    lines.append("- Cosmos receives `target_feature` only; explicit target masks are removed for generation.")
    lines.append("- This architecture does not enable the target-feature cross-attention branch.")
    lines.append("- Therefore the primary explainability signals are feature saliency, gated latent residual strength, and keep/zero/drop generation deltas.")
    lines.append("")

    if feature:
        agg = feature.get("aggregate_mean", {})
        lines.append("## Feature Probe")
        lines.append("")
        lines.append("| Map | AUC | Inside/Outside | Pointing | Top10 Inside |")
        lines.append("|---|---:|---:|---:|---:|")
        for name in ["feature_norm", "centered_energy", "pca1_abs", "oracle_linear_probe"]:
            metrics = agg.get(name, {})
            lines.append(
                f"| `{name}` | {fmt(metrics.get('auc'))} | {fmt(metrics.get('inside_outside_ratio'))} | "
                f"{fmt(metrics.get('pointing_game'))} | {fmt(metrics.get('top10_inside_fraction'))} |"
            )
        lines.append("")
        lines.append("Interpretation: high centered/PCA/probe scores mean the InstructSAM dense hidden field contains target-location information even when raw feature norm is not target-aligned.")
        lines.append("")

    if gate:
        values = gate.get("gate_values", {})
        lines.append("## Latent Gate Probe")
        lines.append("")
        lines.append("| Key | raw | tanh(raw) |")
        lines.append("|---|---:|---:|")
        for key in sorted(values):
            if key.startswith("net_ema."):
                continue
            item = values[key]
            raw = item.get("raw", [None])[0]
            tanh = item.get("tanh", [None])[0]
            lines.append(f"| `{key}` | {fmt(raw)} | {fmt(tanh)} |")
        lines.append("")
        lines.append("Interpretation: larger tanh(gate) means the latent grounding branch has a stronger direct residual effect on DiT tokens.")
        lines.append("")

    if gen:
        variants = gen.get("variants", {})
        lines.append("## Generation Ablation")
        lines.append("")
        lines.append("| Variant | Mean Abs RGB vs keep | Target Abs | Background Abs | Target/Background |")
        lines.append("|---|---:|---:|---:|---:|")
        for name in variants:
            m = variants[name]
            lines.append(
                f"| `{name}` | {fmt(m.get('mean_abs_rgb'))} | {fmt(m.get('target_mask_mean_abs_rgb'))} | "
                f"{fmt(m.get('background_mean_abs_rgb'))} | {fmt(m.get('target_to_background_diff_ratio'))} |"
            )
        lines.append("")
        lines.append("Interpretation: keep-vs-zero/drop measures whether target features affect generation. A target/background ratio > 1 indicates localized target-region influence; < 1 indicates more global scene/action influence.")
        lines.append("")

    if eval_summary:
        lines.append("## Artifact Paths")
        lines.append("")
        for key in ["run_root", "contact_sheet", "diff_sheet", "metrics"]:
            if key in eval_summary:
                lines.append(f"- `{key}`: `{eval_summary[key]}`")
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
