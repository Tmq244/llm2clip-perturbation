#!/usr/bin/env python
"""Generate summary figures and a short report for semantic perturbation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "clip": "CLIP",
    "llm2clip": "LLM2CLIP",
    "original_caption": "Original",
    "semantic_distractor": "Semantic distractor",
    "object_swap": "Object swap",
    "attribute_spatial_swap": "Color/spatial swap",
}

COLORS = {
    "clip": "#4C78A8",
    "llm2clip": "#F58518",
}


def load_summary(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line.replace("NaN", "null")))
    return pd.DataFrame(records)


def save_bar(ax, path: Path, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def grouped_bars(ax, data: pd.DataFrame, x_col: str, y_col: str, order: list[str], ylabel: str) -> None:
    width = 0.36
    x = np.arange(len(order))
    for offset, model in [(-width / 2, "clip"), (width / 2, "llm2clip")]:
        values = []
        for item in order:
            subset = data[(data["model"] == model) & (data[x_col] == item)]
            values.append(float(subset[y_col].iloc[0]) if len(subset) else np.nan)
        bars = ax.bar(x + offset, values, width, label=LABELS[model], color=COLORS[model])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[item] for item in order], rotation=15, ha="right")
    ax.set_ylabel(ylabel)


def plot_original_similarity(summary: pd.DataFrame, figures_dir: Path) -> None:
    data = summary[summary["perturbation"] == "original_caption"]
    fig, ax = plt.subplots(figsize=(6, 4))
    models = ["clip", "llm2clip"]
    values = [float(data[data["model"] == model]["s_original_mean"].iloc[0]) for model in models]
    bars = ax.bar([LABELS[m] for m in models], values, color=[COLORS[m] for m in models])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, max(values) * 1.25)
    save_bar(ax, figures_dir / "original_similarity.png", "Original Caption Similarity", "Mean cosine similarity")


def plot_delta(summary: pd.DataFrame, figures_dir: Path) -> None:
    order = ["semantic_distractor", "object_swap", "attribute_spatial_swap"]
    data = summary[summary["perturbation"].isin(order)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    grouped_bars(ax, data, "perturbation", "delta_mean", order, "Mean delta: s_original - s_corrupted")
    ax.axhline(0, color="#333333", linewidth=0.8)
    save_bar(ax, figures_dir / "delta_mean_by_perturbation.png", "Sensitivity to Semantic Perturbations", "Mean delta")


def plot_positive_rate(summary: pd.DataFrame, figures_dir: Path) -> None:
    order = ["semantic_distractor", "object_swap", "attribute_spatial_swap"]
    data = summary[summary["perturbation"].isin(order)].copy()
    data["delta_positive_rate"] = data["delta_positive_rate"] * 100
    fig, ax = plt.subplots(figsize=(8, 4.6))
    grouped_bars(ax, data, "perturbation", "delta_positive_rate", order, "Positive delta rate (%)")
    ax.set_ylim(0, 110)
    save_bar(ax, figures_dir / "positive_rate_by_perturbation.png", "How Often Perturbations Reduce Similarity", "Positive delta rate (%)")


def plot_similarity_before_after(summary: pd.DataFrame, figures_dir: Path) -> None:
    order = ["semantic_distractor", "object_swap", "attribute_spatial_swap"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, model in zip(axes, ["clip", "llm2clip"]):
        data = summary[(summary["model"] == model) & (summary["perturbation"].isin(order))]
        x = np.arange(len(order))
        original = [float(data[data["perturbation"] == p]["s_original_mean"].iloc[0]) for p in order]
        corrupted = [float(data[data["perturbation"] == p]["s_corrupted_mean"].iloc[0]) for p in order]
        ax.plot(x, original, marker="o", label="Original", color="#54A24B", linewidth=2)
        ax.plot(x, corrupted, marker="o", label="Corrupted", color="#E45756", linewidth=2)
        ax.set_title(LABELS[model])
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[p] for p in order], rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Mean cosine similarity")
    plt.suptitle("Similarity Before vs. After Perturbation", fontsize=13)
    plt.tight_layout()
    plt.savefig(figures_dir / "similarity_before_after.png", dpi=180)
    plt.close()


def ratio(summary: pd.DataFrame, perturbation: str) -> float:
    clip = summary[(summary["model"] == "clip") & (summary["perturbation"] == perturbation)]["delta_mean"].iloc[0]
    llm = summary[(summary["model"] == "llm2clip") & (summary["perturbation"] == perturbation)]["delta_mean"].iloc[0]
    return float(llm / clip)


def write_report(summary: pd.DataFrame, output_dir: Path, figures_dir: Path) -> None:
    rows = {(r.model, r.perturbation): r for r in summary.itertuples(index=False)}
    rel_fig = figures_dir.relative_to(output_dir)
    object_ratio = ratio(summary, "object_swap")
    attr_ratio = ratio(summary, "attribute_spatial_swap")
    distractor_ratio = ratio(summary, "semantic_distractor")

    # The distractor direction flips depending on which model has the larger delta:
    # a ratio < 1.0 means LLM2CLIP's similarity drops LESS, i.e. it is more robust.
    if distractor_ratio >= 1.0:
        distractor_clause = (
            "Since this ratio is above 1.0, LLM2CLIP is more affected by the explicit unrelated description "
            "(its similarity drops further); that can be read as stronger sensitivity to added semantics, "
            "but lower robustness if the extra description should be ignored."
        )
        takeaway_text = (
            "LLM2CLIP improves semantic sensitivity and clean caption alignment, especially for object and "
            "attribute changes. The tradeoff is stronger sensitivity to extra unrelated text, which may be "
            "useful or undesirable depending on whether that text should be treated as part of the query."
        )
    else:
        distractor_clause = (
            "Since this ratio is below 1.0, LLM2CLIP's similarity drops LESS under the distractor: it is more "
            "robust to the explicitly-labelled unrelated description, consistent with following the "
            "'ignore'/'unrelated' instructions that CLIP's encoder cannot."
        )
        takeaway_text = (
            "LLM2CLIP improves semantic sensitivity and clean caption alignment, especially for object and "
            "attribute changes, while remaining more robust to explicitly-labelled distractor text — it better "
            "preserves the target caption similarity, consistent with following the natural-language instructions."
        )

    report = f"""# Semantic Perturbation Report

## Setup

Models compared:

- CLIP: `openai/clip-vit-large-patch14-336`
- LLM2CLIP: `microsoft/LLM2CLIP-Openai-L-14-336` + `microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned`

Dataset: MSCOCO 2014 5k image-text retrieval split. The metric for perturbation experiments is:

```text
delta = cosine(image_emb, text_emb(original_caption)) - cosine(image_emb, text_emb(corrupted_caption))
```

For object and color/spatial swaps, larger `delta` means the model is more sensitive to the semantic error. For semantic distractors, larger `delta` means the model is more affected by the additional unrelated description; lower `delta` means better robustness to that extra text.

## Figures

![Original caption similarity]({rel_fig}/original_similarity.png)

![Delta by perturbation]({rel_fig}/delta_mean_by_perturbation.png)

![Positive delta rate]({rel_fig}/positive_rate_by_perturbation.png)

![Similarity before and after perturbation]({rel_fig}/similarity_before_after.png)

## Summary Table

| Model | Perturbation | n | s_original_mean | s_corrupted_mean | delta_mean | positive_rate |
|---|---:|---:|---:|---:|---:|---:|
"""
    for _, row in summary.iterrows():
        s_corrupted = "" if pd.isna(row["s_corrupted_mean"]) else f"{row['s_corrupted_mean']:.4f}"
        delta = "" if pd.isna(row["delta_mean"]) else f"{row['delta_mean']:.4f}"
        positive = "" if pd.isna(row["delta_positive_rate"]) else f"{row['delta_positive_rate']:.3f}"
        report += (
            f"| {LABELS[row['model']]} | {LABELS[row['perturbation']]} | {int(row['n'])} | "
            f"{row['s_original_mean']:.4f} | {s_corrupted} | {delta} | {positive} |\n"
        )

    report += f"""
## Findings

1. LLM2CLIP has higher original-caption similarity than CLIP: `{rows[('llm2clip', 'original_caption')].s_original_mean:.4f}` vs `{rows[('clip', 'original_caption')].s_original_mean:.4f}`. This suggests stronger average alignment for clean image-caption pairs under this cosine setup.

2. LLM2CLIP is more sensitive to object swaps. Its mean delta is `{rows[('llm2clip', 'object_swap')].delta_mean:.4f}`, compared with CLIP's `{rows[('clip', 'object_swap')].delta_mean:.4f}`, about `{object_ratio:.2f}x` larger.

3. LLM2CLIP is also more sensitive to color/spatial swaps. Its mean delta is `{rows[('llm2clip', 'attribute_spatial_swap')].delta_mean:.4f}`, about `{attr_ratio:.2f}x` CLIP's `{rows[('clip', 'attribute_spatial_swap')].delta_mean:.4f}`. The absolute delta is still much smaller than object swap, so fine-grained attribute/spatial reasoning remains harder.

4. Semantic distractors reveal a sensitivity/robustness tradeoff. LLM2CLIP's mean delta is `{rows[('llm2clip', 'semantic_distractor')].delta_mean:.4f}`, about `{distractor_ratio:.2f}x` CLIP's `{rows[('clip', 'semantic_distractor')].delta_mean:.4f}`. {distractor_clause}

## Takeaway

{takeaway_text}
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("outputs/semantic_perturbation_eval/summary.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_perturbation_eval"))
    args = parser.parse_args()

    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(args.summary)

    plot_original_similarity(summary, figures_dir)
    plot_delta(summary, figures_dir)
    plot_positive_rate(summary, figures_dir)
    plot_similarity_before_after(summary, figures_dir)
    write_report(summary, args.output_dir, figures_dir)
    print(f"Wrote figures to {figures_dir}")
    print(f"Wrote report to {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
