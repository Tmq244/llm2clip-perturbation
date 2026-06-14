#!/usr/bin/env python
"""Create focused figures for semantic perturbation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PERTURBATIONS = ["semantic_distractor", "object_swap", "attribute_spatial_swap"]
MODELS = ["clip", "llm2clip"]
LABELS = {
    "clip": "CLIP",
    "llm2clip": "LLM2CLIP",
    "semantic_distractor": "Semantic distractor",
    "object_swap": "Object swap",
    "attribute_spatial_swap": "Color/spatial swap",
}
COLORS = {"clip": "#4C78A8", "llm2clip": "#F58518"}


def load_summary(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line.replace("NaN", "null")))
    return pd.DataFrame(rows)


def finish(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=13, pad=12)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(PERTURBATIONS))
    width = 0.36
    for offset, model in [(-width / 2, "clip"), (width / 2, "llm2clip")]:
        data = summary[(summary.model == model) & (summary.perturbation.isin(PERTURBATIONS))]
        means, lows, highs = [], [], []
        for perturbation in PERTURBATIONS:
            row = data[data.perturbation == perturbation].iloc[0]
            means.append(row.delta_mean)
            lows.append(row.delta_mean - row.delta_q25)
            highs.append(row.delta_q75 - row.delta_mean)
        bars = ax.bar(x + offset, means, width, yerr=[lows, highs], capsize=4, label=LABELS[model], color=COLORS[model])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[p] for p in PERTURBATIONS], rotation=10, ha="right")
    finish(ax, "Sensitivity to Perturbations", "Mean delta = s_original - s_corrupted")
    save(fig, out / "01_sensitivity_to_perturbation.png")


def plot_before_after(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    y_max = summary[summary.perturbation.isin(PERTURBATIONS)]["s_original_mean"].max() + 0.03

    for ax, perturbation in zip(axes, PERTURBATIONS):
        x = np.arange(len(MODELS))
        width = 0.34
        originals, corrupteds = [], []
        for model in MODELS:
            row = summary[(summary.model == model) & (summary.perturbation == perturbation)].iloc[0]
            originals.append(row.s_original_mean)
            corrupteds.append(row.s_corrupted_mean)

        original_colors = [COLORS[m] for m in MODELS]
        perturbed_colors = [COLORS[m] for m in MODELS]
        b1 = ax.bar(
            x - width / 2,
            originals,
            width,
            label="Original",
            color=original_colors,
            alpha=0.95,
        )
        b2 = ax.bar(
            x + width / 2,
            corrupteds,
            width,
            label="Perturbed",
            color=perturbed_colors,
            alpha=0.45,
            hatch="//",
            edgecolor=perturbed_colors,
            linewidth=0.8,
        )
        ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
        ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)

        ax.set_title(LABELS[perturbation], fontsize=12, pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in MODELS])
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Mean cosine similarity", fontsize=11)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#777777", alpha=0.95, label="Original"),
        Patch(facecolor="#777777", alpha=0.45, hatch="//", label="Perturbed"),
        Patch(facecolor=COLORS["clip"], label="CLIP"),
        Patch(facecolor=COLORS["llm2clip"], label="LLM2CLIP"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Similarity With vs. Without Perturbation", fontsize=14, y=1.08)
    save(fig, out / "02_similarity_original_vs_perturbed.png")


def plot_positive_rate(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(PERTURBATIONS))
    width = 0.36
    for offset, model in [(-width / 2, "clip"), (width / 2, "llm2clip")]:
        vals = []
        for perturbation in PERTURBATIONS:
            row = summary[(summary.model == model) & (summary.perturbation == perturbation)].iloc[0]
            vals.append(row.delta_positive_rate * 100)
        bars = ax.bar(x + offset, vals, width, label=LABELS[model], color=COLORS[model])
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[p] for p in PERTURBATIONS], rotation=10, ha="right")
    finish(ax, "Fraction of Samples Where Perturbation Reduces Similarity", "Positive delta rate")
    save(fig, out / "03_positive_delta_rate.png")


def load_details(output_dir: Path) -> pd.DataFrame:
    frames = []
    for model in MODELS:
        for perturbation in PERTURBATIONS:
            path = output_dir / f"{model}_{perturbation}.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["model"] = model
                df["perturbation"] = perturbation
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_delta_distribution(details: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    positions, data, colors, labels = [], [], [], []
    pos = 1
    for perturbation in PERTURBATIONS:
        for model in MODELS:
            vals = details[(details.model == model) & (details.perturbation == perturbation)]["delta"].dropna()
            data.append(vals.sample(min(len(vals), 4000), random_state=7).to_numpy())
            positions.append(pos)
            colors.append(COLORS[model])
            labels.append(f"{LABELS[perturbation]}\n{LABELS[model]}")
            pos += 1
        pos += 0.7
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    finish(ax, "Delta Distribution by Perturbation", "delta")
    save(fig, out / "04_delta_distribution_boxplot.png")


def plot_semantic_templates(details: pd.DataFrame, out: Path) -> None:
    data = details[details.perturbation == "semantic_distractor"].copy()
    if "template_name" not in data.columns:
        return
    grouped = data.groupby(["model", "template_id", "template_name"], as_index=False).agg(
        delta_mean=("delta", "mean"),
        positive_rate=("delta", lambda x: (x > 0).mean() * 100),
    )
    names = grouped.sort_values("template_id")["template_name"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    width = 0.36
    for offset, model in [(-width / 2, "clip"), (width / 2, "llm2clip")]:
        vals = []
        for name in names:
            vals.append(grouped[(grouped.model == model) & (grouped.template_name == name)].delta_mean.iloc[0])
        bars = ax.bar(x + offset, vals, width, label=LABELS[model], color=COLORS[model])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    finish(ax, "Semantic Distractor Sensitivity by Template", "Mean delta")
    save(fig, out / "05_semantic_distractor_by_template.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_perturbation_eval"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()

    summary_path = args.summary or args.output_dir / "summary.jsonl"
    figures_dir = args.figures_dir or args.output_dir / "figures_v2"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    details = load_details(args.output_dir)
    plot_sensitivity(summary, figures_dir)
    plot_before_after(summary, figures_dir)
    plot_positive_rate(summary, figures_dir)
    plot_delta_distribution(details, figures_dir)
    plot_semantic_templates(details, figures_dir)
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
