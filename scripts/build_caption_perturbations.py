#!/usr/bin/env python
"""Build semantic caption perturbations for MSCOCO retrieval experiments."""

from __future__ import annotations

import argparse
import ast
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


OBJECT_SWAPS = {
    "person": "dog",
    "man": "woman",
    "woman": "man",
    "boy": "girl",
    "girl": "boy",
    "child": "adult",
    "kid": "adult",
    "dog": "cat",
    "cat": "dog",
    "horse": "bicycle",
    "bicycle": "horse",
    "bike": "horse",
    "motorcycle": "bicycle",
    "moped": "bicycle",
    "car": "bus",
    "bus": "car",
    "truck": "train",
    "train": "truck",
    "airplane": "boat",
    "plane": "boat",
    "boat": "airplane",
    "bird": "kite",
    "kite": "bird",
    "chair": "bed",
    "bed": "chair",
    "table": "bench",
    "bench": "table",
    "cup": "bottle",
    "bottle": "cup",
    "plate": "bowl",
    "bowl": "plate",
    "banana": "pizza",
    "pizza": "banana",
    "sandwich": "cake",
    "cake": "sandwich",
    "laptop": "book",
    "keyboard": "book",
    "phone": "remote",
    "remote": "phone",
    "clock": "tv",
    "television": "clock",
    "tv": "clock",
    "umbrella": "skateboard",
    "skateboard": "umbrella",
    "surfboard": "snowboard",
    "snowboard": "surfboard",
    "skis": "surfboard",
    "tennis racket": "baseball bat",
    "baseball bat": "tennis racket",
}

COLOR_SWAPS = {
    "red": "blue",
    "blue": "red",
    "green": "yellow",
    "yellow": "green",
    "black": "white",
    "white": "black",
    "brown": "gray",
    "grey": "brown",
    "gray": "brown",
    "orange": "purple",
    "purple": "orange",
    "pink": "black",
}

SEMANTIC_DISTRACTOR_TEMPLATES = [
    (
        "target_unrelated",
        "Target image description: {caption} Unrelated description: {distractor}",
    ),
    (
        "revert target_unrelated",
        "Unrelated description: {distractor} Target image description: {caption}",
    ),
    (
        "caption_unrelated_note",
        "{caption} Unrelated note: {distractor}",
    ),
    (
        "image_shows_unrelated_sentence",
        "The image shows: {caption} The following sentence is unrelated to the image: {distractor}",
    ),
    (
        "ignore_unrelated_actual",
        "Ignore the following unrelated sentence: {distractor} The actual image shows: {caption}",
    ),
]


SPATIAL_SWAPS = {
    "in front of": "behind",
    "behind": "in front of",
    "next to": "far from",
    "beside": "away from",
    "near": "far from",
    "above": "under",
    "below": "over",
    "under": "over",
    "over": "under",
    "on top of": "under",
    "on": "under",
    "in": "outside",
    "inside": "outside",
    "outside": "inside",
    "left of": "right of",
    "right of": "left of",
}


@dataclass(frozen=True)
class Replacement:
    text: str
    source: str
    target: str
    category: str
    applied: bool
    template_id: int | str = ""
    template_name: str = ""


def parse_captions(raw_value: str) -> list[str]:
    captions = ast.literal_eval(raw_value)
    return [str(caption).strip() for caption in captions]


def compile_pattern(keys: list[str]) -> re.Pattern[str]:
    escaped = [re.escape(key) for key in sorted(keys, key=len, reverse=True)]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


OBJECT_PATTERN = compile_pattern(list(OBJECT_SWAPS))
ATTRIBUTE_PATTERN = compile_pattern(list(COLOR_SWAPS) + list(SPATIAL_SWAPS))


def match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target.capitalize()
    return target


def preserve_simple_plural(source: str, target: str) -> str:
    source_lower = source.lower()
    if source_lower.endswith("s") and not target.endswith("s") and " " not in target:
        return target + "s"
    return target


def replace_once(text: str, mapping: dict[str, str], pattern: re.Pattern[str], category: str) -> Replacement:
    match = pattern.search(text)
    if match is None:
        return Replacement("", "", "", category, False)

    source = match.group(0)
    target = mapping[source.lower()]
    target = preserve_simple_plural(source, target)
    target = match_case(source, target)
    corrupted = text[: match.start()] + target + text[match.end() :]
    return Replacement(corrupted, source, target, category, True)


def add_semantic_distractor(
    text: str,
    image_id: int,
    caption_pool: list[tuple[int, str]],
    rng: random.Random,
    template_id: int,
) -> Replacement:
    candidates = [caption for other_image_id, caption in caption_pool if other_image_id != image_id]
    distractor = rng.choice(candidates)
    template_name, template = SEMANTIC_DISTRACTOR_TEMPLATES[template_id]
    corrupted = template.format(caption=text.strip(), distractor=distractor.strip())
    return Replacement(
        corrupted,
        "",
        distractor,
        "semantic_distractor",
        True,
        template_id=template_id,
        template_name=template_name,
    )


def build_rows(input_csv: Path, seed: int) -> list[dict[str, object]]:
    frame = pd.read_csv(input_csv)
    expanded: list[dict[str, object]] = []

    for _, row in frame.iterrows():
        captions = parse_captions(row["raw"])
        for caption_index, caption in enumerate(captions):
            expanded.append(
                {
                    "image_id": int(row["cocoid"]),
                    "filename": row["filename"],
                    "caption_index": caption_index,
                    "caption": caption,
                }
            )

    caption_pool = [(int(row["image_id"]), str(row["caption"])) for row in expanded]
    rng = random.Random(seed)
    output_rows: list[dict[str, object]] = []

    for row_id, row in enumerate(expanded):
        image_id = int(row["image_id"])
        caption = str(row["caption"])

        perturbations = [
            add_semantic_distractor(
                caption,
                image_id,
                caption_pool,
                rng,
                template_id=row_id % len(SEMANTIC_DISTRACTOR_TEMPLATES),
            ),
            replace_once(caption, OBJECT_SWAPS, OBJECT_PATTERN, "object_swap"),
            replace_once(
                caption,
                {**COLOR_SWAPS, **SPATIAL_SWAPS},
                ATTRIBUTE_PATTERN,
                "attribute_spatial_swap",
            ),
        ]

        for perturbation in perturbations:
            output_rows.append(
                {
                    "row_id": row_id,
                    "image_id": image_id,
                    "filename": row["filename"],
                    "caption_index": row["caption_index"],
                    "perturbation": perturbation.category,
                    "applied": perturbation.applied,
                    "source": perturbation.source,
                    "target": perturbation.target,
                    "template_id": perturbation.template_id,
                    "template_name": perturbation.template_name,
                    "caption_original": caption,
                    "caption_corrupted": perturbation.text,
                }
            )

    return output_rows


def write_split_files(rows: list[dict[str, object]], split_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    split_dir.mkdir(parents=True, exist_ok=True)
    for perturbation, group in frame[frame["applied"].astype(bool)].groupby("perturbation"):
        group.to_csv(split_dir / f"{perturbation}.csv", index=False)


def write_summary(rows: list[dict[str, object]], summary_path: Path) -> None:
    summary = (
        pd.DataFrame(rows)
        .groupby("perturbation", as_index=False)
        .agg(total=("applied", "size"), applied=("applied", "sum"))
    )
    summary["coverage"] = summary["applied"] / summary["total"]
    summary.to_csv(summary_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("datasets/mscoco_2014_5k_test_image_text_retrieval/test_5k_mscoco_2014.csv"),
    )
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/caption_perturbations.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/caption_perturbations_summary.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("outputs/perturbations_by_type"))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rows = build_rows(args.input_csv, args.seed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_summary(rows, args.summary_csv)
    write_split_files(rows, args.split_dir)
    print(f"Wrote {len(rows)} perturbation rows to {args.output_csv}")
    print(f"Wrote summary to {args.summary_csv}")
    print(f"Wrote applied-only split files to {args.split_dir}")


if __name__ == "__main__":
    main()
