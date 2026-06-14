#!/usr/bin/env python
"""Evaluate CLIP/LLM2CLIP sensitivity to semantic caption perturbations."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPModel,
    BitsAndBytesConfig,
)


PERTURBATION_FILES = {
    "semantic_distractor": "semantic_distractor.csv",
    "object_swap": "object_swap.csv",
    "attribute_spatial_swap": "attribute_spatial_swap.csv",
}


@dataclass
class EvalBatch:
    frame: pd.DataFrame
    image_paths: list[Path]
    original_texts: list[str]
    corrupted_texts: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["clip", "llm2clip", "both"], default="both")
    parser.add_argument("--clip-model-path", default="models/clip-vit-large-patch14-336")
    parser.add_argument("--llm2clip-vision-path", default="models/LLM2CLIP-Openai-L-14-336")
    parser.add_argument("--llm2clip-text-path", default="models/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned")
    parser.add_argument("--perturbation-dir", type=Path, default=Path("outputs/perturbations_by_type"))
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("datasets/mscoco_2014_5k_test_image_text_retrieval/images_mscoco_2014_5k_test"),
    )
    parser.add_argument(
        "--image-zip",
        type=Path,
        default=Path("datasets/mscoco_2014_5k_test_image_text_retrieval/images_mscoco_2014_5k_test.zip"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_perturbation_eval"))
    parser.add_argument("--perturbations", nargs="+", choices=list(PERTURBATION_FILES), default=list(PERTURBATION_FILES))
    parser.add_argument("--skip-original-caption", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--llm-device-map", default="auto", help="auto, single, none, or any Transformers device_map string")
    parser.add_argument("--llm-load-in-8bit", action="store_true")
    parser.add_argument("--llm-load-in-4bit", action="store_true")
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--llm-text-batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def ensure_images(image_dir: Path, image_zip: Path) -> None:
    if image_dir.exists() and any(image_dir.glob("*.jpg")):
        return
    if not image_zip.exists():
        raise FileNotFoundError(f"Image directory not found and zip is missing: {image_zip}")
    image_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(image_zip) as archive:
        archive.extractall(image_dir.parent)


def batched(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_image_paths(frame: pd.DataFrame, image_dir: Path) -> list[Path]:
    paths = [image_dir / filename for filename in frame["filename"].tolist()]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} image files. First missing files:\n{preview}")
    return paths


def load_perturbation(path: Path, limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "applied" in frame.columns:
        frame = frame[frame["applied"].astype(bool)].copy()
    if limit is not None:
        frame = frame.head(limit).copy()
    return frame.reset_index(drop=True)


def make_eval_batch(frame: pd.DataFrame, image_dir: Path) -> EvalBatch:
    return EvalBatch(
        frame=frame,
        image_paths=resolve_image_paths(frame, image_dir),
        original_texts=frame["caption_original"].astype(str).tolist(),
        corrupted_texts=frame["caption_corrupted"].astype(str).tolist(),
    )


def load_original_captions(perturbation_dir: Path, limit: int | None) -> pd.DataFrame:
    source_path = perturbation_dir / PERTURBATION_FILES["semantic_distractor"]
    frame = pd.read_csv(source_path)
    columns = ["row_id", "image_id", "filename", "caption_index", "caption_original"]
    frame = frame[columns].drop_duplicates("row_id").copy()
    frame["perturbation"] = "original_caption"
    if limit is not None:
        frame = frame.head(limit).copy()
    return frame.reset_index(drop=True)


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.float(), dim=-1)


class ClipEvaluator:
    name = "clip"

    def __init__(self, model_path: str, device: str):
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def encode_images(self, image_paths: list[Path], batch_size: int) -> torch.Tensor:
        features = []
        for batch_paths in tqdm(list(batched(image_paths, batch_size)), desc="CLIP images"):
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float16)
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                batch_features = self.model.get_image_features(pixel_values=pixel_values)
            features.append(normalize(batch_features).cpu())
        return torch.cat(features)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str], batch_size: int) -> torch.Tensor:
        features = []
        for text_batch in tqdm(list(batched(texts, batch_size)), desc="CLIP texts"):
            inputs = self.processor(text=text_batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                batch_features = self.model.get_text_features(**inputs)
            features.append(normalize(batch_features).cpu())
        return torch.cat(features)


class LLM2ClipEvaluator:
    name = "llm2clip"

    def __init__(
        self,
        vision_path: str,
        text_path: str,
        device: str,
        llm_device_map: str,
        load_in_8bit: bool,
        load_in_4bit: bool,
        max_length: int,
        clip_model_path: str,
    ):
        from llm2vec import LLM2Vec

        self.device = torch.device(device)
        self.max_length = max_length
        self.processor = CLIPImageProcessor.from_pretrained(clip_model_path, local_files_only=True)
        self.vision_model = AutoModel.from_pretrained(
            vision_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device)
        self.vision_model.eval()

        config = AutoConfig.from_pretrained(text_path, trust_remote_code=True, local_files_only=True)
        # The HF config points to McGill-NLP remote code; use the local copy for offline runs.
        config.auto_map = {"AutoModel": "modeling_llama_encoder.LlamaEncoderModel"}
        kwargs = {
            "config": config,
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if llm_device_map == "single":
            device_index = self.device.index if self.device.index is not None else 0
            parsed_device_map = {"": device_index}
        elif llm_device_map.lower() in {"none", ""}:
            parsed_device_map = None
        else:
            parsed_device_map = llm_device_map

        if load_in_8bit or load_in_4bit:
            if parsed_device_map is not None:
                kwargs["device_map"] = parsed_device_map
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
        elif parsed_device_map is not None:
            kwargs["device_map"] = parsed_device_map
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float16

        llm_model = AutoModel.from_pretrained(text_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(text_path, local_files_only=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        llm_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.l2v = LLM2Vec(
            llm_model,
            tokenizer,
            pooling_mode="mean",
            max_length=max_length,
            doc_max_length=max_length,
        )

    @torch.inference_mode()
    def encode_images(self, image_paths: list[Path], batch_size: int) -> torch.Tensor:
        features = []
        for batch_paths in tqdm(list(batched(image_paths, batch_size)), desc="LLM2CLIP images"):
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device, dtype=torch.float16)
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                batch_features = self.vision_model.get_image_features(pixel_values)
            features.append(normalize(batch_features).cpu())
        return torch.cat(features)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str], batch_size: int) -> torch.Tensor:
        features = []
        for text_batch in tqdm(list(batched(texts, batch_size)), desc="LLM2Vec texts"):
            converted = [self.l2v._convert_to_str("", text) for text in text_batch]
            prepared = [self.l2v.prepare_for_tokenization(text) for text in converted]
            token_features = self.l2v.tokenize(prepared)
            token_features = {key: value.to(self.device) for key, value in token_features.items()}
            llm_features = self.l2v.forward(token_features).detach().to(self.device, dtype=torch.float16)
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                batch_features = self.vision_model.get_text_features(llm_features)
            features.append(normalize(batch_features).cpu())
        return torch.cat(features)


def cosine_for_pairs(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    return (image_features * text_features).sum(dim=-1)


def summarize(frame: pd.DataFrame, model_name: str, perturbation: str) -> dict[str, object]:
    delta = frame["delta"].to_numpy()
    positive = delta > 0
    return {
        "model": model_name,
        "perturbation": perturbation,
        "n": int(len(frame)),
        "delta_mean": float(frame["delta"].mean()),
        "delta_std": float(frame["delta"].std(ddof=0)),
        "delta_median": float(frame["delta"].median()),
        "delta_q25": float(frame["delta"].quantile(0.25)),
        "delta_q75": float(frame["delta"].quantile(0.75)),
        "delta_positive_rate": float(positive.mean()) if len(delta) else math.nan,
        "s_original_mean": float(frame["s_original"].mean()),
        "s_corrupted_mean": float(frame["s_corrupted"].mean()),
    }


def summarize_original(frame: pd.DataFrame, model_name: str) -> dict[str, object]:
    return {
        "model": model_name,
        "perturbation": "original_caption",
        "n": int(len(frame)),
        "delta_mean": math.nan,
        "delta_std": math.nan,
        "delta_median": math.nan,
        "delta_q25": math.nan,
        "delta_q75": math.nan,
        "delta_positive_rate": math.nan,
        "s_original_mean": float(frame["s_original"].mean()),
        "s_corrupted_mean": math.nan,
    }


def run_original(evaluator, frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    image_paths = resolve_image_paths(frame, args.image_dir)
    texts = frame["caption_original"].astype(str).tolist()
    text_batch_size = args.llm_text_batch_size if evaluator.name == "llm2clip" else args.text_batch_size

    image_features = evaluator.encode_images(image_paths, args.image_batch_size)
    text_features = evaluator.encode_texts(texts, text_batch_size)

    result = frame.copy()
    result["model"] = evaluator.name
    result["s_original"] = cosine_for_pairs(image_features, text_features).numpy()
    return result, summarize_original(result, evaluator.name)


def run_one(evaluator, batch: EvalBatch, args: argparse.Namespace, perturbation: str) -> tuple[pd.DataFrame, dict[str, object]]:
    image_features = evaluator.encode_images(batch.image_paths, args.image_batch_size)
    text_batch_size = args.llm_text_batch_size if evaluator.name == "llm2clip" else args.text_batch_size
    original_features = evaluator.encode_texts(batch.original_texts, text_batch_size)
    corrupted_features = evaluator.encode_texts(batch.corrupted_texts, text_batch_size)

    result = batch.frame.copy()
    result["model"] = evaluator.name
    result["s_original"] = cosine_for_pairs(image_features, original_features).numpy()
    result["s_corrupted"] = cosine_for_pairs(image_features, corrupted_features).numpy()
    result["delta"] = result["s_original"] - result["s_corrupted"]
    return result, summarize(result, evaluator.name, perturbation)


def save_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_summary(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = []
    valid_perturbations = {"original_caption", *PERTURBATION_FILES}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line.replace("NaN", "null"))
            if record.get("perturbation") in valid_perturbations:
                records.append(record)
    return records


def merge_summary_records(
    existing_records: list[dict[str, object]],
    new_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged = {(record["model"], record["perturbation"]): record for record in existing_records}
    for record in new_records:
        merged[(record["model"], record["perturbation"])] = record

    model_order = {"clip": 0, "llm2clip": 1}
    perturbation_order = {
        "original_caption": 0,
        "semantic_distractor": 1,
        "object_swap": 2,
        "attribute_spatial_swap": 3,
    }
    return sorted(
        merged.values(),
        key=lambda record: (
            perturbation_order.get(record["perturbation"], 99),
            model_order.get(record["model"], 99),
        ),
    )


def main() -> None:
    args = parse_args()
    ensure_images(args.image_dir, args.image_zip)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluators = []
    if args.model in {"clip", "both"}:
        evaluators.append(ClipEvaluator(args.clip_model_path, args.device))
    if args.model in {"llm2clip", "both"}:
        evaluators.append(
            LLM2ClipEvaluator(
                args.llm2clip_vision_path,
                args.llm2clip_text_path,
                args.device,
                args.llm_device_map,
                args.llm_load_in_8bit,
                args.llm_load_in_4bit,
                args.max_length,
                args.clip_model_path,
            )
        )

    summary_records = []
    if not args.skip_original_caption:
        original_frame = load_original_captions(args.perturbation_dir, args.limit)
        for evaluator in evaluators:
            result, summary = run_original(evaluator, original_frame, args)
            result_path = args.output_dir / f"{evaluator.name}_original_caption.csv"
            result.to_csv(result_path, index=False)
            summary_records.append(summary)
            print(json.dumps(summary, indent=2))

    for perturbation in args.perturbations:
        input_csv = args.perturbation_dir / PERTURBATION_FILES[perturbation]
        frame = load_perturbation(input_csv, args.limit)
        batch = make_eval_batch(frame, args.image_dir)
        for evaluator in evaluators:
            result, summary = run_one(evaluator, batch, args, perturbation)
            result_path = args.output_dir / f"{evaluator.name}_{perturbation}.csv"
            result.to_csv(result_path, index=False)
            summary_records.append(summary)
            print(json.dumps(summary, indent=2))

    summary_path = args.output_dir / "summary.jsonl"
    existing_records = load_existing_summary(summary_path)
    merged_records = merge_summary_records(existing_records, summary_records)
    summary = pd.DataFrame(merged_records)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    save_jsonl(summary_path, merged_records)
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
