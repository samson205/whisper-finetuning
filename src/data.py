import json
from typing import Any, Optional
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, Audio


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch


def load_manifest_as_dataset(manifest_path: Path, clips_dir: Path) -> Dataset:
    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            rows.append({
                "audio": str(clips_dir.parent / entry["audio_filepath"]),
                "sentence": entry["text"],
            })
    dataset = Dataset.from_list(rows)
    return dataset.cast_column("audio", Audio(sampling_rate=16000))


def prepare_example(example: dict, processor: Any) -> Optional[dict]:
    audio = example["audio"]
    if len(audio["array"]) == 0:
        return None
    example["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    example["labels"] = processor.tokenizer(example["sentence"]).input_ids
    if len(example["labels"]) > 225:
        example["labels"] = example["labels"][:225]
    return example
