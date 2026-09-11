import json
import random
import logging
from typing import Any, Optional
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, Audio

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


def load_manifest_as_dataset(manifest_path: Path, clips_dir: Path) -> Dataset:
    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            rows.append(
                {
                    "audio": str(clips_dir / entry["audio_filepath"]),
                    "sentence": entry["text"],
                    "category": entry.get("category", "generic"),
                }
            )
    dataset = Dataset.from_list(rows)
    return dataset.cast_column("audio", Audio(sampling_rate=16000))


def stratified_train_eval_split(
    dataset: Dataset, test_size: float = 0.15, seed: int = 42
) -> tuple[Dataset, Dataset]:
    random.seed(seed)

    indicies_by_category = {}
    for idx, category in enumerate(dataset["category"]):
        indicies_by_category.setdefault(category, []).append(idx)

    train_indicies, eval_indicies = [], []
    for category, indicies in indicies_by_category.items():
        shuffled = indicies.copy()
        random.shuffle(shuffled)
        n_eval = max(1, round(len(shuffled) * test_size))
        eval_indicies.extend(shuffled[:n_eval])
        train_indicies.extend(shuffled[n_eval:])
        logger.info(
            "category=%s train=%d eval=%d", category, len(shuffled) - n_eval, n_eval
        )

    random.shuffle(train_indicies)
    random.shuffle(eval_indicies)
    return dataset.select(train_indicies), dataset.select(eval_indicies)


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


def prepare_batch_augmented(batch: dict, processor: Any, augmenter) -> dict:
    input_features_list, labels_list = [], []
    for audio, sentence in zip(batch["audio"], batch["sentence"]):
        waveform = augmenter(audio["array"], audio["sampling_rate"])
        feats = processor.feature_extractor(
            waveform, sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        input_features_list.append(feats)

        labels = processor.tokenizer(sentence).input_ids
        if len(labels) > 225:
            labels = labels[:225]
        labels_list.append(labels)

    return {"input_features": input_features_list, "labels": labels_list}
