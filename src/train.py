import logging
import functools
from pathlib import Path

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.config import settings
from src.data import load_manifest_as_dataset
from src.model import load_model_with_lora, load_processor, merge_adapters
from src.metrics import load_metrics, compute_metrics
from src.data import prepare_example, DataCollatorSpeechSeq2SeqWithPadding

logger = logging.getLogger(__name__)


def run_training(manifest_path: Path, clips_dir: Path, output_dir: Path) -> None:
    # 1. Данные
    logger.info("Загрузка manifest...")
    full_dataset = load_manifest_as_dataset(manifest_path, clips_dir)

    split = full_dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info("Train: %d, Eval: %d", len(train_dataset), len(eval_dataset))

    # 2. Модель
    model = load_model_with_lora(settings.MODEL_NAME)
    processor = load_processor(settings.MODEL_NAME)
    
    # 3. Препроцессинг
    logger.info("Извлечение признаков из аудио...")
    train_dataset = train_dataset.map(lambda ex: prepare_example(ex, processor), remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(lambda ex: prepare_example(ex, processor), remove_columns=eval_dataset.column_names)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 4. Метрика
    wer_metric = load_metrics()
    compute_metrics_fn = functools.partial(compute_metrics, processor=processor, wer_metric=wer_metric)

    # 5. Трейнер
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-lora-checkpoints",

        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=5,
        fp16=False,
        max_grad_norm=1.0,
        gradient_checkpointing=True,

        eval_strategy="epoch",
        # eval_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        # save_steps=10,

        per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_first_step=True,
        weight_decay=0.01,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    # TODO: вынести до подключения LoRA
    logger.info("WER базовой модели (до обучения)...")
    base_metrics = trainer.evaluate()
    logger.info("WER ДО обучения: %.4f", base_metrics["eval_wer"])

    logger.info("Запуск обучения...")
    trainer.train()

    final_metrics = trainer.evaluate()
    logger.info("WER ПОСЛЕ обучения: %.4f", final_metrics["eval_wer"])
    logger.info(
        "Изменение: %.4f -> %.4f (%s)",
        base_metrics["eval_wer"], final_metrics["eval_wer"],
        "улучшение" if final_metrics["eval_wer"] < base_metrics["eval_wer"] else "ухудшение/без изменений",
    )

    # 6. Сохранение и слияние адаптеров
    trainer.model.save_pretrained(settings.FINAL_CHECKPOINT_DIR)
    logger.info("Итоговые адаптеры сохранены в %s", settings.FINAL_CHECKPOINT_DIR)

    merge_adapters(settings.FINAL_CHECKPOINT_DIR, settings.MODEL_NAME, output_dir)
    