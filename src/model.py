import logging
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def load_processor(model_name: str) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(model_name, language="russian", task="transcribe")


def load_model_with_lora(model_name: str, processor, lora_r: int, lora_alpha: int):
    logger.info("Загрузка базовой модели %s...", model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name, quantization_config=bnb_config,)
    model.config.use_cache = False
    model.generation_config.language = "ru" # type: ignore
    model.generation_config.task = "transcribe" # type: ignore
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="russian", task="transcribe") # type: ignore
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    target_modules = [
        name for name, _ in model.named_modules()
        if (name.startswith("model.encoder") or name.startswith("model.decoder"))
        and "encoder_attn" not in name
        and any(
            name.endswith(suffix) for suffix in (
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
            )
        )
    ]

    logger.info("LoRA target modules count=%d)", len(target_modules))

    assert all("encoder_attn" not in name for name in target_modules)
    assert all(name.endswith(("q_proj", "k_proj", "v_proj", "out_proj")) for name in target_modules)
    assert len(target_modules) > 0

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def merge_adapters(adapter_dir: Path, model_name: str, output_dir: Path) -> None:
    logger.info("Слияние модели с LoRA адаптерами")
    if not adapter_dir.exists() or not any(adapter_dir.iterdir()):
        raise FileNotFoundError(f"Адаптеры не найдены в {adapter_dir}")

    logger.info("Загрузка базовой модели %s на CPU в float16...", model_name)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    logger.info("Загрузка LoRA-адаптеров из %s...", adapter_dir)
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    logger.info("Слияние весов (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload() # type: ignore

    logger.info("Сохранение готовой модели в %s...", output_dir)
    merged_model.save_pretrained(str(output_dir))

    # Сохраняем процессор (токенизатор)
    processor = WhisperProcessor.from_pretrained(model_name, language="russian", task="transcribe")
    processor.save_pretrained(str(output_dir))

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    feature_extractor.save_pretrained(str(output_dir))

    logger.info("Слияние весов завершено, можно запускать ct2-transformers-converter")
    