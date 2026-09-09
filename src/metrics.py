import logging

import evaluate

logger = logging.getLogger(__name__)


def load_metrics():
    return evaluate.load("wer")


def compute_metrics(pred, processor, wer_metric) -> dict:
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    for i in range(min(5, len(pred_str))):
        logger.info("Пример #%s", i + 1)
        logger.info("  Target: %s", label_str[i])
        logger.info("  Model: %s", pred_str[i])
        logger.info("-" * 80)

    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}
