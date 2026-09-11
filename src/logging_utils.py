import logging
from pathlib import Path

import transformers.utils.logging as tul
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path) -> None:
    log_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    tul.set_verbosity_info()
    tul.enable_default_handler()
    tul.add_handler(file_handler)


class FileLoggingCallback(TrainerCallback):
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs
    ):
        if logs is None:
            return
        logger.info("step=%s %s", state.global_step, logs)
