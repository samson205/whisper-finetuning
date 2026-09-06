from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    MODEL_NAME: str = "openai/whisper-large-v3"
    TEMP_CHECKPOINTS_DIR: Path = Path("./whisper-lora-checkpoints")
    FINAL_CHECKPOINT_DIR: Path = Path("./final_lora_adapters")


settings = Settings()
