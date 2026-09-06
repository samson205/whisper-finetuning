# Whisper Finetuning

A project for QLoRA or LoRA fine-tuning of Whisper models.

## Manifest Format

```json
{"audio_filepath": "example1.wav", "text": "Hello, my name is..."}
{"audio_filepath": "example2.wav", "text": "What's your name?"}
```

## Usage

```bash
git clone https://github.com/samson205/whisper-finetuning
cd whisper-finetuning
```

```bash
python -m src.cli --manifest path/to/manifest.jsonl --clips-dir path/to/clips_dir --output-dir path/to/output_dir
```
