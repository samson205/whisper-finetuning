FROM python:3.12-slim

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch==2.13.0 torchaudio==2.11.0 torchcodec==0.16.0 bitsandbytes==0.50.2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENTRYPOINT ["python3", "-m", "src.cli"]