import random
from pathlib import Path

import numpy as np
import torch
import torchaudio.transforms as T
from audiomentations import Gain, AddGaussianSNR, AddBackgroundNoise, Compose


class AudioAugmenter:
    def __init__(
        self, sr: int = 16000, p_augment: float = 0.6, noise_dir: Path | None = None
    ) -> None:
        self._sr = sr
        self._p_augment = p_augment

        transforms = [
            Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
            AddGaussianSNR(min_snr_db=10, max_snr_db=30, p=0.4),
        ]
        if noise_dir:
            transforms.append(
                AddBackgroundNoise(
                    sounds_path=noise_dir, min_snr_db=5, max_snr_db=20, p=0.4
                )
            )
        self._pipeline = Compose(transforms)

    def _speed_perturb(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        speed_factor = random.choice([0.9, 0.95, 1.0, 1.05, 1.1])
        if speed_factor == 1.0:
            return waveform
        speed_transform = T.Speed(sr, speed_factor)
        augmented, _ = speed_transform(waveform)
        return augmented

    def __call__(self, np_array: np.ndarray, sr: int) -> np.ndarray:
        if random.random() > self._p_augment:
            return np_array

        waveform = torch.from_numpy(np_array).float().unsqueeze(0)
        waveform = self._speed_perturb(waveform, sr)

        np_wave = waveform.squeeze(0).numpy().astype(np.float32)
        np_wave = self._pipeline(np_wave, sr)
        return np_wave
