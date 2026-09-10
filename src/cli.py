import argparse
from pathlib import Path

from src.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=Path("training.log"))
    parser.add_argument("--noise-dir", type=Path, required=False)
    parser.add_argument("--p-augment", type=float, default=0.6)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--memory-fraction", type=float, default=0.5)
    args = parser.parse_args()

    run_training(args.manifest, args.clips_dir, args.output_dir, args.log_file, args.noise_dir, args.p_augment, args.gpu_device, args.memory_fraction)

if __name__ == "__main__":
    main()
