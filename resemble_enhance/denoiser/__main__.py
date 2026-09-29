import argparse
from pathlib import Path

import torch

from ..audio_io import load_audio, save_audio
from .inference import denoise


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=Path, help="Path to input audio folder")
    parser.add_argument("out_dir", type=Path, help="Output folder")
    parser.add_argument("--run_dir", type=Path, default="runs/denoiser", help="Path to run folder")
    parser.add_argument("--suffix", type=str, default=".wav", help="File suffix")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    for path in args.in_dir.glob(f"**/*{args.suffix}"):
        print(f"Processing {path} ..")
        dwav, sr = load_audio(path)
        hwav, sr = denoise(dwav[0], sr, args.run_dir, args.device)
        out_path = args.out_dir / path.relative_to(args.in_dir)
        save_audio(out_path, hwav[None], sr)


if __name__ == "__main__":
    main()
