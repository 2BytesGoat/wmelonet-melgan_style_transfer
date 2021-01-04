from mel2wav import MelVocoder

from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch

from mel2wav.utils import save_sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=Path, required=False)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vocoder = MelVocoder(args.load_path, github=True)

    args.save_path.mkdir(exist_ok=True, parents=True)

    for i, fname in tqdm(enumerate(args.folder.glob("*.wav"))):
        wavname = fname.name
        wav, sr = librosa.core.load(fname)

        mel = vocoder(torch.from_numpy(wav)[None])
        recons = vocoder.inverse(mel).squeeze().cpu()

        save_sample(args.save_path / wavname, sr, recons)


if __name__ == "__main__":
    main()
