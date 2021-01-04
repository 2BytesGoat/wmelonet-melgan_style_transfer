import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    parent_path = Path(args.file_path)
    assert parent_path.exists()

    for genre_folder in parent_path.glob('*'):
        genre_type = genre_folder.name
        for idx, sound_file in enumerate((parent_path / genre_type).glob('*.wav')):
            if idx < 70:
                with open(parent_path / "train_files.txt", mode='a') as f:
                    f.write(genre_type + '/' + sound_file.name + '\n')
            else:
                with open(parent_path / "test_files.txt", mode='a') as f:
                    f.write(genre_type + '/' + sound_file.name + '\n')

