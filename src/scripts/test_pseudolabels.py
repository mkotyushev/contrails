import argparse
import cv2
import numpy as np
from pathlib import Path

from src.data.datasets import N_TIMES, LABELED_TIME_INDEX
from src.utils.utils import temp_seed


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(list(Path(args.input_dir).glob('*.npy')))

    assert len(paths) == 22385
    print(f'Total records: {len(paths)}')

    with temp_seed(0):
        sample_paths = np.random.choice(paths, args.num_samples, replace=False)
    
    for path in sample_paths:
        mask = np.load(path)

        assert mask.shape == (256, 256, N_TIMES)
        assert mask.dtype == np.uint8
        print(path, mask.shape, mask.dtype, mask.min(), mask.max(), np.unique(mask))
        
        cv2.imwrite(str(args.output_dir / f'{path.stem}.png'), mask[:, :, LABELED_TIME_INDEX])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path, help='directory with .npy pseudolabels')
    parser.add_argument('output_dir', type=Path, help='directory to save .png images')
    parser.add_argument('--num_samples', type=int, default=50)
    args = parser.parse_args()
    main(args)
