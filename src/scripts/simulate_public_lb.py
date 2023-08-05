import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

from src.data.datasets import LABELED_TIME_INDEX 


def main(args):
    paths = sorted([list(d.glob('*.png')) for d in args.input_dirs])
    print(f'Total records: {len(paths)}')

    # Read data
    targets, preds = [], []
    for path in tqdm(paths, desc='reading data'):
        if f'_{LABELED_TIME_INDEX}_' not in path.stem:
            continue

        record_id = path.stem.split('_')[0]

        # Get target path
        target_path = args.dataset_dir / 'train' / record_id / 'human_pixel_masks.npy'
        if not target_path.exists():
            target_path = args.dataset_dir / 'validation' / record_id / 'human_pixel_masks.npy'
        assert target_path.exists()
        
        # Load data
        target = np.load(target_path)[..., 0]
        pred = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) / 255.0

        assert target.shape == pred.shape == (256, 256), (target.shape, pred.shape)
        assert target.dtype == np.int32, target.dtype
        assert pred.dtype == np.float64, pred.dtype

        targets.append(target)
        preds.append(pred)

    # Score full dataset
    metric = BinaryF1Score(threshold=args.threshold)
    target = np.concatenate([targets[i] for i in range(len(targets))])
    pred = np.concatenate([preds[i] for i in range(len(preds))])
    metric.update(torch.tensor(pred).flatten(), torch.tensor(target).flatten())
    print(f'Full dataset score: {metric.compute().item()}')
    metric.reset()

    # Bootstrap
    metric_values = []
    for _ in  tqdm(range(args.n_bootstrap), desc='bootstrap'):
        # Sample
        sample_indices = np.random.choice(len(paths), args.n_samples, replace=args.replace)
        target = np.concatenate([targets[i] for i in sample_indices])
        pred = np.concatenate([preds[i] for i in sample_indices])

        # Update metric
        metric.update(torch.tensor(pred).flatten(), torch.tensor(target).flatten())
        metric_values.append(metric.compute().item())
        metric.reset()

    # Plot histogram
    plt.hist(metric_values, bins=50, density=True)
    plt.savefig(f'simulate_public_lb_hist-{args.input_dir.name}-{args.threshold:.2f}-{args.n_samples}-{args.n_bootstrap}.png')

    # Print statistics
    print(f'Bootstrap mean: {np.mean(metric_values)}')
    print(f'Bootstrap std: {np.std(metric_values)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path, help='directory with dataset')
    parser.add_argument('input_dirs', type=Path, nargs='+', help='directory(-es) with .png predictions')
    parser.add_argument('--threshold', type=float, default=0.3, help='threshold to binarize predictions')
    parser.add_argument('--n_samples', type=int, default=279, help='number of samples to use')
    parser.add_argument('--n_bootstrap', type=int, default=1000, help='number of bootstrap tries')
    parser.add_argument('--replace', action='store_true', help='whether sample with replacement or not')
    args = parser.parse_args()
    main(args)
