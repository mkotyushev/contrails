# input_dir contains pseudolabels as images, name is {record_id}_{time_idx}_{model_postfix}.png
# - content is grayscale image of shape (H, W)
# - there may be multiple images for each record_id and time_idx pair corresponding to different models

# output name is name is {record_id}.npy, should be formed as follows:
# - single file for each record_id
# - content is np.uint8 array of shape (H, W, N_TIMES)
# - obtained as average of all models

# use pathlib for path manipulations

import logging
import cv2
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from src.data.datasets import N_TIMES


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get grouped files
    record_id_to_time_idx_to_files = defaultdict(lambda: defaultdict(list))
    for file_path in args.input_dir.glob('*.png'):
        record_id = file_path.stem.split('_')[0]
        time_idx = int(file_path.stem.split('_')[1])
        record_id_to_time_idx_to_files[record_id][time_idx].append(file_path)

    logger.info(
        f'Found {len(record_id_to_time_idx_to_files)} records, '
        f'total {sum(len(time_idx_to_files) for time_idx_to_files in record_id_to_time_idx_to_files.values())} images')

    # Iterate over grouped files
    for record_id, time_idx_to_files in tqdm(record_id_to_time_idx_to_files.items()):
        preds_per_record = np.full((256, 256, N_TIMES), np.nan, dtype=np.float32)
        for time_idx, files in time_idx_to_files.items():
            preds_per_time_idx = []
            for file_path in files:
                pred = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)  # (H, W)
                preds_per_time_idx.append(pred)
            preds_per_time_idx = np.stack(preds_per_time_idx, axis=-1).astype(np.float32).mean(-1)  # (H, W)
            preds_per_record[:, :, time_idx] = preds_per_time_idx
        
        assert np.all(np.isfinite(preds_per_record))
        
        if args.threshold is not None:
            preds_per_record = (preds_per_record > args.threshold)
        preds_per_record = preds_per_record.astype(np.uint8)  # (H, W, N_TIMES)
        np.save(str(args.output_dir / f'{record_id}.npy'), preds_per_record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.75)
    args = parser.parse_args()
    main(args)
