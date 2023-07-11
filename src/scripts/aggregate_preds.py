# input_dir contains preds as images, name is {record_id}_{time_idx}_{model_postfix}.png
# - content is grayscale image of shape (H, W)
# - there may be multiple images for each record_id and time_idx pair corresponding to different models

# output name is rle encoded average of all preds for each record_id

# use pathlib for path manipulations

# Usage example: python src/scripts/aggregate_preds.py --input_dir preds --output_path submission.csv --threshold 0.75

import logging
import cv2
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.data.datasets import LABELED_TIME_INDEX
from src.utils.utils import list_to_string, rle_encode


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get grouped files
    record_id_to_files = defaultdict(list)
    for file_path in args.input_dir.glob('*.png'):
        record_id = file_path.stem.split('_')[0]
        time_idx = int(file_path.stem.split('_')[1])
        if time_idx != LABELED_TIME_INDEX:
            continue
        record_id_to_files[record_id].append(file_path)

    logger.info(
        f'Found {len(record_id_to_files)} records, '
        f'total {sum(len(files) for files in record_id_to_files.values())} images')

    # Iterate over grouped files
    prediction_info = []
    for record_id, files in tqdm(record_id_to_files.items()):
        preds = []
        for file_path in files:
            pred = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)  # (H, W)
            preds.append(pred)
        preds = np.stack(preds, axis=-1).astype(np.float32).mean(-1)  # (H, W)
        preds = (preds > args.threshold)

        prediction_info.append({
            'record_id': record_id,
            'encoded_pixels': list_to_string(rle_encode(preds))
        })

    df = pd.DataFrame(prediction_info)
    df = df.sort_values(by=['record_id'])
    df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.75)
    args = parser.parse_args()
    main(args)
