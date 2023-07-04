import argparse
import numpy as np
import networkx as nx
import numpy as np
import json
import pandas as pd
from scipy.spatial import KDTree
from pathlib import Path

from src.data.datasets import ContrailsDataset


# https://stackoverflow.com/questions/69016985/
# finding-duplicates-with-tolerance-and-assign-to-a-set-in-pandas
def group_neighbors(df, tol, p=np.inf, show=False):
    r = np.linalg.norm(np.ones(len(tol)), p)
    kd = KDTree(df[tol.index] / tol)
    g = nx.Graph([
        (i, j)
        for i, neighbors in enumerate(kd.query_ball_tree(kd, r=r, p=p))
        for j in neighbors
    ])
    if show:
        nx.draw_networkx(g)
    ix, id_ = np.array([
        (j, i)
        for i, s in enumerate(nx.connected_components(g))
        for j in s
    ]).T
    id_[ix] = id_.copy()
    return df.assign(set_id=id_)


def build_split_info(data_dir):
    # Spatio(-temporal) grouping:
    # records which are close in space (and time) are grouped together
    with open(data_dir / 'train_metadata.json', 'r') as f:
        train_metadata = json.load(f)
    with open(data_dir / 'validation_metadata.json', 'r') as f:
        validation_metadata = json.load(f)
    df = pd.DataFrame(train_metadata + validation_metadata)
    
    # Spatial
    cols_to_group = ['row_min', 'row_size', 'col_min', 'col_size']
    tol = pd.Series([1e3, 1e2, 1e3, 1e2], index=cols_to_group)
    df = group_neighbors(df, tol)
    df['set_id_spatial'] = df['set_id']

    # Spatiotemporal
    cols_to_group = ['row_min', 'row_size', 'col_min', 'col_size', 'timestamp']
    tol = pd.Series([1e3, 1e2, 1e3, 1e2, 3600], index=cols_to_group)
    df = group_neighbors(df, tol)
    df['set_id_spatiotemporal'] = df['set_id']

    # Stratification by number of pixels in mask:
    # for each mask calculate number of pixels and split into 10 bins
    # then stratify by these bins
    record_dirs = []
    for d in [data_dir / 'train', data_dir / 'validation']:
        record_dirs += [path for path in d.iterdir() if path.is_dir()]

    dataset = ContrailsDataset(
        record_dirs=record_dirs, 
        shared_cache=None,
        transform=None,
        transform_mix=None,
        transform_cpp=None,
        is_mask_empty=None,

        band_ids=[],
        mask_type='voting50',
        mmap=True,
        conversion_type='ash',
        quantize=True,
        stats_precomputed=False,
    )

    mask_sums, pathes = [], []
    for item in dataset:
        mask_sums.append(item['mask'].sum())
        pathes.append(item['path'])

    df_mask_sums = pd.DataFrame({'path': pathes, 'mask_sum': mask_sums})
    df_mask_sums['record_id'] = df_mask_sums['path'].str.split('/').str[-1]

    # Simple > 0 condition or split into 10 bins
    df_mask_sums['mask_sum_qcut'] = pd.qcut(df_mask_sums['mask_sum'], 10, duplicates='drop')
    df_mask_sums['mask_sum_qcut_code'] = df_mask_sums['mask_sum_qcut'].cat.codes
    df_mask_sums['mask_sum_g0'] = df_mask_sums['mask_sum'] > 0

    # Merge
    df = df.merge(df_mask_sums, on='record_id', how='left')

    return df


def main(args):
    df = build_split_info(args.data_dir)
    df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default='../data')
    parser.add_argument('--output_path', type=str, default='./split_info.csv')
    args = parser.parse_args()
    main(args)