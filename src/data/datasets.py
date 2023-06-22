from copy import deepcopy
import logging
import numpy as np
from pathlib import Path
from typing import Any, List, Optional, Tuple, Literal

from src.data.transforms import CopyPastePositive


BANDS = (8, 9, 10, 11, 12, 13, 14, 15, 16)
LABELED_TIME_INDEX = 4
N_TIMES = 8
# https://www.kaggle.com/code/inversion/visualizing-contrails
_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)
# from train & validation data (a small dataleakage if using k-fold)
MEAN, STD, MIN, MAX = (260.1077976678496, 21.99711806156345, 178.84055, 336.71628)
QUANTILES = {
    0.05: 225.89370874509805,
    0.1: 232.08491384313726,
    0.15: 235.79963690196078,
    0.2: 238.8952394509804,
    0.25: 242.60996250980392,
    0.3: 245.70556505882354,
    0.35: 249.42028811764706,
    0.4: 252.51589066666668,
    0.45: 256.23061372549023,
    0.5: 259.3262162745098,
    0.55: 263.0409393333333,
    0.6: 267.37478290196077,
    0.65: 271.0895059607843,
    0.7: 274.1851085098039,
    0.75: 279.1380725882353,
    0.8: 283.47191615686273,
    0.85: 287.1866392156863,
    0.9: 290.2822417647059,
    0.95: 293.3778443137255,
}


logger = logging.getLogger(__name__)


def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_images(data, type_='all', mult=1.0, precomputed=False):
    if not data:
        return None
    band11 = data[11]
    band14 = data[14]
    band15 = data[15]
    if type_.startswith('false'):
        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)
        images = np.clip(np.stack([r, g, b], axis=2), 0, 1) * mult
        if type_ == 'falseq':
            assert mult == 255.0
            images = images.astype(np.uint8)
    elif type_ == 'minmax3':
        r = (band11 - band11.min()) / (band11.max() - band11.min())
        g = (band14 - band14.min()) / (band14.max() - band14.min())
        b = (band15 - band15.min()) / (band15.max() - band15.min())
        images = np.stack([r, g, b], axis=2) * mult
    elif type_ == 'minmax1':
        r = g = b = (band11 - band11.min()) / (band11.max() - band11.min())
        images = np.stack([r, g, b], axis=2) * mult
    elif type_ in ['all', 'minmaxall', 'quantilesall', 'meanstdall']:
        bands = []
        for band in sorted(data.keys()):
            bands.append(data[band])
        images = np.stack(bands, axis=2)

        if type_ == 'all':
            subtract, divide = 0, 1
        else:
            if precomputed:
                if type_ == 'minmaxall':
                    subtract = MIN
                    divide = MAX - subtract
                elif type_ == 'quantilesall':
                    subtract = QUANTILES[0.05]
                    divide = QUANTILES[0.95] - subtract
                elif type_ == 'meanstdall':
                    subtract = MEAN
                    divide = STD
            else:
                if type_ == 'minmaxall':
                    subtract = images.min()
                    divide = images.max() - subtract
                elif type_ == 'quantilesall':
                    subtract = np.quantile(images, 0.05)
                    divide = np.quantile(images, 0.95) - subtract
                elif type_ == 'meanstdall':
                    subtract = images.mean()
                    divide = images.std()
        images = (images - subtract) / divide
        
    return images


class ContrailsDataset:
    def __init__(
        self, 
        record_dirs: List[Path], 
        shared_cache: Optional[Any] = None,
        transform=None,
        # Args below change the way images are loaded
        # thus change cache behavior.
        # See ContrailsDatamodule.make_cache for details.
        *,
        band_ids: Tuple[int] = BANDS,
        mask_type: Literal['voting50', 'mean', 'weighted'] = 'voting50',
        propagate_mask: bool = False,
        mmap: bool = False,
        conversion_type: Literal[
            'false', 
            'falseq', 
            'minmax3', 
            'minmax1', 
            'all', 
            'minmaxall', 
            'quantilesall', 
            'meanstdall'
        ] = 'falseq',
        stats_precomputed: bool = False,
        is_mask_empty: Optional[List[bool]] = None,
        enable_cpp_aug: bool = False,
    ):
        self.record_dirs = record_dirs
        self.records = None

        assert all(band in BANDS for band in band_ids)
        self.band_ids = band_ids

        assert mask_type in ('voting50', 'mean', 'weighted')
        self.mask_type = mask_type

        self.propagate_mask = propagate_mask
        self.mmap = mmap
        self.transform = transform
        self.conversion_type = conversion_type
        self.stats_precomputed = stats_precomputed

        self.is_mask_empty = is_mask_empty
        self.non_empty_mask_indices = None
        self.transform_cpp = None
        if enable_cpp_aug:
            assert is_mask_empty is not None, \
                'if enable_cpp_aug = True, is_mask_empty must be provided'
            self.non_empty_mask_indices = np.where(~np.array(is_mask_empty))[0]
            self.transform_cpp = CopyPastePositive(always_apply=True, p=1.0)

        self.shared_cache = shared_cache
    
    def __len__(self):
        if self.propagate_mask:
            return len(self.record_dirs) * N_TIMES
        return len(self.record_dirs)

    def _get_item(self, idx, record_dir):
        # Get path and indices
        if self.propagate_mask:
            # Load all the times to use in propagation
            time_idx = idx % N_TIMES
            time_indices = np.arange(N_TIMES)
        else:
            # Load only the labeled time
            time_idx = 0  # only single time index is loaded
            time_indices = [LABELED_TIME_INDEX]
        # Load bands
        # data: dict, key is band from self.band_ids, value 
        # is float numpy array of shape (H, W, T)
        data = {}
        for band_id in self.band_ids:
            data[band_id] = np.load(
                record_dir / f'band_{band_id:02}.npy',
                mmap_mode='r' if self.mmap else None
            )[..., time_indices]

        # Convert to numpy array
        image = get_images(
            data, 
            type_=self.conversion_type, 
            mult=255.0,
            precomputed=self.stats_precomputed,
        )  # (H, W, C, T)
        
        # Load masks (if available)
        human_pixel_masks = None
        if (record_dir / 'human_pixel_masks.npy').exists():
            human_pixel_masks = np.load(
                record_dir / 'human_pixel_masks.npy'
            ) > 0  # (H, W, 1)
        
        human_individual_masks = None
        if (
            self.mask_type in ['mean', 'weighted'] and 
            (record_dir / 'human_individual_masks.npy').exists()
        ):
            human_individual_masks = np.load(
                record_dir / 'human_individual_masks.npy'
            ) > 0  # (H, W, 1, N)
        
        # Get single mask
        if human_pixel_masks is not None:
            if self.mask_type == 'voting50' or human_individual_masks is None:
                if self.mask_type != 'voting50' and human_individual_masks is None:
                    logger.warning(
                        f'{self.mask_type} mask type is not available '
                        'because human_individual_masks.npy is not found, '
                        'using "voting50" mask instead.'
                    )
                mask = human_pixel_masks
            else:
                # Mask is weighted average of individual masks
                if self.mask_type == 'mean':
                    mask = human_individual_masks.mean(axis=-1)  # (H, W, 1)
                else:
                    # Weights are equal to IoU of individual mask 
                    # and pixel mask
                    intersection = np.logical_and(
                        human_pixel_masks[..., None], 
                        human_individual_masks,
                        axis=-1
                    )  # (H, W, 1, N)
                    union = np.logical_or(
                        human_pixel_masks[..., None],
                        human_individual_masks,
                        axis=-1
                    )  # (H, W, 1, N)
                    iou = intersection.sum(axis=(0, 1)) / union.sum(axis=(0, 1))  # (1, N)
                    mask = (iou[None, None, ...] * human_individual_masks).sum(axis=-1)  # (H, W, 1)
            mask = mask.astype(np.uint8)  # (H, W, 1)

        # Propagate mask
        if self.propagate_mask:
            raise NotImplementedError(
                'Mask propagation is not implemented yet.'
            )
        else:
            assert mask is None or mask.shape[-1] == 1
            assert image is None or image.shape[-1] == 1

        # Select time index
        if image is not None:
            image = image[..., time_idx]  # (H, W, C, T) -> (H, W, C)
        if mask is not None:
            mask = mask[..., time_idx]  # (H, W, T) -> (H, W)
        
        # Prepare output
        output = {
            'image': image,  # (H, W, C)
            'mask': mask,  # (H, W)
            'path': str(record_dir),
        }
        
        return output
    
    def _get_item_cached(self, idx):
        # Get path and indices
        if self.propagate_mask:
            record_idx = idx // N_TIMES
        else:
            record_idx = idx
        record_dir = self.record_dirs[record_idx]
        
        # Get item from cache or load it
        if self.shared_cache is not None and record_dir in self.shared_cache:
            output = self.shared_cache[record_dir]
        else:
            logger.debug(f'Cache miss, adding: {record_dir}')
            output = self._get_item(idx, record_dir)
            if self.shared_cache is not None:
                self.shared_cache[record_dir] = deepcopy(output)
        
        return output

    def __getitem__(self, idx):
        output = self._get_item_cached(idx)

        # If mask is empty and augmentation is enabled,
        # apply copy-paste-positive augmentation
        if self.transform_cpp is not None and self.is_mask_empty[idx]:
            # Sample random record with non-empty mask
            random_idx = np.random.choice(self.non_empty_mask_indices)
            random_output = self._get_item_cached(random_idx)

            # Apply augmentation
            output = self.transform_cpp(
                **{
                    **output,  
                    **{f'{k}1': v for k, v in random_output.items()}
                  }
            )

            # Remove unnecessary keys
            output = {
                k: v for k, v in output.items() 
                if not (k.endswith('1') and k[:-1] in output)
            }         

        # Apply transform
        if self.transform is not None:
            output = self.transform(**output)

        return output
