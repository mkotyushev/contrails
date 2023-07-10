from copy import deepcopy
import logging
import numpy as np
from pathlib import Path
from typing import Any, List, Optional, Tuple, Literal


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


def get_images(data, type_='all', quantize=False, precomputed=False):
    """Converts data to numpy array of shape (H, W, C, T).

    args:
        data: dict, key is band from self.band_ids, value
            is float numpy array of shape (H, W, T)
        
        type_: str, one of ['ash', 'minmax3', 'minmax1', 
            'all', 'minmaxall', 'quantilesall', 'meanstdall']
            
            'ash': 
                ash color scheme (see https://www.kaggle.com/
                code/inversion/visualizing-contrails)
            'minmax3': 
                min-max normalization for each band separately
                for bands 11, 14, 15
            'minmax1':
                min-max normalization for single band 11
            'all':
                no normalization, all provided bands are stacked
                into single image
            'minmaxall':
                min-max normalization for all provided bands by 
                either precomputed (on train + val) min and max 
                values or by min and max values of provided record
            'quantilesall':
                q05-q95 normalization for all provided bands by
                either precomputed (on train + val) quantiles or
                values or by q05 and q95 values of provided record
            'meanstdall':
                mean-std normalization for all provided bands by
                either precomputed (on train + val) mean and std
                values or by mean and std values of provided record
        
        quantize: bool, if True, converts images to uint8
            could only be used for type_ not in ['all', 
            'quantilesall', 'meanstdall'] because the result
            range for these types is not [0, 255]
        
        precomputed: bool, if True, uses precomputed values
            for normalization (see args type_)
    """
    if not data:
        return None
    
    if type_ in ['all', 'quantilesall', 'meanstdall']:
        assert not quantize, \
            f'quantize=True is not supported for type_="{type_}": ' \
            f'the result range is not [0, 255]'

    band11 = data[11]
    band14 = data[14]
    band15 = data[15]
    if type_.startswith('ash'):
        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)
        images = np.stack([r, g, b], axis=2) * 255.0
    elif type_ == 'minmax3':
        r = (band11 - band11.min()) / (band11.max() - band11.min())
        g = (band14 - band14.min()) / (band14.max() - band14.min())
        b = (band15 - band15.min()) / (band15.max() - band15.min())
        images = np.stack([r, g, b], axis=2) * 255.0
    elif type_ == 'minmax1':
        r = g = b = (band11 - band11.min()) / (band11.max() - band11.min())
        images = np.stack([r, g, b], axis=2) * 255.0
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
                    divide = (MAX - subtract) / 255.0
                elif type_ == 'quantilesall':
                    subtract = QUANTILES[0.05]
                    divide = QUANTILES[0.95] - subtract
                elif type_ == 'meanstdall':
                    subtract = MEAN
                    divide = STD
            else:
                if type_ == 'minmaxall':
                    subtract = images.min()
                    divide = (images.max() - subtract) / 255.0
                elif type_ == 'quantilesall':
                    subtract = np.quantile(images, 0.05)
                    divide = np.quantile(images, 0.95) - subtract
                elif type_ == 'meanstdall':
                    subtract = images.mean()
                    divide = images.std()
        images = (images - subtract) / divide
    else:
        raise ValueError(f'Unknown type_="{type_}"')

    if quantize:
        images = np.clip(images, 0, 255)
        images = images.astype(np.uint8)
        
    return images


class ContrailsDataset:
    def __init__(
        self, 
        record_dirs: List[Path], 
        shared_cache: Optional[Any] = None,
        transform=None,
        transform_mix=None,
        transform_cpp=None,
        is_mask_empty: Optional[List[bool]] = None,
        mmap: bool = False,
        # Args below change the way images are loaded
        # thus change cache behavior.
        # See ContrailsDatamodule.make_cache for details.
        *,
        band_ids: Optional[Tuple[int]] = None,
        mask_type: Literal['voting50', 'mean', 'weighted', None] = 'voting50',
        use_not_labeled: bool = False,
        pseudolabels_path: Optional[Path] = None,
        conversion_type: Literal[
            'ash', 
            'minmax3', 
            'minmax1', 
            'all', 
            'minmaxall', 
            'quantilesall', 
            'meanstdall'
        ] = 'ash',
        quantize: bool = True,
        stats_precomputed: bool = False,
    ):
        self.record_dirs = record_dirs
        self.records = None
        self.mmap = mmap

        if band_ids is None:
            band_ids = tuple(BANDS)
        assert all(band in BANDS for band in band_ids)
        self.band_ids = band_ids

        assert mask_type is None or mask_type in ('voting50', 'mean', 'weighted')
        self.mask_type = mask_type

        if use_not_labeled and mask_type is not None:
            assert pseudolabels_path is not None, \
                'pseudolabels_path must be provided if use_not_labeled=True ' \
                'and mask_type is not None'
        self.use_not_labeled = use_not_labeled
        self.pseudolabels_path = pseudolabels_path
        self.transform = transform
        self.transform_mix = transform_mix
        self.transform_cpp = transform_cpp
        
        if (
            stats_precomputed and 
            conversion_type not in ['minmaxall', 'quantilesall', 'meanstdall']
        ):
            logger.warning(
                f'precomputed is not used for type_="{conversion_type}"'
            )
        self.conversion_type = conversion_type
        self.quantize = quantize
        self.stats_precomputed = stats_precomputed

        self.is_mask_empty = is_mask_empty
        self.non_empty_mask_indices = None
        if transform_cpp is not None:
            assert is_mask_empty is not None, \
                'if transform_cpp is not None, is_mask_empty must be provided'
            self.non_empty_mask_indices = np.where(~np.array(is_mask_empty))[0]

        self.shared_cache = shared_cache
    
    def __len__(self):
        if self.use_not_labeled:
            return len(self.record_dirs) * N_TIMES
        return len(self.record_dirs)

    def _get_item(self, time_idx, record_dir):        
        # Load bands
        # data: dict, key is band from self.band_ids, value 
        # is float numpy array of shape (H, W, T)
        data = {}
        for band_id in self.band_ids:
            data[band_id] = np.load(
                record_dir / f'band_{band_id:02}.npy',
                mmap_mode='r' if self.mmap else None
            )[..., time_idx]

        # Convert to numpy array
        image = get_images(
            data, 
            type_=self.conversion_type, 
            quantize=self.quantize,
            precomputed=self.stats_precomputed,
        )  # (H, W, C, T)
        
        # Load masks (if available)
        human_pixel_masks = None
        if (
            self.mask_type is not None and 
            (record_dir / 'human_pixel_masks.npy').exists()
        ):
            human_pixel_masks = np.load(
                record_dir / 'human_pixel_masks.npy'
            ) > 0  # (H, W, 1)
        
        human_individual_masks = None
        if (
            self.mask_type is not None and
            self.mask_type in ['mean', 'weighted'] and 
            (record_dir / 'human_individual_masks.npy').exists()
        ):
            human_individual_masks = np.load(
                record_dir / 'human_individual_masks.npy'
            ) > 0  # (H, W, 1, N)

        pseudolabel_masks = None
        record_id = record_dir.name
        if (
            self.mask_type is not None and
            self.use_not_labeled and 
            self.pseudolabels_path is not None and 
            (self.pseudolabels_path / f'{record_id}.npy').exists()
        ):
            pseudolabel_masks = np.load(
                self.pseudolabels_path / f'{record_id}.npy'
            )[..., time_idx]  # (H, W, 1)
        
        # Get single mask
        mask = None
        if human_pixel_masks is not None:
            if self.mask_type == 'voting50' or human_individual_masks is None:
                if self.mask_type != 'voting50' and human_individual_masks is None:
                    logger.debug(
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
                    cat_masks = np.concatenate(
                        [
                            np.tile(
                                human_pixel_masks[..., None], 
                                (1, 1, 1, human_individual_masks.shape[-1])
                            ), 
                            human_individual_masks,
                        ],
                        axis=2
                    )
                    intersection = np.all(cat_masks, axis=2)  # (H, W, 1, N)
                    union = np.any(cat_masks, axis=2)  # (H, W, 1, N)
                    union_sum = union.sum(axis=(0, 1))  # (1, N)
                    
                    iou = np.divide(
                        intersection.sum(axis=(0, 1)), 
                        union_sum,
                        out=np.zeros_like(union_sum, dtype=np.float32),
                        where=union_sum > 0, 
                    )  # (1, N)

                    weight = iou[None, None, ...]
                    weight_sum = weight.sum()
                    if not np.isclose(weight_sum, 0.0):
                        mask = (weight * human_individual_masks).sum(axis=-1) / weight_sum  # (H, W, 1)

                        # Hard 1 on voting 50 mask
                        mask[human_pixel_masks > 0] = 1.0
                    else:
                        mask = human_pixel_masks

            # Convert to uint8
            # - if binary (voting50), 0 and 255
            # - if not binary (mean, weighted), in range [0..255]
            mask = mask.astype(np.float32)  # (H, W, 1)
            mask = np.clip(mask, 0, 1)
            mask = (mask * 255).astype(np.uint8)  # (H, W, 1)

        # Get masks for all times
        if mask is None and pseudolabel_masks is not None:
            mask = pseudolabel_masks
        
        # Prepare output
        output = {
            'image': image,  # (H, W, C)
            'mask': mask,  # (H, W)
            'path': str(record_dir),
            'time_idx': time_idx if self.use_not_labeled else LABELED_TIME_INDEX,
        }
        
        return output
    
    def _get_item_cached(self, idx):
        # Get path and indices
        if self.use_not_labeled:
            record_idx = idx // N_TIMES
        else:
            record_idx = idx
        record_dir = self.record_dirs[record_idx]

        # Get path and indices
        if self.use_not_labeled:
            time_idx = idx % N_TIMES
        else:
            time_idx = LABELED_TIME_INDEX  # only single time index is loaded
        
        # Get item from cache or load it
        if self.shared_cache is not None and (time_idx, record_dir) in self.shared_cache:
            output = self.shared_cache[(time_idx, record_dir)]
        else:
            logger.debug(f'Cache miss, adding: {record_dir}')
            output = self._get_item(time_idx, record_dir)
            if self.shared_cache is not None:
                self.shared_cache[(time_idx, record_dir)] = deepcopy(output)
        
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

        # Apply mix transform
        if self.transform_mix is not None:
            # Sample random record
            random_idx = np.random.choice(len(self))
            random_output = self._get_item_cached(random_idx)

            # Apply augmentation
            output = self.transform_mix(
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

        # Convert mask from uint8 range [0..255] 
        # to float range [0, 1]
        if output['mask'] is not None:
            output['mask'] = output['mask'].astype(np.float32) / 255.0
        else:
            del output['mask']

        # TODO: check which way is better:
        # - cpp and mix then single transform
        # - single transform to both components of cpp and mix
        # Apply single transform
        if self.transform is not None:
            output = self.transform(**output)

        return output
