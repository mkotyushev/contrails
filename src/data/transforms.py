import itertools
import math
import random
import logging
import numpy as np
import torch
from albumentations import RandomResizedCrop
from typing import Dict, List, Literal
from torchvision.transforms import functional as F_torchvision
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.data.datasets import N_TIMES


logger = logging.getLogger(__name__)


###################################################################
##################### CV ##########################################
###################################################################

class CopyPastePositive:
    """Copy masked area from one image to another.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
        self, 
        mask_positive_value: int = 255,
        always_apply=True,
        p=1.0, 
    ):
        self.mask_positive_value = mask_positive_value
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs

        mask = \
            np.isclose(kwargs['mask'], self.mask_positive_value) & \
            (~np.isclose(kwargs['mask1'], self.mask_positive_value))

        # TODO: copy mask as it, but smooth edges of mask and 
        # apply to image as weighted average of two images
        kwargs['image'][mask] = kwargs['image1'][mask]
        kwargs['mask'][mask] = kwargs['mask1'][mask]

        return kwargs


# https://github.com/albumentations-team/albumentations/pull/1409/files
class MixUp:
    def __init__(
        self,
        alpha = 32.,
        beta = 32.,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        self.alpha = alpha
        self.beta = beta
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs

        h1, w1, _ = kwargs['image'].shape
        h2, w2, _ = kwargs['image1'].shape
        if h1 != h2 or w1 != w2:
            raise ValueError("MixUp transformation expects both images to have identical shape.")
        
        r = np.random.beta(self.alpha, self.beta)
        
        kwargs['image'] = (kwargs['image'] * r + kwargs['image1'] * (1 - r)).astype(kwargs['image'].dtype)
        kwargs['mask'] = (kwargs['mask'] * r + kwargs['mask1'] * (1 - r)).astype(kwargs['mask'].dtype)
        
        return kwargs


class CutMix:
    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        self.width = width
        self.height = height
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs

        h, w, _ = kwargs['image'].shape
        h1, w1, _ = kwargs['image1'].shape
        if (
            h < self.height or 
            w < self.width or 
            h1 < self.height or 
            w1 < self.width
        ):
            raise ValueError("CutMix transformation expects both images to be at least {}x{} pixels.".format(self.max_height, self.max_width))

        # Get random bbox
        h_start = random.randint(0, h - self.height)
        w_start = random.randint(0, w - self.width)
        h_end = h_start + self.height
        w_end = w_start + self.width

        # Copy image and mask region
        kwargs['image'][h_start:h_end, w_start:w_end] = kwargs['image1'][h_start:h_end, w_start:w_end]
        kwargs['mask'][h_start:h_end, w_start:w_end] = kwargs['mask1'][h_start:h_end, w_start:w_end]
        
        return kwargs


class TtaHorizontalFlip:
    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W)
        return batch.flip(3)
    
    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        # (N, C = 1, H, W)
        return batch_pred.flip(3)

    def __repr__(self) -> str:
        return f"TtaHorizontalFlip()"


class TtaVerticalFlip:
    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W)
        return batch.flip(2)
    
    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        # (N, C = 1, H, W)
        return batch_pred.flip(2)
    
    def __repr__(self) -> str:
        return f"TtaVerticalFlip()"


class TtaRotate90:
    def __init__(self, n_rot) -> None:
        assert n_rot % 4 != 0, f"n_rot should not be divisible by 4. Got {n_rot}"
        self.n_rot = n_rot % 4
    
    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W)
        return batch.rot90(self.n_rot, (2, 3))

    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        # (N, C = 1, H, W)
        return batch_pred.rot90(-self.n_rot, (2, 3))
    
    def __repr__(self) -> str:
        return f"TtaRotate90(n_rot={self.n_rot})"


class TtaRotate:
    def __init__(self, limit_degrees: int = 90, fill_value=0.0) -> None:
        self.limit_degrees = limit_degrees
        self.fill_value = fill_value
        self.angle = None

    @staticmethod
    def _rotate(batch: torch.Tensor, angle: int, fill: float=0) -> torch.Tensor:
        # (N, C, H, W)
        return F_torchvision.rotate(
            batch,
            angle,
            interpolation=F_torchvision.InterpolationMode.BILINEAR,
            expand=False,
            fill=fill,
        )

    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W)
        assert self.angle is None, "TtaRotate should be applied only once."
        self.angle = random.randint(-self.limit_degrees, self.limit_degrees) 
        return TtaRotate._rotate(batch, self.angle, fill=self.fill_value)

    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        # (N, C = 1, H, W)
        assert self.angle is not None, "TtaRotate should be applied before TtaRotate.apply_inverse_to_pred."
        # Fill with NaNs to ignore them in averaging
        batch_pred = TtaRotate._rotate(batch_pred, -self.angle, fill=torch.nan)
        self.angle = None
        return batch_pred

    def __repr__(self) -> str:
        return f"TtaRotate(limit_degrees={self.limit_degrees}, fill_value={self.fill_value})"


class TtaShift:
    """Shift image by (dx, dy) pixels."""
    def __init__(self, dx: int, dy: int, fill_value=0.0) -> None:
        self.dx = dx
        self.dy = dy
        self.fill_value = fill_value

    @staticmethod
    def _shift(batch: torch.Tensor, dx: int, dy: int, fill: float=0) -> torch.Tensor:
        # (N, C, H, W)
        return F_torchvision.affine(
            batch,
            angle=0,
            translate=[dx, dy],
            scale=1.0,
            shear=0.0,
            interpolation=F_torchvision.InterpolationMode.BILINEAR,
            fill=fill,
        )

    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W)
        return TtaShift._shift(batch, self.dx, self.dy, fill=self.fill_value)
    
    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        # (N, C = 1, H, W)

        # Seemingly there is a bug in torchvision.affine:
        # if fill=torch.nan all the image is filled with nans,
        # so fill manually with masking
        mask = torch.zeros_like(batch_pred).bool()
        mask = TtaShift._shift(mask, -self.dx, -self.dy, fill=True)
        batch_pred = TtaShift._shift(batch_pred, -self.dx, -self.dy, fill=0)
        batch_pred[mask] = torch.nan
        
        return batch_pred
    
    def __repr__(self) -> str:
        return f"TtaShift(dx={self.dx}, dy={self.dy}, fill_value={self.fill_value})"


class TtaScale:
    """Scale image by factor."""
    def __init__(self, factor: float) -> None:
        self.factor = factor
        self.original_shape = None

    @staticmethod
    def _scale(batch: torch.Tensor, size: float) -> torch.Tensor:
        # (N, C, H, W)
        return F_torchvision.resize(
            batch,
            size=size,
            interpolation=F_torchvision.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W)
        assert self.original_shape is None, "TtaScale should be applied only once."
        self.original_shape = batch.shape[-2:]
        return TtaScale._scale(
            batch, 
            (self.factor * torch.tensor(batch.shape[-2:])).long(), 
        )
    
    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        # (N, C = 1, H, W)
        result = TtaScale._scale(
            batch_pred, 
            self.original_shape, 
        )
        self.original_shape = None
        return result
    
    def __repr__(self) -> str:
        return f"TtaScale(factor={self.factor})"


class Tta:
    def __init__(
        self, 
        model, 
        do_tta, 
        aggr='mean',
        single_index=None,
        n_random_replays=1, 
        use_hflip=True, 
        use_vflip=True, 
        rotate90_indices=None,
        shift_params=None,
        scale_factors=None,
    ):
        self.do_tta = do_tta

        assert aggr in ['mean', 'min', 'max'], \
            f"aggr should be 'mean', 'min' or 'max'. Got {aggr}"
        self.aggr = aggr

        assert (
            n_random_replays > 0 or 
            use_hflip or 
            use_vflip or 
            rotate90_indices is not None or
            shift_params is not None or
            scale_factors is not None
        ), \
            "At least one of n_random_replays > 0, "\
            "use_hflip or use_vflip or rotate90_indices is not None should be True."
        assert rotate90_indices is None or all([i > 0 and i <= 3 for i in rotate90_indices]), \
            f"rotate90_indices should be in [0, 3]. Got {rotate90_indices}"
        assert rotate90_indices is None or len(rotate90_indices) == len(set(rotate90_indices)), \
            f"rotate90_indices should not contain duplicates. Got {rotate90_indices}"
        self.model = model
        
        # Imagenet normalization during training is assumed
        fill_value = (0.0 - sum(IMAGENET_DEFAULT_MEAN) / 3) / (sum(IMAGENET_DEFAULT_STD) / 3)
        
        # All possible combinations of
        # - flips
        # - rotations on 90 degrees
        # - n_replays rotations on random angle
        rotates90 = [None]
        if rotate90_indices is not None:
            rotates90 = [None] + [
                TtaRotate90(i) for i in rotate90_indices
            ]
        flips = [None]
        if use_hflip:
            flips += [
                TtaHorizontalFlip(),
            ]
        if use_vflip:
            flips += [
                TtaVerticalFlip(),
            ]
        rotates = [None]
        if n_random_replays > 0:
            rotates = [None] + [
                TtaRotate(limit_degrees=45, fill_value=fill_value) 
                for _ in range(n_random_replays)
            ]
        shifts = [None]
        if shift_params is not None:
            shifts = [None] + [
                TtaShift(dx, dy, fill_value=fill_value) 
                for dx, dy in shift_params
            ]
        scales = [None]
        if scale_factors is not None:
            scales = [None] + [
                TtaScale(factor) 
                for factor in scale_factors
            ]
        self.transforms = [
            rotates90,
            flips,
            rotates,
            shifts,
            scales,
        ]

        # Select single transform if needed
        if single_index is not None:
            product = list(itertools.product(*self.transforms))
            assert single_index >= 0 and single_index < len(product), \
                f"single_index should be in [0, {len(product) - 1}]. Got {single_index}"
            self.transforms = [[t] for t in product[single_index]]
            logger.info(f"Selected single (of {len(product)}) transform chain: {product[single_index]}")

    def predict(self, batch: torch.Tensor) -> List[torch.Tensor]:
        preds = []

        # Apply TTA
        for transform_chain in itertools.product(*self.transforms):
            # Direct transform
            batch_aug = batch.clone()
            for transform in transform_chain:
                if transform is not None:
                    batch_aug = transform.apply(batch_aug)
            
            # Predict
            pred_aug = self.model(batch_aug)
            if pred_aug.ndim == 3:
                pred_aug = pred_aug.unsqueeze(1)

            # Inverse transform
            # Note: order of transforms is reversed
            for transform in reversed(transform_chain):
                if transform is not None:
                    pred_aug = transform.apply_inverse_to_pred(pred_aug)
            
            preds.append(pred_aug)

        return preds

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if not self.do_tta:
            return self.model(batch)

        preds = self.predict(batch)
        
        # Average predictions, ignoring NaNs
        preds = torch.stack(preds, dim=0)

        if self.aggr == 'mean':
            preds = torch.nanmean(preds, dim=0)
        elif self.aggr == 'min':
            preds, _ = torch.min(
                torch.where(
                    preds.isnan(),
                    torch.tensor(float('inf'), device=preds.device),
                    preds
                ), 
                dim=0
            )
        elif self.aggr == 'max':
            preds, _ = torch.max(
                torch.where(
                    preds.isnan(),
                    torch.tensor(float('-inf'), device=preds.device),
                    preds
                ), 
                dim=0
            )

        # Remove nans (could be there if full nan / 
        # a single transform with nan fill value was applied)
        preds[preds.isnan()] = 0.0
        
        return preds


class RandomSubsequence:
    def __init__(
        self, 
        num_frames,
        always_apply=True,
        p=1.0, 
    ):
        assert num_frames > 1, f'num_frames should be > 1. Got {num_frames}'
        assert num_frames <= N_TIMES, f'num_frames should be <= {N_TIMES}. Got {num_frames}'
        self.num_frames = num_frames
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs

        if self.num_frames == N_TIMES:
            return kwargs
        
        start_index = random.randint(0, N_TIMES - self.num_frames)

        # image.shape = (H, W, C' = C * N_TIMES) -> (H, W, C, N_TIMES) -> (H, W, C, self.num_frames)
        image = kwargs["image"]
        image = image.reshape(*image.shape[:-1], -1, N_TIMES)
        image = image[..., start_index:start_index + self.num_frames]
        kwargs["image"] = image.reshape(*image.shape[:-2], -1)

        # mask.shape = (H, W, N_TIMES) -> (H, W, self.num_frames)
        kwargs["mask"] = kwargs["mask"][..., start_index:start_index + self.num_frames]

        return kwargs


class RandomResizedCropUniformArea(RandomResizedCrop):
    """Same as RandomResizedCrop, but with uniformly distributed area 
    of the crop instead of uniformly distributed scale.
    """
    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        area = img.shape[0] * img.shape[1]

        for _attempt in range(10):
            target_area = random.uniform(self.scale[0] * area, self.scale[1] * area)
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))  # skipcq: PTC-W0028
            h = int(round(math.sqrt(target_area / aspect_ratio)))  # skipcq: PTC-W0028

            if 0 < w <= img.shape[1] and 0 < h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return {
                    "crop_height": h,
                    "crop_width": w,
                    "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
                    "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
                }

        # Fallback to central crop
        in_ratio = img.shape[1] / img.shape[0]
        if in_ratio < min(self.ratio):
            w = img.shape[1]
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = img.shape[0]
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
            "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
        }


class SelectConcatTransform:
    def __init__(
        self, 
        cat_mode: Literal['spatial', 'channel'] = 'spatial',
        num_total_frames=None,
        time_indices=None,
        fill_value=0.0,
        always_apply=True,
        p=1.0, 
    ) -> None:
        assert num_total_frames is None or num_total_frames > 1, \
            f'num_total_frames should be > 1. Got {num_total_frames}'
        assert num_total_frames is None or num_total_frames <= N_TIMES, \
            f'num_total_frames should be <= {N_TIMES}. Got {num_total_frames}'
        assert time_indices is None or all([i >= 0 and i < N_TIMES for i in time_indices]), \
            f"time_indices should be in [0, {N_TIMES}]. Got {time_indices}"
        assert time_indices is None or len(time_indices) == len(set(time_indices)), \
            f"time_indices should not contain duplicates. Got {time_indices}"
        assert time_indices is None or sorted(time_indices) == time_indices, \
            f"time_indices should be sorted. Got {time_indices}"
        
        assert num_total_frames is not None or time_indices is not None, \
            "At least one of num_total_frames is not None or time_indices is not None should be True."
        if num_total_frames is not None and time_indices is not None:
            assert num_total_frames >= len(time_indices), \
                f"num_total_frames should be >= len(time_indices). Got {num_total_frames} and {len(time_indices)}"
        self.num_total_frames = num_total_frames

        self.time_indices = time_indices
        self.cat_mode = cat_mode
        self.fill_value = fill_value
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs

        # Reshape: (H, W, C' = C * N_TIMES) -> (H, W, C, N_TIMES)
        image = kwargs["image"]
        image = image.reshape(*image.shape[:-1], -1, N_TIMES)

        # Fill time_indices: pre-defined + random
        time_indices = []

        if self.time_indices is not None:
            time_indices += self.time_indices
        
        if self.num_total_frames is not None and len(time_indices) < self.num_total_frames:
            num_random_frames = self.num_total_frames - len(time_indices)
            random_time_indices += random.sample(
                [i for i in range(N_TIMES) if i not in time_indices],
                num_random_frames
            )
            random_time_indices = sorted(random_time_indices)
            time_indices += random_time_indices
        
        assert len(time_indices) > 0, "len(time_indices) should be > 0"

        # Select frames: (H, W, C, N_TIMES) -> (H, W, C, len(time_indices))
        image = image[..., self.time_indices]

        # Concatenate frames
        if self.cat_mode == 'channel':
            # Same way as in dataset: stack frames along channel axis

            # (H, W, C, len(self.time_indices)) -> (H, W, C * len(self.time_indices))
            image = image.reshape(*image.shape[:-2], -1)
        elif self.cat_mode == 'spatial':
            # Group frames into square spatial grid
            H, W, C, N = image.shape
            assert H == W

            # Pad len(self.time_indices) to nearest square:
            # image: (H, W, C, T)
            # image_new: (n_rows, H, n_cols, W, C)
            n_rows = n_cols = int(np.ceil(np.sqrt(len(N))))
            image_new = np.full(
                (H, n_rows, W, n_cols, C),
                fill_value=self.fill_value,
                dtype=image.dtype
            )

            for i, t in enumerate(N):
                row_index = i // n_cols
                col_index = i % n_cols
                image_new[row_index, :, col_index, :] = image[..., t]

            # (n_rows, H, n_cols, W, C) -> (n_rows * H, n_cols * W, C)
            image = image_new.reshape(n_rows * H, n_cols * W, C)

        # Set new image: (H, W, C) or (H, W, C')
        kwargs["image"] = image

        return kwargs
