import itertools
import random
import numpy as np
import torch
from typing import Dict, List
from torchvision.transforms import functional as F_torchvision
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.data.datasets import N_TIMES

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


class TtaVerticalFlip:
    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W)
        return batch.flip(2)
    
    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        # (N, C = 1, H, W)
        return batch_pred.flip(2)


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
        return TtaShift._shift(batch_pred, -self.dx, -self.dy, fill=torch.nan)


class TtaScale:
    """Scale image by factor."""
    def __init__(self, factor: float, fill_value=0.0) -> None:
        self.factor = factor
        self.fill_value = fill_value

    @staticmethod
    def _scale(batch: torch.Tensor, factor: float, fill: float=0) -> torch.Tensor:
        # (N, C, H, W)
        return F_torchvision.resize(
            batch,
            size=(factor * torch.tensor(batch.shape[-2:])).long(),
            interpolation=F_torchvision.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W)
        return TtaScale._scale(batch, self.factor, fill=self.fill_value)
    
    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        # (N, C = 1, H, W)
        return TtaScale._scale(batch_pred, 1 / self.factor, fill=torch.nan)


class Tta:
    def __init__(
        self, 
        model, 
        do_tta, 
        aggr='mean',
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
                TtaScale(factor, fill_value=fill_value) 
                for factor in scale_factors
            ]
        self.transforms = [
            rotates90,
            flips,
            rotates,
            shifts,
            scales,
        ]

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
