import itertools
import random
import numpy as np
import torch
from typing import Dict, List
from torchvision.transforms import functional as F_torchvision
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

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
        mask_index: int = 2,
        always_apply=True,
        p=1.0, 
    ):
        self.mask_index = mask_index
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs

        mask = (kwargs['masks'][self.mask_index] > 0) & (kwargs['masks1'][self.mask_index] <= 0)

        kwargs['image'][mask] = kwargs['image1'][mask]
        for i in range(len(kwargs['masks'])):
            kwargs['masks'][i][mask] = kwargs['masks1'][i][mask]

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
        
        kwargs['image'] = (kwargs['image'] * r + kwargs['image1'] * (1 - r))
        for i in range(len(kwargs['masks'])):
            kwargs['masks'][i] = (kwargs['masks'][i] * r + kwargs['masks1'][i] * (1 - r))
        
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

        # Copy image and masks region
        kwargs['image'][h_start:h_end, w_start:w_end] = kwargs['image1'][h_start:h_end, w_start:w_end]
        for i in range(len(kwargs['masks'])):
            kwargs['masks'][i][h_start:h_end, w_start:w_end] = kwargs['masks1'][i][h_start:h_end, w_start:w_end]
        
        return kwargs



class TtaHorizontalFlip:
    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W, D)
        return batch.flip(3)
    
    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        # (N, C = 1, H, W)
        return batch_pred.flip(3)


class TtaVerticalFlip:
    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W, D)
        return batch.flip(2)
    
    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        # (N, C = 1, H, W)
        return batch_pred.flip(2)


class TtaRotate90:
    def __init__(self, n_rot) -> None:
        assert n_rot % 4 != 0, f"n_rot should not be divisible by 4. Got {n_rot}"
        self.n_rot = n_rot % 4
    
    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W, D)
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
        N, C, H, W, D = batch.shape
        # (N, C, H, W, D) -> (N, C, D, H, W)
        batch = batch.permute(0, 1, 4, 2, 3)
        # (N, C, D, H, W) -> (N, C * D, H, W)
        batch = batch.reshape(N, C * D, H, W)
        batch = F_torchvision.rotate(
            batch,
            angle,
            interpolation=F_torchvision.InterpolationMode.BILINEAR,
            expand=False,
            fill=fill,
        )
        # (N, C * D, H, W) -> (N, C, D, H, W)
        batch = batch.reshape(N, C, D, H, W)
        # (N, C, D, H, W) -> (N, C, H, W, D)
        batch = batch.permute(0, 1, 3, 4, 2)
        return batch

    def apply(self, batch: torch.Tensor) -> torch.Tensor:
        assert self.angle is None, "TtaRotate should be applied only once."
        # (N, C, H, W, D)
        self.angle = random.randint(-self.limit_degrees, self.limit_degrees) 
        return TtaRotate._rotate(batch, self.angle, fill=self.fill_value)

    def apply_inverse_to_pred(self, batch_pred: torch.Tensor) -> torch.Tensor:
        assert self.angle is not None, "TtaRotate should be applied before TtaRotate.apply_inverse_to_pred."
        # (N, C = 1, H, W) -> (N, C = 1, H, W, D = 1)
        batch_pred = batch_pred.unsqueeze(-1)
        # Fill with NaNs to ignore them in averaging
        batch_pred = TtaRotate._rotate(batch_pred, -self.angle, fill=torch.nan)
        # (N, C = 1, H, W, D = 1) -> (N, C = 1, H, W)
        batch_pred = batch_pred.squeeze(-1)
        self.angle = None
        return batch_pred


class Tta:
    def __init__(self, model, n_random_replays=1, use_hflip=True, use_vflip=True, rotate90_indices=None):
        assert n_random_replays > 0 or use_hflip or use_vflip or rotate90_indices is not None, \
            "At least one of n_random_replays > 0, "\
            "use_hflip or use_vflip or rotate90_indices is not None should be True."
        assert all([i > 0 and i <= 3 for i in rotate90_indices]), \
            f"rotate90_indices should be in [0, 3]. Got {rotate90_indices}"
        assert len(rotate90_indices) == len(set(rotate90_indices)), \
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
        self.transforms = [
            rotates90,
            flips,
            rotates,
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
        preds = self.predict(batch)
        
        # Average predictions, ignoring NaNs
        preds = torch.stack(preds, dim=0)
        preds = torch.nanmean(preds, dim=0)
        
        return preds
