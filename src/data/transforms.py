import itertools
import random
import numpy as np
import torch
from typing import Dict, List, Literal
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
    def __init__(self, model, do_tta, n_random_replays=1, use_hflip=True, use_vflip=True, rotate90_indices=None):
        self.do_tta = do_tta

        assert n_random_replays > 0 or use_hflip or use_vflip or rotate90_indices is not None, \
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
        if not self.do_tta:
            return self.model(batch)

        preds = self.predict(batch)
        
        # Average predictions, ignoring NaNs
        preds = torch.stack(preds, dim=0)
        preds = torch.nanmean(preds, dim=0)
        
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
        assert time_indices is None or all([i > 0 and i <= N_TIMES for i in time_indices]), \
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
        if self.cat_mode == 'spatial':
            # Same way as in dataset: stack frames along channel axis

            # (H, W, C, len(self.time_indices)) -> (H, W, C * len(self.time_indices))
            image = image.reshape(*image.shape[:-2], -1)
        else:
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