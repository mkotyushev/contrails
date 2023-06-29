import cv2
import logging
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from multiprocessing.managers import MakeProxyType, SyncManager
from torch.utils.data import default_collate
from weakref import proxy
from pathlib import Path
from patchify import NonUniformStepSizeError, unpatchify
from collections import defaultdict
from typing import Dict, Optional, Union, Tuple
from lightning import Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from timm.layers.format import nhwc_to, Format
from torchvision.utils import make_grid


logger = logging.getLogger(__name__)


###################################################################
########################## General Utils ##########################
###################################################################

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """Add argument links to parser.

        Example:
            parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        """
        return


class MyLightningCLISweep(LightningCLI):
    """Implement args binding for sweeps.

    Sweep configs currently only support cartesian product of args
    and not args binding. This is a workaround to bind args.

    E. g. if some value of `backbone_name` imply usage of `compile`
    and some not, grid search over `backbone_name` and `compile` is
    not possible. This can be solved by binding `compile` to `backbone_name`.
    """
    def add_arguments_to_parser(self, parser):
        """Add argument links to parser.

        Example:
            parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        """
        return


    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes."""
        device_to_batch_size_divider = {
            'NVIDIA GeForce RTX 3090': 1,
            'NVIDIA GeForce RTX 3080 Ti Laptop GPU': 2,
            'Tesla T4': 2,
        }
        backbone_name_to_batch_params = {
            'convnextv2_base.fcmae_ft_in22k_in1k_384': {
                'batch_size': 64,
                'accumulate_grad_batches': 1,
            },
            'eva02_B_ade_seg_upernet_sz512': {
                'batch_size': 64,
                'accumulate_grad_batches': 1,
            },
            'maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k': {
                'batch_size': 32,
                'accumulate_grad_batches': 2,
            },
            'nvidia/mit-b5': {
                'batch_size': 64,
                'accumulate_grad_batches': 1,
            },
        }

        # Force not deterministic training
        if (
            self.config['fit']['model']['init_args']['backbone_name'].startswith('nvidia') or
            self.config['fit']['model']['init_args']['backbone_name'].startswith('eva02')
        ):
            if self.config['fit']['trainer']['deterministic']:
                logger.warning(
                    'Deterministic training is not supported with NVIDIA or Eva02 models. '
                    'Setting `deterministic=False`.'
                )
            self.config['fit']['trainer']['deterministic'] = False

        # Force not compile
        if (
            self.config['fit']['model']['init_args']['backbone_name'].startswith('convnext') or
            self.config['fit']['model']['init_args']['backbone_name'].startswith('eva02')
        ):
            if self.config['fit']['model']['init_args']['compile']:
                logger.warning(
                    'compile is not supported with convnext or Eva02 models. '
                    'Setting `compile=False`.'
                )
            self.config['fit']['model']['init_args']['compile'] = False

        # Set LR (needed for sweeps)
        if self.config['fit']['model']['init_args']['lr'] is not None:
            self.config['fit']['model']['init_args']['optimizer_init']['init_args']['lr'] = \
                self.config['fit']['model']['init_args']['lr']

        # Overrride batch params (needed for different machines)
        device_name = torch.cuda.get_device_name()
        if device_name not in device_to_batch_size_divider:
            logger.warning(
                f'Unknown device {device_name}. '
                f'Using default batch size and accumulate_grad_batches.'
            )
            device_to_batch_size_divider[device_name] = 1

        self.config['fit']['data']['init_args']['batch_size'] = \
            backbone_name_to_batch_params[
                self.config['fit']['model']['init_args']['backbone_name']
            ]['batch_size'] // \
            device_to_batch_size_divider[device_name]
        self.config['fit']['trainer']['accumulate_grad_batches'] = \
            backbone_name_to_batch_params[
                self.config['fit']['model']['init_args']['backbone_name']
            ]['accumulate_grad_batches'] * \
            device_to_batch_size_divider[device_name]


class TrainerWandb(Trainer):
    """Hotfix for wandb logger saving config & artifacts to project root dir
    and not in experiment dir."""
    @property
    def log_dir(self) -> Optional[str]:
        """The directory for the current experiment. Use this to save images to, etc...

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                img = ...
                save_img(img, self.trainer.log_dir)
        """
        if len(self.loggers) > 0:
            if isinstance(self.loggers[0], WandbLogger):
                dirpath = self.loggers[0]._experiment.dir
            elif not isinstance(self.loggers[0], TensorBoardLogger):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath


class ModelCheckpointNoSave(ModelCheckpoint):
    def best_epoch(self) -> int:
        # exmple: epoch=10-step=1452.ckpt
        return int(self.best_model_path.split('=')[-2].split('-')[0])
    
    def ith_epoch_score(self, i: int) -> Optional[float]:
        # exmple: epoch=10-step=1452.ckpt
        ith_epoch_filepath_list = [
            filepath 
            for filepath in self.best_k_models.keys()
            if f'epoch={i}-' in filepath
        ]
        
        # Not found
        if not ith_epoch_filepath_list:
            return None
    
        ith_epoch_filepath = ith_epoch_filepath_list[-1]
        return self.best_k_models[ith_epoch_filepath]

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))
    
    def on_validation_end(self, trainer, pl_module) -> None:
        super().on_validation_end(trainer, pl_module)

        pl_module.logger.experiment.log(
            {
                'best_epoch': self.best_epoch(), 
                f'best_{self.monitor}': self.best_model_score,
            }
        )


class TempSetContextManager:
    def __init__(self, obj, attr, value):
        self.obj = obj
        self.attr = attr
        self.value = value

    def __enter__(self):
        self.old_value = getattr(self.obj, self.attr)
        setattr(self.obj, self.attr, self.value)

    def __exit__(self, *args):
        setattr(self.obj, self.attr, self.old_value)



def state_norm(module: torch.nn.Module, norm_type: Union[float, int, str], group_separator: str = "/") -> Dict[str, float]:
    """Compute each state dict tensor's norm and their overall norm.

    The overall norm is computed over all tensor together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the tensor norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the tensor viewed
            as a single vector.
    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")

    norms = {
        f"state_{norm_type}_norm{group_separator}{name}": p.data.float().norm(norm_type)
        for name, p in module.state_dict().items()
        if not 'num_batches_tracked' in name
    }
    if norms:
        total_norm = torch.tensor(list(norms.values())).norm(norm_type)
        norms[f"state_{norm_type}_norm_total"] = total_norm
    return norms


###################################################################
##################### CV ##########################################
###################################################################

# Segmentation

# https://github.com/bnsreenu/python_for_microscopists/blob/master/
# 229_smooth_predictions_by_blending_patches/smooth_tiled_predictions.py
def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size / 4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def _spline_window_2d(h, w, power=2):
    h_wind = _spline_window(h, power)
    w_wind = _spline_window(w, power)
    return h_wind[:, None] * w_wind[None, :]


def _unpatchify2d_avg(  # pylint: disable=too-many-locals
    patches: np.ndarray, imsize: Tuple[int, int], weight_mode='uniform',
) -> np.ndarray:
    assert len(patches.shape) == 4
    assert weight_mode in ['uniform', 'spline']

    i_h, i_w = imsize
    image = np.zeros(imsize, dtype=np.float32)
    weights = np.zeros(imsize, dtype=np.float32)

    n_h, n_w, p_h, p_w = patches.shape

    s_w = 0 if n_w <= 1 else (i_w - p_w) / (n_w - 1)
    s_h = 0 if n_h <= 1 else (i_h - p_h) / (n_h - 1)

    # The step size should be same for all patches, otherwise the patches are unable
    # to reconstruct into a image
    if int(s_w) != s_w:
        raise NonUniformStepSizeError(i_w, n_w, p_w, s_w)
    if int(s_h) != s_h:
        raise NonUniformStepSizeError(i_h, n_h, p_h, s_h)
    s_w = int(s_w)
    s_h = int(s_h)

    weight = 1  # uniform
    if weight_mode == 'spline':
        weight = _spline_window_2d(p_h, p_w, power=2)

    # For each patch, add it to the image at the right location
    for i in range(n_h):
        for j in range(n_w):
            image[i * s_h : i * s_h + p_h, j * s_w : j * s_w + p_w] += (patches[i, j] * weight)
            weights[i * s_h : i * s_h + p_h, j * s_w : j * s_w + p_w] += weight

    # Average
    weights = np.where(np.isclose(weights, 0.0), 1.0, weights)
    image /= weights

    image = image.astype(patches.dtype)

    return image, weights


class PredictionTargetPreviewAgg(nn.Module):
    """Aggregate prediction and target patches to images with downscaling."""
    def __init__(
        self, 
        preview_downscale: Optional[int] = 4, 
        metrics=None, 
        input_std=1, 
        input_mean=0, 
        fill_value=0,
        overlap_avg_weight_mode='uniform',
    ):
        super().__init__()
        self.preview_downscale = preview_downscale
        self.metrics = metrics
        self.previews = {}
        self.shapes = {}
        self.shapes_before_padding = {}
        self.input_std = input_std
        self.input_mean = input_mean
        self.fill_value = fill_value
        self.overlap_avg_weight_mode = overlap_avg_weight_mode

    def reset(self):
        # Note: metrics are reset in compute()
        self.previews = {}
        self.shapes = {}
        self.shapes_before_padding = {}

    def update(
        self, 
        arrays: Dict[str, torch.Tensor | np.ndarray],
        pathes: list[str], 
        patch_size: torch.LongTensor | np.ndarray,
        indices: torch.LongTensor | np.ndarray, 
        shape_patches: torch.LongTensor | np.ndarray,
        shape_original: torch.LongTensor | np.ndarray,
        shape_before_padding: torch.LongTensor,
    ):
        # To CPU & types
        for name in arrays:
            if isinstance(arrays[name], torch.Tensor):
                arrays[name] = arrays[name].cpu().numpy()
            
            if name == 'input':
                arrays[name] = ((arrays[name] * self.input_std + self.input_mean) * 255).astype(np.uint8)
            elif name == 'probas':
                arrays[name] = arrays[name].astype(np.float32)
            elif name == ['mask', 'target']:
                arrays[name] = arrays[name].astype(np.uint8)
            else:
                # Do not convert type
                pass

        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        if isinstance(shape_patches, torch.Tensor):
            shape_patches = shape_patches.cpu().numpy()
        if isinstance(shape_before_padding, torch.Tensor):
            shape_before_padding = shape_before_padding.cpu().numpy()

        indices, shape_patches, shape_before_padding = \
            indices.astype(np.int64), \
            shape_patches.astype(np.int64), \
            shape_before_padding.astype(np.int64)
    
        # Place patches on the preview images
        B = arrays[list(arrays.keys())[0]].shape[0]
        for i in range(B):
            path = Path(pathes[i])
            path = str(path.relative_to(path.parent.parent))
            shape = [
                *shape_patches[i].tolist(),
                *patch_size,
            ]

            self.shapes[path] = shape_original[i].tolist()[:2]
            self.shapes_before_padding[path] = shape_before_padding[i].tolist()[:2]
            patch_index_w, patch_index_h = indices[i].tolist()

            for name, value in arrays.items():
                key = f'{name}|{path}'
                if key not in self.previews:
                    self.previews[key] = np.full(shape, fill_value=self.fill_value, dtype=arrays[name].dtype)
                    if name.startswith('probas'):
                        # Needed to calculate average from sum
                        # hack to not change dict size later, actually computed in compute()
                        self.previews[f'counts|{path}'] = None
                self.previews[key][patch_index_h, patch_index_w] = value[i]
    
    def compute(self):
        # Unpatchify
        for name in self.previews:
            path = name.split('|')[-1]
            shape_original = self.shapes[path]
            if name.startswith('probas'):
                # Average overlapping patches
                self.previews[name], counts = _unpatchify2d_avg(
                    self.previews[name], 
                    shape_original,
                    weight_mode=self.overlap_avg_weight_mode,
                )
                self.previews[name.replace('probas', 'counts')] = counts.astype(np.uint8)
            elif name.startswith('counts'):
                # Do nothing
                pass
            else:
                # Just unpatchify
                self.previews[name] = unpatchify(
                    self.previews[name], 
                    shape_original
                )

        # Zero probas out where mask is zero
        for name in self.previews:
            if name.startswith('probas'):
                mask = self.previews[name.replace('probas', 'mask')] == 0
                self.previews[name][mask] = 0

        # Crop to shape before padding
        for name in self.previews:
            path = name.split('|')[-1]
            shape_before_padding = self.shapes_before_padding[path]
            self.previews[name] = self.previews[name][
                :shape_before_padding[0], 
                :shape_before_padding[1],
            ]

        # Compute metrics if available
        metric_values = None
        if self.metrics is not None:
            preds, targets = [], []
            for name in self.previews:
                if name.startswith('probas'):
                    path = name.split('|')[-1]
                    mask = self.previews[f'mask|{path}'] > 0
                    pred = self.previews[name][mask].flatten()
                    target = self.previews[f'target|{path}'][mask].flatten()

                    preds.append(pred)
                    targets.append(target)
            preds = torch.from_numpy(np.concatenate(preds))
            targets = torch.from_numpy(np.concatenate(targets))

            metric_values = {}
            for metric_name, metric in self.metrics.items():
                metric.update(preds, targets)
                metric_values[metric_name] = metric.compute()
                metric.reset()
        
        # Downscale and get captions
        captions, previews = [], []
        for name, preview in self.previews.items():
            if self.preview_downscale is not None:
                preview = cv2.resize(
                    preview,
                    dsize=(0, 0),
                    fx=1 / self.preview_downscale, 
                    fy=1 / self.preview_downscale, 
                    interpolation=cv2.INTER_LINEAR, 
                )
            captions.append(name)
            previews.append(preview)

        return metric_values, captions, previews
    

class PredictionTargetPreviewGrid(nn.Module):
    """Aggregate prediction and target patches to images with downscaling."""
    def __init__(self, preview_downscale: int = 4, n_images: int = 4):
        super().__init__()
        self.preview_downscale = preview_downscale
        self.n_images = n_images
        self.previews = defaultdict(list)

    def reset(self):
        self.previews = defaultdict(list)

    def update(
        self, 
        input: torch.Tensor,
        probas: torch.Tensor, 
        target: torch.Tensor, 
    ):
        # Add images until grid is full
        for i in range(probas.shape[0]):
            if len(self.previews['input']) >= self.n_images:
                return

            # Get preview images
            inp = F.interpolate(
                input[i].float().unsqueeze(0),
                scale_factor=1 / self.preview_downscale, 
                mode='bilinear',
                align_corners=False, 
            ).cpu()
            proba = F.interpolate(
                probas[i].float().unsqueeze(0).unsqueeze(1),  # interpolate as (N, C, H, W)
                scale_factor=1 / self.preview_downscale, 
                mode='bilinear', 
                align_corners=False, 
            ).cpu()
            targ = F.interpolate(
                target[i].float().unsqueeze(0).unsqueeze(1),  # interpolate as (N, C, H, W)
                scale_factor=1 / self.preview_downscale,
                mode='bilinear',
                align_corners=False, 
            ).cpu()

            self.previews['input'].append(inp)
            self.previews['proba'].append((proba * 255).byte())
            self.previews['target'].append((targ * 255).byte())
    
    def compute(self):
        captions = list(self.previews.keys())
        preview_grids = [
            make_grid(
                torch.cat(v, dim=0), 
                nrow=int(self.n_images ** 0.5)
            ).float()
            for v in self.previews.values()
        ]

        return captions, preview_grids


class FeatureExtractorWrapper(nn.Module):
    def __init__(self, model, format: Format | str = 'NHWC'):
        super().__init__()
        self.model = model
        self.output_stride = 32
        self.format = format if isinstance(format, Format) else Format(format)

    def __iter__(self):
        return iter(self.model)
    
    def forward(self, x):
        if self.format == Format('NHWC'):
            features = [nhwc_to(y, Format('NCHW')) for y in self.model(x)]
        else:
            features = self.model(x)
        return features


class Eva02Wrapper(nn.Module):
    def __init__(self, model, scale_factor=4):
        super().__init__()
        self.model = model
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
    
    def forward(self, x):
        x = self.model(x)
        # # Remove the channel (single class) dimension
        # # (B, 1, H, W) -> (B, H, W)
        # x = x.squeeze(1)
        # Upsample to original size
        x = self.upsampling(x)
        return x


class HfWrapper(nn.Module):
    def __init__(self, model, scale_factor=4):
        super().__init__()
        self.model = model
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
    
    def forward(self, x):
        x = self.model(x)['logits']
        # Upsample to original size
        x = self.upsampling(x)
        return x


def get_feature_channels(model, input_shape, output_format='NHWC'):
    is_training = model.training
    model.eval()
    
    x = torch.randn(1, *input_shape).to(next(model.parameters()).device)
    with torch.no_grad():
        y = model(x)
    channel_index = output_format.find('C')
    assert channel_index != -1, \
        f'output_format {output_format} not supported, must contain C'
    assert all(len(output_format) == len(y_.shape) for y_ in y), \
        f'output_format {output_format} does not match output shape {y[0].shape}'
    result = tuple(y_.shape[channel_index] for y_ in y)
    logger.info(f'feature channels: {result}')
    
    model.train(is_training)
    
    return result


def contrails_collate_fn(batch):
    """Collate function for surface volume dataset.
    batch: list of dicts of key:str, value: np.ndarray | list | None
    output: dict of torch.Tensor
    """
    output = defaultdict(list)
    for sample in batch:
        for k, v in sample.items():
            if v is None:
                continue
            output[k].append(v)
    
    for k, v in output.items():
        if isinstance(v[0], str) or v[0].dtype == object:
            output[k] = v
        else:
            output[k] = default_collate(v)
    
    return output


class CacheDictWithSave(dict):
    """Cache dict that saves itself to disk when full."""
    def __init__(self, record_dirs, cache_save_path: Optional[Path] = None, *args, **kwargs):
        self.total_expected_records = len(record_dirs)
        self.cache_save_path = cache_save_path
        self.cache_already_on_disk = False
        
        super().__init__(*args, **kwargs)

        if self.cache_save_path is not None and self.cache_save_path.exists():
            logger.info(f'Loading cache from {self.cache_save_path}')
            self.load()
            assert len(self) >= self.total_expected_records, \
                f'Cache loaded from {self.cache_save_path} has {len(self)} records, ' \
                f'but {self.total_expected_records} were expected.'
            assert all(d in self for d in record_dirs)

    def __setitem__(self, index, value):
        # Hack to allow setting items in joblib.load()
        initialized = (
            hasattr(self, 'total_expected_records') and
            hasattr(self, 'cache_save_path') and
            hasattr(self, 'cache_already_on_disk')
        )
        if not initialized:
            super().__setitem__(index, value)
            return
        
        if len(self) >= self.total_expected_records + 1:
            logger.warning(
                f'More records than expected '
                f'({len(self)} >= {self.total_expected_records + 1}) '
                f'in cache. Will be added, but not saved to disk.'
            )
        super().__setitem__(index, value)
        if (
            not self.cache_already_on_disk and 
            len(self) >= self.total_expected_records and 
            self.cache_save_path is not None
        ):
            self.save()

    def load(self):
        cache = joblib.load(self.cache_save_path)
        self.update(cache)
        self.cache_already_on_disk = True

    def save(self):
        assert not self.cache_already_on_disk, \
            f'cache_already_on_disk = True, but save() was called. ' \
            f'This should not happen.'
        assert not self.cache_save_path.exists(), \
            f'Cache save path {self.cache_save_path} already exists ' \
            f'but was not loaded from disk (cache_already_on_disk = False). ' \
            f'This should not happen.'

        logger.info(f'Saving cache to {self.cache_save_path}')
        joblib.dump(self, self.cache_save_path)
        self.cache_already_on_disk = True

CacheDictWithSaveProxy = MakeProxyType("CacheDictWithSaveProxy", [
    '__contains__', '__delitem__', '__getitem__', '__len__',
    '__setitem__', 'clear', 'copy', 'default_factory', 'fromkeys',
    'get', 'items', 'keys', 'pop', 'popitem', 'setdefault',
    'update', 'values'])

# Register proxy on the main process
SyncManager.register("CacheDictWithSaveProxy", CacheDictWithSave, CacheDictWithSaveProxy)
