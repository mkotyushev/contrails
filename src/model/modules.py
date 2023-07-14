import logging
from pathlib import Path
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from unittest.mock import patch
from copy import deepcopy
from torch.nn import ModuleDict
from lightning import LightningModule
from typing import Any, Dict, Literal, Mapping, Optional, Union
from torch import Tensor
from lightning.pytorch.cli import instantiate_class
from torchmetrics.classification import BinaryF1Score 
from torchmetrics import Metric
from lightning.pytorch.utilities import grad_norm
from torchvision.ops import sigmoid_focal_loss
from transformers import (
    AutoBackbone,
    TimmBackbone,
    TimmBackboneConfig,
    ConvNextV2Backbone,
    ConvNextV2Config,
    UperNetConfig,
    UperNetForSemanticSegmentation,
    SegformerForSemanticSegmentation, 
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
)
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelDecoder,
    Mask2FormerPixelLevelModuleOutput,
)
from mmseg.models import build_segmentor
from mmengine.runner import load_checkpoint
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmengine.config import Config

from src.data.datasets import N_TIMES, LABELED_TIME_INDEX
from src.model.modeling_video_mask2former import (
    VideoMask2FormerForVideoSegmentation,
    VideoMask2FormerPixelDecoder,
    VideoMask2FormerPixelLevelModuleOutput,
)
from src.model.smp_old import UnetOld
from src.data.transforms import Tta
from src.utils.mechanic import mechanize
from src.utils.utils import (
    FeatureExtractorWrapper, 
    UpsampleWrapper,
    PredictionTargetPreviewAgg, 
    PredictionTargetPreviewGrid, 
    get_feature_channels, 
    state_norm,
)


logger = logging.getLogger(__name__)


class BaseModule(LightningModule):
    def __init__(
        self, 
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: int = 0,
        lr_layer_decay: Union[float, Dict[str, float]] = 1.0,
        n_bootstrap: int = 1000,
        skip_nan: bool = False,
        prog_bar_names: Optional[list] = None,
        mechanize: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.metrics = None
        self.cat_metrics = None

        self.tta = None

        self.configure_metrics()

    def compute_loss_preds(self, batch, *args, **kwargs):
        """Compute losses and predictions."""

    def configure_metrics(self):
        """Configure task-specific metrics."""

    def bootstrap_metric(self, probas, targets, metric: Metric):
        """Calculate metric on bootstrap samples."""

    @staticmethod
    def check_batch_dims(batch):
        assert all(map(lambda x: len(x) == len(batch[0]), batch)), \
            f'All entities in batch must have the same length, got ' \
            f'{list(map(len, batch))}'

    def remove_nans(self, y, y_pred):
        nan_mask = torch.isnan(y_pred)
        
        if nan_mask.ndim > 1:
            nan_mask = nan_mask.any(dim=1)
        
        if nan_mask.any():
            if not self.hparams.skip_nan:
                raise ValueError(
                    f'Got {nan_mask.sum()} / {nan_mask.numel()} nan values in update_metrics. '
                    f'Use skip_nan=True to skip them.'
                )
            logger.warning(
                f'Got {nan_mask.sum()} / {nan_mask.numel()} nan values in update_metrics. '
                f'Dropping them & corresponding targets.'
            )
            y_pred = y_pred[~nan_mask]
            y = y[~nan_mask]
        return y, y_pred

    def extract_targets_and_probas_for_metric(self, preds, batch):
        """Extract preds and targets from batch.
        Could be overriden for custom batch / prediction structure.
        """
        y, y_pred = batch[1].detach(), preds[:, 1].detach().float()
        y, y_pred = self.remove_nans(y, y_pred)
        y_pred = torch.softmax(y_pred, dim=1)
        return y, y_pred

    def update_metrics(self, span, preds, batch):
        """Update train metrics."""
        y, y_proba = self.extract_targets_and_probas_for_metric(preds, batch)
        self.cat_metrics[span]['probas'].update(y_proba)
        self.cat_metrics[span]['targets'].update(y)

    def on_train_epoch_start(self) -> None:
        """Called in the training loop at the very beginning of the epoch."""
        # Unfreeze all layers if freeze period is over
        if self.hparams.finetuning is not None:
            # TODO change to >= somehow
            if self.current_epoch == self.hparams.finetuning['unfreeze_before_epoch']:
                self.unfreeze()

    def unfreeze_only_selected(self):
        """
        Unfreeze only layers selected by 
        model.finetuning.unfreeze_layer_names_*.
        """
        if self.hparams.finetuning is not None:
            for name, param in self.named_parameters():
                selected = False

                if 'unfreeze_layer_names_startswith' in self.hparams.finetuning:
                    selected = selected or any(
                        name.startswith(pattern) 
                        for pattern in self.hparams.finetuning['unfreeze_layer_names_startswith']
                    )

                if 'unfreeze_layer_names_contains' in self.hparams.finetuning:
                    selected = selected or any(
                        pattern in name
                        for pattern in self.hparams.finetuning['unfreeze_layer_names_contains']
                    )
                logger.info(f'Param {name}\'s requires_grad == {selected}.')
                param.requires_grad = selected

    def training_step(self, batch, batch_idx, **kwargs):
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        for loss_name, loss in losses.items():
            self.log(
                f'tl_{loss_name}', 
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch[0].shape[0],
            )
        self.update_metrics('t_metrics', preds, batch)

        # Handle nan in loss
        has_nan = False
        if torch.isnan(total_loss):
            has_nan = True
            logger.warning(
                f'Loss is nan at epoch {self.current_epoch} '
                f'step {self.global_step}.'
            )
        for loss_name, loss in losses.items():
            if torch.isnan(loss):
                has_nan = True
                logger.warning(
                    f'Loss {loss_name} is nan at epoch {self.current_epoch} '
                    f'step {self.global_step}.'
                )
        if has_nan:
            return None
        
        return total_loss
    
    def validation_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        assert dataloader_idx is None or dataloader_idx == 0, 'Only one val dataloader is supported.'
        for loss_name, loss in losses.items():
            self.log(
                f'vl_{loss_name}', 
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                batch_size=batch[0].shape[0],
            )
        self.log(
            f'vl', 
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=batch[0].shape[0],
        )
        self.update_metrics('v_metrics', preds, batch)
        return total_loss

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        _, _, preds = self.compute_loss_preds(batch, **kwargs)
        return preds

    def log_metrics_and_reset(
        self, 
        prefix, 
        on_step=False, 
        on_epoch=True, 
        prog_bar_names=None,
        reset=True,
    ):
        # Get metric span: train or val
        span = None
        if prefix == 'train':
            span = 't_metrics'
        elif prefix in ['val', 'val_ds']:
            span = 'v_metrics'
        
        # Get concatenated preds and targets
        # and reset them
        probas, targets = \
            self.cat_metrics[span]['probas'].compute().cpu(),  \
            self.cat_metrics[span]['targets'].compute().cpu()
        if reset:
            self.cat_metrics[span]['probas'].reset()
            self.cat_metrics[span]['targets'].reset()

        # Calculate and log metrics
        for name, metric in self.metrics.items():
            metric_value = None
            if prefix == 'val_ds':  # bootstrap
                if self.hparams.n_bootstrap > 0:
                    metric_value = self.bootstrap_metric(probas[:, 1], targets, metric)
                else:
                    logger.warning(
                        f'prefix == val_ds but n_bootstrap == 0. '
                        f'No bootstrap metrics will be calculated '
                        f'and logged.'
                    )
            else:
                metric.update(probas[:, 1], targets)
                metric_value = metric.compute()
                metric.reset()
            
            prog_bar = False
            if prog_bar_names is not None:
                prog_bar = (name in prog_bar_names)

            if metric_value is not None:
                self.log(
                    f'{prefix}_{name}',
                    metric_value,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=prog_bar,
                )

    def on_train_epoch_end(self) -> None:
        """Called in the training loop at the very end of the epoch."""
    
    def on_validation_epoch_end(self) -> None:
        """Called in the validation loop at the very end of the epoch."""

    def get_lr_decayed(self, lr, layer_index, layer_name):
        """
        Get lr decayed by 
            - layer index as (self.hparams.lr_layer_decay ** layer_index) if
              self.hparams.lr_layer_decay is float 
              (useful e. g. when new parameters are in classifer head)
            - layer name as self.hparams.lr_layer_decay[layer_name] if
              self.hparams.lr_layer_decay is dict
              (useful e. g. when pretrained parameters are at few start layers 
              and new parameters are the most part of the model)
        """
        if isinstance(self.hparams.lr_layer_decay, dict):
            for key in self.hparams.lr_layer_decay:
                if layer_name.startswith(key):
                    return lr * self.hparams.lr_layer_decay[key]
            return lr
        elif isinstance(self.hparams.lr_layer_decay, float):
            if self.hparams.lr_layer_decay == 1.0:
                return lr
            else:
                return lr * (self.hparams.lr_layer_decay ** layer_index)

    def build_parameter_groups(self):
        """Get parameter groups for optimizer."""
        if self.hparams.lr_layer_decay == 1.0:
            return self.parameters()
        names, params = list(zip(*self.named_parameters()))
        num_layers = len(params)
        grouped_parameters = [
            {
                'params': param, 
                'lr': self.get_lr_decayed(
                    self.hparams.optimizer_init['init_args']['lr'], 
                    num_layers - layer_index - 1,
                    name
                )
            } for layer_index, (name, param) in enumerate(self.named_parameters())
        ]
        logger.info(
            f'Number of layers: {num_layers}, '
            f'min lr: {names[0]}, {grouped_parameters[0]["lr"]}, '
            f'max lr: {names[-1]}, {grouped_parameters[-1]["lr"]}'
        )
        return grouped_parameters

    def configure_optimizer(self):
        if not self.hparams.mechanize:
            optimizer = instantiate_class(args=self.build_parameter_groups(), init=self.hparams.optimizer_init)
            return optimizer
        else:
            # similar to instantiate_class, but with mechanize
            args, init = self.build_parameter_groups(), self.hparams.optimizer_init
            kwargs = init.get("init_args", {})
            if not isinstance(args, tuple):
                args = (args,)
            class_module, class_name = init["class_path"].rsplit(".", 1)
            module = __import__(class_module, fromlist=[class_name])
            args_class = getattr(module, class_name)
            
            optimizer = mechanize(args_class)(*args, **kwargs)
            
            return optimizer

    def configure_lr_scheduler(self, optimizer):
        # Convert milestones from total persents to steps
        # for PiecewiceFactorsLRScheduler
        if (
            'PiecewiceFactorsLRScheduler' in self.hparams.lr_scheduler_init['class_path'] and
            self.hparams.pl_lrs_cfg['interval'] == 'step'
        ):
            total_steps = len(self.trainer.fit_loop._data_source.dataloader()) * self.trainer.max_epochs
            grad_accum_steps = self.trainer.accumulate_grad_batches
            self.hparams.lr_scheduler_init['init_args']['milestones'] = [
                int(milestone * total_steps / grad_accum_steps) 
                for milestone in self.hparams.lr_scheduler_init['init_args']['milestones']
            ]
        elif (
            'PiecewiceFactorsLRScheduler' in self.hparams.lr_scheduler_init['class_path'] and
            self.hparams.pl_lrs_cfg['interval'] == 'epoch'
        ):
            self.hparams.lr_scheduler_init['init_args']['milestones'] = [
                int(milestone * self.trainer.max_epochs) 
                for milestone in self.hparams.lr_scheduler_init['init_args']['milestones']
            ]
        
        scheduler = instantiate_class(args=optimizer, init=self.hparams.lr_scheduler_init)
        scheduler = {
            "scheduler": scheduler,
            **self.hparams.pl_lrs_cfg,
        }

        return scheduler

    def configure_optimizers(self):
        optimizer = self.configure_optimizer()
        if self.hparams.lr_scheduler_init is None:
            return optimizer

        scheduler = self.configure_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms."""
        if self.hparams.log_norm_verbose == 0:
            return
        
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        if self.hparams.log_norm_verbose > 1:
            self.log_dict(norms)
        else:
            if 'grad_2.0_norm_total' in norms:
                self.log('grad_2.0_norm_total', norms['grad_2.0_norm_total'])

        norms = state_norm(self, norm_type=2)
        if self.hparams.log_norm_verbose > 1:
            self.log_dict(norms)
        else:
            if 'state_2.0_norm_total' in norms:
                self.log('state_2.0_norm_total', norms['state_2.0_norm_total'])


backbone_name_to_params = {
    'swinv2': {
        'window_size': (8, 8, 8),
        # TODO: SWIN v2 has patch size 4, upsampling at 
        # the last step degrades quality
        'upsampling': 4,
        'decoder_channels': (256, 128, 64),
        'decoder_mid_channels': (256, 128, 64),
        'decoder_out_channels': (256, 128, 64),
        'scale_factors': (2, 2, 2),
        'format': 'NHWC',
    },
    'convnext': {
        'upsampling': 4,
        'decoder_channels': (256, 128, 64),
        'decoder_mid_channels': (256, 128, 64),
        'decoder_out_channels': (256, 128, 64),
        'format': 'NCHW',
    },
    'convnextv2': {
        'upsampling': 4,
        'decoder_channels': (256, 128, 64),
        'decoder_mid_channels': (256, 128, 64),
        'decoder_out_channels': (256, 128, 64),
        'format': 'NCHW',
    },
    'maxvit': {
        'upsampling': 4,
        'decoder_channels': (256, 128, 64),
        'decoder_mid_channels': (256, 128, 64),
        'decoder_out_channels': (256, 128, 64),
        'format': 'NCHW',
    },
    'caformer': {
        'upsampling': 4,
        'decoder_channels': (256, 128, 64),
        'decoder_mid_channels': (256, 128, 64),
        'decoder_out_channels': (256, 128, 64),
        'format': 'NCHW',
    },
    'mit-b5': {
        'upsampling': 4,
        'decoder_channels': (256, 128, 64),
        'decoder_mid_channels': (256, 128, 64),
        'decoder_out_channels': (256, 128, 64),
        'format': 'NCHW',
    },
    'efficientnetv2': {
        'upsampling': 4,
        'decoder_channels': (256, 128, 64),
        'decoder_mid_channels': (256, 128, 64),
        'decoder_out_channels': (256, 128, 64),
        'format': 'NCHW',
    },
    'efficientnet': {
        'upsampling': 4,
        'decoder_channels': (256, 128, 64),
        'decoder_mid_channels': (256, 128, 64),
        'decoder_out_channels': (256, 128, 64),
        'format': 'NCHW',
    },
}


eva02_backbone_name_to_params = {
    'eva02_B_ade_seg_upernet_sz512': {
        'cfg_path': './lib/EVA/EVA-02/seg/configs/eva02/upernet/upernet_eva02_base_12_512_slide_60k.py',
        'ckpt_path': './pretrained/eva02_B_ade_seg_upernet_sz512.pth',
        'img_size': 512,
    },
    'eva02_L_ade_seg_upernet_sz512': {
        'cfg_path': './lib/EVA/EVA-02/seg/configs/eva02/upernet/upernet_eva02_large_24_512_slide_80k.py',
        'ckpt_path': './pretrained/eva02_L_ade_seg_upernet_sz512.pth',
        'img_size': 512,
    },
    'eva02_L_ade_seg_upernet_sz640': {
        'cfg_path': './lib/EVA/EVA-02/seg/configs/eva02/upernet/upernetpro_eva02_large_24_640_slide_80k.py',
        'ckpt_path': './pretrained/eva02_L_ade_seg_upernet_sz640.pth',
        'img_size': 640,
    },
}
def build_segmentation_eva02(
    backbone_name,
    architecture,
    in_channels, 
    pretrained=True,  # on inference loaded by lightning
    grad_checkpointing=False,
    img_size=384,
    xattn=False,
    postprocess=None,
):
    assert architecture == 'upernet'

    # To register custom models
    from src.model import eva2

    def load_checkpoint_to_model_eva02(checkpoint, model):
        """Modify checkpoint for different img_size & load it to model.

        Interpolate pos_embed weight along dim 1 -- number of patches 
        + 1 for cls token -- which is different for different img_size.
        """
        # Get pos_embed weight
        pos_embed = checkpoint['state_dict']['backbone.pos_embed']
        pretrained_n_patches = pos_embed.shape[1] - 1

        # Extract all except first pos embedding corresponding to cls token
        cls_pos_embed = pos_embed[:, 0, :].unsqueeze(1)
        pos_embed = pos_embed[:, 1:, :]

        # Permute and reshapre to square to interpolate 
        # along last dims:
        # (1, N, E) -> (1, E, N) -> (1, E, sqrt(N), sqrt(N))
        assert int(pos_embed.shape[1] ** 0.5) ** 2 == pos_embed.shape[1], \
            f'checkpoint pos_embed.shape[1] must be a square, got {pos_embed.shape[1]}'
        assert int((model.backbone.pos_embed.shape[1] - 1) ** 0.5) ** 2 == model.backbone.pos_embed.shape[1] - 1, \
            f'model pos_embed.shape[1] must be a square + 1, got {model.backbone.pos_embed.shape[1]}'
        pos_embed = pos_embed.permute(0, 2, 1)
        pos_embed = pos_embed.reshape(
            (
                pos_embed.shape[0],
                pos_embed.shape[1],
                int(pretrained_n_patches ** 0.5),
                int(pretrained_n_patches ** 0.5),
            )
        )

        # Interpolate
        current_n_patches = model.backbone.pos_embed.shape[1] - 1
        pos_embed = F.interpolate(
            pos_embed, 
            size=(
                int(current_n_patches ** 0.5), 
                int(current_n_patches ** 0.5)
            ), 
            mode='bilinear',
        )

        # Reshape back:
        # (1, E, sqrt(N), sqrt(N)) -> (1, E, N) -> (1, N, E)
        pos_embed = pos_embed.flatten(2)
        pos_embed = pos_embed.permute(0, 2, 1)

        # Assign to model
        checkpoint['state_dict']['backbone.pos_embed'] = torch.cat(
            [cls_pos_embed, pos_embed], dim=1
        )

        _load_checkpoint_to_model(
            model, 
            checkpoint, 
            strict=False, 
            logger=None, 
            revise_keys=[(r'^module\.', '')],
        )
    
    # Get config
    cfg_path = eva02_backbone_name_to_params[backbone_name]['cfg_path']
    cfg = Config.fromfile(cfg_path)
    cfg.model.decode_head.num_classes = 1
    cfg.model.auxiliary_head.num_classes = 1
    cfg.model.backbone.xattn = xattn

    # Build model & load checkpoint
    if pretrained:
        ckpt_path = eva02_backbone_name_to_params[backbone_name]['ckpt_path']
        if img_size == eva02_backbone_name_to_params[backbone_name]['img_size']:
            # Simply load checkpoint with mmseg built-in function
            model = build_segmentor(
                cfg.model,
                train_cfg=cfg.get('train_cfg'),
                test_cfg=cfg.get('test_cfg'),
            )
            load_checkpoint(model, ckpt_path, map_location='cuda')
        else:
            # Change img_size
            cfg.model.backbone.img_size = img_size

            # Manually load weights
            checkpoint = _load_checkpoint(ckpt_path, map_location='cuda', logger=None)

            # Create model
            model = build_segmentor(
                cfg.model,
                train_cfg=cfg.get('train_cfg'),
                test_cfg=cfg.get('test_cfg'),
            )

            # Load checkpoint with interpolation of pos_embed
            load_checkpoint_to_model_eva02(checkpoint, model)
    
    # Patch first conv from 3 to in_channels
    smp.encoders._utils.patch_first_conv(
        model, 
        new_in_channels=in_channels,
        default_in_channels=3, 
        pretrained=True,
    )
    
    # Set grad checkpointing
    model.use_checkpoint = grad_checkpointing
    model = UpsampleWrapper(model, scale_factor=4, postprocess=postprocess)

    return model


mmseg_params = {
    ('upernet', 'internimage-b'): {
        'cfg_path': './lib/InternImage/segmentation/configs/ade20k/upernet_internimage_b_512_160k_ade20k.py',
        'ckpt_path': 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_b_512_160k_ade20k.pth',
    }
}
def build_segmentation_mmseg(
    backbone_name,
    architecture,
    in_channels, 
    grad_checkpointing=False,
    postprocess=None,
):
    assert (architecture, backbone_name) in mmseg_params, \
        f'unknown architecture {architecture} and ' \
        f'backbone {backbone_name} combination for mmseg'
    
    # To register custom models
    import sys
    sys.path.insert(0, '/workspace/contrails/lib/InternImage/segmentation')
    from src.model import internimage
    
    # Get config
    cfg_path = mmseg_params[(architecture, backbone_name)]['cfg_path']
    cfg = Config.fromfile(cfg_path)
    cfg.model.decode_head.num_classes = 1
    cfg.model.auxiliary_head.num_classes = 1

    # Build model
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'),
    )

    # Load checkpoint
    load_checkpoint(
        model, 
        mmseg_params[(architecture, backbone_name)]['ckpt_path'], 
        map_location='cuda',
    )

    # Patch first conv from 3 to in_channels
    if in_channels != 3:
        smp.encoders._utils.patch_first_conv(
            model, 
            new_in_channels=in_channels,
            default_in_channels=3, 
            pretrained=True,
        )
    
    # Set grad checkpointing
    model.use_checkpoint = grad_checkpointing
    model = UpsampleWrapper(model, scale_factor=4, postprocess=postprocess)

    return model


def build_segmentation_smp(
    backbone_name, 
    architecture,
    in_channels=1, 
    pretrained=True,
    postprocess=None,
):
    """Build segmentation model."""
    encoder_weights = "imagenet" if pretrained else None
    if architecture == 'unet':
        model = smp.Unet(
            encoder_name=backbone_name,     # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
    else:
        raise ValueError(f'unknown architecture {architecture} for SMP')
    
    # For postprocessing
    if postprocess is not None:
        model = UpsampleWrapper(model, scale_factor=1, postprocess=postprocess)

    return model


def build_segmentation_smp_old(
    backbone_name, 
    architecture,
    in_channels=1, 
    decoder_attention_type=None, 
    img_size=256,
    grad_checkpointing=False,
    pretrained=True,
    postprocess=None,
):
    """Build segmentation model."""
    assert architecture == 'unet'

    if backbone_name.startswith('tf_'):
        backbone_param_key = backbone_name.split('_')[1]
    else:
        backbone_param_key = backbone_name.split('_')[0]
    create_model_kwargs = {}
    if backbone_param_key in ['swinv2', 'maxvit']:
        create_model_kwargs['img_size'] = img_size

    encoder = timm.create_model(
        backbone_name, 
        features_only=True,
        pretrained=pretrained,
        **create_model_kwargs,
    )

    smp.encoders._utils.patch_first_conv(
        encoder, 
        new_in_channels=in_channels,
        default_in_channels=3, 
        pretrained=pretrained,
    )
    encoder.set_grad_checkpointing(grad_checkpointing)

    encoder = FeatureExtractorWrapper(
        encoder, 
        format=backbone_name_to_params[backbone_param_key]['format']
    )
    model = UnetOld(
        encoder=encoder,
        encoder_channels=get_feature_channels(
            encoder, 
            input_shape=(in_channels, img_size, img_size),
            output_format='NCHW',
        ),
        decoder_channels=backbone_name_to_params[backbone_param_key]['decoder_channels'],
        classes=1,
        upsampling=backbone_name_to_params[backbone_param_key]['upsampling'],
        decoder_attention_type=decoder_attention_type,
    )

    # For postprocessing
    if postprocess is not None:
        model = UpsampleWrapper(model, scale_factor=1, postprocess=postprocess)

    return model


class VideoMask2FormerPixelLevelModuleAnyBackbone(nn.Module):
    def __init__(self, config: Mask2FormerConfig):
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://arxiv.org/abs/2112.01527). It runs the input image through a backbone and a pixel
        decoder, generating multi-scale feature maps and pixel embeddings.

        Args:
            config ([`Mask2FormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()

        self.encoder = AutoBackbone.from_config(config.backbone_config)
        self.decoder = VideoMask2FormerPixelDecoder(config, feature_channels=self.encoder.channels)

    def forward(
        self, pixel_values: Tensor, output_hidden_states: bool = False
    ) -> VideoMask2FormerPixelLevelModuleOutput:
        backbone_features = self.encoder(pixel_values).feature_maps
        decoder_output = self.decoder(backbone_features, output_hidden_states=output_hidden_states)

        return VideoMask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=backbone_features[-1],
            encoder_hidden_states=tuple(backbone_features) if output_hidden_states else None,
            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        )


class Mask2FormerPixelLevelModuleAnyBackbone(nn.Module):
    def __init__(self, config: Mask2FormerConfig):
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://arxiv.org/abs/2112.01527). It runs the input image through a backbone and a pixel
        decoder, generating multi-scale feature maps and pixel embeddings.

        Args:
            config ([`Mask2FormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()

        self.encoder = AutoBackbone.from_config(config.backbone_config)
        self.decoder = Mask2FormerPixelDecoder(config, feature_channels=self.encoder.channels)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> Mask2FormerPixelLevelModuleOutput:
        backbone_features = self.encoder(pixel_values).feature_maps
        decoder_output = self.decoder(backbone_features, output_hidden_states=output_hidden_states)

        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=backbone_features[-1],
            encoder_hidden_states=tuple(backbone_features) if output_hidden_states else None,
            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        )

    
def build_segmentation_hf(
    backbone_name, 
    architecture='upernet',
    in_channels=1, 
    grad_checkpointing=False,
    pretrained=True,
    postprocess=None,
):
    if grad_checkpointing:
        logger.warning(
            'grad_checkpointing is not supported for nvidia models, '
            'setting grad_checkpointing=False.'
        )
    # TODO: fix pretrained not used for all the options
    if architecture == 'segformer':
        model = SegformerForSemanticSegmentation.from_pretrained(
            backbone_name,
            num_labels=1,
            ignore_mismatched_sizes=False,
            num_channels=3,
        )
    elif architecture == 'upernet':
        if 'openmmlab' in backbone_name:
            # Native full OpenMMlab models
            model = UperNetForSemanticSegmentation.from_pretrained(
                backbone_name,
                num_labels=1,
                # carefully check 'size mismatch' warnings in logs 
                # it should consist only decode_head and auxiliary_head
                # missmatch due to different number of classes
                ignore_mismatched_sizes=True,
            )
        else:
            # Pretrained backbone with random initialization as per 
            # https://huggingface.co/docs/transformers/main/en/model_doc/upernet#usage
            if backbone_name.startswith('facebook/convnextv2'):
                # Well supported by upernet
                backbone_config = ConvNextV2Config.from_pretrained(
                    backbone_name,
                    out_features=["stage1", "stage2", "stage3", "stage4"],
                )
                backbone = ConvNextV2Backbone.from_pretrained(
                    backbone_name,
                    out_features=["stage1", "stage2", "stage3", "stage4"],
                )
            else:
                backbone_config = TimmBackboneConfig(
                    backbone_name,
                    use_pretrained_backbone=pretrained,
                    out_indices=[0, 1, 2, 3],
                )
                backbone = TimmBackbone(
                    backbone_config,
                )

            config = UperNetConfig(
                backbone_config=backbone_config, 
                num_labels=1, 
                use_auxiliary_head=False,
            )
            model = UperNetForSemanticSegmentation(config)

            # Load pretrained backbone explicitly
            model.backbone.load_state_dict(backbone.state_dict())
    elif architecture == 'mask2former':
        if 'facebook/mask2former' in backbone_name:
            # Native full mask2former models
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                backbone_name,
                num_labels=1,
                # num_queries=1,  # TODO check if num_queries=1 better than default
                # carefully check 'size mismatch' warnings in logs 
                # it should consist only decode_head and auxiliary_head
                # missmatch due to different number of classes
                ignore_mismatched_sizes=True,
            )
        else:
            # Pretrained backbone with random initialization as per 
            # https://huggingface.co/docs/transformers/main/en/model_doc/mask2former#usage
            if backbone_name.startswith('facebook/convnextv2'):
                # Well supported by upernet
                backbone_config = ConvNextV2Config.from_pretrained(
                    backbone_name,
                    out_features=["stage1", "stage2", "stage3", "stage4"],
                )
                backbone = ConvNextV2Backbone.from_pretrained(
                    backbone_name,
                    out_features=["stage1", "stage2", "stage3", "stage4"],
                )
            else:
                backbone_config = TimmBackboneConfig(
                    backbone_name,
                    use_pretrained_backbone=pretrained,
                    out_indices=[0, 1, 2, 3],
                )
                backbone = TimmBackbone.from_pretrained(
                    backbone_name,
                )

            config = Mask2FormerConfig(
                backbone_config=backbone_config, 
                num_labels=1, 
                use_auxiliary_head=False,
            )

            # Hack to use any backbone
            with patch(
                'transformers.models.mask2former.modeling_mask2former.Mask2FormerPixelLevelModule', 
                Mask2FormerPixelLevelModuleAnyBackbone
            ):
                model = Mask2FormerForUniversalSegmentation(config)

            # Load pretrained backbone explicitly
            model.model.pixel_level_module.encoder.load_state_dict(backbone.state_dict())
    elif architecture == 'video_mask2former':
        if 'shivi/video-mask2former' in backbone_name:
            # Native full video mask2former models
            model = VideoMask2FormerForVideoSegmentation.from_pretrained(
                backbone_name,
                num_labels=1,
                # num_queries=1,  # TODO check if num_queries=1 better than default
                # carefully check 'size mismatch' warnings in logs 
                # it should consist only decode_head and auxiliary_head
                # missmatch due to different number of classes
                ignore_mismatched_sizes=True,
            )
        else:
            # Pretrained backbone with random initialization as per 
            backbone_config = TimmBackboneConfig(
                backbone_name,
                use_pretrained_backbone=pretrained,
                out_indices=[0, 1, 2, 3],
            )
            backbone = TimmBackbone.from_pretrained(
                backbone_name,
            )

            config = Mask2FormerConfig(
                backbone_config=backbone_config, 
                num_labels=1, 
                use_auxiliary_head=False,
            )
            config.num_frames = N_TIMES

            # Hack to use any backbone
            with patch(
                'src.model.modeling_video_mask2former.VideoMask2FormerPixelLevelModule', 
                VideoMask2FormerPixelLevelModuleAnyBackbone
            ):
                model = VideoMask2FormerForVideoSegmentation(config)

            # Load pretrained backbone explicitly
            model.model.pixel_level_module.encoder.load_state_dict(backbone.state_dict())
    else:
        raise ValueError(f'unknown architecture {architecture} for HF')
    
    # Final scaling is different for different models
    scale_factor = 1
    if architecture == 'segformer':
        scale_factor = 4
    elif 'mask2former' in architecture:
        if 'facebook/mask2former' in backbone_name or 'shivi/video-mask2former' in backbone_name:
            scale_factor = 4
        else:
            scale_factor = 2

    n_frames = None
    if architecture == 'video_mask2former':
        n_frames = N_TIMES

    model = UpsampleWrapper(
        model, 
        n_frames=n_frames, 
        scale_factor=scale_factor, 
        postprocess=postprocess,
    )

    # Patch first conv from 3 to in_channels
    if in_channels != 3:
        smp.encoders._utils.patch_first_conv(
            model, 
            new_in_channels=in_channels,
            default_in_channels=3, 
            pretrained=pretrained,
        )

    return model


def dice_with_logits_loss(input, target, smooth=1.0, pos_weight=1.0):
    """Dice loss for logits."""
    input = torch.sigmoid(input)

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection * pos_weight + smooth) /
              (iflat.sum() + pos_weight * tflat.sum() + smooth))

# https://github.com/Mehrdad-Noori/Brain-Tumor-Segmentation/blob/
# ebfd12d666dfd6a1743bc4935c6f68a8e26370e3/loss.py
def generalized_dice_loss(y_true, y_pred):
    """
    Generalized Dice Score
    https://arxiv.org/pdf/1707.03237
    
    """
    eps = 1e-7
    y_true    = y_true.flatten(1)
    y_pred    = y_pred.flatten(1)
    sum_p     = y_pred.sum(-1)
    sum_r     = y_true.sum(-1)
    sum_pr    = (y_true * y_pred).sum(-1)
    weights   = 1 / (sum_r ** 2 + eps)
    generalized_dice = (2 * (weights * sum_pr).sum()) / ((weights * (sum_r + sum_p)).sum())
    return 1 - generalized_dice


def parse_loss_name(loss_name):
    # Get loss names and weights
    name_to_weight = {}
    for item in loss_name.split('+'):
        name, weight = item.split('=')
        name_to_weight[name] = float(weight)

    assert len(set(name_to_weight.keys())) == len(name_to_weight.keys()), \
        f'Loss names must be unique, got {name_to_weight.keys()}'
    assert all(map(lambda x: x >= 0, name_to_weight.values())), \
        f'Loss weights must be non-negative, got {name_to_weight.values()}'

    # Remove zero weights
    name_to_weight = {name: weight for name, weight in name_to_weight.items() if weight > 0}

    # Normalize weights
    weight_sum = sum(name_to_weight.values())
    for name in name_to_weight:
        name_to_weight[name] /= weight_sum

    return name_to_weight


class SegmentationModule(BaseModule):
    def __init__(
        self, 
        library: Literal['smp_old', 'smp', 'hf', 'eva', 'mmseg'] = 'smp',
        architecture: Literal['unet', 'upernet', 'segformer', 'mask2former', 'video_mask2former'] = 'unet',
        backbone_name: str = 'timm-efficientnet-b5',
        in_channels: int = 6,
        log_preview_every_n_epochs: int = 10,
        tta_params: Dict[str, Any] = None,
        pretrained: bool = True,
        label_smoothing: float = 0.0,
        pos_weight: Optional[float] = None,
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: int = 0,
        grad_checkpointing: bool = False,
        lr_layer_decay: Union[float, Dict[str, float]] = 1.0,
        n_bootstrap: int = 1000,
        skip_nan: bool = False,
        prog_bar_names: Optional[list] = None,
        mechanize: bool = False,
        img_size=256,
        loss_name: str = 'bce=1.0',
        compile: bool = False,
        lr: float = 1e-3,
        postprocess: Literal['cnn', 'erosion', None] = None,
        pretrained_ckpt_path: Optional[Path] = None,
    ):
        super().__init__(
            optimizer_init=optimizer_init,
            lr_scheduler_init=lr_scheduler_init,
            pl_lrs_cfg=pl_lrs_cfg,
            finetuning=finetuning,
            log_norm_verbose=log_norm_verbose,
            lr_layer_decay=lr_layer_decay,
            n_bootstrap=n_bootstrap,
            skip_nan=skip_nan,
            prog_bar_names=prog_bar_names,
            mechanize=mechanize,
        )
        self.save_hyperparameters()

        torch.set_float32_matmul_precision('medium')

        if library == 'eva':
            self.model = build_segmentation_eva02(
                architecture=architecture,
                in_channels=in_channels, 
                backbone_name=backbone_name,
                pretrained=pretrained,
                grad_checkpointing=grad_checkpointing,
                img_size=img_size,
                xattn=not compile,  # xattn is not supported by torch.compile
                postprocess=postprocess,
            )
        elif library == 'hf':
            self.model = build_segmentation_hf(
                backbone_name=backbone_name, 
                architecture=architecture,
                in_channels=in_channels,
                grad_checkpointing=grad_checkpointing,
                pretrained=pretrained,
                postprocess=postprocess,
            )
        elif library == 'smp':
            self.model = build_segmentation_smp(
                backbone_name=backbone_name, 
                architecture=architecture,
                in_channels=in_channels,
                pretrained=pretrained,
                postprocess=postprocess,
            )
        elif library == 'smp_old':
            self.model = build_segmentation_smp_old(
                backbone_name, 
                architecture=architecture,
                in_channels=in_channels,
                decoder_attention_type=None,
                img_size=img_size,
                grad_checkpointing=grad_checkpointing,
                pretrained=pretrained,
                postprocess=postprocess,
            )
        elif library == 'mmseg':
            self.model = build_segmentation_mmseg(
                backbone_name, 
                architecture=architecture,
                in_channels=in_channels,
                grad_checkpointing=grad_checkpointing,
                postprocess=postprocess,
            )
        else:
            raise ValueError(f'unknown library {library}')

        if compile:
            self.model = torch.compile(self.model)

        if finetuning is not None and finetuning['unfreeze_before_epoch'] == 0:
            self.unfreeze()
        else:
            self.unfreeze_only_selected()

        logger.info(f'Following losses will be used: {parse_loss_name(loss_name)}')

        if tta_params is None:
            tta_params = {}
        self.tta = Tta(
            model=self.model, 
            do_tta=len(tta_params) > 0, 
            **tta_params
        )

        # Load only weights from checkpoint if provided
        if pretrained_ckpt_path is not None:
            logger.info(f'Loading weights from {pretrained_ckpt_path}')
            checkpoint = torch.load(pretrained_ckpt_path)
            logger.info(str(self.load_state_dict(checkpoint['state_dict'], strict=False)))

    def compute_loss_preds(self, batch, only_labeled=False, *args, **kwargs):
        """Compute losses and predictions."""
        preds = self.tta(batch['image'])
        
        if 'mask' not in batch:
            return None, None, preds
        
        if preds.ndim == 4 and only_labeled:
            # video_mask2former, on validation use only one truly labeled frame
            # and not pseudo-labeled
            preds = preds[..., LABELED_TIME_INDEX]
            batch['mask'] = batch['mask'][..., LABELED_TIME_INDEX]

        losses = {}
        for loss_name, loss_weight in parse_loss_name(self.hparams.loss_name).items():
            if loss_name == 'bce':
                pos_weight = None
                if self.hparams.pos_weight is not None:
                    pos_weight = torch.tensor(self.hparams.pos_weight)
                loss_value = F.binary_cross_entropy_with_logits(
                    preds.squeeze(1).float().flatten(),
                    batch['mask'].float().flatten(),
                    reduction='mean',
                    pos_weight=pos_weight,
                )
            elif loss_name == 'focal':
                alpha = -1
                if self.hparams.pos_weight is not None:
                    # alpha is in [0, 1] and pos_weight is absolute 
                    # assuming negative weight is 1.0
                    # so it need to be converted
                    # e. g. 99 -> 0.99, 0.1 -> ~0.09, 0.01 -> ~0.01
                    alpha = self.hparams.pos_weight / (1.0 + self.hparams.pos_weight)
                loss_value = sigmoid_focal_loss(
                    preds.squeeze(1).float().flatten(),
                    batch['mask'].float().flatten(),
                    reduction='mean',
                    alpha=alpha,
                )
            elif loss_name == 'dice':
                loss_fn = smp.losses.DiceLoss(mode="binary", smooth=1.0)
                loss_value = loss_fn(preds, batch['mask'])
            elif loss_name == 'gdl':
                loss_value = generalized_dice_loss(
                    preds.squeeze(1).float(),
                    batch['mask'].float(),
                )
            
            losses[loss_name] = loss_value * loss_weight
        
        total_loss = sum(losses.values())
        return total_loss, losses, preds

    def configure_metrics(self):
        """Configure task-specific metrics."""
        metrics = ModuleDict(
            {
                'preview': PredictionTargetPreviewGrid(preview_downscale=4, n_images=9),
                'dice': BinaryF1Score(),
                'dice_pos': BinaryF1Score(ignore_index=0),
            }
        )
        self.metrics = ModuleDict(
            {
                't_metrics': deepcopy(metrics),
                'v_metrics': deepcopy(metrics),
                'v_tta_metrics': deepcopy(metrics),
            }
        )
        self.cat_metrics = None

    def training_step(self, batch, batch_idx, **kwargs):
        total_loss, losses, preds = self.compute_loss_preds(batch, only_labeled=False, **kwargs)
        for loss_name, loss in losses.items():
            self.log(
                f'tl_{loss_name}', 
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch['image'].shape[0],
            )
        self.log(
            f'tl', 
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch['image'].shape[0],
        )

        # Handle nan in loss
        has_nan = False
        if torch.isnan(total_loss):
            has_nan = True
            logger.warning(
                f'Loss is nan at epoch {self.current_epoch} '
                f'step {self.global_step}.'
            )
        for loss_name, loss in losses.items():
            if torch.isnan(loss):
                has_nan = True
                logger.warning(
                    f'Loss {loss_name} is nan at epoch {self.current_epoch} '
                    f'step {self.global_step}.'
                )
        if has_nan:
            return None
        
        y, y_pred = self.extract_targets_and_probas_for_metric(preds, batch)
        for metric_name, metric in self.metrics['t_metrics'].items():
            if isinstance(metric, PredictionTargetPreviewGrid):  # Epoch-level
                metric.update(
                    batch['image'][:, :3, ...],
                    y_pred, 
                    y, 
                )
            else:
                metric.update(y_pred.flatten(), y.flatten())
                self.log(
                    f't_{metric_name}',
                    metric.compute(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch['image'].shape[0],
                )
                metric.reset()
        
        return total_loss
    
    def validation_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        loss_prefix = 'vl'
        total_loss, losses, preds = self.compute_loss_preds(batch, only_labeled=True, **kwargs)
        assert dataloader_idx is None or dataloader_idx == 0, 'Only one val dataloader is supported.'
        for loss_name, loss in losses.items():
            self.log(
                f'{loss_prefix}_{loss_name}', 
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                batch_size=batch['image'].shape[0],
            )
        self.log(
            f'vl', 
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=batch['image'].shape[0],
        )
        
        span_prefix = 'v'
        y, y_pred = self.extract_targets_and_probas_for_metric(preds, batch)
        for metric in self.metrics[f'{span_prefix}_metrics'].values():
            if isinstance(metric, PredictionTargetPreviewAgg) and batch['indices'] is not None:
                metric.update(
                    arrays={
                        'input': batch['image'][:, :3, ...],
                        'probas': y_pred,
                        'target': y,
                    },
                    pathes=batch['path'],
                    patch_size=y_pred.shape[-2:],
                    indices=batch['indices'], 
                    shape_patches=batch['shape_patches'],
                    shape_original=batch['shape_original'],
                    shape_before_padding=batch['shape_before_padding'],
                )
            elif isinstance(metric, PredictionTargetPreviewGrid):  # Epoch-level
                metric.update(
                    batch['image'][:, :3, ...],
                    y_pred, 
                    y, 
                )
            else:
                metric.update(y_pred.flatten(), y.flatten())
        return total_loss
    
    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        # Prevent storing predictions in _PredictionLoop._predict_step
        self.trainer.predict_loop.return_predictions = False
        _, _, preds = self.compute_loss_preds(batch, only_labeled=True, **kwargs)
        return preds

    def on_train_epoch_end(self) -> None:
        """Called in the training loop at the very end of the epoch."""
        if self.metrics is None:
            return

        for metric_name, metric in self.metrics['t_metrics'].items():
            if isinstance(metric, PredictionTargetPreviewGrid):
                captions, previews = metric.compute()
                if self.current_epoch % self.hparams.log_preview_every_n_epochs == 0:
                    self.trainer.logger.log_image(
                        key=f't_{metric_name}',	
                        images=previews,
                        caption=captions,
                        step=self.current_epoch,
                    )
            metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Called in the validation loop at the very end of the epoch."""
        if self.metrics is None:
            return

        metric_prefix = span_prefix = 'v'
        for metric_name, metric in self.metrics[f'{span_prefix}_metrics'].items():
            if isinstance(metric, PredictionTargetPreviewAgg):
                metric_values, captions, previews = metric.compute()
                if self.current_epoch % self.hparams.log_preview_every_n_epochs == 0:
                    self.trainer.logger.log_image(
                        key=f'{metric_prefix}_{metric_name}',	
                        images=previews,
                        caption=captions,
                        step=self.current_epoch,
                    )
                for name, value in metric_values.items():
                    self.log(
                        f'{metric_prefix}_{name}',
                        value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
            elif isinstance(metric, PredictionTargetPreviewGrid):
                captions, previews = metric.compute()
                metric.reset()
                if self.current_epoch % self.hparams.log_preview_every_n_epochs == 0:
                    self.trainer.logger.log_image(
                        key=f'v_{metric_name}',	
                        images=previews,
                        caption=captions,
                        step=self.current_epoch,
                    )
            else:
                self.log(
                    f'{metric_prefix}_{metric_name}',
                    metric.compute(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
            metric.reset()

    def extract_targets_and_probas_for_metric(self, preds, batch):
        """Extract preds and targets from batch.
        Could be overriden for custom batch / prediction structure.
        """
        y = None
        y_pred = torch.sigmoid(preds.detach().float()).squeeze(1)
        
        if 'mask' in batch:
            y = torch.isclose(
                batch['mask'].detach(), 
                torch.tensor(1.0, device=y_pred.device)
            ).long()
            y, y_pred = self.remove_nans(y, y_pred)

        if y_pred.ndim == 4:
            # video_mask2former: select only single frame
            y_pred = y_pred[..., LABELED_TIME_INDEX]
            if y is not None:
                y = y[..., LABELED_TIME_INDEX]
        
        return y, y_pred
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if not self.hparams.compile and all(['_orig_mod' in k for k in state_dict.keys()]):
            # Trained with compile=True, load with compile=False
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        elif self.hparams.compile and not all(['_orig_mod' in k for k in state_dict.keys()]):
            # Trained with compile=False, load with compile=True
            state_dict = {
                '.'.join([k.split('.')[0]] + ['_orig_mod'] + [k.split('.')[1:]])
                : v for k, v in state_dict.items()
            }
        return super().load_state_dict(state_dict, strict)
