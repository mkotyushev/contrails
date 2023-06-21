import logging
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn import ModuleDict
from lightning import LightningModule
from typing import Any, Dict, Literal, Optional, Union
from torch import Tensor
from lightning.pytorch.cli import instantiate_class
from torchmetrics import Dice, Metric
from lightning.pytorch.utilities import grad_norm
from torchvision.ops import sigmoid_focal_loss
from transformers import SegformerForSemanticSegmentation

from src.data.transforms import Tta
from src.model.smp import Unet, patch_first_conv, DiceLoss
from src.utils.mechanic import mechanize
from src.utils.utils import (
    FeatureExtractorWrapper, 
    Eva02Wrapper,
    HfWrapper,
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
        total_loss, losses, preds = self.compute_loss_preds(batch, tta=False, **kwargs)
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
        total_loss, losses, preds = self.compute_loss_preds(batch, tta=self.tta is not None, **kwargs)
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
        _, _, preds = self.compute_loss_preds(batch, tta=self.tta is not None, **kwargs)
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
}


eva02_backbone_name_to_params = {
    'eva02_B_ade_seg_upernet_sz512': {
        'cfg_path': './lib/EVA/EVA-02/seg/configs/eva02/upernet/upernet_eva02_base_12_512_slide_60k.py',
        'ckpt_path': './pretrained/eva02_B_ade_seg_upernet_sz512.pth',
    },
    'eva02_L_ade_seg_upernet_sz640': {
        'cfg_path': './lib/EVA/EVA-02/seg/configs/eva02/upernet/upernetpro_eva02_large_24_640_slide_80k.py',
        'ckpt_path': './pretrained/eva02_L_ade_seg_upernet_sz640.pth',
    },
}
def build_segmentation_eva02(
    backbone_name,
    in_channels, 
    pretrained=True,  # on inference loaded by lightning
    grad_checkpointing=False,
    img_size=384,
):
    import sys

    sys.path.insert(0, '/workspace/contrails/lib/EVA/EVA-02/seg')
    from backbone import eva2
    from mmseg.models import build_segmentor
    from mmengine.runner import load_checkpoint
    from mmengine.config import Config
    
    # Get config
    cfg_path = eva02_backbone_name_to_params[backbone_name]['cfg_path']
    cfg = Config.fromfile(cfg_path)
    cfg.model.decode_head.num_classes = 1
    cfg.model.auxiliary_head.num_classes = 1
    cfg.model.backbone.img_size = img_size

    # Build model & load checkpoint
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if pretrained:
        ckpt_path = eva02_backbone_name_to_params[backbone_name]['ckpt_path']
        load_checkpoint(model, ckpt_path, map_location='cuda')
    
    # Patch first conv from 3 to in_channels
    patch_first_conv(
        model, 
        new_in_channels=in_channels,
        default_in_channels=3, 
        pretrained=True,
        conv_type=nn.Conv2d,
    )
    
    # Set grad checkpointing
    model.use_checkpoint = grad_checkpointing
    model = Eva02Wrapper(model, scale_factor=4)

    return model


def build_segmentation_timm(
    backbone_name, 
    in_channels=1, 
    decoder_attention_type=None, 
    img_size=256,
    grad_checkpointing=False,
    pretrained=True,
):
    """Build segmentation model."""
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

    patch_first_conv(
        encoder, 
        new_in_channels=in_channels,
        default_in_channels=3, 
        pretrained=pretrained,
        conv_type=nn.Conv2d,
    )
    encoder.set_grad_checkpointing(grad_checkpointing)

    encoder = FeatureExtractorWrapper(
        encoder, 
        format=backbone_name_to_params[backbone_param_key]['format']
    )
    model = Unet(
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

    return model


def build_segmentation_hf(
    backbone_name, 
    in_channels=1, 
    grad_checkpointing=False,
    pretrained=True,
):
    if grad_checkpointing:
        logger.warning(
            'grad_checkpointing is not supported for nvidia models, '
            'setting grad_checkpointing=False.'
        )
    model = SegformerForSemanticSegmentation.from_pretrained(
        backbone_name,
        num_labels=1,
        ignore_mismatched_sizes=False,
        num_channels=3,
    )
    model = HfWrapper(model)

    # Patch first conv from 3 to in_channels
    patch_first_conv(
        model, 
        new_in_channels=in_channels,
        default_in_channels=3, 
        pretrained=pretrained,
        conv_type=nn.Conv2d,
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


# https://arxiv.org/pdf/1707.03237.pdf
def gdl(input, target, smooth=1.0, weight=None):
    pos_target = (target > 0).float().flatten()
    pos_proba = torch.sigmoid(input).flatten()

    neg_target = (target == 0).float().flatten()
    neg_proba = 1 - pos_proba

    if weight is None:
        weight = torch.tensor([1.0, 1.0], dtype=torch.float32, device=input.device)
    loss = 1 - 2 * (
        weight[0] * (neg_target * neg_proba).sum() + 
        weight[1] * (pos_target * pos_proba).sum() + 
        smooth
    ) / (
        weight[0] * (neg_target + neg_proba).sum() + 
        weight[1] * (pos_target + pos_proba).sum() + 
        smooth
    )
    return loss.mean()


class SegmentationModule(BaseModule):
    def __init__(
        self, 
        decoder_attention_type: Literal[None, 'scse'] = None,
        backbone_name: str = 'swinv2_tiny_window8_256.ms_in1k',
        in_channels: int = 6,
        log_preview_every_n_epochs: int = 10,
        tta_each_n_epochs: int = 10,
        tta_params: Dict[str, Any] = None,
        pretrained: bool = True,
        label_smoothing: float = 0.0,
        pos_weight: float = 1.0,
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
        loss_name: str = 'bce',
        compile: bool = False,
        lr: float = 1e-3,
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

        if backbone_name.startswith('eva02'):
            self.model = build_segmentation_eva02(
                in_channels=in_channels, 
                backbone_name=backbone_name,
                pretrained=pretrained,
                grad_checkpointing=grad_checkpointing,
                img_size=img_size,
            )
        elif backbone_name.startswith('nvidia'):
            self.model = build_segmentation_hf(
                backbone_name, 
                in_channels=in_channels,
                grad_checkpointing=grad_checkpointing,
                pretrained=pretrained,
            )
        else:
            self.model = build_segmentation_timm(
                backbone_name, 
                in_channels=in_channels,
                decoder_attention_type=decoder_attention_type,
                img_size=img_size,
                grad_checkpointing=grad_checkpointing,
                pretrained=pretrained,
            )

        if compile:
            self.model = torch.compile(self.model, mode="reduce-overhead")

        if finetuning is not None and finetuning['unfreeze_before_epoch'] == 0:
            self.unfreeze()
        else:
            self.unfreeze_only_selected()

        assert loss_name in ['bce', 'focal', 'dice', 'gdl'], f'Unknown loss name {loss_name}.'
        assert tta_each_n_epochs != 0, \
            'tta_each_n_epochs == 0 is not supported, use tta_each_n_epochs == -1 to disable TTA.'
        if tta_params is None:
            tta_params = {}
        self.tta = \
            None \
            if tta_each_n_epochs < 0 else \
            Tta(model=self.model, **tta_params)

    def compute_loss_preds(self, batch, tta=False, *args, **kwargs):
        """Compute losses and predictions."""
        if tta:
            assert self.tta is not None, 'tta is not initialized.'
            preds = self.tta(batch['image'])
        else:
            preds = self.model(batch['image'])
        
        if 'mask' not in batch:
            return None, None, preds

        if self.hparams.loss_name == 'bce':
            # 3d_acs_weights outputs probabilities, not logits
            weight = torch.where(
                batch['mask'] == 1,
                torch.tensor(self.hparams.pos_weight, dtype=torch.float32, device=batch['mask'].device),
                torch.tensor(1.0, dtype=torch.float32, device=batch['mask'].device),
            ).flatten()
            loss_value = F.binary_cross_entropy_with_logits(
                preds.squeeze(1).float().flatten(),
                batch['mask'].float().flatten(),
                reduction='mean',
                weight=weight,
            )
        elif self.hparams.loss_name == 'focal':
            loss_value = sigmoid_focal_loss(
                preds.squeeze(1).float().flatten(),
                batch['mask'].float().flatten(),
                reduction='mean',
                # alpha is in [0, 1] with negative weight assigned to 1 - alpha 
                # and pos_weight is absolute assuming negative weight is 1.0
                # so it need to be converted
                alpha=self.hparams.pos_weight / (1.0 + self.hparams.pos_weight),
            )
        elif self.hparams.loss_name == 'dice':
            loss_fn = DiceLoss(mode="binary", smooth=1.0)
            loss_value = loss_fn(preds, batch['mask'])
        elif self.hparams.loss_name == 'gdl':
            loss_value = gdl(
                preds.squeeze(1).float().flatten(),
                batch['mask'].float().flatten(),
                weight=torch.tensor([1.0, self.hparams.pos_weight], dtype=torch.float32, device=batch['mask'].device),
            )
        
        losses = {
            self.hparams.loss_name: loss_value,
        }
        total_loss = sum(losses.values())
        return total_loss, losses, preds

    def configure_metrics(self):
        """Configure task-specific metrics."""
        metrics = ModuleDict(
            {
                'preview': PredictionTargetPreviewGrid(preview_downscale=4, n_images=9),
                'dice': Dice(average='micro'),
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
        total_loss, losses, preds = self.compute_loss_preds(batch, tta=False, **kwargs)
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
    
    def _validation_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, tta=False, **kwargs) -> Tensor:
        loss_prefix = 'vl' if not tta else 'vl_tta'
        total_loss, losses, preds = self.compute_loss_preds(batch, tta=tta, **kwargs)
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
        
        span_prefix = 'v' if not tta else 'v_tta'
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
    
    def validation_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        tta = self.tta is not None and self.current_epoch % self.hparams.tta_each_n_epochs == 0
        result = self._validation_step(batch, batch_idx, dataloader_idx, tta=False, **kwargs)
        if tta:
            self._validation_step(batch, batch_idx, dataloader_idx, tta=True, **kwargs)
        return result

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        _, _, preds = self.compute_loss_preds(batch, tta=self.tta is not None, **kwargs)
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

    def _on_validation_epoch_end(self, tta) -> None:
        """Called in the validation loop at the very end of the epoch."""
        if self.metrics is None:
            return

        metric_prefix = span_prefix = 'v' if not tta else 'v_tta'
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

    def on_validation_epoch_end(self) -> None:
        tta = self.tta is not None and self.current_epoch % self.hparams.tta_each_n_epochs == 0
        self._on_validation_epoch_end(tta=False)
        if tta:
            self._on_validation_epoch_end(tta=True)

    def extract_targets_and_probas_for_metric(self, preds, batch):
        """Extract preds and targets from batch.
        Could be overriden for custom batch / prediction structure.
        """
        y = None
        y_pred = torch.sigmoid(preds.detach().float()).squeeze(1)
        
        if 'mask' in batch:
            y = batch['mask'].detach()
            y, y_pred = self.remove_nans(y, y_pred)
        
        return y, y_pred
