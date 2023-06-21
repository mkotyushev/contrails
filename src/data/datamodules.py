import logging
import albumentations as A
from pathlib import Path
from typing import List, Optional
from lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from src.data.datasets import ContrailsDataset, BANDS
from src.data.transforms import (
    CopyPastePositive,
    CutMix,
    MixUp,
)
from src.utils.utils import contrails_collate_fn


logger = logging.getLogger(__name__)


class ContrailsDatamodule(LightningDataModule):
    """Base datamodule for contrails data."""
    def __init__(
        self,
        data_dirs: List[Path] | Path = '/workspace/data/train',
        data_dirs_test: Optional[List[Path] | Path] = None,	
        mmap: bool = False,
        num_folds: Optional[int] = 5,
        fold_index: Optional[int] = None,
        random_state: int = 0,
        img_size: int = 256,
        band_ids=BANDS,
        mix_transform_name: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        use_online_val_test: bool = False,
    ):
        super().__init__()

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        if isinstance(data_dirs_test, str):
            data_dirs_test = [data_dirs_test]
        self.save_hyperparameters()

        assert (
            num_folds is None and fold_index is None or
            num_folds is not None and fold_index is not None
        ), 'num_folds and fold_index must be both None or both not None'

        assert mix_transform_name is None or \
            mix_transform_name in ['cutmix', 'mixup', 'cpp'], \
            'mix_transform_name must be one of [cutmix, mixup, cpp]'

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_transform = None
        self.train_transform_mix = None
        self.val_transform = None
        self.test_transform = None

        self.train_volume_mean = 0
        self.train_volume_std = 1

        self.collate_fn = contrails_collate_fn

    def build_transforms(self) -> None:        
        # If mix is used, then train_transform_mix is used
        # (additional costly sampling & augmentation from dataset)
        # and post transform is done in train_transform_mix
        # otherwise post transform is done in train_transform 
        post_transform = []
        if self.hparams.mix_transform_name is None:
            post_transform = [
                ToTensorV2(),
            ]

        self.train_transform = A.Compose(
            [
                A.Resize(
                    height=self.hparams.img_size,
                    width=self.hparams.img_size,
                    always_apply=True,
                ),
                A.Rotate(
                    p=0.5, 
                    limit=45, 
                    crop_border=False,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=[10, 50]),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                    ], 
                    p=0.4
                ),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.CoarseDropout(
                    max_holes=1, 
                    max_width=int(self.hparams.img_size * 0.3), 
                    max_height=int(self.hparams.img_size * 0.3), 
                    mask_fill_value=0, p=0.5
                ),
                A.Normalize(
                    max_pixel_value=255,
                    mean=self.train_volume_mean,
                    std=self.train_volume_std,
                    always_apply=True,
                ),
                *post_transform,
            ],
        )
        self.train_transform_mix = None
        if self.hparams.mix_transform_name is not None:
            if self.hparams.mix_transform_name == 'cutmix':
                mix_transform = CutMix(
                    width=int(self.hparams.img_size * 0.3), 
                    height=int(self.hparams.img_size * 0.3), 
                    p=1.0,
                    always_apply=False,
                )
            elif self.hparams.mix_transform_name == 'mixup':
                mix_transform = MixUp(
                    alpha=3.0, 
                    beta=3.0, 
                    p=1.0,
                    always_apply=False,
                )
            elif self.hparams.mix_transform_name == 'cpp':
                mix_transform = CopyPastePositive(
                    mask_index=2, 
                    p=1.0, 
                    always_apply=False,
                )
            self.train_transform_mix = A.Compose(
                [
                    mix_transform,
                    ToTensorV2(),
                ],
            )
        self.val_transform = self.test_transform = A.Compose(
            [
                A.Resize(
                    height=self.hparams.img_size,
                    width=self.hparams.img_size,
                    always_apply=True,
                ),
                A.Normalize(
                    max_pixel_value=255,
                    mean=self.train_volume_mean,
                    std=self.train_volume_std,
                    always_apply=True,
                ),
                ToTensorV2(),
            ],
        )

    def setup(self, stage: str = None) -> None:
        self.build_transforms()

        # K-fold split or full train
        train_record_dirs, val_record_dirs = [], []
        if self.hparams.data_dirs is not None:
            # List all dirs in each data_dir
            dirs = []
            for data_dir in self.hparams.data_dirs:
                dirs += [path for path in data_dir.iterdir() if path.is_dir()]
            
            if self.hparams.fold_index is not None:
                # Split train dirs to train and val
                kfold = KFold(
                    n_splits=self.hparams.num_folds,
                    shuffle=True,
                    random_state=self.hparams.random_state,
                )
                for i, (train_index, val_index) in enumerate(kfold.split(dirs)):
                    if i == self.hparams.fold_index:
                        train_record_dirs = [dirs[i] for i in train_index]
                        val_record_dirs = [dirs[i] for i in val_index]
                        break
            else:
                train_record_dirs = dirs

        # Train
        if self.train_dataset is None and train_record_dirs:
            self.train_dataset = ContrailsDataset(
                record_dirs=train_record_dirs, 
                band_ids=self.hparams.band_ids,
                mask_type='voting50',
                propagate_mask=False,
                mmap=self.hparams.mmap,
                transform=self.train_transform,
            )

        if self.val_dataset is None and val_record_dirs:
            self.val_dataset = ContrailsDataset(
                record_dirs=val_record_dirs, 
                band_ids=self.hparams.band_ids,
                mask_type='voting50',
                propagate_mask=False,
                mmap=self.hparams.mmap,
                transform=self.val_transform,
            )

        if self.test_dataset is None and self.hparams.data_dirs_test is not None:
            # List all dirs in each data_dir
            test_record_dirs = []
            for data_dir in self.hparams.data_dirs_test:
                test_record_dirs += [path for path in data_dir.iterdir() if path.is_dir()]
            
            self.val_dataset = ContrailsDataset(
                record_dirs=test_record_dirs, 
                band_ids=self.hparams.band_ids,
                mask_type='voting50',
                propagate_mask=False,
                mmap=self.hparams.mmap,
                transform=self.test_transform,
            )

    def reset_transforms(self):
        self.build_transforms()

        if self.train_dataset is not None:
            self.train_dataset.transform = self.train_transform
            self.train_dataset.transform_mix = self.train_transform_mix
        if self.val_dataset is not None:
            self.val_dataset.transform = self.val_transform
        if self.test_dataset is not None:
            self.test_dataset.transform = self.test_transform

    def train_dataloader(self) -> DataLoader:
        sampler, shuffle = None, True      
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=self.collate_fn,
            sampler=sampler,
            shuffle=shuffle,
            drop_last=True,  # for compiling
        )

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=self.collate_fn,
            shuffle=False
        )
        
        return val_dataloader

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "test dataset is not defined"
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=self.collate_fn,
            shuffle=False
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()