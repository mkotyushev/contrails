import hashlib
import logging
import albumentations as A
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional
from lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.data.datasets import MEAN, STD, ContrailsDataset, BANDS
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
        num_folds: Optional[int] = 5,
        fold_index: Optional[int] = None,
        random_state: int = 0,
        img_size: int = 256,
        dataset_kwargs: Optional[dict] = None,
        mix_transform_name: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        use_online_val_test: bool = False,
        cache_dir: Optional[Path] = None,
    ):
        super().__init__()

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        if isinstance(data_dirs_test, str):
            data_dirs_test = [data_dirs_test]
        if dataset_kwargs is None:
            dataset_kwargs = {}
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

        self.train_volume_mean = IMAGENET_DEFAULT_MEAN
        self.train_volume_std = IMAGENET_DEFAULT_STD

        self.collate_fn = contrails_collate_fn
        self.cache = None

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
                    max_pixel_value=255.0,
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

    def make_cache(self, total_expected_records) -> None:
        if self.hparams.cache_dir is None:
            return
        
        self.hparams.cache_dir.mkdir(parents=True, exist_ok=True)

        # Name the cache with md5 hash of 
        # /workspace/contrails/src/data/datasets.py file
        # and ContrailsDataset parameters
        # to avoid using cache when the dataset handling 
        # is changed.
        with open(Path(__file__).parent / 'datasets.py', 'rb') as f:
            content = f.read() + str(self.hparams.dataset_kwargs).encode()
            datasets_file_hash = hashlib.md5(content).hexdigest()
        cache_save_path = self.hparams.cache_dir / f'{datasets_file_hash}.joblib'

        # Check that only one cache file is in the cache dir
        # and its name is the same as the one we are going to create
        cache_files = list(self.hparams.cache_dir.iterdir())
        assert len(cache_files) <= 1, \
            f"More than one cache files found in {cache_save_path} " \
            "which is not advised due to high disk space consumption. " \
            "Please delete all cache files "
        assert len(cache_files) == 0 or cache_files[0] == cache_save_path, \
            f"Cache file {cache_files[0]} is not the same as the one " \
            f"we are going to create {cache_save_path}. " \
            "Please delete all cache files of previous runs."
        
        self.cache = mp.Manager().CacheDictWithSaveProxy(
            total_expected_records=total_expected_records,
            cache_save_path=cache_save_path,
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

        # List all dirs in each data_dir
        test_record_dirs = []
        if self.hparams.data_dirs_test is not None:
            for data_dir in self.hparams.data_dirs_test:
                test_record_dirs += [path for path in data_dir.iterdir() if path.is_dir()]

        # Create shared cache.
        self.make_cache(
            total_expected_records=(
                len(train_record_dirs) + 
                len(val_record_dirs) + 
                len(test_record_dirs)
            )
        )

        # Train
        if self.train_dataset is None and train_record_dirs:
            self.train_dataset = ContrailsDataset(
                record_dirs=train_record_dirs, 
                transform=self.train_transform,
                shared_cache=self.cache,
                **self.hparams.dataset_kwargs,
            )

        if self.val_dataset is None and val_record_dirs:
            self.val_dataset = ContrailsDataset(
                record_dirs=val_record_dirs, 
                transform=self.val_transform,
                shared_cache=self.cache,
                **self.hparams.dataset_kwargs,
            )

        if self.test_dataset is None and test_record_dirs:
            self.test_dataset = ContrailsDataset(
                record_dirs=test_record_dirs, 
                transform=self.test_transform,
                shared_cache=self.cache,
                **self.hparams.dataset_kwargs,
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
