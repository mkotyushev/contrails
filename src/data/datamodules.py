import hashlib
import logging
import math
import albumentations as A
import multiprocessing as mp
import pandas as pd
import yaml
import git
from copy import deepcopy
from pathlib import Path
from typing import List, Literal, Optional
from lightning import LightningDataModule
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
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
from src.utils.randaugment import RandAugment


logger = logging.getLogger(__name__)


class ContrailsDatamodule(LightningDataModule):
    """Base datamodule for contrails data."""
    def __init__(
        self,
        data_dirs: List[Path] | Path = '/workspace/data/train',
        data_dirs_test: Optional[List[Path] | Path] = None,	
        num_folds: Optional[int] = 6,
        fold_index: Optional[int] = None,
        fold_index_outer: Optional[int] = None,
        random_state: int = 0,
        img_size: int = 256,
        dataset_kwargs: Optional[dict] = None,
        randaugment_num_ops: int = 2,
        randaugment_magnitude: int = 9,
        mix_transform_name: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        use_online_val_test: bool = False,
        mmap: bool = True,
        disable_cache: bool = False,
        cache_dir: Optional[Path] = None,
        empty_mask_strategy: Literal['cpp', 'drop', 'drop_only_train'] | None = None,
        split_info_path: Optional[Path] = None,
        scale_factor: int = 1,
        to_predict: Literal['test', 'val', 'train'] = 'test',
        remove_pseudolabels_from_val_test: bool = True,
    ):
        super().__init__()

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        if isinstance(data_dirs_test, str):
            data_dirs_test = [data_dirs_test]
        if dataset_kwargs is None:
            dataset_kwargs = {}
        logger.info(f"dataset_kwargs types: {[type(v) for v in dataset_kwargs.values()]}")
        logger.info(f"dataset_kwargs: {dataset_kwargs}")
        
        self.save_hyperparameters()

        assert scale_factor >= 1, 'scale_factor must be >= 1'
        assert (
            num_folds is None and fold_index is None or
            num_folds is not None and fold_index is not None
        ), 'num_folds and fold_index must be both None or both not None'

        assert mix_transform_name is None or \
            mix_transform_name in ['cutmix', 'mixup'], \
            'mix_transform_name must be one of [cutmix, mixup]'

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_transform = None
        self.train_transform_mix = None
        self.train_transform_cpp = None

        self.val_transform = None
        self.test_transform = None

        self.train_volume_mean = sum(IMAGENET_DEFAULT_MEAN) / len(IMAGENET_DEFAULT_MEAN)
        self.train_volume_std = sum(IMAGENET_DEFAULT_STD) / len(IMAGENET_DEFAULT_STD)

        self.collate_fn = contrails_collate_fn
        self.cache = None

    def build_transforms(self) -> None:
        # Train augmentations
        if self.hparams.scale_factor == 1:
            train_resize_transform = A.Resize(
                height=self.hparams.img_size,
                width=self.hparams.img_size,
                always_apply=True,
            )
        else:
            train_resize_transform = A.RandomResizedCrop(
                height=self.hparams.img_size,
                width=self.hparams.img_size,
                scale=(1 / self.hparams.scale_factor, 1.0),
                always_apply=True,
            )

        self.train_transform = A.Compose(
            [
                train_resize_transform,
                RandAugment(self.hparams.randaugment_num_ops, self.hparams.randaugment_magnitude),
                A.Normalize(
                    max_pixel_value=255.0,
                    mean=self.train_volume_mean,
                    std=self.train_volume_std,
                    always_apply=True,
                ),
                ToTensorV2(),
            ],
        )

        # Train copy-paste augmentation
        if self.hparams.empty_mask_strategy == 'cpp':
            self.train_transform_cpp = CopyPastePositive(
                mask_positive_value=255,
                always_apply=False, 
                p=0.5,
            )

        # Train mix augmentation
        if self.hparams.mix_transform_name is not None:
            if self.hparams.mix_transform_name == 'cutmix':
                self.train_transform_mix = CutMix(
                    width=int(self.hparams.img_size * 0.3), 
                    height=int(self.hparams.img_size * 0.3), 
                    p=0.5,
                    always_apply=False,
                )
            elif self.hparams.mix_transform_name == 'mixup':
                self.train_transform_mix = MixUp(
                    alpha=3.0, 
                    beta=3.0, 
                    p=0.5,
                    always_apply=False,
                )
        
        # Val and test transformations
        self.val_transform = self.test_transform = A.Compose(
            [
                A.Resize(
                    height=self.hparams.scale_factor * self.hparams.img_size,
                    width=self.hparams.scale_factor * self.hparams.img_size,
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

        if self.hparams.to_predict == 'train':
            self.train_transform = self.val_transform
            self.train_transform_cpp = None
            self.train_transform_mix = None

    def setup(self, stage: str = None) -> None:
        self.build_transforms()

        # Split on train, val and test
        train_record_dirs, val_record_dirs, test_record_dirs = [], [], []
        if self.hparams.data_dirs is not None:
            # List all dirs in each data_dir
            dirs = []
            for data_dir in self.hparams.data_dirs:
                dirs += [path for path in sorted(data_dir.iterdir()) if path.is_dir()]
            
            (
                train_record_dirs, \
                val_record_dirs, 
                test_record_dirs, 
                train_is_mask_empty, 
                *_,
            ) = self.split_train_val_test(dirs)

        # Use test dirs if given
        if self.hparams.data_dirs_test is not None:
            if self.hparams.fold_index_outer is not None:
                logger.warning(
                    f'Using test data_dirs_test instead of '
                    f'fold_index_outer {self.hparams.fold_index_outer}'
                )
            test_record_dirs = []
            for data_dir in self.hparams.data_dirs_test:
                test_record_dirs += [path for path in data_dir.iterdir() if path.is_dir()]

        # Create shared cache.
        if not self.hparams.disable_cache:
            use_not_labeled = self.hparams.dataset_kwargs.get('use_not_labeled', False)
            self.make_cache(
                record_dirs=train_record_dirs + val_record_dirs + test_record_dirs,
                use_not_labeled=use_not_labeled
            )

        # Train
        if self.train_dataset is None and train_record_dirs:
            self.train_dataset = ContrailsDataset(
                record_dirs=train_record_dirs, 
                transform=self.train_transform,
                transform_mix=self.train_transform_mix,
                transform_cpp=self.train_transform_cpp,
                shared_cache=self.cache,
                is_mask_empty=train_is_mask_empty,
                mmap=self.hparams.mmap,
                **self.hparams.dataset_kwargs,
            )

        # Remove pseudolabels from val & test
        # Note: for real training remove_pseudolabels_from_val_test should be 
        # set to True to avoid biasing the model with pseudolabels.
        # But to create proper cache reusable by all the folds, 
        # remove_pseudolabels_from_val_test need to be set to False
        # for create_cache.py script
        val_test_dataset_kwargs = deepcopy(self.hparams.dataset_kwargs)
        if self.hparams.remove_pseudolabels_from_val_test:
            val_test_dataset_kwargs['use_not_labeled'] = False
            val_test_dataset_kwargs['pseudolabels_path'] = None

        if self.val_dataset is None and val_record_dirs:
            self.val_dataset = ContrailsDataset(
                record_dirs=val_record_dirs, 
                transform=self.val_transform,
                transform_mix=None,
                transform_cpp=None,
                shared_cache=self.cache,
                is_mask_empty=None,
                mmap=self.hparams.mmap,
                **val_test_dataset_kwargs,
            )

        if self.test_dataset is None and test_record_dirs:
            self.test_dataset = ContrailsDataset(
                record_dirs=test_record_dirs, 
                transform=self.test_transform,
                transform_mix=None,
                transform_cpp=None,
                shared_cache=self.cache,
                is_mask_empty=None,
                mmap=self.hparams.mmap,
                **val_test_dataset_kwargs,
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

    def make_cache(self, record_dirs, use_not_labeled) -> None:
        cache_save_path = None
        if self.hparams.cache_dir is not None:
            # Name the cache with md5 hash of 
            # /workspace/contrails/src/data/datasets.py file
            # and ContrailsDataset parameters
            # to avoid using cache when the dataset handling 
            # is changed.
            with open(Path(__file__).parent / 'datasets.py', 'rb') as f:
                datasets_content = f.read()
                datasets_file_hash = hashlib.md5(
                    datasets_content + 
                    str(self.hparams.dataset_kwargs).encode()
                ).hexdigest()
            cache_save_path = self.hparams.cache_dir / f'{datasets_file_hash}.joblib'

        self.cache = mp.Manager().CacheDictWithSaveProxy(
            record_dirs=record_dirs,
            cache_save_path=cache_save_path,
            use_not_labeled=use_not_labeled,
        )

        if cache_save_path is None:
            return
        
        self.hparams.cache_dir.mkdir(parents=True, exist_ok=True)

        # Check that only one cache file is in the cache dir
        # and its name is the same as the one we are going to create
        cache_files = list(self.hparams.cache_dir.glob('*.joblib'))
        assert len(cache_files) <= 1, \
            f"More than one cache files found in {cache_save_path} " \
            "which is not advised due to high disk space consumption. " \
            "Please delete all cache files "
        assert len(cache_files) == 0 or cache_files[0] == cache_save_path, \
            f"Cache file {cache_files[0]} is not the same as the one " \
            f"we are going to create {cache_save_path}. " \
            "Please delete all cache files of previous runs."

        # Copy datasets.py to cache dir and
        # save cache info to a file
        # to ease debugging
        (self.hparams.cache_dir / 'datasets.py').write_bytes(datasets_content)
        
        with open(self.hparams.cache_dir / 'cache_info.yaml', 'w') as f:
            commit_id, dirty = None, None
            try:
                commit_id = git.Repo(search_parent_directories=True).head.object.hexsha
                dirty = git.Repo(search_parent_directories=True).is_dirty()
            except git.exc.InvalidGitRepositoryError:
                logger.warning("Not a git repository")
            
            cache_info = {
                'dataset_kwargs': self.hparams.dataset_kwargs, 
                'commit_id': commit_id,
                'dirty': dirty,
            }
            yaml.dump(cache_info, f, default_flow_style=False)
    
    def split_train_val_test(self, dirs):
        # Load split info from file
        df_full = pd.read_csv(self.hparams.split_info_path)
        df_full['path'] = df_full['path'].astype(str)
        df_full['is_mask_empty'] = ~df_full['mask_sum_g0']

        # Merge with given dirs
        df_dirs = pd.DataFrame(dirs, columns=['path'])
        df_dirs['path'] = df_dirs['path'].astype(str)
        df = df_dirs.merge(
            df_full, 
            on='path', 
            how='left',
        )

        is_mask_empty = df['is_mask_empty'].values

        if self.hparams.empty_mask_strategy == 'drop':
            dirs = [
                d for d, is_empty in zip(dirs, is_mask_empty) 
                if not is_empty
            ]
            is_mask_empty = [False] * len(dirs)

        train_record_dirs, val_record_dirs, test_record_dirs = [], [], []
        train_is_mask_empty, val_is_mask_empty, test_is_mask_empty = [], [], []
        if self.hparams.fold_index_outer is None:
            if self.hparams.fold_index is None:
                # Simply use all the dirs as train
                train_record_dirs = dirs
                train_is_mask_empty = is_mask_empty
            else:
                # Split train dirs to train and val
                # stratified by mask_sum_qcut_code
                # and grouped by spatial set_id
                # and num_folds folds
                kfold = StratifiedGroupKFold(
                    n_splits=self.hparams.num_folds,
                    shuffle=True,
                    random_state=self.hparams.random_state,
                )
                for i, (train_index, val_index) in enumerate(
                    kfold.split(
                        df, 
                        df['mask_sum_qcut_code'], 
                        df['set_id_spatial']
                    )
                ):
                    if i == self.hparams.fold_index:
                        train_record_dirs = [dirs[i] for i in train_index]
                        val_record_dirs = [dirs[i] for i in val_index]
                        train_is_mask_empty = [is_mask_empty[i] for i in train_index]
                        val_is_mask_empty = [is_mask_empty[i] for i in val_index]
                        break
        else:
            # Split train dirs to train + val and test
            # stratified by mask_sum_qcut_code
            # and grouped by spatial set_id
            # and num_folds folds.
            
            # Then split train + val on train and val
            # using the same stratification scheme
            # and num_folds - 1 folds.
            kfold_outer = StratifiedGroupKFold(
                n_splits=self.hparams.num_folds,
                shuffle=True,
                random_state=self.hparams.random_state,
            )
            for i, (train_val_index, test_index) in enumerate(
                kfold_outer.split(
                    df, 
                    df['mask_sum_qcut_code'], 
                    df['set_id_spatial']
                )
            ):
                if i == self.hparams.fold_index_outer:
                    train_val_record_dirs = [dirs[i] for i in train_val_index]
                    test_record_dirs = [dirs[i] for i in test_index]
                    train_val_is_mask_empty = [is_mask_empty[i] for i in train_val_index]
                    test_is_mask_empty = [is_mask_empty[i] for i in test_index]

                    if self.hparams.fold_index is None:
                        # Use all the train + val as train
                        train_record_dirs = train_val_record_dirs
                        train_is_mask_empty = train_val_is_mask_empty
                        break

                    # Split train + val on train and val
                    df_outer = df.iloc[train_val_index].copy()
                    
                    kfold_inner = StratifiedGroupKFold(
                        n_splits=self.hparams.num_folds - 1,
                        shuffle=True,
                        random_state=self.hparams.random_state,
                    )
                    for j, (train_index, val_index) in enumerate(
                        kfold_inner.split(
                            df_outer, 
                            df_outer['mask_sum_qcut_code'], 
                            df_outer['set_id_spatial']
                        )
                    ):
                        if j == self.hparams.fold_index:
                            train_record_dirs = [train_val_record_dirs[i] for i in train_index]
                            val_record_dirs = [train_val_record_dirs[i] for i in val_index]
                            train_is_mask_empty = [train_val_is_mask_empty[i] for i in train_index]
                            val_is_mask_empty = [train_val_is_mask_empty[i] for i in val_index]
                            break
                    break
        
        # Drop empty masks from train
        if self.hparams.empty_mask_strategy == 'drop_only_train':
            train_record_dirs = [
                d for d, is_empty in zip(train_record_dirs, train_is_mask_empty) 
                if not is_empty
            ]
            train_is_mask_empty = [False] * len(train_record_dirs)

        return (
            train_record_dirs, 
            val_record_dirs, 
            test_record_dirs, 
            train_is_mask_empty, 
            val_is_mask_empty, 
            test_is_mask_empty,
        )

    def train_dataloader(self) -> DataLoader:
        sampler, shuffle, drop_last = None, True, True
        if self.hparams.to_predict == 'train':
            shuffle = False

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
            drop_last=drop_last,  # for compiling
        )

    def val_dataloader(self) -> DataLoader:
        batch_size = math.floor(self.hparams.batch_size // self.hparams.scale_factor ** 2)
        batch_size = max(batch_size, 1)
        val_dataloader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=batch_size, 
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
        batch_size = math.floor(self.hparams.batch_size // self.hparams.scale_factor ** 2)
        batch_size = max(batch_size, 1)
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=self.collate_fn,
            shuffle=False
        )

    def predict_dataloader(self) -> DataLoader:
        if self.hparams.to_predict == 'train':
            return self.train_dataloader()
        elif self.hparams.to_predict == 'val':
            return self.val_dataloader()
        elif self.hparams.to_predict == 'test':
            return self.test_dataloader()
