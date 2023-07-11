import logging
import os
import torch
import torch._dynamo
from tqdm import tqdm


LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(
    level=LOGLEVEL,
    format = '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s'
)

if LOGLEVEL == 'DEBUG':
    torch._dynamo.config.verbose = True

from src.utils.utils import MyLightningCLI, TrainerWandb

# Usage example:
# python src/scripts/create_cache.py --config run/configs/common.yaml --model.init_args.backbone_name tu-tf_efficientnet_b5 --model.init_args.compile False --data.init_args.cache_dir another_cache

def cli_main():
    cli = MyLightningCLI(
        trainer_class=TrainerWandb, 
        save_config_kwargs={
            'config_filename': 'config_pl.yaml',
            'overwrite': True,
        },
        run=False,
    )

    assert not cli.datamodule.hparams.remove_pseudolabels_from_val_test, \
        "remove_pseudolabels_from_val_test should be False here to create full cache"

    cli.datamodule.setup()
    
    if cli.datamodule.val_dataset is not None:
        cli.datamodule.val_dataset.transform = None
        cli.datamodule.val_dataset.transform_cpp = None
        cli.datamodule.val_dataset.transform_mix = None
        for _ in tqdm(cli.datamodule.val_dataloader()):
            pass

    if cli.datamodule.test_dataset is not None:
        cli.datamodule.test_dataset.transform = None
        cli.datamodule.test_dataset.transform_cpp = None
        cli.datamodule.test_dataset.transform_mix = None
        for _ in tqdm(cli.datamodule.test_dataloader()):
            pass

    if cli.datamodule.train_dataset is not None:
        cli.datamodule.train_dataset.transform = None
        cli.datamodule.train_dataset.transform_cpp = None
        cli.datamodule.train_dataset.transform_mix = None
        # Due to random sampling it is ~ 1.75 epochs
        # to fill the cache
        for i in range(2):
            for _ in tqdm(cli.datamodule.train_dataloader()):
                pass
    

if __name__ == "__main__":
    cli_main()