import logging
import os
import torch
import torch._dynamo


LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(
    level=LOGLEVEL,
    format = '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s'
)

if LOGLEVEL == 'DEBUG':
    torch._dynamo.config.verbose = True

from src.utils.utils import MyLightningCLISweep, TrainerWandb


def cli_main():
    cli = MyLightningCLISweep(
        trainer_class=TrainerWandb, 
        save_config_kwargs={
            'config_filename': 'config_pl.yaml',
            'overwrite': True,
        }
    )


if __name__ == "__main__":
    cli_main()