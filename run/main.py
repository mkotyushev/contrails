import logging
import os

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(
    level=LOGLEVEL,
    format = '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s'
)

from src.utils.utils import MyLightningCLI, TrainerWandb


def cli_main():
    cli = MyLightningCLI(
        trainer_class=TrainerWandb, 
        save_config_kwargs={
            'config_filename': 'config_pl.yaml',
            'overwrite': True,
        }
    )

    # Best scores
    scores = {}
    for cb in cli.trainer.checkpoint_callbacks:
        if cb.best_model_score is not None:
            scores[f'best_{cb.monitor}'] = cb.best_model_score.item()
    cli.model.log_dict(scores)


if __name__ == "__main__":
    cli_main()