from typing import List, Tuple

import hydra
import pyrootutils
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger

from src import utils


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(root / "configs"),
    "config_name": "eval.yaml",
}
log = utils.get_pylogger(__name__)


@utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies
    extra utilities before and after the call.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated
        objects.
    """

    assert cfg.ckpt_path

    log.info('Using config: \n%s', OmegaConf.to_yaml(cfg))

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, _recursive_=False
    )

    # Init lightning model
    log.info(f"Instantiating lightning model <{cfg.module._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.module, _recursive_=False
    )

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(
        cfg.get("logger")
    )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Log metadata
    log.info("Logging metadata!")
    utils.log_metadata(cfg)

    if cfg.get("predict"):
        log.info("Starting predicting!")
        predictions = trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
        )
        utils.save_predictions(
            predictions=predictions,
            dirname=cfg.paths.output_dir,
            **cfg.extras.predictions_saving_params,
        )

    else:
        log.info("Starting testing!")
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
        )

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


if __name__ == "__main__":
    main()
