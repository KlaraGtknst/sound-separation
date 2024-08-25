from typing import Any, List, Optional, Tuple

import hydra
import pyrootutils
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import LearningRateMonitor

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
    "config_name": "train.yaml",
}
log = utils.get_pylogger(__name__)


@utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best
    weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator which applies
    extra utilities before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated
        objects.
    """

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

    datamodule.prepare_data()
    datamodule.setup("fit")

    # Init lightning model
    if cfg.module.scheduler.extras.interval == "epoch":
        cfg.module.scheduler.scheduler["total_steps"] = (len(datamodule.train_dataloader())  * cfg.trainer.max_epochs) // cfg.datamodule.loaders.train.batch_size
    else:
        cfg.module.scheduler.scheduler["total_steps"] = len(datamodule.train_dataloader())  * cfg.trainer.max_epochs
    log.info(f"Instantiating lightning model <{cfg.module._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.module, _recursive_=False
    )

    # Init callbacks
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(
        cfg.get("callbacks")
    )
    callbacks.append(LearningRateMonitor("step"))

    # Init loggers
    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(
        cfg.get("logger")
    )

    # Init lightning ddp plugins
    log.info("Instantiating plugins...")
    plugins: Optional[List[Any]] = utils.instantiate_plugins(cfg)

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, plugins=plugins
    )

    # Send parameters from cfg to all lightning loggers
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Train the model
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    # Test the model
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning(
                "Best ckpt not found! Using current weights for testing..."
            )
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # Save state dicts for best and last checkpoints
    if cfg.get("save_state_dict"):
        log.info("Starting saving state dicts!")
        utils.save_state_dicts(
            trainer=trainer,
            model=model,
            dirname=cfg.paths.output_dir,
            **cfg.extras.state_dict_saving_params,
        )

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_dict, object_dict





if __name__ == "__main__":
    main()
