from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from src.modules.metrics import load_metrics


class BaseModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Model loop (model_step)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        network: DictConfig,
        loss: DictConfig,
        metrics: DictConfig,
        output_activation: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        logging: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """LightningModule with standalone train, val and test dataloaders.

        Args:
            network (DictConfig): Network config.
            optimizer (DictConfig): Optimizer config.
            scheduler (DictConfig): Scheduler config.
            logging (DictConfig): Logging config.
            args (Any): Additional arguments for pytorch_lightning.LightningModule.
            kwargs (Any): Additional keyword arguments for pytorch_lightning.LightningModule.
        """

        super().__init__(*args, **kwargs)
        self.model = hydra.utils.instantiate(network.model)
        self.loss_cfg = loss
        self.metrics_cfg = metrics
        self.output_activation_cfg = output_activation
        self.opt_cfg = optimizer
        self.slr_cfg = scheduler
        self.logging_cfg = logging

        self.loss = hydra.utils.instantiate(self.loss_cfg)
        self.output_activation = hydra.utils.instantiate(self.output_activation_cfg, _partial_=True)

        main_metric, valid_metric_best, add_metrics = load_metrics(self.metrics_cfg)
        self.train_metric = main_metric.clone()
        self.train_add_metrics = add_metrics.clone(prefix="train/")
        self.valid_metric = main_metric.clone()
        self.valid_metric_best = valid_metric_best.clone()
        self.valid_add_metrics = add_metrics.clone(prefix="valid/")
        self.test_metric = main_metric.clone()
        self.test_add_metrics = add_metrics.clone(prefix="test/")

        self.save_hyperparameters(logger=False)

    def forward(self, x: Any, *args, **kwargs) -> Any:
        return self.model.forward(x)

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        logits = self.forward(batch["input"], *args, **kwargs)
        loss = self.loss(logits, batch["label"])
        preds = self.output_activation(logits)
        return loss, preds, batch["label"]

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before
        # training starts, so we need to make sure valid_metric_best doesn't store
        # accuracy from these checks
        self.valid_metric_best.reset()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f"train/{self.loss.__class__.__name__}",
            loss,
            **self.logging_cfg,
        )

        self.train_metric(preds, targets)
        self.log(
            f"train/{self.train_metric.__class__.__name__}",
            self.train_metric,
            **self.logging_cfg,
        )

        self.train_add_metrics(preds, targets)
        self.log_dict(self.train_add_metrics, **self.logging_cfg)

        # Lightning keeps track of `training_step` outputs and metrics on GPU for
        # optimization purposes. This works well for medium size datasets, but
        # becomes an issue with larger ones. It might show up as a CPU memory leak
        # during training step. Keep it in mind.
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f"valid/{self.loss.__class__.__name__}",
            loss,
            **self.logging_cfg,
        )

        self.valid_metric(preds, targets)
        self.log(
            f"valid/{self.valid_metric.__class__.__name__}",
            self.valid_metric,
            **self.logging_cfg,
        )

        self.valid_add_metrics(preds, targets)
        self.log_dict(self.valid_add_metrics, **self.logging_cfg)
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        valid_metric = self.valid_metric.compute()  # get current valid metric
        self.valid_metric_best(valid_metric)  # update best so far valid metric
        # log `valid_metric_best` as a value through `.compute()` method, instead
        # of as a metric object otherwise metric would be reset by lightning
        # after each epoch
        self.log(
            f"valid/{self.valid_metric.__class__.__name__}_best",
            self.valid_metric_best.compute(),
            **self.logging_cfg,
        )

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f"test/{self.loss.__class__.__name__}", loss, **self.logging_cfg
        )

        self.test_metric(preds, targets)
        self.log(
            f"test/{self.test_metric.__class__.__name__}",
            self.test_metric,
            **self.logging_cfg,
        )

        self.test_add_metrics(preds, targets)
        self.log_dict(self.test_add_metrics, **self.logging_cfg)
        return {"loss": loss}

    def on_test_epoch_end(self) -> None:
        pass

    def configure_optimizers(self) -> Any:
        optimizer: torch.optim = hydra.utils.instantiate(
            self.opt_cfg, params=self.parameters(), _convert_="partial"
        )
        if self.slr_cfg.get("scheduler"):
            scheduler: torch.optim.lr_scheduler = hydra.utils.instantiate(
                self.slr_cfg.scheduler,
                optimizer=optimizer,
                _convert_="partial",
            )
            lr_scheduler_dict = {"scheduler": scheduler}
            if self.slr_cfg.get("extras"):
                for key, value in self.slr_cfg.get("extras").items():
                    lr_scheduler_dict[key] = value
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        return {"optimizer": optimizer}

