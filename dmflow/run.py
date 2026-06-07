"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import resource
import sys
from pathlib import Path
from typing import List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

import wandb
from diffcsp.common.utils import log_hyperparameters
from dmflow.model.eval_utils import register_omega_conf_resolvers
from dmflow.model.model_pl import MaterialsRFMLitModule
from dmflow.utils import load_state_dict_from_checkpoint

# https://github.com/Project-MONAI/MONAI/issues/701#issuecomment-767330310
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


WANDB_MODE = os.environ.get("WANDB_MODE", "")


register_omega_conf_resolvers()


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if (WANDB_MODE.lower() != "disabled") and ("lr_monitor" in cfg.logging):
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath="./",
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

    if "every_n_epochs_checkpoint" in cfg.train:
        hydra.utils.log.info(
            f"Adding callback <ModelCheckpoint> for every {cfg.train.every_n_epochs_checkpoint.every_n_epochs} epochs"
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath="every_n_epochs",
                every_n_epochs=cfg.train.every_n_epochs_checkpoint.every_n_epochs,
                save_top_k=cfg.train.every_n_epochs_checkpoint.save_top_k,
                verbose=cfg.train.every_n_epochs_checkpoint.verbose,
                save_last=cfg.train.every_n_epochs_checkpoint.save_last,
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Fast development run <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing single-process configuration!"
        )
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    # hydra_dir = Path(HydraConfig.get().run.dir)
    hydra_dir = Path.cwd()
    hydra.utils.log.info(f"Hydra Directory is {hydra_dir.resolve()}")

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    get_model = MaterialsRFMLitModule
    hydra.utils.log.info(f"Instantiating <{get_model}>")
    model = get_model(cfg)

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    do_wandb_log = (WANDB_MODE.lower() != "disabled") and ("wandb" in cfg.logging)
    if do_wandb_log:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            settings=wandb.Settings(start_method="fork"),
            tags=cfg.core.tags,
        )
        hydra.utils.log.info(f"W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    ckpt_path = cfg.get("ckpt_path", None)
    resume_from_checkpoint = cfg.get("resume_from_checkpoint", False)
    if resume_from_checkpoint and not ckpt_path:
        raise ValueError(
            "If `resume_from_checkpoint` is True, `ckpt_path` must be provided."
        )

    if not resume_from_checkpoint and ckpt_path:
        hydra.utils.log.info(f"Loading model state_dict from {ckpt_path}")
        state_dict = load_state_dict_from_checkpoint(ckpt_path)
        model.load_state_dict(state_dict=state_dict, strict=True)

    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        # default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        # progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        **cfg.train.pl_trainer,
    )

    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path if resume_from_checkpoint else None,
    )

    if do_wandb_log:
        hydra.utils.log.info(
            "W&B is no longer watching <{cfg.logging.wandb_watch.log}>!"
        )
        wandb_logger.experiment.unwatch(model)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(
    config_path="../conf",
    config_name="default",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
