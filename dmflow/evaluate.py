"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import ast
from copy import deepcopy
import json
from numbers import Number
from pathlib import Path
import sys
from typing import Any, Literal, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

import wandb
from dmflow.data.dataset import GenDataset
from dmflow.model.eval_utils import (
    CSPDataset,
    get_loaders,
    load_cfg,
    load_date_from_wandb,
    load_group_from_wandb,
    load_id_from_wandb,
    load_model,
    load_project_from_wandb,
    register_omega_conf_resolvers,
)
from dmflow.old_eval.generation_metrics import compute_generation_metrics
from dmflow.old_eval.lattice_metrics import compute_lattice_metrics
from dmflow.old_eval.reconstruction_metrics import compute_reconstruction_metrics

TASKS_TYPE = Literal[
    "reconstruct", "recon_trajectory", "generate", "gen_trajectory", "pred"
]
TASKS = deepcopy(TASKS_TYPE.__args__)
STAGE_TYPE = Literal["train", "val", "test"]
STAGES = deepcopy(STAGE_TYPE.__args__)
register_omega_conf_resolvers()


class TorchPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Path | str,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch",
    ):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Sequence[Any] | None,
    ) -> None:
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(
            predictions, self.output_dir / f"predictions_{trainer.global_rank:02d}.pt"
        )

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(
            batch_indices,
            self.output_dir / f"batch_indices_{trainer.global_rank:02d}.pt",
        )


@click.group()
def cli():
    pass


@cli.command()
@click.argument("checkpoint", type=Path)
@click.option(
    "--stage",
    type=click.Choice(STAGES, case_sensitive=False),
    default="val",
)
@click.option("--batch_size", type=int, default=16384)
@click.option("--num_evals", type=int, default=1)
@click.option("--limit_predict_batches", type=str, default="1.")
@click.option("--num_steps", type=int, default=None)
@click.option(
    "--div_mode",
    type=click.Choice(["exact", "rademacher"], case_sensitive=False),
    default=None,
)
@click.option(
    "--single_gpu/--multi_gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="use one gpu, not ddp",
)
def nll(
    checkpoint: Path,
    stage: STAGE_TYPE,
    batch_size: int,
    num_evals: int,
    limit_predict_batches: str,
    num_steps: int | None,
    div_mode: str | None,
    single_gpu: bool,
) -> None:
    raise NotImplementedError(
        "there are currently base distributions which make this unappealing."
    )


def get_target_dir(checkpoint: Path, subdir: str | None) -> Path:
    if subdir:
        target_dir = checkpoint.parent / subdir
    else:
        target_dir = checkpoint.parent
    return target_dir.resolve()


def _parse_limit_predict_batches(value: str) -> int | float:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (int, float)):
        raise ValueError("--limit_predict_batches must be a number.")
    return parsed


def _validate_num_evals(num_evals: int) -> None:
    if num_evals <= 0:
        raise ValueError("--num_evals must be positive.")


def _set_stage_batch_size(cfg, stage: STAGE_TYPE, batch_size: int | None) -> str:
    stage = stage.lower()
    if batch_size is None:
        batch_size = getattr(cfg.data.datamodule.batch_size, stage)
        click.echo(f"Using {batch_size=} from config.")
    else:
        setattr(cfg.data.datamodule.batch_size, stage, batch_size)
        click.echo(f"Using custom {batch_size=}.")
    return stage


def _apply_inference_options(
    cfg,
    num_steps: int | None,
    inference_anneal_slope: float | None,
    inference_anneal_offset: float | None,
    inference_anneal_types: bool,
    inference_anneal_coords: bool,
    inference_anneal_lattice: bool,
    compute_traj_velo_norms: bool | None = None,
    entire_traj: bool = False,
) -> None:
    if num_steps is not None:
        cfg.integrate.num_steps = num_steps
    if inference_anneal_slope is not None:
        cfg.integrate.inference_anneal_slope = inference_anneal_slope
    if inference_anneal_offset is not None:
        if not 0 <= inference_anneal_offset < 1:
            raise ValueError("--inference_anneal_offset must be in [0, 1).")
        cfg.integrate.inference_anneal_offset = inference_anneal_offset
    if compute_traj_velo_norms:
        cfg.integrate.compute_traj_velo_norms = compute_traj_velo_norms

    cfg.integrate.inference_anneal_types = inference_anneal_types
    cfg.integrate.inference_anneal_coords = inference_anneal_coords
    cfg.integrate.inference_anneal_lattice = inference_anneal_lattice
    if entire_traj:
        cfg.integrate.entire_traj = True


def _prediction_trainer(
    single_gpu: bool,
    pred_writer: TorchPredictionWriter,
    limit_predict_batches: str | None = None,
) -> pl.Trainer:
    trainer_kwargs = {
        "accelerator": "gpu",
        "callbacks": [pred_writer],
    }
    if limit_predict_batches is not None:
        trainer_kwargs["limit_predict_batches"] = _parse_limit_predict_batches(
            limit_predict_batches
        )

    if single_gpu:
        return pl.Trainer(devices=1, **trainer_kwargs)
    return pl.Trainer(strategy="ddp", devices="auto", **trainer_kwargs)


def _run_predictions(
    model: pl.LightningModule,
    loader,
    checkpoint: Path,
    target_dir: Path,
    directories: Sequence[str],
    num_steps: int,
    single_gpu: bool,
    limit_predict_batches: str | None = None,
) -> None:
    for directory in directories:
        output_dir = target_dir / directory
        pred_writer = TorchPredictionWriter(
            output_dir=output_dir, write_interval="epoch"
        )
        (output_dir / "num_steps.txt").write_text(str(num_steps))
        trainer = _prediction_trainer(
            single_gpu=single_gpu,
            pred_writer=pred_writer,
            limit_predict_batches=limit_predict_batches,
        )
        trainer.predict(
            model,
            dataloaders=loader,
            return_predictions=False,
            ckpt_path=checkpoint,
        )


@cli.command()
@click.argument("checkpoint", type=Path)
@click.option("--stage", type=click.Choice(STAGES, case_sensitive=False), default="val")
@click.option("--batch_size", type=int, default=16384)
@click.option("--num_evals", type=int, default=1)
@click.option("--limit_predict_batches", type=str, default="1.")
@click.option("--num_steps", type=int, default=None)
@click.option(
    "--single_gpu/--multi_gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="use one gpu, not ddp",
)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option(
    "--inference_anneal_slope",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_offset",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_types/--no-inference_anneal_types",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--inference_anneal_coords/--no-inference_anneal_coords",
    is_flag=True,
    show_default=True,
    default=True,
)
@click.option(
    "--inference_anneal_lattice/--no-inference_anneal_lattice",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--compute_traj_velo_norms",
    is_flag=True,
    show_default=True,
    default=False,
)
def reconstruct(
    checkpoint: Path,
    stage: STAGE_TYPE,
    batch_size: int | None,
    num_evals: int,
    limit_predict_batches: str,
    num_steps: int | None,
    single_gpu: bool,
    subdir: str,
    inference_anneal_slope: float | None,
    inference_anneal_offset: float | None,
    inference_anneal_types: bool,
    inference_anneal_coords: bool,
    inference_anneal_lattice: bool,
    compute_traj_velo_norms: bool | None,
) -> None:
    cfg, model = load_model(checkpoint)
    if "null" not in cfg.model.manifold_getter.atom_type_manifold:
        raise ValueError(
            f"you cannot do reconstruction with an unconditional atom_type_manifold {cfg.model.manifold_getter.atom_type_manifold=}"
        )

    stage = _set_stage_batch_size(cfg, stage, batch_size)
    _apply_inference_options(
        cfg,
        num_steps,
        inference_anneal_slope,
        inference_anneal_offset,
        inference_anneal_types,
        inference_anneal_coords,
        inference_anneal_lattice,
        compute_traj_velo_norms,
    )

    loaders = get_loaders(cfg)
    loader = loaders[STAGES.index(stage)]
    target_dir = get_target_dir(checkpoint, subdir)
    _validate_num_evals(num_evals)
    directories = [f"reconstruct_{i:02d}" for i in range(num_evals)]
    _run_predictions(
        model=model,
        loader=loader,
        checkpoint=checkpoint,
        target_dir=target_dir,
        directories=directories,
        num_steps=cfg.integrate.num_steps,
        single_gpu=single_gpu,
        limit_predict_batches=limit_predict_batches,
    )


@cli.command(name="recon_trajectory")
@click.argument("checkpoint", type=Path)
@click.option("--stage", type=click.Choice(STAGES, case_sensitive=False), default="val")
@click.option("--batch_size", type=int, default=16384)
@click.option("--num_evals", type=int, default=1)
@click.option("--limit_predict_batches", type=str, default="1.")
@click.option("--num_steps", type=int, default=None)
@click.option(
    "--single_gpu/--multi_gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="use one gpu, not ddp",
)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option(
    "--inference_anneal_slope",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_offset",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_types/--no-inference_anneal_types",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--inference_anneal_coords/--no-inference_anneal_coords",
    is_flag=True,
    show_default=True,
    default=True,
)
@click.option(
    "--inference_anneal_lattice/--no-inference_anneal_lattice",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--compute_traj_velo_norms",
    is_flag=True,
    show_default=True,
    default=False,
)
def recon_trajectory(
    checkpoint: Path,
    stage: STAGE_TYPE,
    batch_size: int | None,
    num_evals: int,
    limit_predict_batches: str,
    num_steps: int | None,
    single_gpu: bool,
    subdir: str,
    inference_anneal_slope: float | None,
    inference_anneal_offset: float | None,
    inference_anneal_types: bool,
    inference_anneal_coords: bool,
    inference_anneal_lattice: bool,
    compute_traj_velo_norms: bool | None,
) -> None:
    cfg, model = load_model(checkpoint)
    if "null" not in cfg.model.manifold_getter.atom_type_manifold:
        raise ValueError(
            f"you cannot do reconstruction with an unconditional atom_type_manifold {cfg.model.manifold_getter.atom_type_manifold=}"
        )

    stage = _set_stage_batch_size(cfg, stage, batch_size)
    _apply_inference_options(
        cfg,
        num_steps,
        inference_anneal_slope,
        inference_anneal_offset,
        inference_anneal_types,
        inference_anneal_coords,
        inference_anneal_lattice,
        compute_traj_velo_norms,
        entire_traj=True,
    )

    loaders = get_loaders(cfg)
    loader = loaders[STAGES.index(stage)]
    target_dir = get_target_dir(checkpoint, subdir)
    _validate_num_evals(num_evals)
    directories = [f"recon_trajectory_{i:02d}" for i in range(num_evals)]
    _run_predictions(
        model=model,
        loader=loader,
        checkpoint=checkpoint,
        target_dir=target_dir,
        directories=directories,
        num_steps=cfg.integrate.num_steps,
        single_gpu=single_gpu,
        limit_predict_batches=limit_predict_batches,
    )


@cli.command()
@click.argument("checkpoint", type=Path)
@click.option("--num_samples", type=int, default=10_000)
@click.option("--batch_size", type=int, default=16384)
@click.option("--num_steps", type=int, default=None)
@click.option(
    "--single_gpu/--multi_gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="use one gpu, not ddp",
)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option("--gen_id", type=int, default=0, help=r"folder name is generate_{gen_id}")
@click.option(
    "--inference_anneal_slope",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_offset",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_types/--no-inference_anneal_types",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--inference_anneal_coords/--no-inference_anneal_coords",
    is_flag=True,
    show_default=True,
    default=True,
)
@click.option(
    "--inference_anneal_lattice/--no-inference_anneal_lattice",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--compute_traj_velo_norms",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--set_norm",
    type=str,
    default="true",
    help="Deprecated; generation mask selection now runs in old_eval_metrics.",
)
def generate(
    checkpoint: Path,
    num_samples: int,
    batch_size: int | None,
    num_steps: int | None,
    single_gpu: bool,
    subdir: str,
    gen_id: int,
    inference_anneal_slope: float | None,
    inference_anneal_offset: float | None,
    inference_anneal_types: bool,
    inference_anneal_coords: bool,
    inference_anneal_lattice: bool,
    compute_traj_velo_norms: bool | None,
    set_norm: str = "true",
) -> None:
    cfg, model = load_model(checkpoint)
    if set_norm.lower() not in {"true", "false"}:
        raise ValueError("--set_norm must be 'true' or 'false'.")
    if "null" in cfg.model.manifold_getter.atom_type_manifold:
        raise ValueError(
            f"you cannot do generation with a conditional atom_type_manifold {cfg.model.manifold_getter.atom_type_manifold=}"
        )

    _apply_inference_options(
        cfg,
        num_steps,
        inference_anneal_slope,
        inference_anneal_offset,
        inference_anneal_types,
        inference_anneal_coords,
        inference_anneal_lattice,
        compute_traj_velo_norms,
    )

    sample_set = GenDataset(dataset=cfg.data.dataset_name, total_num=num_samples)
    loader = DataLoader(sample_set, batch_size=batch_size)
    target_dir = get_target_dir(checkpoint, subdir)
    _run_predictions(
        model=model,
        loader=loader,
        checkpoint=checkpoint,
        target_dir=target_dir,
        directories=[f"generate_{gen_id:02d}"],
        num_steps=cfg.integrate.num_steps,
        single_gpu=single_gpu,
    )


@cli.command(name="gen_trajectory")
@click.argument("checkpoint", type=Path)
@click.option("--num_samples", type=int, default=256)
@click.option("--batch_size", type=int, default=256)
@click.option("--num_steps", type=int, default=None)
@click.option(
    "--single_gpu/--multi_gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="use one gpu, not ddp",
)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option("--gen_id", type=int, default=0, help=r"folder name is generate_{gen_id}")
@click.option(
    "--inference_anneal_slope",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_offset",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_types/--no-inference_anneal_types",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--inference_anneal_coords/--no-inference_anneal_coords",
    is_flag=True,
    show_default=True,
    default=True,
)
@click.option(
    "--inference_anneal_lattice/--no-inference_anneal_lattice",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--compute_traj_velo_norms",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--set_norm",
    type=str,
    default="true",
    help="Deprecated; generation mask selection now runs in old_eval_metrics.",
)
def gen_trajectory(
    checkpoint: Path,
    num_samples: int,
    batch_size: int | None,
    num_steps: int | None,
    single_gpu: bool,
    subdir: str,
    gen_id: int,
    inference_anneal_slope: float | None,
    inference_anneal_offset: float | None,
    inference_anneal_types: bool,
    inference_anneal_coords: bool,
    inference_anneal_lattice: bool,
    compute_traj_velo_norms: bool | None,
    set_norm: str = "true",
) -> None:
    cfg, model = load_model(checkpoint)
    if set_norm.lower() not in {"true", "false"}:
        raise ValueError("--set_norm must be 'true' or 'false'.")
    if "null" in cfg.model.manifold_getter.atom_type_manifold:
        raise ValueError(
            f"you cannot do generation with a conditional atom_type_manifold {cfg.model.manifold_getter.atom_type_manifold=}"
        )

    _apply_inference_options(
        cfg,
        num_steps,
        inference_anneal_slope,
        inference_anneal_offset,
        inference_anneal_types,
        inference_anneal_coords,
        inference_anneal_lattice,
        compute_traj_velo_norms,
        entire_traj=True,
    )

    sample_set = GenDataset(dataset=cfg.data.dataset_name, total_num=num_samples)
    loader = DataLoader(sample_set, batch_size=batch_size)
    target_dir = get_target_dir(checkpoint, subdir)
    _run_predictions(
        model=model,
        loader=loader,
        checkpoint=checkpoint,
        target_dir=target_dir,
        directories=[f"gen_trajectory_{gen_id:02d}"],
        num_steps=cfg.integrate.num_steps,
        single_gpu=single_gpu,
    )


@cli.command()
@click.argument("checkpoint", type=Path)
@click.argument("atom_types_path", type=Path)
@click.option("--batch_size", type=int, default=16384)
@click.option("--num_steps", type=int, default=None)
@click.option(
    "--single_gpu/--multi_gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="use one gpu, not ddp",
)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option("--pred_id", type=int, default=0, help=r"folder name is pred_{pred_id}")
@click.option(
    "--inference_anneal_slope",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_offset",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_types/--no-inference_anneal_types",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--inference_anneal_coords/--no-inference_anneal_coords",
    is_flag=True,
    show_default=True,
    default=True,
)
@click.option(
    "--inference_anneal_lattice/--no-inference_anneal_lattice",
    is_flag=True,
    show_default=True,
    default=False,
)
def predict(
    checkpoint: Path,
    atom_types_path: Path,
    batch_size: int | None,
    num_steps: int | None,
    single_gpu: bool,
    subdir: str,
    pred_id: int,
    inference_anneal_slope: float | None,
    inference_anneal_offset: float | None,
    inference_anneal_types: bool,
    inference_anneal_coords: bool,
    inference_anneal_lattice: bool,
) -> None:
    cfg, model = load_model(checkpoint)
    _apply_inference_options(
        cfg,
        num_steps,
        inference_anneal_slope,
        inference_anneal_offset,
        inference_anneal_types,
        inference_anneal_coords,
        inference_anneal_lattice,
    )

    atom_types = ast.literal_eval(atom_types_path.read_text())
    dataset = CSPDataset(atom_types)
    loader = DataLoader(dataset, batch_size=batch_size)
    target_dir = get_target_dir(checkpoint, subdir)
    _run_predictions(
        model=model,
        loader=loader,
        checkpoint=checkpoint,
        target_dir=target_dir,
        directories=[f"pred_{pred_id:02d}"],
        num_steps=cfg.integrate.num_steps,
        single_gpu=single_gpu,
    )


CONSOLIDATED_KEYS = {
    "atom_types",
    "frac_coords",
    "lattices",
    "lengths",
    "angles",
    "pd_weights",
    "pd_frac_coords",
    "sd_weights",
    "sd_onehot",
    "pd_onehot",
}


def _get_consolidated_path(directory: Path, task: str) -> Path:
    return directory / f"consolidated_{task}.pt"


def _list_of_dicts_to_dict_of_lists(
    lod: list[dict[str, torch.Tensor | Batch]],
    keys_to_ignore: tuple[str] = (),
) -> dict[str, list[torch.Tensor] | list[Data]]:
    out = {k: [] for k in lod[0].keys() if k not in keys_to_ignore}
    for key, val in out.items():
        for d in lod:
            if isinstance(d[key], Batch):
                val.append(d[key].to_data_list())
            else:
                val.append(d[key])
    return out


def _consolidate(
    target_dir: Path, task: TASKS_TYPE
) -> dict[str, list[dict[str, torch.Tensor | list[Data] | list[int]]]] | None:
    pattern = f"{task}_??"
    directories = sorted(list(target_dir.glob(pattern)))

    if not directories:
        click.echo(f"No directories found for pattern {pattern}.")
        return None
    click.echo(f"Consolidating {len(directories)} directories for pattern {pattern}.")

    num_stepss = []
    out_by_eval = []
    for directory in directories:
        preds = sorted(list(directory.glob("predictions_??.pt")))
        batches = sorted(list(directory.glob("batch_indices_??.pt")))
        if len(preds) != len(batches):
            raise ValueError(
                f"Found {len(preds)} prediction files but {len(batches)} batch-index "
                f"files in {directory}."
            )
        if not preds:
            raise FileNotFoundError(f"No prediction files found in {directory}.")

        keys = torch.load(preds[0], map_location="cpu")[0][0].keys()
        out = {k: [] for k in keys}
        order = []

        for pred, batch in zip(preds, batches):
            pred = torch.load(pred, map_location="cpu")
            batch = torch.load(batch, map_location="cpu")

            for pp, bb in zip(pred, batch):
                for p, b in zip(pp, bb):
                    for k, v in p.items():
                        if isinstance(v, torch.Tensor):
                            out[k].append(v)
                        elif isinstance(v, Data):
                            out[k].extend(v.to_data_list())
                        else:
                            raise TypeError(
                                f"Cannot consolidate value with {type(v)=}."
                            )
                    order.extend(b)

        for k, v in out.items():
            if k == "input_data_batch":
                out[k] = Batch.from_data_list(v)
            elif k in CONSOLIDATED_KEYS:
                if "trajectory" in task:
                    out[k] = torch.concat(v, dim=1)
                else:
                    out[k] = torch.concat(v, dim=0)
            elif k == "num_atoms":
                out[k] = torch.concat(v, dim=0)
            else:
                raise ValueError(f"Cannot consolidate unknown key {k!r}.")
        num_stepss.append(int((directory / "num_steps.txt").read_text()))
        out["batch_indices"] = torch.tensor(order)
        out_by_eval.append(out)

    if any(num_stepss[0] != value for value in num_stepss):
        raise ValueError(f"Not all num_steps values match: {num_stepss=}.")
    num_steps = num_stepss[0]

    (_get_consolidated_path(target_dir, task).parent / "num_steps.txt").write_text(
        str(num_steps)
    )
    out_by_eval = _list_of_dicts_to_dict_of_lists(out_by_eval)
    torch.save(out_by_eval, _get_consolidated_path(target_dir, task))
    return out_by_eval


def _create_eval_pt(
    consolidated: dict[str, torch.Tensor] | None,
    target_dir: Path,
    filename: str,
) -> Path:
    # consolidated = {k: v.reshape(-1, v.shape[-1]) if k != "lattices" else v.reshape(-1, *v.shape[-2:]) for k, v in r.items()}
    if consolidated is None:
        raise ValueError(
            f"you cannot try to save an eval_pt with no data, {consolidated=}"
        )
    consolidated = {k: v[0] for k, v in consolidated.items()}
    consolidated["eval_setting"] = None
    path = target_dir / filename
    torch.save(consolidated, path)
    return path


@cli.command()
@click.argument("checkpoint", type=Path)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option(
    "--path_eval_pt",
    show_default=True,
    default=None,
    help="select a path to save the eval_pt, otherwise it is not saved",
)
@click.option(
    "--task_to_save",
    default=None,
    help="if it is ambiguous, you can select the task for path_eval_pt",
)
def consolidate(
    checkpoint: Path,
    subdir: str,
    path_eval_pt: str | None,
    task_to_save: TASKS_TYPE | None,
) -> None:
    target_dir = get_target_dir(checkpoint, subdir)
    r = _consolidate(target_dir, "reconstruct")
    rt = _consolidate(target_dir, "recon_trajectory")
    g = _consolidate(target_dir, "generate")
    gt = _consolidate(target_dir, "gen_trajectory")
    p = _consolidate(target_dir, "pred")

    consolidations = {k: v for k, v in zip(TASKS, [r, rt, g, gt, p])}
    did_consolidate = {k: v is not None for k, v in consolidations.items()}

    if any(did_consolidate.values()):
        if task_to_save is not None:
            click.echo(f"Saving {task_to_save}.")
            consolidated = consolidations[task_to_save]
        elif sum(did_consolidate.values()) == 1:
            consolidated_task = list(consolidations.keys())[
                list((did_consolidate.values())).index(True)
            ]
            click.echo(f"Only {consolidated_task} was consolidated.")
            consolidated = consolidations[consolidated_task]
        else:
            raise ValueError(
                "More than one task was consolidated. Pass --task_to_save to select "
                f"the eval_pt source. {did_consolidate=}"
            )
    else:
        raise ValueError(
            "Nothing was consolidated, so the program cannot save an eval_pt."
        )

    if sum(did_consolidate.values()) > 0 and path_eval_pt is not None:
        path = _create_eval_pt(consolidated, target_dir, path_eval_pt)
        click.echo("eval_pt:")
        click.echo(path)


def _reconstruction_metrics_wandb(
    target_dir: Path,
    consolidated_reconstruction_path: Path,
    global_step: int,
    stage: STAGE_TYPE,
) -> dict[str, float]:
    recon_metrics = {}
    if consolidated_reconstruction_path.exists():
        tmp, num_evals = compute_reconstruction_metrics(
            consolidated_reconstruction_path,
            multi_eval=False,
            metrics_path=target_dir / f"old_eval_metrics_reconstruct_single.json",
            ground_truth_path=None,
        )
        recon_metrics.update(
            {f"{stage}/" + k + f"_{num_evals:02d}": v for k, v in tmp.items()}
        )

        recon_num_steps = int(
            (consolidated_reconstruction_path.parent / "num_steps.txt").read_text()
        )
        recon_metrics.update({f"{stage}/recon_num_steps": recon_num_steps})
        recon_metrics.update({"trainer/global_step": global_step})
    return recon_metrics


def _generation_metrics_wandb(
    target_dir: Path,
    consolidated_generation_path: Path,
    gt_dataset_path: Path,
    global_step: int,
    eval_model_name: Literal["cod_sd", "cod_pd", "cod_spd"],
    n_subsamples: int,
    stage: STAGE_TYPE,
    set_norm: bool,
) -> dict[str, float]:
    gen_metrics = {}
    if consolidated_generation_path.exists():
        tmp = compute_generation_metrics(
            path=consolidated_generation_path,
            metrics_path=target_dir / f"old_eval_metrics_generate.json",
            ground_truth_path=gt_dataset_path,
            eval_model_name=eval_model_name,
            n_subsamples=n_subsamples,
            set_norm=set_norm,
        )
        gen_metrics.update({f"{stage}/" + k: v for k, v in tmp.items()})
        gen_num_steps = int(
            (consolidated_generation_path.parent / "num_steps.txt").read_text()
        )
        gen_metrics.update({f"{stage}/gen_n_subsamples": n_subsamples})
        gen_metrics.update({f"{stage}/gen_num_steps": gen_num_steps})
        gen_metrics.update({"trainer/global_step": global_step})
    return gen_metrics


def _jsonable_metric_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _jsonable_metric_value(value.detach().cpu().item())
        return value.detach().cpu().tolist()

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable_metric_value(item())
        except (TypeError, ValueError, RuntimeError):
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except (TypeError, ValueError, RuntimeError):
            pass

    return value


def _echo_metrics(title: str, metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    click.echo("")
    click.echo(f"======= {title} metrics =======")
    click.echo(
        json.dumps(
            {k: _jsonable_metric_value(v) for k, v in sorted(metrics.items())},
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


@cli.command(name="old_eval_metrics")
@click.argument("checkpoint", type=Path)
@click.option(
    "--do_not_log_wandb",
    is_flag=True,
    show_default=True,
    default=False,
    help="do not log results in the wandb training run",
)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option(
    "--gen_subsamples",
    type=int,
    default=1_000,
    help="gen metrics are on this many subsamples",
)
@click.option("--stage", type=click.Choice(STAGES, case_sensitive=False), default="val")
@click.option(
    "--set_norm/--no-set_norm",
    is_flag=True,
    show_default=True,
    default=True,
    help="normalize generated raw disorder weights before selecting masks",
)
def old_eval_metrics(
    checkpoint: Path,
    do_not_log_wandb: bool,
    subdir: str,
    gen_subsamples: int,
    stage: STAGE_TYPE,
    set_norm: bool,
) -> None:
    log_wandb = not do_not_log_wandb
    target_dir = get_target_dir(checkpoint, subdir)

    chkp = torch.load(checkpoint, map_location="cpu")
    global_step = chkp["global_step"]

    click.echo("")
    click.echo("======= reconstruction =======")
    click.echo("")
    consolidated_reconstruction_path = target_dir / _get_consolidated_path(
        target_dir, "reconstruct"
    )
    if consolidated_reconstruction_path.exists():
        recon_metrics = _reconstruction_metrics_wandb(
            target_dir, consolidated_reconstruction_path, global_step, stage
        )
    else:
        recon_metrics = {}
        click.echo(f"{consolidated_reconstruction_path=} not found")
        click.echo("Not performing reconstruction metrics.")

    click.echo("")
    click.echo("======= generation =======")
    click.echo("")

    consolidated_generation_path = target_dir / _get_consolidated_path(
        target_dir, "generate"
    )
    if consolidated_generation_path.exists():
        cfg = load_cfg(checkpoint)
        gen_metrics = _generation_metrics_wandb(
            target_dir=target_dir,
            consolidated_generation_path=consolidated_generation_path,
            gt_dataset_path=cfg.data.datamodule.datasets[stage][0].save_path,
            global_step=global_step,
            eval_model_name=cfg.data.eval_model_name,
            n_subsamples=gen_subsamples,
            stage=stage,
            set_norm=set_norm,
        )
    else:
        gen_metrics = {}
        click.echo(f"{consolidated_generation_path=} not found")
        click.echo("Not performing generation metrics.")

        click.echo("")

    if (
        not consolidated_reconstruction_path.exists()
        and not consolidated_generation_path.exists()
    ):
        raise FileNotFoundError(
            f"Neither {consolidated_reconstruction_path=} nor {consolidated_generation_path=} exists."
        )

    _echo_metrics("reconstruction", recon_metrics)
    _echo_metrics("generation", gen_metrics)

    if log_wandb:
        cfg = load_cfg(checkpoint)
        wandb_config = cfg.logging.wandb
        wandb_config.project = load_project_from_wandb(checkpoint)
        wandb_config.group = load_group_from_wandb(checkpoint)
        wandb_config.job_type = "cdvae_metrics"
        wandb_config.tags = [
            load_date_from_wandb(checkpoint),
            load_id_from_wandb(checkpoint),
        ]
        wandb_config = dict(wandb_config)
        del wandb_config["log_model"]
        wandb.init(**wandb_config)
        wandb.log(recon_metrics, global_step)
        wandb.log(gen_metrics, global_step)
        wandb.finish()


@cli.command(name="lattice_metrics")
@click.argument("checkpoint", type=Path)
# @click.option(
#     "--do_not_log_wandb",
#     is_flag=True,
#     show_default=True,
#     default=False,
#     help="do not log results in the wandb training run",
# )
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option("--stage", type=click.Choice(STAGES, case_sensitive=False), default="val")
def lattice_metrics(
    checkpoint: Path,
    # do_not_log_wandb: bool,
    subdir: str,
    stage: STAGE_TYPE,
) -> None:
    # log_wandb = not do_not_log_wandb
    target_dir = get_target_dir(checkpoint, subdir)

    consolidated_reconstruction_path = target_dir / _get_consolidated_path(
        target_dir, "reconstruct"
    )
    if consolidated_reconstruction_path.exists():
        compute_lattice_metrics(
            consolidated_reconstruction_path,
            metrics_path=target_dir / f"lattice_metrics_reconstruct_single.json",
        )  # right now this returns nothing since it just plots the distribution

    consolidated_generation_path = target_dir / _get_consolidated_path(
        target_dir, "generate"
    )
    if consolidated_generation_path.exists():
        cfg = load_cfg(checkpoint)
        compute_lattice_metrics(
            consolidated_generation_path,
            metrics_path=target_dir / f"lattice_metrics_generate.json",
            ground_truth_path=cfg.data.datamodule.datasets[stage][0].save_path,
        )  # right now this returns nothing since it just plots the distribution

    if (
        not consolidated_reconstruction_path.exists()
        and not consolidated_generation_path.exists()
    ):
        raise FileNotFoundError(
            f"Neither {consolidated_reconstruction_path=} nor {consolidated_generation_path=} exist."
        )


if __name__ == "__main__":
    cli()
