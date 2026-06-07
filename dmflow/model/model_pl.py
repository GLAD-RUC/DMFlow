"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations
from copy import deepcopy

import warnings
from functools import partial
from typing import Any, Literal

import hydra
import pytorch_lightning as pl
import torch
from geoopt.manifolds.euclidean import Euclidean
from geoopt.manifolds.product import ProductManifold
from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.func import jvp, vjp
from torch_geometric.data import Data
from torchmetrics import MeanMetric, MinMetric
from torch_geometric.utils import to_dense_batch


from diffcsp.common.data_utils import lattices_to_params_shape

from manifm.ema import EMA
from manifm.model_pl import ManifoldFMLitModule, div_fn, output_and_div
from manifm.solvers import projx_integrator, projx_integrator_return_last
from dmflow.data import NUM_ATOMIC_TYPES
from dmflow.model.arch import CSPNet, ProjectedConjugatedCSPNet
from dmflow.model.solvers import (
    projx_cond_integrator_return_last,
    projx_integrate_xt_to_x1,
)
from dmflow.model.standardize import get_affine_stats
from dmflow.rfm.manifold_getter import Dims, ManifoldGetter
from dmflow.rfm.manifolds.spd import SPDGivenN, SPDNonIsotropicRandom
from dmflow.rfm.vmap import VMapManifolds
from dmflow.utils import masked_mean


def output_and_div(
    vecfield: callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    v: torch.Tensor | None = None,
    div_mode: Literal["exact", "rademacher"] = "exact",
) -> tuple[torch.Tensor, torch.Tensor]:
    if div_mode == "exact":
        dx = vecfield(x)
        div = div_fn(vecfield)(x)
    else:
        dx, vjpfunc = vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div


class MaterialsRFMLitModule(ManifoldFMLitModule):
    def __init__(self, cfg: DictConfig):
        pl.LightningModule.__init__(self)
        self.cfg = cfg
        self.save_hyperparameters()

        self.manifold_getter = ManifoldGetter(
            atom_type_manifold=cfg.model.manifold_getter.atom_type_manifold,
            coord_manifold=cfg.model.manifold_getter.coord_manifold,
            lattice_manifold=cfg.model.manifold_getter.lattice_manifold,
            weight_manifold=cfg.model.manifold_getter.weight_manifold,
            pd_frac_coords_manifold=cfg.model.manifold_getter.pd_frac_coords_manifold,
            dataset=cfg.data.dataset_name,
            analog_bits_scale=cfg.model.manifold_getter.get("analog_bits_scale", None),
            length_inner_coef=cfg.model.manifold_getter.get("length_inner_coef", None),
            emb_sd_info=cfg.model.get("emb_sd_info", False),
        )

        if "null" in cfg.model.manifold_getter.atom_type_manifold:
            cost_type = 0.0
            cost_cross_ent = 0.0
            warnings.warn(
                f"since {cfg.model.manifold_getter.atom_type_manifold=}, set {cost_type=} & {cost_cross_ent=}",
                PossibleUserWarning,
            )
        else:
            cost_type = cfg.model.cost_type
            cost_cross_ent = cfg.model.cost_cross_ent

        self.costs = {
            "loss_a": cost_type,
            "loss_f": cfg.model.cost_coord,
            "loss_l": cfg.model.cost_lattice,
            "loss_ce": cost_cross_ent,
            "loss_pd_w": (
                cfg.model.cost_pd_weight
                if "null" not in cfg.model.manifold_getter.weight_manifold
                else 0.0
            ),
            "loss_pd_f": (
                cfg.model.cost_pd_frac_coords
                if "null" not in cfg.model.manifold_getter.pd_frac_coords_manifold
                else 0.0
            ),
        }
        if cfg.model.affine_combine_costs:
            total_cost = sum([v for v in self.costs.values()])
            self.costs = {k: v / total_cost for k, v in self.costs.items()}

        model: CSPNet = hydra.utils.instantiate(
            self.cfg.vectorfield,
            emb_pd_info=self.cfg.model.emb_pd_info,
            _convert_="partial",
        )
        # Model of the vector field.
        cspnet = ProjectedConjugatedCSPNet(
            cspnet=model,
            manifold_getter=self.manifold_getter,
            lattice_affine_stats=get_affine_stats(
                cfg.data.dataset_name, self.manifold_getter.lattice_manifold
            ),
            coord_affine_stats=get_affine_stats(
                cfg.data.dataset_name, self.manifold_getter.coord_manifold
            ),
        )
        if cfg.optim.get("ema_decay", None) is None:
            self.model = cspnet
        else:
            self.model = EMA(
                cspnet,
                cfg.optim.ema_decay,
            )

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        base_metrics = [
            "loss",
            "loss_a",
            "loss_f",
            "loss_l",
            "loss_ce",
            "loss_pd_w",
            "loss_pd_f",
            "unscaled/loss_a",
            "unscaled/loss_f",
            "unscaled/loss_l",
            "unscaled/loss_ce",
            "unscaled/loss_pd_w",
            "unscaled/loss_pd_f",
        ]

        self.train_metrics = {k: MeanMetric() for k in base_metrics}
        self.val_metrics = {k: MeanMetric() for k in base_metrics}
        self.test_metrics = {k: MeanMetric() for k in base_metrics + ["nll"]}
        # for logging best so far validation accuracy
        self.val_metrics_best = {k: MinMetric() for k in base_metrics}

        if self.cfg.val.compute_nll:
            self.val_metrics["nll"] = MeanMetric()
            self.val_metrics_best["nll"] = MinMetric()

    @staticmethod
    def _annealing_schedule(
        t: torch.Tensor, slope: float, intercept: float
    ) -> torch.Tensor:
        return slope * torch.nn.functional.relu(t - intercept) + 1.0

    @torch.no_grad()
    def sample(
        self,
        batch: Data,
        x0: torch.Tensor = None,
        num_steps: int = 1_000,
        entire_traj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            x1,  ## [B, max_atom_num * (\sum d_i)]
            manifold,
            a_manifold,
            f_manifold,
            l_manifold,
            pd_w_manifold,
            pd_f_manifold,
            dims,
            mask_a_or_f,
        ) = self.manifold_getter(
            batch.batch,
            batch.atom_types,
            batch.frac_coords,
            batch.lengths,
            batch.angles,
            batch.sd_weights,
            batch.pd_weights,
            batch.pd_frac_coords,
            split_manifold=True,
        )
        if x0 is None:
            if self.cfg.base_distribution_from_data:
                x0 = self.manifold_getter(
                    batch.batch,
                    batch.atom_types,
                    batch.frac_coords,
                    batch.lengths,
                    batch.angles,
                    batch.sd_weights,
                    batch.pd_weights,
                    batch.pd_frac_coords,
                    split_manifold=True,
                )[0]
            else:
                x0 = manifold.random(*x1.shape, dtype=x1.dtype, device=x1.device)
        else:
            x0 = x0.to(x1)

        return self.finish_sampling(
            x0=x0,
            manifold=manifold,
            a_manifold=a_manifold,
            f_manifold=f_manifold,
            l_manifold=l_manifold,
            pd_w_manifold=pd_w_manifold,
            pd_f_manifold=pd_f_manifold,
            dims=dims,
            num_atoms=batch.num_atoms,
            node2graph=batch.batch,
            mask_a_or_f=mask_a_or_f,
            num_steps=num_steps,
            entire_traj=entire_traj,
        )

    @torch.no_grad()
    def gen_sample(
        self,
        node2graph: torch.LongTensor,
        dim_coords: int,
        dim_pd_weight: int,
        x0: torch.Tensor = None,
        num_steps: int = 1_000,
        entire_traj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            shape,
            manifold,
            a_manifold,
            f_manifold,
            l_manifold,
            pd_w_manifold,
            pd_f_manifold,
            dims,
            mask_a_or_f,
        ) = self.manifold_getter.from_empty_batch(
            node2graph, dim_coords, dim_pd_weight, split_manifold=True
        )
        num_atoms = self.manifold_getter._get_num_atoms(mask_a_or_f)

        if x0 is None:
            assert (
                not self.cfg.base_distribution_from_data
            ), "Need to sample from the base distribution"
            x0 = manifold.random(*shape, device=node2graph.device)
        else:
            x0 = x0.to(device=node2graph.device)

        return self.finish_sampling(
            x0=x0,
            manifold=manifold,
            a_manifold=a_manifold,
            f_manifold=f_manifold,
            l_manifold=l_manifold,
            pd_w_manifold=pd_w_manifold,
            pd_f_manifold=pd_f_manifold,
            dims=dims,
            num_atoms=num_atoms,
            node2graph=node2graph,
            mask_a_or_f=mask_a_or_f,
            num_steps=num_steps,
            entire_traj=entire_traj,
        )

    @torch.no_grad()
    def pred_sample(
        self,
        node2graph: torch.LongTensor,
        atom_types: torch.LongTensor,
        dim_coords: int,
        x0: torch.Tensor = None,
        num_steps: int = 1_000,
        entire_traj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            shape,
            manifold,
            a_manifold,
            f_manifold,
            l_manifold,
            dims,
            mask_a_or_f,
        ) = self.manifold_getter.from_only_atom_types(
            node2graph, atom_types, dim_coords, split_manifold=True
        )
        num_atoms = self.manifold_getter._get_num_atoms(mask_a_or_f)

        if x0 is None:
            assert (
                not self.cfg.base_distribution_from_data
            ), "Need to sample from the base distribution"
            x0 = manifold.random(*shape, device=node2graph.device)
        else:
            x0 = x0.to(device=node2graph.device)

        return self.finish_sampling(
            x0=x0,
            manifold=manifold,
            a_manifold=a_manifold,
            f_manifold=f_manifold,
            l_manifold=l_manifold,
            dims=dims,
            num_atoms=num_atoms,
            node2graph=node2graph,
            mask_a_or_f=mask_a_or_f,
            num_steps=num_steps,
            entire_traj=entire_traj,
        )

    @torch.no_grad()
    def finish_sampling(
        self,
        x0: torch.Tensor,
        manifold: VMapManifolds,
        a_manifold: VMapManifolds,
        f_manifold: VMapManifolds,
        l_manifold: VMapManifolds,
        pd_w_manifold: VMapManifolds,
        pd_f_manifold: VMapManifolds,
        dims: Dims,
        num_atoms: torch.LongTensor,
        node2graph: torch.LongTensor,  # aka batch.batch
        mask_a_or_f: torch.BoolTensor,
        num_steps: int,
        entire_traj: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vecfield = partial(
            self.vecfield,
            num_atoms=num_atoms,
            node2graph=node2graph,
            dims=dims,
            mask_a_or_f=mask_a_or_f,
        )

        ## default to False, so that we do not compute the entire trajectory
        compute_traj_velo_norms = self.cfg.integrate.get(
            "compute_traj_velo_norms", False
        )

        c = self.cfg.integrate.get("inference_anneal_slope", 0.0)
        b = self.cfg.integrate.get("inference_anneal_offset", 0.0)

        anneal_types = self.cfg.integrate.get("inference_anneal_types", False)
        anneal_coords = self.cfg.integrate.get("inference_anneal_coords", True)
        anneal_lattice = self.cfg.integrate.get("inference_anneal_lattice", False)

        def scheduled_fn_to_integrate(
            t: torch.Tensor, x: torch.Tensor, cond: torch.Tensor | None = None
        ) -> torch.Tensor:

            ## annealing_factor = c * ReLU(t - b) + 1.0
            anneal_factor = self._annealing_schedule(t, c, b)
            out = vecfield(
                t=torch.atleast_2d(t),
                x=torch.atleast_2d(x),
                manifold=manifold,
                cond=torch.atleast_2d(cond) if isinstance(cond, torch.Tensor) else cond,
            )

            # dims order: dims.a, dims.f, dims.l,  dims.pd_w, dims.pd_f
            if anneal_types:
                out[:, : dims.a].mul_(anneal_factor)
            if anneal_coords:
                # out[:, dims.a : -dims.l].mul_(anneal_factor)
                out[:, dims.a : dims.a + dims.f].mul_(anneal_factor)
            if anneal_lattice:
                # out[:, -dims.l :].mul_(anneal_factor)
                out[:, dims.a + dims.f : dims.a + dims.f + dims.l].mul_(anneal_factor)
            return out

        if self.cfg.model.get("self_cond", False):
            x1 = projx_cond_integrator_return_last(
                manifold,
                scheduled_fn_to_integrate,
                x0,
                t=torch.linspace(0, 1, num_steps + 1).to(x0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                local_coords=False,
                pbar=True,
            )
            return x1

        elif entire_traj or compute_traj_velo_norms:
            xs, vs = projx_integrator(
                manifold,
                scheduled_fn_to_integrate,
                x0,
                t=torch.linspace(0, 1, num_steps + 1).to(x0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                pbar=True,
            )
        else:
            x1 = projx_integrator_return_last(
                manifold,
                scheduled_fn_to_integrate,
                x0,
                t=torch.linspace(0, 1, num_steps + 1).to(x0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                local_coords=False,
                pbar=True,
            )
            return x1

        if compute_traj_velo_norms:
            ## calculate the velocity norms for each manifold component
            manifolds_info = [
                (dims.a, a_manifold),
                (dims.f, f_manifold),
                (dims.l, l_manifold),
                (dims.pd_w, pd_w_manifold),
                (dims.pd_f, pd_f_manifold),
            ]

            norms = []
            start = 0
            for dim_size, manifold_obj in manifolds_info:
                end = start + dim_size
                norm = manifold_obj.inner(
                    xs[..., start:end],
                    vs[..., start:end],
                    vs[..., start:end],
                    data_in_dim=1,
                )
                norms.append(norm)
                start = end

            norm_a, norm_f, norm_l, norm_pd_w, norm_pd_f = norms

        if entire_traj and compute_traj_velo_norms:
            return xs, norm_a, norm_f, norm_l, norm_pd_w, norm_pd_f
        elif entire_traj and not compute_traj_velo_norms:
            return xs
        elif not entire_traj and compute_traj_velo_norms:
            return xs[0], norm_a, norm_f, norm_l, norm_pd_w, norm_pd_f
        else:
            # this should happen due to logic above
            return xs[0]

    @torch.no_grad()
    def compute_exact_loglikelihood(
        self,
        batch: Data,
        stage: str,
        t1: float = 1.0,
        return_projx_error: bool = False,
        num_steps: int = 1_000,
    ):
        """Computes the negative log-likelihood of a batch of data."""
        x1, manifold, dims, mask_a_or_f = self.manifold_getter(
            batch.batch,
            batch.atom_types,
            batch.frac_coords,
            batch.lengths,
            batch.angles,
            split_manifold=False,
        )
        dim = sum(dims)

        nfe = [0]

        div_mode = self.cfg.integrate.get(
            "div_mode", "rademacher"
        )  # alternative: exact

        v = None
        if div_mode == "rademacher":
            v = torch.randint(low=0, high=2, size=x1.shape).to(x1) * 2 - 1

        vecfield = partial(
            self.vecfield,
            num_atoms=batch.num_atoms,
            node2graph=batch.batch,
            dims=dims,
            mask_a_or_f=mask_a_or_f,
        )

        def odefunc(t, tensor):
            nfe[0] += 1
            t = t.to(tensor)
            x = tensor[..., :dim]
            l_vecfield = lambda x: vecfield(
                t=torch.atleast_2d(t), x=torch.atleast_2d(x), manifold=manifold
            )
            dx, div = output_and_div(l_vecfield, x, v=v, div_mode=div_mode)

            if hasattr(manifold, "logdetG"):
                corr = jvp(manifold.logdetG, (x,), (dx,))[1]
                div = div + 0.5 * corr.to(div)

            div = div.reshape(-1, 1)
            del t, x
            return torch.cat([dx, div], dim=-1)

        # Solve ODE on the product manifold of data manifold x euclidean.
        prod_manis = [
            ProductManifold((m, dim), (Euclidean(), 1)) for m in manifold.manifolds
        ]
        product_man = VMapManifolds(prod_manis)
        state1 = torch.cat([x1, torch.zeros_like(x1[..., :1])], dim=-1)

        with torch.no_grad():
            state0 = projx_integrator_return_last(
                product_man,
                odefunc,
                state1,
                t=torch.linspace(t1, 0, num_steps + 1).to(x1),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                local_coords=False,
                pbar=True,
            )

        # log number of function evaluations
        # self.log(
        #     f"{stage}/nfe",
        #     float(nfe[0]),
        #     prog_bar=True,
        #     logger=True,
        #     batch_size=batch.batch_size,
        # )

        x0, logdetjac = state0[..., :dim], state0[..., -1]

        x0_ = x0
        x0 = manifold.projx(x0)

        # log how close the final solution is to the manifold.
        integ_error = (x0[..., :dim] - x0_[..., :dim]).abs().max()
        # self.log(f"{stage}/integ_error", integ_error, batch_size=batch.batch_size)

        logp0 = manifold.base_logprob(x0)
        logp1 = logp0 + logdetjac

        if self.cfg.get("normalize_loglik", True):
            logp1 = logp1 / batch.num_atoms

        # Mask out those that left the manifold
        masked_logp1 = logp1
        if isinstance(manifold, (SPDGivenN, SPDNonIsotropicRandom)):
            mask = integ_error < 1e-5
            # self.log(
            #     f"{stage}/frac_within_manifold",
            #     mask.sum() / mask.nelement(),
            #     batch_size=batch.batch_size,
            # )
            masked_logp1 = logp1[mask]

        if return_projx_error:
            return logp1, integ_error
        else:
            return masked_logp1

    def loss_fn(self, batch: Data, *args, **kwargs) -> dict[str, torch.Tensor]:
        return self.rfm_loss_fn(batch, *args, **kwargs)

    def rfm_loss_fn(
        self, batch: Data, x0: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:

        (
            x1,  ## [B, max_atom_num * (\sum d_i)]
            manifold,
            a_manifold,
            f_manifold,
            l_manifold,
            pd_w_manifold,
            pd_f_manifold,
            dims,
            mask_a_or_f,
        ) = self.manifold_getter(
            batch.batch,
            batch.atom_types,
            batch.frac_coords,
            batch.lengths,
            batch.angles,
            batch.sd_weights,
            batch.pd_weights,
            batch.pd_frac_coords,
            split_manifold=True,
        )

        if x0 is None:
            if self.cfg.base_distribution_from_data:
                x0 = self.manifold_getter(
                    batch.batch,
                    batch.atom_types,
                    batch.frac_coords,
                    batch.lengths,
                    batch.angles,
                    batch.sd_weights,
                    batch.pd_weights,
                    batch.pd_frac_coords,
                    split_manifold=False,
                )[0]
            else:
                ## [B, max_atom_num * (\sum d_i)]
                x0 = manifold.random(*x1.shape, dtype=x1.dtype, device=x1.device)

        vecfield = partial(
            self.vecfield,
            num_atoms=batch.num_atoms,
            node2graph=batch.batch,
            dims=dims,
            mask_a_or_f=mask_a_or_f,
        )

        N = x1.shape[0]

        t = torch.rand(N, dtype=x1.dtype, device=x1.device).reshape(-1, 1)

        x1 = manifold.projx(x1)

        x_t, u_t = manifold.cond_u(x0, x1, t)

        x_t = x_t.reshape(N, x0.shape[-1])
        u_t = u_t.reshape(N, x0.shape[-1])

        # this projects out the mean from the tangent vectors
        # our model cannot predict it, so keeping it in inflates the loss
        u_t = manifold.proju(x_t, u_t)

        cond = None
        if self.cfg.model.self_cond:
            with torch.no_grad():
                if torch.rand((1)) < 0.5:
                    cond = projx_integrate_xt_to_x1(
                        manifold,
                        lambda t, x: vecfield(
                            t=torch.atleast_2d(t),
                            x=torch.atleast_2d(x),
                            manifold=manifold,
                        ),
                        x_t,
                        t,
                    ).detach_()

        u_t_pred = vecfield(t=t, x=x_t, manifold=manifold, cond=cond)

        diff = u_t_pred - u_t

        max_num_atoms = mask_a_or_f.size(-1)

        dim_feat_a = dims.a / max_num_atoms  ## NUM_ATOMIC_BITS
        dim_feat_f = dims.f / max_num_atoms  ## 3
        dim_feat_pd_w = dims.pd_w / max_num_atoms
        dim_feat_pd_f = dims.pd_f / max_num_atoms

        if self.costs["loss_ce"] > 0.0:
            x1_pred_t = projx_integrate_xt_to_x1(
                manifold,
                None,  # already computed velocity
                x_t,
                t,
                u_t_pred,
            )
            a1_pred_t = x1_pred_t[:, : dims.a].reshape(N * max_num_atoms, -1)
            a1 = x1[:, : dims.a].reshape(N * max_num_atoms, -1)
            if self.manifold_getter.atom_type_manifold == "analog_bits":
                loss_ce = torch.einsum("bi,bi->b", a1_pred_t, a1).reshape(
                    N, max_num_atoms
                )
                loss_ce = torch.nn.functional.logsigmoid(loss_ce) * mask_a_or_f
            elif self.manifold_getter.atom_type_manifold == "simplex":
                a1 = self.manifold_getter._inverse_atomic_one_hot(a1)
                loss_ce = torch.nn.functional.cross_entropy(
                    a1_pred_t,
                    a1,
                    reduce=False,
                ).reshape(N, max_num_atoms)
                loss_ce = loss_ce * mask_a_or_f
            else:
                raise ValueError(
                    f"{self.manifold_getter.atom_type_manifold=} cannot do cross entropy"
                )
            loss_ce = (loss_ce.sum(dim=-1) / mask_a_or_f.sum(dim=-1)).mean()
        else:
            loss_ce = torch.Tensor([0.0]).squeeze().to(diff)

        ## [B, max_num_atoms, 1]
        disorder_types_dense, _ = to_dense_batch(batch.disorder_types, batch.batch)

        # sd_type_mask = (disorder_types_dense == 2)  # [B, max_num_atoms, 1]
        # sd_type_mask = torch.ones_like(disorder_types_dense, device=disorder_types_dense.device).bool()  # use all atoms
        sd_type_mask = None

        # pd_weight_mask = (disorder_types_dense == 3)  # [B, max_num_atoms, 1]
        pd_weight_mask = None

        pd_frac_coords_mask = disorder_types_dense == 3

        if "null" in self.manifold_getter.atom_type_manifold:
            sd_type_mask = None
        if "null" in self.manifold_getter.weight_manifold:
            pd_weight_mask = None
        if "null" in self.manifold_getter.pd_frac_coords_manifold:
            pd_frac_coords_mask = None

        manifolds = [
            (dims.a, a_manifold, dim_feat_a, sd_type_mask),
            (dims.f, f_manifold, dim_feat_f, None),
            (dims.l, l_manifold, dims.l, None),
            (dims.pd_w, pd_w_manifold, dim_feat_pd_w, pd_weight_mask),
            (dims.pd_f, pd_f_manifold, dim_feat_pd_f, pd_frac_coords_mask),
        ]

        # calculate loss for all manifolds
        losses, start = [], 0
        for dim_size, manifold, dim_feat, atom_mask in manifolds:
            end = start + dim_size
            if atom_mask is None:
                loss = (
                    manifold.inner(
                        x_t[:, start:end], diff[:, start:end], diff[:, start:end]
                    ).mean()
                    / dim_feat
                )
            else:
                selected_diff = diff[:, start:end]
                masked_diff = selected_diff * atom_mask.repeat(
                    1, 1, int(dim_feat)
                ).flatten(1)
                loss = (
                    manifold.inner(x_t[:, start:end], masked_diff, masked_diff).mean()
                    / dim_feat
                )
            losses.append(loss)
            start = end

        loss_a, loss_f, loss_l, loss_pd_w, loss_pd_f = losses

        loss = (
            self.costs["loss_a"] * loss_a
            + self.costs["loss_f"] * loss_f
            + self.costs["loss_l"] * loss_l
            + self.costs["loss_ce"] * loss_ce
            + self.costs["loss_pd_w"] * loss_pd_w
            + self.costs["loss_pd_f"] * loss_pd_f
        )

        return {
            "loss": loss,
            "loss_a": self.costs["loss_a"] * loss_a,
            "loss_f": self.costs["loss_f"] * loss_f,
            "loss_l": self.costs["loss_l"] * loss_l,
            "loss_ce": self.costs["loss_ce"] * loss_ce,
            "loss_pd_w": self.costs["loss_pd_w"] * loss_pd_w,
            "loss_pd_f": self.costs["loss_pd_f"] * loss_pd_f,
            "unscaled/loss_a": loss_a,
            "unscaled/loss_f": loss_f,
            "unscaled/loss_l": loss_l,
            "unscaled/loss_ce": loss_ce,
            "unscaled/loss_pd_w": loss_pd_w,
            "unscaled/loss_pd_f": loss_pd_f,
        }

    def training_step(self, batch: Data, batch_idx: int):
        loss_dict = self.loss_fn(batch)
        for metric in self.train_metrics.values():
            metric.to(batch.batch.device)
        if torch.isfinite(loss_dict["loss"]):
            # log train metrics

            for k, v in loss_dict.items():
                self.log(
                    f"train/{k}",
                    v,
                    batch_size=batch.batch_size,
                    sync_dist=True,
                )
                # self.train_metrics[k].update(v.cpu())
                self.train_metrics[k].update(v)
        else:
            warnings.warn(
                f"Skipping iteration because loss is {loss_dict['loss'].item()}.",
                RuntimeWarning,
            )
            return None

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return loss_dict

    def training_epoch_end(self, outputs: list[Any]):
        # def on_train_epoch_end(self, outputs: list[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        for train_metric in self.train_metrics.values():
            train_metric.reset()

    def shared_eval_step(
        self,
        batch: Data,
        batch_idx: int,
        stage: Literal["val", "test"],
        compute_loss: bool,
        compute_nll: bool,
    ) -> dict[str, torch.Tensor]:
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        for metric in metrics.values():
            metric.to(batch.batch.device)

        if hasattr(self, f"{stage}_metrics_best"):
            metrics_best = getattr(self, f"{stage}_metrics_best")
            for metric in metrics_best.values():
                metric.to(batch.batch.device)

        out = {}

        if compute_loss:
            loss_dict = self.loss_fn(batch)
            for k, v in loss_dict.items():
                self.log(
                    f"{stage}/{k}",
                    v,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch.batch_size,
                    sync_dist=True,
                )
                # metrics[k].update(v.cpu())
                metrics[k].update(v)
            out.update(loss_dict)

        if compute_nll:
            nll_dict = {}
            logprob = self.compute_exact_loglikelihood(
                batch,
                stage,
                num_steps=self.cfg.integrate.get("num_steps", 1_000),
            )
            nll = -logprob.mean()
            nll_dict[f"{stage}/nll"] = nll
            nll_dict[f"{stage}/nll_num_steps"] = self.cfg.integrate.num_steps
            # self.log(
            #     f"{stage}/nll",
            #     nll,
            #     prog_bar=True,
            #     batch_size=batch.batch_size,
            # )
            # self.log(
            #     f"{stage}/nll_num_steps",
            #     self.cfg.integrate.num_steps,
            #     prog_bar=True,
            #     batch_size=batch.batch_size,
            # )
            self.logger.log_metrics(nll_dict, step=self.global_step)
            metrics["nll"].update(nll.cpu())
            # metrics["nll"].update(nll)
            out.update(nll_dict)

        return out

    def compute_reconstruction(
        self,
        batch: Data,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        *_, dims, mask_a_or_f = self.manifold_getter(
            batch.batch,
            batch.atom_types,
            batch.frac_coords,
            batch.lengths,
            batch.angles,
            batch.sd_weights,
            batch.pd_weights,
            batch.pd_frac_coords,
            split_manifold=False,
        )
        if self.cfg.base_distribution_from_data:
            x0 = self.manifold_getter(
                batch.batch,
                batch.atom_types,
                batch.frac_coords,
                batch.lengths,
                batch.angles,
                batch.sd_weights,
                batch.pd_weights,
                batch.pd_frac_coords,
                split_manifold=True,
            )[0]
        else:
            x0 = None

        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l, norms_pd_w, norms_pd_f = self.sample(
                batch, num_steps=num_steps, x0=x0
            )
            norms = {
                "norms_a": norms_a,
                "norms_f": norms_f,
                "norms_l": norms_l,
                "norms_pd_w": norms_pd_w,
                "norms_pd_f": norms_pd_f,
            }
        else:
            recon = self.sample(batch, num_steps=num_steps, x0=x0)
            norms = {}

        (
            atom_types,
            frac_coords,
            lattices,
            pd_weights,
            pd_frac_coords,
        ) = self.manifold_getter.flatrep_to_crystal(recon, dims, mask_a_or_f)
        lengths, angles = lattices_to_params_shape(lattices)

        out = {
            "atom_types": atom_types,
            "frac_coords": frac_coords,
            "lattices": lattices,
            "lengths": lengths,
            "angles": angles,
            "pd_weights": pd_weights,
            "pd_frac_coords": pd_frac_coords,
            "num_atoms": batch.num_atoms,
            "input_data_batch": batch,
        }
        out.update(norms)
        return out

    def compute_recon_trajectory(
        self,
        batch: Data,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        *_, dims, mask_a_or_f = self.manifold_getter(
            batch.batch,
            batch.atom_types,
            batch.frac_coords,
            batch.lengths,
            batch.angles,
            split_manifold=False,
        )
        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l = self.sample(
                batch, num_steps=num_steps, entire_traj=True
            )
            norms = {"norms_a": norms_a, "norms_f": norms_f, "norms_l": norms_l}
        else:
            recon = self.sample(batch, num_steps=num_steps, entire_traj=True)
            norms = {}

        atom_types, frac_coords, lattices = [], [], []
        lengths, angles = [], []
        for recon_step in recon:
            a, f, l = self.manifold_getter.flatrep_to_crystal(
                recon_step, dims, mask_a_or_f
            )
            lns, angs = lattices_to_params_shape(l)
            atom_types.append(a)
            frac_coords.append(f)
            lattices.append(l)
            lengths.append(lns)
            angles.append(angs)
        out = {
            "atom_types": torch.stack(atom_types, dim=0),
            "frac_coords": torch.stack(frac_coords, dim=0),
            "lattices": torch.stack(lattices, dim=0),
            "lengths": torch.stack(lengths, dim=0),
            "angles": torch.stack(angles, dim=0),
            "num_atoms": batch.num_atoms,
            "input_data_batch": batch,
        }
        out.update(norms)
        return out

    def compute_generation(
        self,
        batch: Data,
        dim_coords: int = 3,
        dim_pd_weight: int | None = None,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:

        *_, dims, mask_a_or_f = self.manifold_getter.from_empty_batch(
            batch.batch, dim_coords, dim_pd_weight, split_manifold=False
        )

        if self.cfg.base_distribution_from_data:
            x0 = self.manifold_getter(
                batch.batch,
                batch.atom_types_initial,
                batch.frac_coords_initial,
                batch.lengths_initial,
                batch.angles_initial,
                split_manifold=True,
            )[0]
        else:
            x0 = None

        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l, norms_pd_w, norms_pd_f = self.gen_sample(
                batch.batch, dim_coords, dim_pd_weight, num_steps=num_steps, x0=x0
            )
            norms = {
                "norms_a": norms_a,
                "norms_f": norms_f,
                "norms_l": norms_l,
                "norms_pd_w": norms_pd_w,
                "norms_pd_f": norms_pd_f,
            }

        else:
            recon = self.gen_sample(
                batch.batch, dim_coords, dim_pd_weight, num_steps=num_steps, x0=x0
            )
            norms = {}
        (
            atom_types,
            frac_coords,
            lattices,
            pd_weights,
            pd_frac_coords,
        ) = self.manifold_getter.flatrep_to_crystal(recon, dims, mask_a_or_f)
        lengths, angles = lattices_to_params_shape(lattices)

        out = {
            "atom_types": atom_types,
            "frac_coords": frac_coords,
            "lattices": lattices,
            "lengths": lengths,
            "angles": angles,
            "pd_weights": pd_weights,
            "pd_frac_coords": pd_frac_coords,
            "num_atoms": batch.num_atoms,
            "input_data_batch": batch,
        }
        out.update(norms)
        return out

    def compute_gen_trajectory(
        self,
        batch: Data,
        dim_coords: int = 3,
        dim_pd_weight: int | None = None,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        *_, dims, mask_a_or_f = self.manifold_getter.from_empty_batch(
            batch.batch, dim_coords, dim_pd_weight, split_manifold=False
        )

        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l = self.gen_sample(
                batch.batch,
                dim_coords,
                num_steps=num_steps,
                entire_traj=True,
            )
            norms = {"norms_a": norms_a, "norms_f": norms_f, "norms_l": norms_l}
        else:
            recon = self.gen_sample(
                batch.batch,
                dim_coords,
                dim_pd_weight,
                num_steps=num_steps,
                entire_traj=True,
            )
            norms = {}

        atom_types, frac_coords, lattices = [], [], []
        lengths, angles = [], []
        pd_weights, pd_frac_coords = [], []
        for recon_step in recon:
            a, f, l, pd_w, pd_f = self.manifold_getter.flatrep_to_crystal(
                recon_step, dims, mask_a_or_f
            )

            lns, angs = lattices_to_params_shape(l)
            atom_types.append(a)
            frac_coords.append(f)
            lattices.append(l)
            lengths.append(lns)
            angles.append(angs)
            pd_weights.append(pd_w)
            pd_frac_coords.append(pd_f)

        out = {
            "atom_types": torch.stack(atom_types, dim=0),
            "frac_coords": torch.stack(frac_coords, dim=0),
            "lattices": torch.stack(lattices, dim=0),
            "lengths": torch.stack(lengths, dim=0),
            "angles": torch.stack(angles, dim=0),
            "pd_weights": torch.stack(pd_weights, dim=0),
            "pd_frac_coords": torch.stack(pd_frac_coords, dim=0),
            "num_atoms": batch.num_atoms,
            "input_data_batch": batch,
        }
        out.update(norms)
        return out

    def compute_prediction(
        self,
        batch: Data,
        dim_coords: int = 3,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        *_, dims, mask_a_or_f = self.manifold_getter.from_only_atom_types(
            batch.batch, batch.atom_types, dim_coords, split_manifold=False
        )
        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l = self.pred_sample(
                batch.batch, batch.atom_types, dim_coords, num_steps=num_steps
            )
            norms = {"norms_a": norms_a, "norms_f": norms_f, "norms_l": norms_l}
        else:
            recon = self.pred_sample(
                batch.batch, batch.atom_types, dim_coords, num_steps=num_steps
            )
            norms = {}
        atom_types, frac_coords, lattices = self.manifold_getter.flatrep_to_crystal(
            recon, dims, mask_a_or_f
        )
        lengths, angles = lattices_to_params_shape(lattices)
        out = {
            "atom_types": atom_types,
            "frac_coords": frac_coords,
            "lattices": lattices,
            "lengths": lengths,
            "angles": angles,
            "num_atoms": batch.num_atoms,
            "input_data_batch": batch,
        }
        out.update(norms)
        return out

    def validation_step(self, batch: Data, batch_idx: int):
        return self.shared_eval_step(
            batch,
            batch_idx,
            stage="val",
            compute_loss=True,
            compute_nll=self.cfg.val.compute_nll,
        )

    def validation_epoch_end(self, outputs: list[Any]):
        # def on_validation_epoch_end(self):
        out = {}
        for key, val_metric in self.val_metrics.items():
            val_metric_value = (
                val_metric.compute()
            )  # get val accuracy from current epoch
            val_metric_best = self.val_metrics_best[key]
            val_metric_best.update(val_metric_value)
            self.log(
                f"val/best/{key}",
                val_metric_best.compute(),
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            val_metric.reset()
            out[key] = val_metric_value
        return out

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_eval_step(
            batch,
            batch_idx,
            stage="test",
            compute_loss=self.cfg.test.get("compute_loss", False),
            compute_nll=self.cfg.test.get("compute_nll", False),
        )

    def test_epoch_end(self, outputs: list[Any]):
        # def on_test_epoch_end(self, outputs: list[Any]):
        for test_metric in self.test_metrics.values():
            test_metric.reset()

    def predict_step(self, batch: Any, batch_idx: int):
        if not hasattr(batch, "frac_coords"):
            if "null" in self.cfg.model.manifold_getter.atom_type_manifold:
                if self.cfg.integrate.get("entire_traj", False):
                    raise NotImplementedError()
                else:
                    return self.compute_prediction(
                        batch,
                        dim_coords=self.cfg.data.get("dim_coords", 3),
                        num_steps=self.cfg.integrate.get("num_steps", 1_000),
                    )
            else:
                if self.cfg.integrate.get("entire_traj", False):
                    return self.compute_gen_trajectory(
                        batch,
                        dim_coords=self.cfg.data.get("dim_coords", 3),
                        dim_pd_weight=self.cfg.data.get("dim_pd_weight", 2),
                        num_steps=self.cfg.integrate.get("num_steps", 1_000),
                    )
                else:
                    return self.compute_generation(
                        batch,
                        dim_coords=self.cfg.data.get("dim_coords", 3),
                        dim_pd_weight=self.cfg.data.get("dim_pd_weight", 2),
                        num_steps=self.cfg.integrate.get("num_steps", 1_000),
                    )
        else:
            # not generating or predicting new structures
            if self.cfg.integrate.get("entire_traj", False):
                return self.compute_recon_trajectory(
                    batch,
                    num_steps=self.cfg.integrate.get("num_steps", 1_000),
                )
            else:
                return self.compute_reconstruction(
                    batch,
                    num_steps=self.cfg.integrate.get("num_steps", 1_000),
                )

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.cfg.optim.optimizer,
            params=self.parameters(),
            _convert_="partial",
        )
        if self.cfg.optim.get("lr_scheduler", None) is not None:
            lr_scheduler = hydra.utils.instantiate(
                self.cfg.optim.lr_scheduler,
                optimizer,
            )
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": lr_scheduler,
                        "interval": "epoch",
                        "monitor": self.cfg.optim.monitor,
                        "frequency": self.cfg.optim.frequency,
                    },
                }
            elif isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": lr_scheduler,
                        "interval": self.cfg.optim.interval,
                    },
                }
            else:
                raise NotImplementedError("unsuported lr_scheduler")
        else:
            return {"optimizer": optimizer}
