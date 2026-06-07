import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from dmflow.rfm.manifolds.masked import MaskedManifold
from geoopt import Euclidean
from geoopt.utils import broadcast_shapes
import torch.nn.functional as F

## https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/manifolds/sphere.py


class Sphere:
    """Represents a hyperpshere in :math:`R^D`. Isometric to the product of 1-D spheres."""

    EPS = {torch.float32: 1e-4, torch.float64: 1e-7}

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        u_norm = u.norm(dim=-1, keepdim=True)
        exp = torch.cos(u_norm) * x + torch.sinc(u_norm / torch.pi) * u
        return exp

        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retr = self.projx(x + u)
        cond = norm_u > self.EPS[norm_u.dtype]

        return torch.where(cond, exp, retr)

    def logmap(self, x: Tensor, y: Tensor, eps=1e-4) -> Tensor:
        u = y - x
        u = u - (x * u).sum(dim=-1, keepdim=True) * x
        u_norm = u.norm(dim=-1, keepdim=True).clamp(min=eps)
        dist = self.dist(x, y, eps).unsqueeze(-1)
        return torch.where(dist > eps, u * dist / u_norm, u)

        u = self.proju(x, y - x)
        dist = self.dist(x, y, keepdim=True)
        cond = dist.gt(self.EPS[x.dtype])
        result = torch.where(
            cond,
            u * dist / u.norm(dim=-1, keepdim=True).clamp_min(self.EPS[x.dtype]),
            u,
        )
        return result

    ## projects a vector onto the sphere (norm = 1)
    def projx(self, x: Tensor, eps=0.0) -> Tensor:
        x = x.clamp(eps, 1 - eps)
        return F.normalize(x, dim=-1)
        return x / x.norm(dim=-1, keepdim=True)

    ## projects a vector onto the tangent space at point x
    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        return u - (x * u).sum(dim=-1, keepdim=True) * x

    def dist(self, x, y, eps=1e-4):
        return torch.acos((x * y).sum(-1).clamp(0, 1 - eps))

    # def dist(self, x: Tensor, y: Tensor, *, keepdim=False) -> Tensor:
    #     inner = (x * y).sum(-1, keepdim=keepdim)
    #     return torch.acos(inner)


## Inherits from Euclidean class to utilize built-in methods (e.g., _check_shape)
class MaskedSphere(Sphere, MaskedManifold, Euclidean):
    """Represents a masked sphere in :math:`R^D`."""

    def __init__(
        self,
        dim_weight: int = -1,
        num_atoms: int = -1,
        max_num_atoms: int = -1,
    ):
        ## nn.Module.__init__ in fact
        super().__init__(ndim=1)
        self.dim_weight = dim_weight
        self.max_num_atoms = max_num_atoms
        self.register_buffer("mask", self._get_mask(num_atoms, max_num_atoms))

    @staticmethod
    def preprocess(x: torch.Tensor) -> torch.Tensor:
        ## project simplex point onto the sphere
        return x.sqrt()

    ## project sphere point onto the simplex
    @staticmethod
    def postprocess(x_hat: torch.Tensor) -> torch.Tensor:
        return x_hat.square()

    @staticmethod
    def base_logprob_simplex(x, eps=1e-4):
        n_class = torch.tensor(x.size(-1), dtype=torch.float)
        return torch.lgamma(n_class)

    ## from simplex to sphere
    @staticmethod
    def base_logprob(x: torch.Tensor, eps=1e-4) -> torch.Tensor:
        n_class = x.size(-1)
        random_simplex_point = MaskedSphere().base_logprob_simplex(x, eps=eps)
        return torch.log(random_simplex_point.sqrt().clamp(min=eps)).sum(-1) + np.log(
            2
        ) * (n_class - 1)

    ## from sphere to simplex
    @staticmethod
    def postprocess_logprob(x, eps=1e-4):
        n_class = torch.tensor(x.size(-1), dtype=torch.float, device=x.device)
        return -torch.log(x.sqrt().clamp(min=eps)).sum(-1) - np.log(2) * (n_class - 1)

    @property
    def dim_m1(self) -> int:
        return self.dim_weight

    def _check_point_on_manifold(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ):

        norm = torch.norm(x, dim=-1)
        if not torch.allclose(norm, torch.ones_like(norm), atol=atol, rtol=rtol):
            return False, "Point is not on the unit sphere (norm != 1)"
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ):
        if x.shape != u.shape:
            return False, f"Shape mismatch: point {x.shape} vs vector {u.shape}"

        ## check point on manifold
        is_on_manifold, error = self._check_point_on_manifold(
            x, u, atol=atol, rtol=rtol
        )
        if not is_on_manifold:
            return False, error

        dot_product = torch.sum(x * u, dim=-1)
        if not torch.allclose(
            dot_product, torch.zeros_like(dot_product), atol=atol, rtol=rtol
        ):
            return False, "Vector is not orthogonal to the sphere's normal vector"

        return True, None

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        # divide by the number of atoms for loss weighting
        return Euclidean(ndim=1).inner(x, u, v, keepdim=keepdim) / self.mask.sum().to(u)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        initial_shape = x.shape
        # [B, max_num_atom * d] -> [B, max_num_atom, d]
        x = self.reshape_and_mask(initial_shape, x)
        x = super().projx(x)
        return self.mask_and_reshape(initial_shape, x)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """removes drift velocity"""
        initial_x_shape = x.shape
        initial_u_shape = u.shape

        u_ = self.reshape_and_mask(initial_u_shape, u)
        x_ = self.reshape_and_mask(initial_x_shape, x)
        out = super().proju(x_, u_)

        out = self.mask_and_reshape(initial_u_shape, out)
        ## Ensure the output shape matches the initial shapes (follow Euclidean().proju)
        target_shape = broadcast_shapes(initial_x_shape, initial_u_shape)
        return out.expand(target_shape)

    @staticmethod
    def sample_simplex(*size, dtype=None, device=None, eps=1e-4) -> torch.Tensor:
        """
        Uniformly sample from a simplex.
        :param sizes: sizes of the Tensor to be returned
        :param device: device to put the Tensor on
        :param eps: small float to avoid instability
        :return: Tensor of shape sizes, with values summing to 1
        """
        x = torch.empty(*size, device=device, dtype=torch.float).exponential_(1)
        p = x / x.sum(dim=-1, keepdim=True)
        p = p.clamp(eps, 1 - eps)
        return p / p.sum(dim=-1, keepdim=True)

    # def random_base(self, *size, dtype=None, device=None) -> torch.Tensor:
    #     assert (
    #         size[-1] == self.max_num_atoms * self.dim_weight
    #     ), "last dimension must be compatible with max_num_atoms and dim_weight"

    #     ## Generate random points on the sphere
    #     ## [B, max_num_atoms * dim_weight]
    #     random_points = torch.randn(*size, dtype=dtype, device=device)
    #     random_points = self.projx(random_points)
    #     return random_points

    def random_base(self, *size, dtype=None, device=None) -> torch.Tensor:
        """
        Sample random points on the sphere's positive orthant.
        This corresponds to the image of uniform simplex points under the sqrt transform.
        """
        assert (
            size[-1] == self.max_num_atoms * self.dim_weight
        ), "last dimension must be compatible with max_num_atoms and dim_weight"

        concentration = torch.ones(self.dim_weight, dtype=dtype, device=device)
        d = torch.distributions.Dirichlet(concentration, validate_args=False)

        shape = d._extended_shape((*size[:-1], self.max_num_atoms))
        concentration_expanded = d.concentration.expand(shape)
        simplex_samples = torch._sample_dirichlet(concentration_expanded)

        sphere_samples = simplex_samples.sqrt()

        return self.mask_and_reshape(size, sphere_samples)

    random = random_base

    def base_logprob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def extra_repr(self):
        return f"""
        dim_weight={self.dim_weight},
        num_atoms={int(self.mask.sum().cpu().item())},
        max_num_atoms={self.max_num_atoms},
        """
