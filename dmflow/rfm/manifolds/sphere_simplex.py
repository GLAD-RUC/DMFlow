"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from geoopt import Sphere, Euclidean
from geoopt.utils import broadcast_shapes
from dmflow.rfm.manifolds.masked import MaskedManifold


class SphereSimplex(MaskedManifold, Sphere):
    """
    Represents a masked sphere manifold for multi-atom systems.

    This manifold is designed to work with points on the unit sphere that can be
    transformed to/from simplex points via square root mapping:
    - Simplex to Sphere: x_sphere = sqrt(x_simplex)
    - Sphere to Simplex: x_simplex = x_sphere^2

    The masking mechanism allows handling variable number of atoms in a batch.
    """

    name = "SphereSimplex"
    reversible = False

    def __init__(self, dim: int, num_atoms: int, max_num_atoms: int):
        """
        Initialize the masked sphere manifold.

        Args:
            dim: Dimension of each sphere (number of categories/features per atom)
            num_atoms: Actual number of atoms to use
            max_num_atoms: Maximum number of atoms (for padding)
        """
        # Initialize parent Sphere manifold
        Sphere.__init__(self)

        # Store dimensions
        self.dim = dim
        self.num_atoms = num_atoms
        self.max_num_atoms = max_num_atoms

        # Create and register mask for active atoms
        self.register_buffer("mask", self._get_mask(num_atoms, max_num_atoms))

    @property
    def dim_m1(self) -> int:
        """Return the dimension of each individual sphere."""
        return self.dim

    @property
    def dim_m2(self) -> int:
        """Return the maximum number of atoms (from MaskedManifold)."""
        return self.max_num_atoms

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> tuple[bool, str | None]:
        """
        Check if a point lies on the masked sphere manifold.

        Args:
            x: Point tensor of shape [..., max_num_atoms * dim]
            atol: Absolute tolerance for norm check
            rtol: Relative tolerance for norm check

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check dimension compatibility
        if x.shape[-1] != self.max_num_atoms * self.dim:
            return (
                False,
                f"Last dimension {x.shape[-1]} != {self.max_num_atoms * self.dim}",
            )

        # Reshape and apply mask
        x_reshaped = self.reshape_and_mask(x.shape, x)

        # Check that each active atom has unit norm
        norms = x_reshaped.norm(dim=-1)
        target = self.mask.squeeze(-1).to(x)  # 1 for active atoms, 0 for masked

        ok = torch.allclose(norms, target, atol=atol, rtol=rtol)
        if not ok:
            return False, f"Points not on unit sphere (norms != 1 for active atoms)"

        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> tuple[bool, str | None]:
        """
        Check if a vector lies in the tangent space at point x.

        For sphere manifold, tangent vectors must be orthogonal to the position vector.

        Args:
            x: Base point on manifold
            u: Vector to check
            atol: Absolute tolerance
            rtol: Relative tolerance

        Returns:
            Tuple of (is_valid, error_message)
        """
        # First check if x is on manifold
        on_manifold, err = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if not on_manifold:
            return False, f"Base point not on manifold: {err}"

        # Reshape both tensors
        x_reshaped = self.reshape_and_mask(x.shape, x)
        u_reshaped = self.reshape_and_mask(u.shape, u)

        # Check orthogonality: <x, u> = 0 for each atom
        inner_products = (x_reshaped * u_reshaped).sum(dim=-1)
        target = torch.zeros_like(inner_products)

        # Only check for active atoms
        masked_inner = self.mask.squeeze(-1) * inner_products
        ok = torch.allclose(masked_inner, target, atol=atol, rtol=rtol)

        if not ok:
            return False, "Vector not orthogonal to sphere normal"

        return True, None

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a point onto the masked sphere manifold.

        Normalizes each atom's vector to unit length.

        Args:
            x: Point to project, shape [..., max_num_atoms * dim]

        Returns:
            Projected point on the sphere
        """
        initial_shape = x.shape

        # Reshape to [..., max_num_atoms, dim]
        x_reshaped = self.reshape_and_mask(initial_shape, x)

        # Normalize each atom to unit sphere (with numerical stability)
        x_norm = x_reshaped.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_projected = x_reshaped / x_norm

        # Apply mask and reshape back
        return self.mask_and_reshape(initial_shape, x_projected)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Project a vector onto the tangent space at point x.

        For sphere: u_tangent = u - <u, x> * x

        Args:
            x: Base point on the sphere
            u: Vector to project

        Returns:
            Projected tangent vector
        """
        initial_x_shape = x.shape
        initial_u_shape = u.shape

        # Reshape tensors
        x_reshaped = self.reshape_and_mask(initial_x_shape, x)
        u_reshaped = self.reshape_and_mask(initial_u_shape, u)

        # Project: u - (u·x)x for each atom
        inner_prod = (u_reshaped * x_reshaped).sum(dim=-1, keepdim=True)
        u_tangent = u_reshaped - inner_prod * x_reshaped

        # Apply mask and reshape
        u_tangent = self.mask_and_reshape(initial_u_shape, u_tangent)

        # Match broadcast shape
        target_shape = broadcast_shapes(initial_x_shape, initial_u_shape)
        return u_tangent.expand(target_shape)

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        """
        Compute inner product on the manifold with proper normalization.

        Args:
            x: Base point (not used for Euclidean metric)
            u: First tangent vector
            v: Second tangent vector (if None, compute <u, u>)
            keepdim: Whether to keep dimensions

        Returns:
            Inner product normalized by number of active atoms
        """
        # Use Euclidean inner product normalized by number of atoms
        inner_prod = Euclidean(ndim=1).inner(x, u, v, keepdim=keepdim)
        return inner_prod / self.mask.sum().to(u)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Exponential map on the sphere.

        exp_x(u) = cos(||u||) * x + sin(||u||) / ||u|| * u

        Args:
            x: Base point on sphere
            u: Tangent vector at x

        Returns:
            Point on sphere
        """
        initial_shape = x.shape

        x_reshaped = self.reshape_and_mask(initial_shape, x)
        u_reshaped = self.reshape_and_mask(u.shape, u)

        # Compute norm with stability
        u_norm = u_reshaped.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Exponential map formula
        exp_point = (
            torch.cos(u_norm) * x_reshaped + torch.sinc(u_norm / math.pi) * u_reshaped
        )

        return self.mask_and_reshape(initial_shape, exp_point)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map on the sphere.

        log_x(y) = arccos(<x, y>) / sin(arccos(<x, y>)) * (y - <x, y> * x)

        Args:
            x: Base point on sphere
            y: Target point on sphere

        Returns:
            Tangent vector at x pointing to y
        """
        initial_shape = x.shape

        x_reshaped = self.reshape_and_mask(initial_shape, x)
        y_reshaped = self.reshape_and_mask(y.shape, y)

        # Compute inner product and distance
        inner = (
            (x_reshaped * y_reshaped)
            .sum(dim=-1, keepdim=True)
            .clamp(-1 + 1e-7, 1 - 1e-7)
        )
        theta = torch.acos(inner)

        # Compute log map with numerical stability
        u = y_reshaped - inner * x_reshaped
        u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Handle small angles
        factor = torch.where(
            theta.abs() > 1e-7, theta / torch.sin(theta), torch.ones_like(theta)
        )

        log_vec = factor * u / u_norm * u_norm  # Simplifies to factor * u

        return self.mask_and_reshape(initial_shape, log_vec)

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        Compute geodesic distance on the sphere.

        dist(x, y) = arccos(<x, y>)

        Args:
            x: First point on sphere
            y: Second point on sphere
            keepdim: Whether to keep dimensions

        Returns:
            Geodesic distance
        """
        initial_shape = x.shape

        x_reshaped = self.reshape_and_mask(initial_shape, x)
        y_reshaped = self.reshape_and_mask(y.shape, y)

        # Compute inner product with clamping for numerical stability
        inner = (x_reshaped * y_reshaped).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)

        # Apply mask and compute distance
        dist = torch.acos(inner) * self.mask.squeeze(-1).to(x)

        # Sum over atoms dimension
        if not keepdim:
            dist = dist.sum(dim=-1)

        return dist

    def random_base(self, *size, dtype=None, device=None) -> torch.Tensor:
        """
        Sample random points from the positive orthant of the sphere.

        This is done by:
        1. Sampling from uniform Dirichlet distribution (simplex)
        2. Taking square root to map to sphere

        Args:
            size: Shape of output tensor
            dtype: Data type
            device: Device to place tensor

        Returns:
            Random points on masked sphere's positive orthant
        """
        assert (
            size[-1] == self.max_num_atoms * self.dim
        ), f"Last dimension must be {self.max_num_atoms * self.dim}"

        # Sample from uniform Dirichlet
        concentration = torch.ones(self.dim, dtype=dtype, device=device)
        d = torch.distributions.Dirichlet(concentration, validate_args=False)

        # Sample for each atom
        shape = d._extended_shape((*size[:-1], self.max_num_atoms))
        concentration_expanded = d.concentration.expand(shape)
        simplex_samples = torch._sample_dirichlet(concentration_expanded)

        # Map to sphere via square root
        sphere_samples = simplex_samples.sqrt()

        # Apply mask and reshape
        return self.mask_and_reshape(size, sphere_samples)

    random = random_base

    @staticmethod
    def preprocess(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Transform simplex points to sphere points.

        Args:
            x: Points on simplex
            eps: Small value for numerical stability

        Returns:
            Points on sphere (positive orthant)
        """
        return (x.clamp(min=eps)).sqrt()

    @staticmethod
    def postprocess(x: torch.Tensor) -> torch.Tensor:
        """
        Transform sphere points to simplex points.

        Args:
            x: Points on sphere (positive orthant)

        Returns:
            Points on simplex
        """
        return x.square()

    def base_logprob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability for base distribution on sphere.

        This accounts for the change of variables from simplex to sphere.

        Args:
            x: Points on sphere

        Returns:
            Log probability
        """
        initial_shape = x.shape
        x_reshaped = self.reshape_and_mask(initial_shape, x)

        # For uniform distribution on simplex transformed to sphere
        # We need the Jacobian of the transformation
        n_class = self.dim

        # Log probability includes:
        # 1. Log gamma term from Dirichlet
        # 2. Jacobian from sqrt transformation: prod(1/(2*sqrt(x_i))) = 2^(-n) * prod(x_i^(-1/2))
        log_prob_per_atom = torch.lgamma(
            torch.tensor(n_class, dtype=x.dtype, device=x.device)
        )

        # Add Jacobian contribution (for active atoms only)
        x_masked = (
            x_reshaped + (1 - self.mask.to(x)) * 1.0
        )  # Avoid log(0) for masked atoms
        jacobian_term = -0.5 * torch.log(x_masked.clamp(min=1e-8)).sum(
            dim=-1
        ) - n_class * math.log(2)

        # Apply mask and sum
        masked_log_prob = self.mask.squeeze(-1).to(x) * (
            log_prob_per_atom + jacobian_term
        )
        return masked_log_prob.sum(dim=-1)

    def metric_normalized(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Normalize tangent vector by metric tensor.

        For sphere with standard metric, this is identity.

        Args:
            x: Base point
            u: Tangent vector

        Returns:
            Normalized tangent vector
        """
        return u

    def logdetG(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log determinant of metric tensor.

        For unit sphere with standard metric, this is zero.

        Args:
            x: Point on manifold

        Returns:
            Log determinant (zeros)
        """
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

    def extra_repr(self) -> str:
        """String representation."""
        return f"""
        dim={self.dim},
        num_atoms={self.num_atoms},
        max_num_atoms={self.max_num_atoms},
        active_atoms={int(self.mask.sum().item())}
        """
