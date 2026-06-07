"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch
from geoopt import Euclidean
from geoopt.utils import size2shape


class EuclideanWithLogProb(Euclidean):
    def random_normal(
        self, *size, mean=0.0, std=1.0, device=None, dtype=None
    ) -> torch.Tensor:
        self._assert_check_shape(size2shape(*size), "x")
        mean = torch.as_tensor(mean, device=device, dtype=dtype)
        std = torch.as_tensor(std, device=mean.device, dtype=mean.dtype)
        return torch.randn(*size, device=mean.device, dtype=mean.dtype) * std + mean

    def random_base(self, *args, **kwargs):
        return self.random_normal(*args, **kwargs)

    random = random_base

    def normal_logprob(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.ndim == 0:
            raise NotImplementedError()
        else:
            dist = torch.distributions.normal.Normal(
                torch.zeros_like(x[-self.ndim :]),
                torch.ones_like(x[-self.ndim :]),
                validate_args=False,  # doesn't work with vmap yet https://github.com/pytorch/functorch/issues/257
            )
            dist = torch.distributions.independent.Independent(dist, 1)
        return dist.log_prob(x)

    def base_logprob(self, *args, **kwargs) -> torch.Tensor:
        return self.normal_logprob(*args, **kwargs)

    def logdetG(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(x).sum(-1)


from dmflow.rfm.manifolds.masked import MaskedManifold


class FlatEuclidean(Euclidean):
    """
    A flat Euclidean space baseline for comparison with simplex manifolds.
    This allows unconstrained flow in Euclidean space without simplex constraints
    during intermediate steps, only applying normalization at the end.
    """

    name = "FlatEuclideanBaseline"
    reversible = True

    def __init__(self):
        """Initialize the flat Euclidean baseline manifold."""
        super().__init__(ndim=1)

    @staticmethod
    def postprocess(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        """
        Alternative post-processing method using direct normalization.

        This method directly divides by the sum, which may be less numerically
        stable than softmax but preserves the original scale relationships.

        Args:
            x: Input tensor to be normalized
            dim: Dimension along which to normalize (default: -1)
            eps: Small epsilon value to prevent division by zero (default: 1e-8)

        Returns:
            Tensor normalized by sum along the specified dimension
        """
        # Clamp to ensure non-negative values
        x_pos = torch.clamp(x, min=0.0)
        # Add epsilon to prevent division by zero
        x_sum = x_pos.sum(dim=dim, keepdim=True) + eps
        return x_pos / x_sum

    def random_base(self, *size, dtype=None, device=None) -> torch.Tensor:
        """
        Generate random samples from the base distribution.

        This generates samples from a standard normal distribution,
        which will later be post-processed to obtain probability distributions.

        Args:
            *size: Shape of the tensor to generate
            dtype: Data type of the tensor
            device: Device to place the tensor on

        Returns:
            Random tensor sampled from standard normal distribution
        """
        return torch.randn(*size, dtype=dtype, device=device)

    random = random_base

    def base_logprob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability under the base distribution.

        Since we use a standard normal base distribution, this computes
        the log probability of a multivariate standard normal.

        Args:
            x: Input tensor

        Returns:
            Log probability under standard normal distribution
        """
        # Standard normal log probability: -0.5 * ||x||^2 - 0.5 * d * log(2π)
        d = x.shape[-1]
        log_prob = -0.5 * torch.sum(x**2, dim=-1) - 0.5 * d * torch.log(
            2 * torch.pi * torch.ones_like(x[..., 0])
        )
        return log_prob


class MultiAtomFlatEuclidean(MaskedManifold, FlatEuclidean):
    """
    Multi-atom flat Euclidean baseline with masking support.

    This class combines the flat Euclidean baseline with masking capabilities
    for handling variable numbers of atoms. It maintains the unconstrained
    nature during flow while supporting the masked atom structure.

    The manifold has shape [..., NUM_ATOMS, NUM_CATEGORIES] conceptually,
    but is flattened to [..., NUM_ATOMS * NUM_CATEGORIES] for processing.
    """

    name = "MultiAtomFlatEuclidean"
    reversible = True

    def __init__(self, num_categories: int, num_atoms: int, max_num_atoms: int):
        """
        Initialize the multi-atom flat Euclidean baseline.

        Args:
            num_categories: Number of categories per atom
            num_atoms: Actual number of atoms
            max_num_atoms: Maximum number of atoms (for padding/masking)
        """
        FlatEuclidean.__init__(self)
        self.max_num_atoms = max_num_atoms
        self.num_categories = num_categories
        self.num_atoms = num_atoms
        self.register_buffer("mask", self._get_mask(num_atoms, max_num_atoms))

    @property
    def dim_m1(self) -> int:
        """Return the number of categories (inner dimension)."""
        return self.num_categories

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        """
        Compute the inner product on the manifold.

        This computes the standard Euclidean inner product but normalizes
        by the number of valid (unmasked) atoms.

        Args:
            x: First tensor (base point, not used in Euclidean space)
            u: First vector
            v: Second vector (if None, computes ||u||^2)
            keepdim: Whether to keep dimensions

        Returns:
            Inner product normalized by number of valid atoms
        """
        return Euclidean(ndim=self.ndim).inner(
            x, u, v, keepdim=keepdim
        ) / self.mask.sum().to(u)

    def random_base(self, *size, dtype=None, device=None) -> torch.Tensor:
        """
        Generate random samples for the multi-atom setting.

        This generates random normal samples with the correct shape
        and applies masking.

        Args:
            *size: Shape tuple, last dimension should be max_num_atoms * num_categories
            dtype: Data type
            device: Device to place tensor on

        Returns:
            Random tensor with proper masking applied
        """
        assert (
            size[-1] == self.max_num_atoms * self.num_categories
        ), "Last dimension must be compatible with max_num_atoms and num_categories"

        # Generate random normal samples
        x = torch.randn(*size, dtype=dtype, device=device)

        # Apply masking
        initial_shape = x.shape
        x_reshaped = self.reshape_and_mask(initial_shape, x)
        return self.mask_and_reshape(initial_shape, x_reshaped)

    random = random_base

    def base_logprob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under the base distribution.

        This computes the standard normal log probability for the
        unmasked entries only.

        Args:
            x: Input tensor

        Returns:
            Log probability under masked standard normal distribution
        """
        initial_shape = x.shape
        x_reshaped = self.reshape_and_mask(initial_shape, x)

        # Compute log probability for valid (masked) entries only
        valid_entries = self.mask.to(x_reshaped).sum()
        log_prob = -0.5 * torch.sum(x_reshaped**2, dim=(-2, -1))
        log_prob -= (
            0.5 * valid_entries * torch.log(2 * torch.pi * torch.ones_like(log_prob))
        )

        return log_prob

    def extra_repr(self):
        """String representation of the manifold parameters."""
        return f"""
        num_atoms={self.num_atoms},
        max_num_atoms={self.max_num_atoms}, 
        num_categories={self.num_categories},
        """

    def metric_normalized(self, _: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Return the metric-normalized vector (identity for Euclidean space).

        Args:
            _: Base point (unused in Euclidean space)
            u: Vector to normalize

        Returns:
            The input vector u (no normalization needed in Euclidean space)
        """
        return u

    def logdetG(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log determinant of the metric tensor.

        For Euclidean space, the metric tensor is identity, so log determinant is 0.

        Args:
            x: Input tensor

        Returns:
            Zeros (log determinant of identity matrix)
        """
        return torch.zeros_like(x[..., 0])
