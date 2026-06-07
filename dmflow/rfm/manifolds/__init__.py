"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from dmflow.rfm.manifolds.analog_bits import MultiAtomAnalogBits
from dmflow.rfm.manifolds.euclidean import EuclideanWithLogProb
from dmflow.rfm.manifolds.flat_torus import (
    FlatTorus01FixFirstAtomToOrigin,
    FlatTorus01FixFirstAtomToOriginWrappedNormal,
)
from dmflow.rfm.manifolds.null import NullManifoldWithDeltaRandom
from dmflow.rfm.manifolds.product import ProductManifoldWithLogProb
from dmflow.rfm.manifolds.simplex import (
    FlatDirichletSimplex,
    MultiAtomFlatDirichletSimplex,
)
