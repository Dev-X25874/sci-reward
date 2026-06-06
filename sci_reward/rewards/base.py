"""Base reward interface."""

from __future__ import annotations

import abc
from typing import Sequence

import chex
import jax.numpy as jnp
import numpy as np


class BaseReward(abc.ABC):
    """
    Abstract base for all sci-reward reward functions.

    The JAX boundary is explicit: RDKit/numpy work happens inside score(),
    and the result crosses into JAX only in batch_score() via jnp.array().
    This keeps jit/vmap clean without tracing through Python callbacks.
    """

    name: str = "base"

    @abc.abstractmethod
    def score(self, smiles: str) -> float:
        """Score a single SMILES string. Returns float in [0, 1]."""

    def batch_score(self, smiles_list: Sequence[str]) -> chex.Array:
        """Score a batch. Returns float32 DeviceArray of shape (N,)."""
        scores = np.array([self.score(s) for s in smiles_list], dtype=np.float32)
        return jnp.array(scores)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
