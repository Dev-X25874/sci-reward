"""Multi-objective reward aggregation."""

from __future__ import annotations

from typing import Literal, Sequence

import chex
import jax
import jax.numpy as jnp

from sci_reward.rewards.base import BaseReward


AggregationMode = Literal["weighted", "pareto", "product", "min"]


class CompositeReward:
    """
    Compose multiple BaseReward instances into a single scalar reward.

    Args:
        rewards:   Reward functions to compose.
        weights:   Per-reward weights (auto-normalized). Uniform if None.
        mode:      Aggregation mode: "weighted", "product", "min", or "pareto".
        gate:      Optional hard gate — if gate.score returns 0.0, the composite
                   score is 0.0 regardless of other rewards. Use ValiditySMILES()
                   to prevent any reward signal on invalid molecules.
        normalize: Min-max normalize each reward column across the batch
                   before aggregation. Useful when reward scales differ widely.
    """

    def __init__(
        self,
        rewards: Sequence[BaseReward],
        weights: Sequence[float] | None = None,
        mode: AggregationMode = "weighted",
        gate: BaseReward | None = None,
        normalize: bool = False,
    ):
        if not rewards:
            raise ValueError("rewards list cannot be empty")
        self.rewards = list(rewards)
        self.mode = mode
        self.gate = gate
        self.normalize = normalize

        n = len(rewards)
        if weights is None:
            self._weights = jnp.ones(n, dtype=jnp.float32) / n
        else:
            if len(weights) != n:
                raise ValueError(f"len(weights)={len(weights)} must equal len(rewards)={n}")
            w = jnp.array(weights, dtype=jnp.float32)
            self._weights = w / w.sum()

    @property
    def weights(self) -> chex.Array:
        return self._weights

    def score_matrix(self, smiles_list: Sequence[str]) -> chex.Array:
        """Per-reward scores. Returns (N, K) float32 array."""
        matrix = jnp.stack([r.batch_score(smiles_list) for r in self.rewards], axis=1)
        return _normalize_columns(matrix) if self.normalize else matrix

    def gate_mask(self, smiles_list: Sequence[str]) -> chex.Array:
        """Boolean mask (N,). True = molecule passes the gate."""
        if self.gate is None:
            return jnp.ones(len(smiles_list), dtype=jnp.bool_)
        return self.gate.batch_score(smiles_list) > 0.5

    def aggregate(self, score_matrix: chex.Array) -> chex.Array:
        """Aggregate (N, K) score matrix to (N,) scalar rewards. Jit-able."""
        if self.mode == "weighted":
            return jnp.dot(score_matrix, self._weights)
        if self.mode == "product":
            return jnp.prod(score_matrix, axis=1)
        if self.mode == "min":
            return jnp.min(score_matrix, axis=1)
        if self.mode == "pareto":
            return _pareto_scalarize(score_matrix, self._weights)
        raise ValueError(f"Unknown aggregation mode: {self.mode!r}")

    def batch_score(self, smiles_list: Sequence[str]) -> chex.Array:
        """Full pipeline: score -> aggregate -> apply gate. Returns (N,) float32."""
        scores = self.aggregate(self.score_matrix(smiles_list))
        if self.gate is not None:
            scores = jnp.where(self.gate_mask(smiles_list), scores, 0.0)
        return scores

    def explain(self, smiles: str) -> dict[str, float]:
        """Per-reward breakdown for a single molecule."""
        result = {r.name: float(r.score(smiles)) for r in self.rewards}
        if self.gate is not None:
            result[f"gate({self.gate.name})"] = float(self.gate.score(smiles))
        matrix = jnp.array([[result[r.name] for r in self.rewards]])
        result["composite"] = float(self.aggregate(matrix)[0])
        return result

    def __repr__(self) -> str:
        return f"CompositeReward(rewards={[r.name for r in self.rewards]}, mode={self.mode!r}, gate={self.gate})"


@jax.jit
def _normalize_columns(matrix: chex.Array) -> chex.Array:
    """Min-max normalize each column. Zero-range columns are left unchanged."""
    col_min = jnp.min(matrix, axis=0, keepdims=True)
    col_max = jnp.max(matrix, axis=0, keepdims=True)
    rng = col_max - col_min
    safe_rng = jnp.where(rng < 1e-8, 1.0, rng)
    return jnp.where(rng < 1e-8, matrix, (matrix - col_min) / safe_rng)


@jax.jit
def _pareto_scalarize(score_matrix: chex.Array, weights: chex.Array) -> chex.Array:
    """
    Pareto-aware scalarization.

    Penalizes dominated molecules: final_i = weighted_i * (1 - dom_fraction_i),
    where dom_fraction_i = fraction of batch members that dominate molecule i.
    Incentivizes diversity across objectives rather than collapsing to one.
    """
    N = score_matrix.shape[0]
    weighted = jnp.dot(score_matrix, weights)

    si = score_matrix[:, None, :]  # (N, 1, K)
    sj = score_matrix[None, :, :]  # (1, N, K)
    dominates = jnp.all(sj >= si, axis=-1) & jnp.any(sj > si, axis=-1)  # (N, N)
    dom_fraction = jnp.sum(dominates, axis=1).astype(jnp.float32) / N

    return weighted * (1.0 - dom_fraction)
