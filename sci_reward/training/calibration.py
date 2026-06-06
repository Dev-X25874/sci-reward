"""Reward calibration and variance estimation.

RunningStats
    Online Welford mean/variance accumulator used to normalize reward signals
    to zero mean, unit variance during RL training.

    Properties mean, variance, and std always return Python float (never
    numpy.float64). normalize() always returns a JAX DeviceArray regardless
    of whether the input is a list, numpy array, or JAX array.

RewardCalibrator
    Temperature-scaling calibrator that learns scalar T and bias b to
    minimize binary cross-entropy on labeled calibration data:
        calibrated = sigmoid(logit(raw_score) / T + b)

RewardVarianceEstimator
    MC-Dropout epistemic uncertainty estimator for BioactivityReward.
    Runs n_samples forward passes with dropout active and returns
    per-molecule mean and variance.
"""

from __future__ import annotations

from typing import Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax

from sci_reward.rewards.base import BaseReward
from sci_reward.rewards.composite import CompositeReward


class RunningStats:
    """
    Online mean/variance via Welford's algorithm.

    Thread-unsafe by design (single-process RL loop).

    All three scalar properties (mean, variance, std) return Python float.
    normalize() always returns a JAX DeviceArray, accepting list, numpy
    array, or JAX array as input.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self._count: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0

    def update(self, values: np.ndarray | list[float]) -> None:
        for v in np.asarray(values, dtype=np.float64).ravel():
            self._count += 1
            delta = v - self._mean
            self._mean += delta / self._count
            self._M2 += delta * (v - self._mean)

    @property
    def mean(self) -> float:
        return float(self._mean)

    @property
    def variance(self) -> float:
        return float(self._M2 / (self._count - 1)) if self._count >= 2 else 1.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance + self.epsilon))

    def normalize(self, values: chex.Array) -> chex.Array:
        """Return (values - mean) / std as a JAX DeviceArray."""
        return (jnp.asarray(values) - self.mean) / self.std

    def reset(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._M2 = 0.0

    def __repr__(self) -> str:
        return (
            f"RunningStats(count={self._count}, "
            f"mean={self._mean:.4f}, std={self.std:.4f})"
        )


class RewardCalibrator:
    """
    Temperature scaling calibrator for reward functions.

    Learns scalar T and bias b to minimize BCE on labeled calibration data:
        calibrated = sigmoid(logit(raw_score) / T + b)

    temperature and bias properties always return Python float.
    """

    def __init__(
        self,
        reward: BaseReward,
        init_temperature: float = 1.0,
        init_bias: float = 0.0,
        learning_rate: float = 1e-2,
        n_steps: int = 200,
    ):
        self.reward = reward
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self._temperature = float(init_temperature)
        self._bias = float(init_bias)
        self._fitted = False

    @staticmethod
    def _loss(
        params: chex.Array, logits: chex.Array, labels: chex.Array
    ) -> chex.Array:
        calibrated = logits / jnp.maximum(params[0], 1e-3) + params[1]
        return jnp.mean(optax.sigmoid_binary_cross_entropy(calibrated, labels))

    def fit(
        self,
        smiles_list: Sequence[str],
        labels: Sequence[float],
        verbose: bool = False,
    ) -> list[float]:
        """Fit temperature and bias on labeled calibration data. Returns loss history."""
        raw = np.clip(
            np.array([self.reward.score(s) for s in smiles_list], dtype=np.float32),
            1e-6,
            1 - 1e-6,
        )
        logits = jnp.array(np.log(raw / (1 - raw)))
        targets = jnp.array(labels, dtype=jnp.float32)
        params = jnp.array([self._temperature, self._bias])

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)
        grad_fn = jax.jit(jax.value_and_grad(self._loss))
        losses = []

        for step in range(self.n_steps):
            loss, grads = grad_fn(params, logits, targets)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))
            if verbose and step % 50 == 0:
                print(
                    f"Step {step:3d} | loss={float(loss):.4f} "
                    f"| T={float(params[0]):.3f} | b={float(params[1]):.3f}"
                )

        self._temperature = float(params[0])
        self._bias = float(params[1])
        self._fitted = True
        return losses

    def transform(self, scores: chex.Array) -> chex.Array:
        s = np.clip(np.array(scores), 1e-6, 1 - 1e-6)
        logits = jnp.array(np.log(s / (1 - s)))
        return jax.nn.sigmoid(logits / max(self._temperature, 1e-3) + self._bias)

    def batch_score(self, smiles_list: Sequence[str]) -> chex.Array:
        raw = self.reward.batch_score(smiles_list)
        return self.transform(raw) if self._fitted else raw

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def bias(self) -> float:
        return self._bias


class RewardVarianceEstimator:
    """
    Epistemic uncertainty via MC Dropout on BioactivityReward.

    Runs the model n_samples times with dropout active and returns
    mean score and variance. High-variance molecules are out-of-distribution.
    """

    def __init__(
        self,
        reward,
        n_samples: int = 20,
        uncertainty_scale: float = 2.0,
        rng_seed: int = 42,
    ):
        self.reward = reward
        self.n_samples = n_samples
        self.uncertainty_scale = uncertainty_scale
        self._rng = jax.random.PRNGKey(rng_seed)

    def estimate(self, smiles_list: Sequence[str]) -> tuple[chex.Array, chex.Array]:
        """Returns (mean_scores, variances), each (N,) float32."""
        from sci_reward.rewards.bioactivity import batch_fingerprints

        fps, valid = batch_fingerprints(smiles_list)
        fps_jnp = jnp.array(fps)

        samples = []
        for _ in range(self.n_samples):
            self._rng, subkey = jax.random.split(self._rng)
            logits = self.reward.model.apply(
                {"params": self.reward.params},
                fps_jnp,
                training=True,
                rngs={"dropout": subkey},
            )
            samples.append(jax.nn.sigmoid(logits))

        stack = jnp.stack(samples, axis=0)  # (n_samples, N)
        valid_mask = jnp.array(valid, dtype=jnp.float32)
        return (
            jnp.mean(stack, axis=0) * valid_mask,
            jnp.var(stack, axis=0) * valid_mask,
        )

    def penalized_score(self, smiles_list: Sequence[str]) -> chex.Array:
        """mean * exp(-scale * variance)."""
        mean, var = self.estimate(smiles_list)
        return mean * jnp.exp(-self.uncertainty_scale * var)


def calibrate_composite(
    composite: CompositeReward,
    calibration_data: list[tuple[str, float]],
    verbose: bool = False,
) -> list[RewardCalibrator]:
    """Fit a RewardCalibrator for each component reward in a CompositeReward."""
    smiles = [d[0] for d in calibration_data]
    labels = [d[1] for d in calibration_data]
    calibrators = []
    for reward in composite.rewards:
        if verbose:
            print(f"Calibrating {reward.name}...")
        cal = RewardCalibrator(reward)
        cal.fit(smiles, labels, verbose=verbose)
        if verbose:
            print(f"  T={cal.temperature:.3f}, b={cal.bias:.3f}")
        calibrators.append(cal)
    return calibrators
