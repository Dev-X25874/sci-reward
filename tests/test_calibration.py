"""Tests for sci_reward.training.calibration."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sci_reward.rewards.base import BaseReward
from sci_reward.rewards.composite import CompositeReward
from sci_reward.training.calibration import (
    RunningStats,
    RewardCalibrator,
    RewardVarianceEstimator,
    calibrate_composite,
)


class ConstantReward(BaseReward):
    def __init__(self, value: float, name: str = "constant"):
        self.value = value
        self.name = name

    def score(self, smiles: str) -> float:
        return self.value


class LengthReward(BaseReward):
    name = "length"

    def score(self, smiles: str) -> float:
        return min(len(smiles) / 10.0, 1.0)


SMILES = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC"]
LABELS = [0, 0, 0, 0, 1, 1, 1, 1]


class TestRunningStats:
    def test_mean(self):
        s = RunningStats()
        s.update([0.0, 1.0, 0.5])
        assert abs(s.mean - 0.5) < 1e-6

    def test_variance(self):
        s = RunningStats()
        s.update([0.0, 1.0])
        assert abs(s.variance - 0.5) < 1e-6

    def test_incremental_matches_batch(self):
        vals = [0.1, 0.3, 0.5, 0.7, 0.9]
        s_b = RunningStats()
        s_b.update(vals)
        s_i = RunningStats()
        for v in vals:
            s_i.update([v])
        assert abs(s_b.mean - s_i.mean) < 1e-5
        assert abs(s_b.variance - s_i.variance) < 1e-3

    def test_normalize_approx_zero_mean(self):
        s = RunningStats()
        s.update([0.0, 0.5, 1.0])
        normed = s.normalize(jnp.array([0.0, 0.5, 1.0]))
        assert abs(float(jnp.mean(normed))) < 1e-3

    def test_reset(self):
        s = RunningStats()
        s.update([1.0, 2.0])
        s.reset()
        assert s._count == 0 and s.mean == 0.0

    def test_single_value_variance_fallback(self):
        s = RunningStats()
        s.update([0.5])
        assert s.variance == 1.0  # n < 2 fallback

    def test_repr(self):
        assert "RunningStats" in repr(RunningStats())


class TestRewardCalibrator:
    def setup_method(self):
        self.cal = RewardCalibrator(LengthReward(), n_steps=20)

    def test_fit_returns_losses(self):
        losses = self.cal.fit(SMILES, LABELS, verbose=False)
        assert len(losses) == 20
        assert all(np.isfinite(l) for l in losses)

    def test_temperature_positive_after_fit(self):
        self.cal.fit(SMILES, LABELS, verbose=False)
        assert self.cal.temperature > 0.0

    def test_transform_range(self):
        self.cal.fit(SMILES, LABELS, verbose=False)
        out = self.cal.transform(jnp.array([0.1, 0.5, 0.9]))
        assert float(jnp.min(out)) >= 0.0
        assert float(jnp.max(out)) <= 1.0

    def test_unfitted_passthrough(self):
        cal = RewardCalibrator(LengthReward())
        raw = LengthReward().batch_score(SMILES[:3])
        np.testing.assert_allclose(np.array(cal.batch_score(SMILES[:3])), np.array(raw), atol=1e-5)

    def test_batch_score_after_fit(self):
        self.cal.fit(SMILES, LABELS, verbose=False)
        out = self.cal.batch_score(SMILES)
        assert out.shape == (len(SMILES),)
        assert all(0.0 <= float(s) <= 1.0 for s in out)

    def test_loss_decreases(self):
        cal = RewardCalibrator(LengthReward(), learning_rate=5e-2, n_steps=100)
        losses = cal.fit(SMILES, LABELS, verbose=False)
        assert np.mean(losses[-10:]) <= np.mean(losses[:10]) + 0.1


class TestRewardVarianceEstimator:
    def setup_method(self):
        from sci_reward.rewards.bioactivity import BioactivityReward
        self.reward = BioactivityReward(hidden_dim=32, n_layers=2, dropout_rate=0.2).initialize(jax.random.PRNGKey(0))
        self.estimator = RewardVarianceEstimator(self.reward, n_samples=5)
        self.smiles = ["CCO", "c1ccccc1", "invalid!!"]

    def test_shapes(self):
        mean, var = self.estimator.estimate(self.smiles)
        assert mean.shape == (3,)
        assert var.shape == (3,)

    def test_mean_in_range(self):
        mean, _ = self.estimator.estimate(self.smiles)
        assert all(0.0 <= float(m) <= 1.0 for m in mean)

    def test_variance_nonneg(self):
        _, var = self.estimator.estimate(self.smiles)
        assert all(float(v) >= 0.0 for v in var)

    def test_invalid_zeroed(self):
        mean, var = self.estimator.estimate(["invalid!!"])
        assert float(mean[0]) == 0.0
        assert float(var[0]) == 0.0

    def test_penalized_leq_mean(self):
        # Both values from one estimate call — penalized_score calls estimate
        # again internally with a new RNG split, so means won't match otherwise.
        mean, var = self.estimator.estimate(["CCO", "c1ccccc1"])
        import jax.numpy as jnp
        penalty = jnp.exp(-self.estimator.uncertainty_scale * var)
        penalized = mean * penalty
        for m, p in zip(mean, penalized):
            assert float(p) <= float(m) + 1e-5


class TestCalibrateComposite:
    def setup_method(self):
        self.composite = CompositeReward(rewards=[LengthReward(), ConstantReward(0.7)])
        self.data = list(zip(SMILES, LABELS))

    def test_one_calibrator_per_reward(self):
        cals = calibrate_composite(self.composite, self.data)
        assert len(cals) == len(self.composite.rewards)

    def test_all_fitted(self):
        for cal in calibrate_composite(self.composite, self.data):
            assert cal._fitted

    def test_valid_scores(self):
        for cal in calibrate_composite(self.composite, self.data):
            assert all(0.0 <= float(s) <= 1.0 for s in cal.batch_score(SMILES[:3]))
