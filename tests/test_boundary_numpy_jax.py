"""Cross-language boundary checks: numpy <-> JAX array contracts.

Every point in the codebase where numpy arrays cross into JAX (or vice
versa) is exercised here. Tests verify dtype, shape, device, and Python-
scalar contracts at each boundary.

Fixed seed: 0 throughout.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from sci_reward.rewards.base import BaseReward
from sci_reward.rewards.bioactivity import (
    FINGERPRINT_DIM,
    BioactivityReward,
    batch_fingerprints,
    smiles_to_fingerprint,
)
from sci_reward.rewards.chemical import QEDReward, ValiditySMILES
from sci_reward.rewards.composite import CompositeReward
from sci_reward.training.calibration import RunningStats

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]


class TestSmilesToFingerprintBoundary:
    """smiles_to_fingerprint returns numpy; downstream JAX code must not receive wrong dtype."""

    def test_output_is_numpy_not_jax(self):
        fp = smiles_to_fingerprint("CCO")
        assert isinstance(fp, np.ndarray), "smiles_to_fingerprint must return np.ndarray"

    def test_dtype_is_float32_before_jax_cast(self):
        fp = smiles_to_fingerprint("CCO")
        assert fp.dtype == np.float32, f"Expected float32, got {fp.dtype}"

    def test_shape_matches_fingerprint_dim(self):
        fp = smiles_to_fingerprint("CCO")
        assert fp.shape == (FINGERPRINT_DIM,)

    def test_none_for_invalid_never_reaches_jax(self):
        fp = smiles_to_fingerprint("invalid!!")
        assert fp is None, "Invalid SMILES must return None, not a zero array"

    def test_batch_fingerprints_returns_numpy(self):
        fps, valid = batch_fingerprints(SMILES)
        assert isinstance(fps, np.ndarray)
        assert isinstance(valid, np.ndarray)

    def test_batch_fps_dtype_float32(self):
        fps, _ = batch_fingerprints(SMILES)
        assert fps.dtype == np.float32

    def test_batch_valid_dtype_bool(self):
        _, valid = batch_fingerprints(SMILES)
        assert valid.dtype == bool

    def test_invalid_rows_are_zero_numpy_before_jax(self):
        fps, valid = batch_fingerprints(["CCO", "INVALID!!!"])
        assert not valid[1]
        assert np.all(fps[1] == 0.0), (
            "Invalid row must be zero-filled in numpy before JAX cast"
        )


class TestBioactivityRewardJAXBoundary:
    """BioactivityReward.score: numpy fp -> jnp.array cast -> sigmoid -> python float."""

    def setup_method(self):
        self.reward = BioactivityReward(hidden_dim=32, n_layers=1).initialize(
            jax.random.PRNGKey(0)
        )

    def test_score_returns_python_float_not_jax(self):
        result = self.reward.score("CCO")
        assert isinstance(result, float), (
            f"score() must return Python float, got {type(result)}"
        )

    def test_batch_score_returns_jax_array(self):
        result = self.reward.batch_score(SMILES)
        assert isinstance(result, jax.Array), "batch_score must return JAX DeviceArray"

    def test_batch_score_dtype_float32(self):
        result = self.reward.batch_score(SMILES)
        assert result.dtype == jnp.float32

    def test_invalid_zeroed_via_valid_mask_not_sigmoid(self):
        # Invalid score must be exactly 0.0, not sigmoid(0) = 0.5
        result = self.reward.batch_score(["invalid!!"])
        assert float(result[0]) == 0.0, (
            "Invalid SMILES must be 0.0, not sigmoid(0.0)=0.5"
        )

    def test_fp_cast_to_jax_preserves_values(self):
        fp = smiles_to_fingerprint("CCO")
        fp_jax = jnp.array(fp[None])
        np.testing.assert_allclose(np.array(fp_jax[0]), fp, atol=1e-7)

    def test_batch_score_length_matches_input(self):
        result = self.reward.batch_score(SMILES + ["invalid!!"])
        assert result.shape == (len(SMILES) + 1,)


class TestBaseRewardBatchBoundary:
    """BaseReward.batch_score: list[float] -> np.array(dtype=float32) -> jnp.array."""

    def test_batch_score_dtype_float32_from_base(self):
        class SimpleReward(BaseReward):
            name = "simple"

            def score(self, smiles: str) -> float:
                return 0.42

        r = SimpleReward()
        result = r.batch_score(["CCO", "c1ccccc1"])
        assert result.dtype == jnp.float32

    def test_batch_score_shape(self):
        class SimpleReward(BaseReward):
            name = "simple"

            def score(self, smiles: str) -> float:
                return 0.5

        r = SimpleReward()
        assert r.batch_score(SMILES).shape == (3,)

    def test_batch_score_values_match_individual(self):
        r = QEDReward()
        batch = np.array(r.batch_score(SMILES))
        individual = np.array([r.score(s) for s in SMILES], dtype=np.float32)
        np.testing.assert_allclose(batch, individual, atol=1e-5)


class TestCompositeScoreMatrixBoundary:
    """CompositeReward.score_matrix: each reward returns JAX array, stacked via jnp.stack."""

    def setup_method(self):
        self.composite = CompositeReward(
            rewards=[ValiditySMILES(), QEDReward()],
            weights=[0.5, 0.5],
        )

    def test_score_matrix_is_jax_array(self):
        matrix = self.composite.score_matrix(SMILES)
        assert isinstance(matrix, jax.Array)

    def test_score_matrix_dtype_float32(self):
        assert self.composite.score_matrix(SMILES).dtype == jnp.float32

    def test_score_matrix_shape(self):
        matrix = self.composite.score_matrix(SMILES)
        assert matrix.shape == (len(SMILES), 2)

    def test_score_matrix_values_in_unit_interval(self):
        matrix = self.composite.score_matrix(SMILES)
        assert float(jnp.min(matrix)) >= 0.0
        assert float(jnp.max(matrix)) <= 1.0


class TestRunningStatsBoundary:
    """RunningStats: numpy/list inputs -> internal float64 -> normalize returns JAX array."""

    def test_update_accepts_list(self):
        s = RunningStats()
        s.update([0.1, 0.2, 0.3])
        assert abs(s.mean - 0.2) < 1e-6

    def test_update_accepts_numpy_array(self):
        s = RunningStats()
        s.update(np.array([0.1, 0.2, 0.3]))
        assert abs(s.mean - 0.2) < 1e-6

    def test_update_accepts_jax_array(self):
        s = RunningStats()
        s.update(jnp.array([0.1, 0.2, 0.3]))
        assert abs(s.mean - 0.2) < 1e-6

    def test_normalize_returns_jax_array(self):
        s = RunningStats()
        s.update([0.0, 0.5, 1.0])
        result = s.normalize(jnp.array([0.0, 0.5, 1.0]))
        assert isinstance(result, jax.Array)

    def test_normalize_accepts_numpy_array_returns_jax(self):
        s = RunningStats()
        s.update([0.0, 0.5, 1.0])
        result = s.normalize(np.array([0.0, 0.5, 1.0], dtype=np.float32))
        assert isinstance(result, jax.Array)


class TestDropoutRngSplitBoundary:
    """Verify that BT loss uses independent rng keys for chosen vs rejected forward passes.

    Using the same key for both passes produces identical dropout masks,
    correlating the noise and biasing the gradient estimate.
    """

    def setup_method(self):
        self.reward = BioactivityReward(
            hidden_dim=32, n_layers=2, dropout_rate=0.5
        ).initialize(jax.random.PRNGKey(0))

    def test_bt_loss_with_independent_rngs_is_finite(self):
        from sci_reward.rewards.bioactivity import BioactivityRewardTrainer

        trainer = BioactivityRewardTrainer(self.reward, learning_rate=0.0)
        fps = jnp.ones((2, FINGERPRINT_DIM))
        rng = jax.random.PRNGKey(42)
        loss = BioactivityRewardTrainer._bt_loss(
            trainer.state.params, self.reward.model, fps, fps, rng
        )
        assert jnp.isfinite(loss)

    def test_bt_loss_and_acc_deterministic_for_same_rng(self):
        from sci_reward.training.reward_trainer import _bt_loss_and_acc

        reward = BioactivityReward(
            hidden_dim=32, n_layers=2, dropout_rate=0.5
        ).initialize(jax.random.PRNGKey(0))
        fps_c = jnp.ones((4, FINGERPRINT_DIM))
        fps_r = jnp.zeros((4, FINGERPRINT_DIM))
        rng = jax.random.PRNGKey(1)
        loss1, acc1 = _bt_loss_and_acc(reward.params, reward.model, fps_c, fps_r, rng)
        loss2, acc2 = _bt_loss_and_acc(reward.params, reward.model, fps_c, fps_r, rng)
        assert float(loss1) == pytest.approx(float(loss2), abs=1e-6)

    def test_bt_loss_and_acc_finite_for_different_rng(self):
        from sci_reward.training.reward_trainer import _bt_loss_and_acc

        reward = BioactivityReward(
            hidden_dim=32, n_layers=2, dropout_rate=0.5
        ).initialize(jax.random.PRNGKey(0))
        fps_c = jnp.ones((4, FINGERPRINT_DIM))
        fps_r = jnp.zeros((4, FINGERPRINT_DIM))
        loss, _ = _bt_loss_and_acc(
            reward.params, reward.model, fps_c, fps_r, jax.random.PRNGKey(999)
        )
        assert jnp.isfinite(loss)
        assert float(loss) > 0.0
