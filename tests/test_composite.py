"""Tests for sci_reward.rewards.composite."""

from __future__ import annotations

import pytest
import jax.numpy as jnp
import numpy as np

from sci_reward.rewards.base import BaseReward
from sci_reward.rewards.composite import (
    CompositeReward,
    _normalize_columns,
    _pareto_scalarize,
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


class BinaryReward(BaseReward):
    name = "binary"

    def score(self, smiles: str) -> float:
        return 1.0 if smiles == "valid" else 0.0


SMILES = ["CCO", "c1ccccc1", "CCCCCC"]


class TestCompositeRewardInit:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            CompositeReward(rewards=[])

    def test_weight_length_mismatch_raises(self):
        with pytest.raises(ValueError, match=r"len\(weights\)"):
            CompositeReward(rewards=[ConstantReward(0.5), ConstantReward(0.3)], weights=[0.5])

    def test_uniform_default_weights(self):
        c = CompositeReward(rewards=[ConstantReward(0.5), ConstantReward(0.3)])
        np.testing.assert_allclose(np.array(c.weights), [0.5, 0.5], atol=1e-6)

    def test_weights_normalized(self):
        c = CompositeReward(rewards=[ConstantReward(0.5), ConstantReward(0.3)], weights=[1.0, 3.0])
        np.testing.assert_allclose(np.array(c.weights), [0.25, 0.75], atol=1e-6)

    def test_repr_contains_names(self):
        c = CompositeReward(rewards=[ConstantReward(0.5, "r1"), ConstantReward(0.3, "r2")])
        assert "r1" in repr(c) and "r2" in repr(c)


class TestScoreMatrix:
    composite = CompositeReward(rewards=[ConstantReward(0.8, "r1"), LengthReward()])

    def test_shape(self):
        assert self.composite.score_matrix(SMILES).shape == (3, 2)

    def test_dtype(self):
        assert self.composite.score_matrix(SMILES).dtype == jnp.float32

    def test_constant_column(self):
        matrix = self.composite.score_matrix(SMILES)
        np.testing.assert_allclose(np.array(matrix[:, 0]), [0.8, 0.8, 0.8], atol=1e-5)

    def test_normalized_in_range(self):
        c = CompositeReward(rewards=[ConstantReward(0.8), LengthReward()], normalize=True)
        matrix = c.score_matrix(SMILES)
        assert float(jnp.min(matrix)) >= -1e-5
        assert float(jnp.max(matrix)) <= 1.0 + 1e-5


class TestAggregationModes:
    smiles = ["CCO", "c1ccccc1"]

    def _make(self, mode):
        return CompositeReward(
            rewards=[ConstantReward(0.8), ConstantReward(0.6)],
            weights=[0.5, 0.5],
            mode=mode,
        )

    def test_weighted(self):
        scores = self._make("weighted").batch_score(self.smiles)
        np.testing.assert_allclose(np.array(scores), [0.7, 0.7], atol=1e-5)

    def test_product(self):
        scores = self._make("product").batch_score(self.smiles)
        np.testing.assert_allclose(np.array(scores), [0.48, 0.48], atol=1e-5)

    def test_min(self):
        scores = self._make("min").batch_score(self.smiles)
        np.testing.assert_allclose(np.array(scores), [0.6, 0.6], atol=1e-5)

    def test_pareto_runs(self):
        scores = self._make("pareto").batch_score(self.smiles)
        assert scores.shape == (2,)
        assert all(0.0 <= float(s) <= 1.0 for s in scores)

    def test_invalid_mode_raises(self):
        c = CompositeReward(rewards=[ConstantReward(0.5)])
        c.mode = "bad_mode"
        with pytest.raises(ValueError, match="Unknown aggregation mode"):
            c.aggregate(jnp.ones((2, 1)))


class TestGate:
    def test_gate_blocks_invalid(self):
        c = CompositeReward(rewards=[ConstantReward(0.9)], gate=BinaryReward())
        scores = c.batch_score(["valid", "invalid"])
        assert float(scores[0]) > 0.0
        assert float(scores[1]) == 0.0

    def test_no_gate_passes_all(self):
        c = CompositeReward(rewards=[ConstantReward(0.5)])
        mask = c.gate_mask(["anything", "goes"])
        assert all(bool(m) for m in mask)

    def test_gate_mask_shape(self):
        c = CompositeReward(rewards=[ConstantReward(0.5)], gate=BinaryReward())
        assert c.gate_mask(["valid", "invalid", "valid"]).shape == (3,)


class TestExplain:
    composite = CompositeReward(
        rewards=[ConstantReward(0.7, "r1"), ConstantReward(0.3, "r2")],
        weights=[0.6, 0.4],
    )

    def test_keys_present(self):
        result = self.composite.explain("CCO")
        assert {"r1", "r2", "composite"} <= result.keys()

    def test_values(self):
        result = self.composite.explain("CCO")
        assert abs(result["r1"] - 0.7) < 1e-5
        assert abs(result["r2"] - 0.3) < 1e-5
        assert abs(result["composite"] - (0.6 * 0.7 + 0.4 * 0.3)) < 1e-5

    def test_gate_key(self):
        c = CompositeReward(
            rewards=[ConstantReward(0.7, "r1"), ConstantReward(0.3, "r2")],
            gate=BinaryReward(),
        )
        assert "gate(binary)" in c.explain("valid")


class TestNormalizeColumns:
    def test_range(self):
        m = jnp.array([[0.1, 0.5], [0.9, 0.2], [0.5, 0.8]])
        n = _normalize_columns(m)
        assert float(jnp.min(n[:, 0])) == pytest.approx(0.0, abs=1e-5)
        assert float(jnp.max(n[:, 0])) == pytest.approx(1.0, abs=1e-5)

    def test_constant_column_no_nan(self):
        m = jnp.array([[0.5, 0.3], [0.5, 0.7], [0.5, 0.5]])
        assert not jnp.any(jnp.isnan(_normalize_columns(m)))

    def test_shape_preserved(self):
        assert _normalize_columns(jnp.ones((5, 3))).shape == (5, 3)


class TestParetoScalarize:
    def test_output_shape(self):
        m = jnp.array([[0.8, 0.6], [0.5, 0.9], [0.3, 0.3]])
        assert _pareto_scalarize(m, jnp.array([0.5, 0.5])).shape == (3,)

    def test_dominated_penalized(self):
        m = jnp.array([[0.8, 0.6], [0.5, 0.9], [0.3, 0.3]], dtype=jnp.float32)
        w = jnp.array([0.5, 0.5])
        result = _pareto_scalarize(m, w)
        # Mol 2 dominated by both others -> score < naive weighted sum (0.3)
        assert float(result[2]) < 0.3 + 1e-5

    def test_dominator_gets_full_score(self):
        m = jnp.array([[1.0, 1.0], [0.5, 0.5], [0.3, 0.7]], dtype=jnp.float32)
        w = jnp.array([0.5, 0.5])
        assert float(_pareto_scalarize(m, w)[0]) == pytest.approx(1.0, abs=1e-5)

    def test_all_equal_no_penalty(self):
        m = jnp.array([[0.5, 0.5]] * 3, dtype=jnp.float32)
        w = jnp.array([0.5, 0.5])
        np.testing.assert_allclose(np.array(_pareto_scalarize(m, w)), [0.5, 0.5, 0.5], atol=1e-5)
