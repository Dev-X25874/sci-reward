"""Tests for sci_reward.rewards.chemical."""

from __future__ import annotations

import pytest
import numpy as np
import jax.numpy as jnp

from sci_reward.rewards.chemical import (
    LipinskiSuiteReward,
    LogPReward,
    MolecularWeightReward,
    QEDReward,
    RingCountReward,
    SAScoreReward,
    ValiditySMILES,
)

VALID = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "Cn1c(=O)c2c(ncn2C)n(c1=O)C"]
INVALID = ["not_a_smiles", "invalid!!", "ZZZZZ"]


class TestValiditySMILES:
    reward = ValiditySMILES()

    def test_valid(self):
        for s in VALID:
            assert self.reward.score(s) == 1.0

    def test_invalid(self):
        for s in INVALID:
            assert self.reward.score(s) == 0.0

    def test_batch_shape_and_dtype(self):
        out = self.reward.batch_score(VALID)
        assert out.shape == (len(VALID),)
        assert out.dtype == jnp.float32

    def test_batch_mixed(self):
        out = self.reward.batch_score(["CCO", "invalid!!", "c1ccccc1"])
        assert float(out[0]) == 1.0
        assert float(out[1]) == 0.0
        assert float(out[2]) == 1.0

    def test_empty_string_zero(self):
        assert ValiditySMILES().score("") == 0.0
        assert ValiditySMILES().score("   ") == 0.0

    def test_sanitize_false_valid(self):
        assert ValiditySMILES(sanitize=False).score("CCO") == 1.0


class TestQEDReward:
    reward = QEDReward()

    def test_range(self):
        for s in VALID:
            assert 0.0 <= self.reward.score(s) <= 1.0

    def test_invalid_zero(self):
        for s in INVALID:
            assert self.reward.score(s) == 0.0

    def test_aspirin_range(self):
        score = self.reward.score("CC(=O)Oc1ccccc1C(=O)O")
        assert 0.4 < score < 0.8

    def test_batch_consistent(self):
        batch = np.array(self.reward.batch_score(VALID))
        individual = np.array([self.reward.score(s) for s in VALID])
        np.testing.assert_allclose(batch, individual, atol=1e-5)


class TestSAScoreReward:
    reward = SAScoreReward()

    def test_range(self):
        for s in VALID:
            assert 0.0 <= self.reward.score(s) <= 1.0

    def test_invalid_zero(self):
        assert self.reward.score("") == 0.0
        assert self.reward.score("invalid!!") == 0.0

    def test_ethanol_easier_than_caffeine(self):
        assert self.reward.score("CCO") >= self.reward.score("Cn1c(=O)c2c(ncn2C)n(c1=O)C")

    def test_batch(self):
        out = self.reward.batch_score(["CCO", "c1ccccc1"])
        assert out.shape == (2,)
        assert all(0.0 <= float(s) <= 1.0 for s in out)


class TestLogPReward:
    reward = LogPReward(target=2.5, std=2.0)

    def test_range(self):
        for s in VALID:
            assert 0.0 <= self.reward.score(s) <= 1.0

    def test_near_target_high_score(self):
        # Toluene LogP ~2.73, close to target 2.5
        assert self.reward.score("Cc1ccccc1") > 0.9

    def test_invalid_zero(self):
        assert self.reward.score("invalid") == 0.0

    def test_custom_target(self):
        # Ethanol LogP ~-0.31, close to 0
        assert LogPReward(target=0.0, std=0.5).score("CCO") > 0.5


class TestMolecularWeightReward:
    reward = MolecularWeightReward(mw_min=160.0, mw_max=500.0)

    def test_in_range(self):
        # Aspirin MW ~180
        assert self.reward.score("CC(=O)Oc1ccccc1C(=O)O") == 1.0

    def test_below_min_soft_penalty(self):
        score = self.reward.score("CCO")  # MW ~46
        assert 0.0 < score < 1.0

    def test_invalid_zero(self):
        assert self.reward.score("invalid!!") == 0.0


class TestRingCountReward:
    reward = RingCountReward(min_rings=1, max_rings=4)

    def test_monocyclic(self):
        assert self.reward.score("c1ccccc1") == 1.0

    def test_no_rings(self):
        assert self.reward.score("CCO") == 0.0

    def test_bicyclic(self):
        assert self.reward.score("c1ccc2ccccc2c1") == 1.0

    def test_invalid_zero(self):
        assert self.reward.score("invalid") == 0.0


class TestLipinskiSuiteReward:
    reward = LipinskiSuiteReward()

    def test_aspirin_passes_all(self):
        assert self.reward.score("CC(=O)Oc1ccccc1C(=O)O") == 1.0

    def test_ibuprofen_passes_all(self):
        assert self.reward.score("CC(C)Cc1ccc(cc1)C(C)C(=O)O") == 1.0

    def test_invalid_zero(self):
        assert self.reward.score("invalid!!") == 0.0

    def test_score_is_valid_fraction(self):
        valid_scores = {0.0, 0.25, 0.5, 0.75, 1.0}
        for s in VALID:
            assert self.reward.score(s) in valid_scores
