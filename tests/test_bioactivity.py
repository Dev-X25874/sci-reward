"""Tests for sci_reward.rewards.bioactivity."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sci_reward.rewards.bioactivity import (
    FINGERPRINT_DIM,
    BioactivityModel,
    BioactivityReward,
    BioactivityRewardTrainer,
    batch_fingerprints,
    smiles_to_fingerprint,
)

VALID_SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
INVALID_SMILES = ["invalid!!", "", "ZZZZZ"]
DRUG_LIKE = ["CC(=O)Oc1ccccc1C(=O)O"] * 5
SIMPLE = ["CCO"] * 5


class TestFingerprints:
    def test_valid_shape_dtype(self):
        fp = smiles_to_fingerprint("CCO")
        assert fp is not None
        assert fp.shape == (FINGERPRINT_DIM,)
        assert fp.dtype == np.float32

    def test_invalid_returns_none(self):
        assert smiles_to_fingerprint("invalid!!") is None

    def test_empty_returns_none(self):
        assert smiles_to_fingerprint("") is None
        assert smiles_to_fingerprint("   ") is None

    def test_binary_values(self):
        fp = smiles_to_fingerprint("CCO")
        assert set(fp.tolist()) <= {0.0, 1.0}

    def test_batch_shape(self):
        fps, valid = batch_fingerprints(["CCO", "invalid!!", "c1ccccc1"])
        assert fps.shape == (3, FINGERPRINT_DIM)
        assert valid.shape == (3,)

    def test_batch_valid_mask(self):
        _, valid = batch_fingerprints(["CCO", "invalid!!", "c1ccccc1"])
        assert valid[0] and not valid[1] and valid[2]

    def test_invalid_rows_zero(self):
        fps, _ = batch_fingerprints(["CCO", "invalid!!"])
        assert np.all(fps[1] == 0.0)


class TestBioactivityModel:
    def setup_method(self):
        self.model = BioactivityModel(hidden_dim=64, n_layers=2)
        dummy = jnp.zeros((1, FINGERPRINT_DIM))
        self.params = self.model.init(jax.random.PRNGKey(0), dummy, training=False)["params"]

    def test_output_shape_single(self):
        out = self.model.apply({"params": self.params}, jnp.zeros((1, FINGERPRINT_DIM)), training=False)
        assert out.shape == (1,)

    def test_output_shape_batch(self):
        out = self.model.apply({"params": self.params}, jnp.zeros((8, FINGERPRINT_DIM)), training=False)
        assert out.shape == (8,)

    def test_output_dtype(self):
        out = self.model.apply({"params": self.params}, jnp.zeros((4, FINGERPRINT_DIM)), training=False)
        assert out.dtype == jnp.float32

    def test_sigmoid_in_range(self):
        logits = self.model.apply({"params": self.params}, jnp.ones((4, FINGERPRINT_DIM)), training=False)
        probs = jax.nn.sigmoid(logits)
        assert float(jnp.min(probs)) >= 0.0
        assert float(jnp.max(probs)) <= 1.0


class TestBioactivityReward:
    def setup_method(self):
        self.reward = BioactivityReward(hidden_dim=64, n_layers=2).initialize(jax.random.PRNGKey(0))

    def test_score_in_range(self):
        assert 0.0 <= self.reward.score("CCO") <= 1.0

    def test_invalid_zero(self):
        assert self.reward.score("invalid!!") == 0.0
        assert self.reward.score("") == 0.0

    def test_batch_shape_dtype(self):
        out = self.reward.batch_score(VALID_SMILES)
        assert out.shape == (len(VALID_SMILES),)
        assert out.dtype == jnp.float32

    def test_invalid_zeroed_in_batch(self):
        out = self.reward.batch_score(["CCO", "invalid!!", "c1ccccc1"])
        assert float(out[1]) == 0.0

    def test_uninitialized_raises(self):
        with pytest.raises(RuntimeError, match="no params"):
            BioactivityReward().score("CCO")

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.flax"
            self.reward.save(path)
            loaded = BioactivityReward.from_pretrained(path, hidden_dim=64, n_layers=2)
            assert abs(self.reward.score("CCO") - loaded.score("CCO")) < 1e-5


class TestBioactivityRewardTrainer:
    def setup_method(self):
        self.reward = BioactivityReward(hidden_dim=32, n_layers=1).initialize(jax.random.PRNGKey(42))
        self.trainer = BioactivityRewardTrainer(self.reward, learning_rate=1e-3)
        self.pairs = list(zip(DRUG_LIKE * 4, SIMPLE * 4))

    def test_returns_losses(self):
        losses = self.trainer.train(self.pairs, n_epochs=2, batch_size=8, verbose=False)
        assert len(losses) == 2
        assert all(np.isfinite(l) for l in losses)

    def test_params_update(self):
        params_before = jax.tree_util.tree_map(np.array, self.reward.params)
        self.trainer.train(self.pairs, n_epochs=3, batch_size=8, verbose=False)
        changed = any(
            not np.allclose(b, np.array(a))
            for b, a in zip(
                jax.tree_util.tree_leaves(params_before),
                jax.tree_util.tree_leaves(self.reward.params),
            )
        )
        assert changed

    def test_auto_initializes_uninit_reward(self):
        reward = BioactivityReward(hidden_dim=32, n_layers=1)
        assert reward.params is None
        BioactivityRewardTrainer(reward)
        assert reward.params is not None
