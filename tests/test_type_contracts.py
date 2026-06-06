"""Runtime type-contract checks for the public API.

Verifies that every public method returns the type declared in its
annotation without running mypy.  Complements the static mypy step in CI.
Fixed seed: 0.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from sci_reward.rewards.base import BaseReward
from sci_reward.rewards.bioactivity import (
    FINGERPRINT_DIM,
    BioactivityModel,
    BioactivityReward,
    batch_fingerprints,
    smiles_to_fingerprint,
)
from sci_reward.rewards.chemical import (
    LipinskiSuiteReward,
    LogPReward,
    MolecularWeightReward,
    QEDReward,
    RingCountReward,
    SAScoreReward,
    ValiditySMILES,
)
from sci_reward.rewards.composite import CompositeReward
from sci_reward.rewards.format import IUPACFormatReward, SMILESFormatReward
from sci_reward.tinker_integration.callback import (
    SciRewardCallback,
    TinkerJobSpec,
    build_chemistry_job,
)
from sci_reward.training.calibration import RewardCalibrator, RewardVarianceEstimator, RunningStats
from sci_reward.training.reward_trainer import PreferenceDataset, RewardTrainer

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]

ALL_REWARDS = [
    ValiditySMILES(),
    QEDReward(),
    SAScoreReward(),
    LogPReward(),
    MolecularWeightReward(),
    RingCountReward(),
    LipinskiSuiteReward(),
    SMILESFormatReward(),
    IUPACFormatReward(),
]


class TestScoreReturnType:
    @pytest.mark.parametrize("reward", ALL_REWARDS, ids=lambda r: r.name)
    def test_score_returns_python_float(self, reward):
        result = reward.score("CCO")
        assert type(result) is float, (
            f"{reward.name}.score() returned {type(result).__name__}, expected float"
        )

    @pytest.mark.parametrize("reward", ALL_REWARDS, ids=lambda r: r.name)
    def test_score_invalid_returns_python_float(self, reward):
        assert type(reward.score("invalid!!")) is float

    def test_bioactivity_score_returns_python_float(self):
        r = BioactivityReward(hidden_dim=32, n_layers=1).initialize(jax.random.PRNGKey(0))
        assert type(r.score("CCO")) is float


class TestBatchScoreReturnType:
    @pytest.mark.parametrize("reward", ALL_REWARDS, ids=lambda r: r.name)
    def test_batch_score_is_jax_array(self, reward):
        result = reward.batch_score(SMILES)
        assert isinstance(result, jax.Array), (
            f"{reward.name}.batch_score() returned {type(result).__name__}"
        )

    @pytest.mark.parametrize("reward", ALL_REWARDS, ids=lambda r: r.name)
    def test_batch_score_dtype_float32(self, reward):
        result = reward.batch_score(SMILES)
        assert result.dtype == jnp.float32, (
            f"{reward.name}.batch_score() dtype={result.dtype}, expected float32"
        )

    def test_composite_batch_score_is_jax_float32(self):
        c = CompositeReward(rewards=[QEDReward(), ValiditySMILES()])
        result = c.batch_score(SMILES)
        assert isinstance(result, jax.Array)
        assert result.dtype == jnp.float32


class TestRunningStatsPropertyTypes:
    def setup_method(self):
        self.s = RunningStats()
        self.s.update([0.1, 0.5, 0.9])

    def test_mean_is_python_float(self):
        assert type(self.s.mean) is float

    def test_variance_is_python_float(self):
        assert type(self.s.variance) is float

    def test_std_is_python_float(self):
        assert type(self.s.std) is float

    def test_normalize_returns_jax_array(self):
        result = self.s.normalize(jnp.array([0.1, 0.5]))
        assert isinstance(result, jax.Array)

    def test_normalize_numpy_input_returns_jax_array(self):
        result = self.s.normalize(np.array([0.1, 0.5], dtype=np.float32))
        assert isinstance(result, jax.Array)


class TestRewardCalibratorPropertyTypes:
    def setup_method(self):
        class LenReward(BaseReward):
            name = "len"

            def score(self, s):
                return min(len(s) / 10.0, 1.0)

        self.cal = RewardCalibrator(LenReward(), n_steps=10)
        self.cal.fit(["C", "CC", "CCC", "CCCC", "CCCCC"], [0.0, 0.0, 1.0, 1.0, 1.0], verbose=False)

    def test_temperature_is_python_float(self):
        assert type(self.cal.temperature) is float

    def test_bias_is_python_float(self):
        assert type(self.cal.bias) is float

    def test_batch_score_is_jax_array(self):
        assert isinstance(self.cal.batch_score(["CCO", "c1ccccc1"]), jax.Array)


class TestCompositeExplainType:
    def test_explain_returns_dict(self):
        c = CompositeReward(rewards=[QEDReward(), ValiditySMILES()], weights=[0.5, 0.5])
        assert isinstance(c.explain("CCO"), dict)

    def test_explain_values_are_python_float(self):
        c = CompositeReward(rewards=[QEDReward(), ValiditySMILES()], weights=[0.5, 0.5])
        for k, v in c.explain("CCO").items():
            assert type(v) is float, (
                f"explain key '{k}' has type {type(v).__name__}, expected float"
            )

    def test_explain_has_composite_key(self):
        c = CompositeReward(rewards=[QEDReward(), ValiditySMILES()])
        assert "composite" in c.explain("CCO")


class TestCallbackReturnType:
    def setup_method(self):
        composite = CompositeReward(
            rewards=[QEDReward(), ValiditySMILES()], weights=[0.5, 0.5]
        )
        self.cb = SciRewardCallback(reward_fn=composite, normalize=False, log_stats=False)

    def test_returns_list(self):
        result = self.cb(["p"] * 3, ["SMILES: CCO"] * 3)
        assert isinstance(result, list)

    def test_each_element_is_float(self):
        result = self.cb(["p"] * 2, ["SMILES: CCO", "SMILES: c1ccccc1"])
        for i, r in enumerate(result):
            assert isinstance(r, float), (
                f"reward[{i}] is {type(r).__name__}, expected float"
            )

    def test_explain_batch_returns_list_of_dicts(self):
        result = self.cb.explain_batch(["SMILES: CCO", "no smiles here"])
        assert isinstance(result, list)
        assert all(isinstance(d, dict) for d in result)


class TestBuildChemistryJobType:
    def test_returns_dict(self):
        assert isinstance(build_chemistry_job(), dict)

    def test_required_keys_present(self):
        spec = build_chemistry_job()
        for key in ("model", "algorithm", "reward_callback", "max_steps"):
            assert key in spec

    def test_lora_alpha_is_float(self):
        spec = build_chemistry_job(lora_rank=64)
        assert isinstance(spec["lora_alpha"], float)
        assert spec["lora_alpha"] == pytest.approx(128.0)

    def test_reward_callback_is_sci_reward_callback(self):
        assert isinstance(build_chemistry_job()["reward_callback"], SciRewardCallback)


class TestPreferenceDatasetTypes:
    def test_iter_batches_yields_jax_float32(self):
        ds = PreferenceDataset(list(zip(SMILES, SMILES[::-1])))
        for fps_c, fps_r in ds.iter_batches(batch_size=2, shuffle=False):
            assert isinstance(fps_c, jax.Array)
            assert isinstance(fps_r, jax.Array)
            assert fps_c.dtype == jnp.float32
            assert fps_r.dtype == jnp.float32

    def test_get_batch_returns_jax_arrays(self):
        ds = PreferenceDataset(list(zip(SMILES, SMILES[::-1])))
        fps_c, fps_r = ds.get_batch(np.array([0]))
        assert isinstance(fps_c, jax.Array)
        assert isinstance(fps_r, jax.Array)


class TestBioactivityModelTypeContracts:
    def setup_method(self):
        self.model = BioactivityModel(hidden_dim=64, n_layers=2)
        dummy = jnp.zeros((1, FINGERPRINT_DIM))
        self.params = self.model.init(
            jax.random.PRNGKey(0), dummy, training=False
        )["params"]

    def test_apply_returns_jax_array(self):
        out = self.model.apply(
            {"params": self.params},
            jnp.zeros((3, FINGERPRINT_DIM)),
            training=False,
        )
        assert isinstance(out, jax.Array)

    def test_apply_dtype_float32(self):
        out = self.model.apply(
            {"params": self.params},
            jnp.zeros((3, FINGERPRINT_DIM)),
            training=False,
        )
        assert out.dtype == jnp.float32

    def test_apply_shape_is_1d(self):
        out = self.model.apply(
            {"params": self.params},
            jnp.zeros((5, FINGERPRINT_DIM)),
            training=False,
        )
        assert out.ndim == 1
        assert out.shape == (5,)

    def test_smiles_to_fingerprint_return_type(self):
        assert isinstance(smiles_to_fingerprint("CCO"), np.ndarray)

    def test_batch_fingerprints_return_types(self):
        fps, valid = batch_fingerprints(SMILES)
        assert isinstance(fps, np.ndarray)
        assert isinstance(valid, np.ndarray)
        assert fps.dtype == np.float32
        assert valid.dtype == bool

    def test_fingerprint_dim_is_int(self):
        assert isinstance(FINGERPRINT_DIM, int)
        assert FINGERPRINT_DIM == 2048
