"""Crash and exception-boundary checks.

Tests that the code raises the expected exception on malformed inputs and
does not crash on edge-case valid inputs.  Fixed seed: 0/42.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from sci_reward.rewards.bioactivity import (
    FINGERPRINT_DIM,
    BioactivityReward,
    BioactivityRewardTrainer,
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
from sci_reward.tinker_integration.callback import SciRewardCallback, build_chemistry_job
from sci_reward.training.calibration import RewardCalibrator, RunningStats
from sci_reward.training.reward_trainer import PreferenceDataset, RewardTrainer


class TestUninitializedRewardCrashes:
    def test_score_before_init_raises_runtime_error(self):
        r = BioactivityReward()
        with pytest.raises(RuntimeError, match="no params"):
            r.score("CCO")

    def test_batch_score_before_init_raises_runtime_error(self):
        r = BioactivityReward()
        with pytest.raises(RuntimeError, match="no params"):
            r.batch_score(["CCO"])

    def test_save_before_init_raises_runtime_error(self):
        r = BioactivityReward()
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(RuntimeError):
                r.save(Path(d) / "model.flax")

    def test_trainer_auto_inits_uninit_reward(self):
        r = BioactivityReward(hidden_dim=32, n_layers=1)
        assert r.params is None
        BioactivityRewardTrainer(r)
        assert r.params is not None


class TestCompositeRewardConstructionErrors:
    def test_empty_rewards_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            CompositeReward(rewards=[])

    def test_weight_count_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match=r"len\(weights\)"):
            CompositeReward(rewards=[QEDReward(), ValiditySMILES()], weights=[0.5])

    def test_invalid_mode_raises_value_error_on_aggregate(self):
        c = CompositeReward(rewards=[QEDReward()])
        c.mode = "nonsense"
        with pytest.raises(ValueError, match="Unknown aggregation mode"):
            c.aggregate(jnp.ones((2, 1)))

    def test_single_reward_no_crash(self):
        c = CompositeReward(rewards=[QEDReward()])
        assert c.batch_score(["CCO"]).shape == (1,)


class TestCallbackConstructionErrors:
    def test_unknown_format_raises_value_error(self):
        composite = CompositeReward(rewards=[QEDReward()])
        with pytest.raises(ValueError, match="Unknown output_format"):
            SciRewardCallback(reward_fn=composite, output_format="inchi")

    def test_smiles_format_no_error(self):
        assert SciRewardCallback(
            reward_fn=CompositeReward(rewards=[QEDReward()]), output_format="smiles"
        ) is not None

    def test_iupac_format_no_error(self):
        assert SciRewardCallback(
            reward_fn=CompositeReward(rewards=[QEDReward()]), output_format="iupac"
        ) is not None

    def test_custom_extract_fn_accepted(self):
        composite = CompositeReward(rewards=[QEDReward()])
        cb = SciRewardCallback(
            reward_fn=composite, extract_fn=lambda x: x.strip() or None
        )
        result = cb(["p"], ["CCO"])
        assert len(result) == 1


class TestPreferenceDatasetErrors:
    def test_from_csv_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.csv"
            with open(path, "w") as f:
                f.write("wrong_col,another\nCCO,CCN\n")
            with pytest.raises(Exception):
                PreferenceDataset.from_csv(path)

    def test_from_json_wrong_key_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.json"
            path.write_text('[{"a": "CCO", "b": "CCN"}]')
            with pytest.raises(Exception):
                PreferenceDataset.from_json(path)

    def test_from_csv_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            PreferenceDataset.from_csv("/nonexistent/path.csv")

    def test_from_json_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            PreferenceDataset.from_json("/nonexistent/path.json")


class TestEdgeCaseNoCrash:
    @pytest.mark.parametrize("s", [
        "", "   ", "\n", "\t",
        "C" * 1000,
        "[H]",
        "[Na+].[Cl-]",
        "C#N",
        "F",
    ])
    def test_validity_no_crash(self, s):
        score = ValiditySMILES().score(s)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("s", ["", "   ", "C" * 1000, "[Na+].[Cl-]"])
    def test_qed_no_crash(self, s):
        score = QEDReward().score(s)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("s", ["", "   ", "C" * 1000])
    def test_smiles_format_no_crash(self, s):
        score = SMILESFormatReward().score(s)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_iupac_format_empty_no_crash(self):
        assert IUPACFormatReward().score("") == 0.0

    def test_iupac_format_very_long_no_crash(self):
        assert 0.0 <= IUPACFormatReward().score("methyl" * 100) <= 1.0

    def test_mol_weight_single_atom_no_crash(self):
        assert isinstance(MolecularWeightReward().score("C"), float)

    def test_ring_count_single_atom_no_crash(self):
        assert RingCountReward().score("C") == 0.0

    def test_lipinski_single_atom_no_crash(self):
        assert 0.0 <= LipinskiSuiteReward().score("C") <= 1.0

    def test_batch_score_empty_list_no_crash(self):
        assert QEDReward().batch_score([]).shape == (0,)

    def test_composite_batch_empty_list_no_crash(self):
        c = CompositeReward(rewards=[QEDReward(), ValiditySMILES()])
        assert c.batch_score([]).shape == (0,)

    def test_running_stats_normalize_before_any_update(self):
        s = RunningStats()
        result = s.normalize(jnp.array([0.5, 0.3]))
        assert jnp.isfinite(result).all()

    def test_bioactivity_single_atom_no_crash(self):
        r = BioactivityReward(hidden_dim=32, n_layers=1).initialize(jax.random.PRNGKey(0))
        assert 0.0 <= r.score("C") <= 1.0

    def test_bioactivity_batch_all_invalid_no_crash(self):
        r = BioactivityReward(hidden_dim=32, n_layers=1).initialize(jax.random.PRNGKey(0))
        result = r.batch_score(["invalid!!", "", "ZZZZZ"])
        assert result.shape == (3,)
        assert all(float(x) == 0.0 for x in result)

    def test_reward_trainer_single_pair_no_crash(self):
        r = BioactivityReward(hidden_dim=32, n_layers=1).initialize(jax.random.PRNGKey(0))
        trainer = RewardTrainer(r, n_epochs=1, batch_size=1)
        ds = PreferenceDataset([("CCO", "CCN")])
        history = trainer.train(ds, verbose=False)
        assert len(history["train_loss"]) == 1

    def test_build_chemistry_job_no_crash(self):
        spec = build_chemistry_job(max_steps=10)
        assert "reward_callback" in spec
        assert "model" in spec

    def test_calibrator_fit_single_sample_no_crash(self):
        from sci_reward.rewards.base import BaseReward

        class LenReward(BaseReward):
            name = "len"

            def score(self, s):
                return min(len(s) / 10.0, 1.0)

        cal = RewardCalibrator(LenReward(), n_steps=5)
        losses = cal.fit(["CCO"], [1.0], verbose=False)
        assert len(losses) == 5
        assert all(np.isfinite(l) for l in losses)

    def test_logp_reward_near_target_no_nan(self):
        r = LogPReward(target=0.0, std=1e-10)
        assert np.isfinite(r.score("CCO"))

    def test_sa_score_taxol_no_crash(self):
        taxol = (
            "CC(=O)O[C@@H]1C[C@@]2(OC(=O)c3ccccc3)"
            "[C@H](OC(C)=O)[C@@H](O)[C@]3(CC[C@@H]2[C@H]1OC(C)=O)"
            "OC(=O)[C@@H](O)[C@@H](NC(=O)c1ccccc1)c1ccccc1"
        )
        assert 0.0 <= SAScoreReward().score(taxol) <= 1.0
