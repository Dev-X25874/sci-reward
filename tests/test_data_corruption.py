"""Data corruption checks.

Tests that state written to disk, passed between components, or accumulated
incrementally is never silently mutated or truncated.  Fixed seed: 0/42.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from sci_reward.rewards.bioactivity import (
    FINGERPRINT_DIM,
    BioactivityReward,
    batch_fingerprints,
)
from sci_reward.rewards.chemical import QEDReward, ValiditySMILES
from sci_reward.rewards.composite import CompositeReward
from sci_reward.tinker_integration.callback import SciRewardCallback, extract_smiles, extract_iupac
from sci_reward.training.calibration import RunningStats
from sci_reward.training.reward_trainer import PreferenceDataset

VALID = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
PAIRS = list(zip(VALID, ["CCN", "CCCC", "Cc1ccccc1"]))


class TestBioactivityRewardSerializationRoundtrip:
    def setup_method(self):
        self.reward = BioactivityReward(hidden_dim=64, n_layers=2).initialize(
            jax.random.PRNGKey(0)
        )

    def test_save_load_score_identical(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.flax"
            self.reward.save(path)
            loaded = BioactivityReward.from_pretrained(path, hidden_dim=64, n_layers=2)
            for s in VALID:
                orig = self.reward.score(s)
                restored = loaded.score(s)
                assert orig == pytest.approx(restored, abs=1e-6), (
                    f"Score changed after roundtrip for '{s}': {orig} -> {restored}"
                )

    def test_save_file_is_non_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.flax"
            self.reward.save(path)
            assert path.stat().st_size > 0

    def test_save_load_batch_score_identical(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.flax"
            self.reward.save(path)
            loaded = BioactivityReward.from_pretrained(path, hidden_dim=64, n_layers=2)
            orig = np.array(self.reward.batch_score(VALID))
            restored = np.array(loaded.batch_score(VALID))
            np.testing.assert_allclose(orig, restored, atol=1e-6)

    def test_save_before_init_raises_runtime_error(self):
        r = BioactivityReward()
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(RuntimeError):
                r.save(Path(d) / "model.flax")

    def test_from_pretrained_wrong_arch_raises_at_inference(self):
        """Architecture mismatch surfaces at inference time, not at load time."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.flax"
            self.reward.save(path)
            loaded = BioactivityReward.from_pretrained(path, hidden_dim=128, n_layers=2)
            with pytest.raises(Exception):
                loaded.score("CCO")


class TestFingerprintDataIntegrity:
    def test_invalid_row_not_overwritten_by_adjacent_valid(self):
        fps, valid = batch_fingerprints(["CCO", "invalid!!", "c1ccccc1"])
        assert not valid[1]
        assert np.all(fps[1] == 0.0), (
            "Invalid row must be zero-filled, not bleed from neighbors"
        )

    def test_valid_fingerprints_not_mutated_by_invalid_neighbor(self):
        fps_mixed, _ = batch_fingerprints(["CCO", "invalid!!"])
        fps_single, _ = batch_fingerprints(["CCO"])
        np.testing.assert_array_equal(fps_mixed[0], fps_single[0])

    def test_fingerprint_is_binary(self):
        fps, _ = batch_fingerprints(["CCO", "c1ccccc1"])
        unique_vals = set(np.unique(fps).tolist())
        assert unique_vals <= {0.0, 1.0}, (
            f"Fingerprint contains non-binary values: {unique_vals}"
        )

    def test_fingerprint_not_all_zero_for_valid(self):
        for s in VALID:
            fp = batch_fingerprints([s])[0][0]
            assert np.any(fp != 0.0), f"Valid SMILES '{s}' produced all-zero fingerprint"


class TestPreferenceDatasetCorruption:
    def test_from_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.csv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["smiles_chosen", "smiles_rejected"])
                writer.writeheader()
                for c, r in PAIRS:
                    writer.writerow({"smiles_chosen": c, "smiles_rejected": r})
            ds = PreferenceDataset.from_csv(path)
            assert len(ds) == len(PAIRS)
            for i, (c, r) in enumerate(PAIRS):
                assert ds.pairs[i][0] == c
                assert ds.pairs[i][1] == r

    def test_from_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.json"
            records = [{"chosen": c, "rejected": r} for c, r in PAIRS]
            with open(path, "w") as f:
                json.dump(records, f)
            ds = PreferenceDataset.from_json(path)
            assert len(ds) == len(PAIRS)
            for i, (c, r) in enumerate(PAIRS):
                assert ds.pairs[i] == (c, r)

    def test_train_val_split_total_size_preserved(self):
        ds = PreferenceDataset(PAIRS * 20)
        train, val = ds.train_val_split(val_fraction=0.2, seed=42)
        assert len(train) + len(val) == len(ds)

    def test_train_val_split_sizes_correct(self):
        n = 100
        pairs = list(zip(VALID * 34, ["CCN"] * n))[:n]
        ds = PreferenceDataset(pairs)
        train, val = ds.train_val_split(val_fraction=0.1, seed=0)
        assert len(val) == max(1, int(n * 0.1))
        assert len(train) + len(val) == n

    def test_get_batch_shape_correct(self):
        ds = PreferenceDataset(PAIRS)
        fps_c, fps_r = ds.get_batch(np.array([0, 1]))
        assert fps_c.shape == (2, FINGERPRINT_DIM)
        assert fps_r.shape == (2, FINGERPRINT_DIM)

    def test_iter_batches_covers_all_samples(self):
        n = 17
        pairs = list(zip(VALID * 6, ["CCN"] * n))[:n]
        ds = PreferenceDataset(pairs)
        total = sum(fps_c.shape[0] for fps_c, _ in ds.iter_batches(batch_size=5, shuffle=False))
        assert total == n

    def test_precomputed_fps_match_on_demand(self):
        ds = PreferenceDataset(PAIRS)
        fps_c_stored = ds._c_fps[0]
        fps_c_live, _ = batch_fingerprints([PAIRS[0][0]])
        np.testing.assert_array_equal(fps_c_stored, fps_c_live[0])


class TestRunningStatsCorruption:
    def test_reset_clears_all_state(self):
        s = RunningStats()
        s.update([1.0, 2.0, 3.0])
        s.reset()
        assert s._count == 0
        assert s.mean == 0.0
        assert s._M2 == 0.0

    def test_update_with_empty_array_is_noop(self):
        s = RunningStats()
        s.update([0.5])
        mean_before = s.mean
        count_before = s._count
        s.update([])
        assert s.mean == mean_before
        assert s._count == count_before

    def test_multiple_updates_match_single_batch(self):
        rng = np.random.default_rng(0)
        vals = rng.random(50).tolist()
        s1 = RunningStats()
        s1.update(vals)
        s2 = RunningStats()
        for chunk in [vals[:10], vals[10:30], vals[30:]]:
            s2.update(chunk)
        assert abs(s1.mean - s2.mean) < 1e-10
        assert abs(s1.variance - s2.variance) < 1e-8

    def test_variance_positive_for_non_constant_data(self):
        s = RunningStats()
        s.update([0.0, 0.5, 1.0])
        assert s.variance > 0.0

    def test_std_never_zero(self):
        s = RunningStats(epsilon=1e-8)
        s.update([0.5, 0.5, 0.5])
        assert s.std > 0.0


class TestCallbackStateCorruption:
    def setup_method(self):
        composite = CompositeReward(
            rewards=[ValiditySMILES(), QEDReward()],
            weights=[0.5, 0.5],
            gate=ValiditySMILES(),
        )
        self.callback = SciRewardCallback(
            reward_fn=composite, normalize=True, log_stats=False
        )

    def test_call_count_increments(self):
        self.callback(["p"], ["SMILES: CCO"])
        self.callback(["p"], ["SMILES: CCO"])
        assert self.callback._call_count == 2

    def test_reset_stats_clears_running_state(self):
        self.callback(["p"], ["SMILES: CCO"])
        self.callback.reset_stats()
        assert self.callback._stats._count == 0

    def test_all_invalid_batch_returns_invalid_reward(self):
        rewards = self.callback(["p"] * 3, ["no smiles here"] * 3)
        assert len(rewards) == 3
        assert all(r == 0.0 for r in rewards)

    def test_return_length_matches_completions(self):
        completions = ["SMILES: CCO", "SMILES: c1ccccc1", "no smiles"]
        rewards = self.callback(["p"] * 3, completions)
        assert len(rewards) == len(completions)

    def test_extraction_failure_gets_invalid_reward(self):
        # A completion with NO extractable SMILES receives invalid_reward=0.0
        # regardless of normalization
        completions = ["no smiles pattern here at all", "SMILES: CCO"]
        rewards = self.callback(["p"] * 2, completions)
        assert rewards[0] == pytest.approx(0.0, abs=1e-7)


class TestExtractionDataIntegrity:
    def test_extract_smiles_from_labeled_output(self):
        assert extract_smiles("SMILES: CCO") == "CCO"

    def test_extract_smiles_from_code_block(self):
        assert extract_smiles("```smiles\nCCO\n```") == "CCO"

    def test_extract_smiles_returns_none_for_plain_text(self):
        result = extract_smiles("The molecule is ethanol which is a small alcohol")
        assert result is None

    def test_extract_smiles_strips_whitespace(self):
        assert extract_smiles("SMILES:   CCO  ") == "CCO"

    def test_extract_iupac_from_labeled_output(self):
        assert extract_iupac("IUPAC name: ethanol") == "ethanol"

    def test_extract_iupac_not_truncated(self):
        name = "2-methylpropan-1-ol"
        assert extract_iupac(f"IUPAC name: {name}") == name

    def test_extract_smiles_labeled_takes_priority(self):
        result = extract_smiles("SMILES: CCO\nCCN")
        assert result == "CCO"
