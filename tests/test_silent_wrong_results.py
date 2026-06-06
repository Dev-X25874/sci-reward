"""Silent wrong-result checks.

Tests that catch plausible inputs producing numerically incorrect outputs
without raising any exception.  Fixed seed: 0/42.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from sci_reward.rewards.bioactivity import (
    FINGERPRINT_DIM,
    BioactivityReward,
    BioactivityRewardTrainer,
    batch_fingerprints,
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
from sci_reward.rewards.composite import CompositeReward, _normalize_columns, _pareto_scalarize
from sci_reward.rewards.format import SMILESFormatReward
from sci_reward.training.calibration import (
    RewardCalibrator,
    RewardVarianceEstimator,
    RunningStats,
)

VALID = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
INVALID = ["invalid!!", "", "   "]


class TestBTLossGradientCorrectness:
    """The BT loss must use independent dropout masks for chosen vs rejected passes."""

    def setup_method(self):
        self.reward = BioactivityReward(
            hidden_dim=32, n_layers=2, dropout_rate=0.5
        ).initialize(jax.random.PRNGKey(0))
        self.trainer = BioactivityRewardTrainer(self.reward, learning_rate=1e-3)

    def test_loss_finite_after_one_epoch(self):
        pairs = list(zip(VALID * 4, ["CCO"] * 12))
        losses = self.trainer.train(pairs, n_epochs=1, batch_size=6, verbose=False)
        assert np.isfinite(losses[0]), f"Loss is not finite: {losses[0]}"

    def test_preference_direction_learned_after_training(self):
        aspirin = ["CC(=O)Oc1ccccc1C(=O)O"] * 16
        ethanol = ["CCO"] * 16
        pairs = list(zip(aspirin, ethanol))
        self.trainer.train(pairs, n_epochs=5, batch_size=8, verbose=False)
        s_asp = self.reward.score("CC(=O)Oc1ccccc1C(=O)O")
        s_eth = self.reward.score("CCO")
        assert s_asp > s_eth, (
            f"Aspirin score {s_asp:.4f} should exceed ethanol {s_eth:.4f} "
            "after preference training"
        )

    def test_permutation_uses_local_rng_not_global(self):
        state_before = np.random.get_state()[1][0]
        pairs = list(zip(VALID * 4, ["CCO"] * 12))
        self.trainer.train(pairs, n_epochs=2, batch_size=6, verbose=False)
        state_after = np.random.get_state()[1][0]
        assert state_before == state_after, (
            "BioactivityRewardTrainer.train must not mutate global numpy rng state"
        )


class TestRewardScoreSilentWrongValues:
    def test_validity_binary_no_intermediate_values(self):
        r = ValiditySMILES()
        for s in VALID:
            assert r.score(s) == 1.0
        for s in INVALID:
            assert r.score(s) == 0.0

    def test_qed_aspirin_known_range(self):
        score = QEDReward().score("CC(=O)Oc1ccccc1C(=O)O")
        assert 0.4 < score < 0.8, f"Aspirin QED {score} outside expected (0.4, 0.8)"

    def test_logp_ethanol_reward_matches_gaussian(self):
        r = LogPReward(target=2.5, std=2.0)
        score = r.score("CCO")
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        logp = Descriptors.MolLogP(Chem.MolFromSmiles("CCO"))
        expected = float(np.exp(-0.5 * ((logp - 2.5) / 2.0) ** 2))
        assert abs(score - expected) < 0.05

    def test_mol_weight_above_max_partial_score(self):
        r = MolecularWeightReward(mw_min=160.0, mw_max=500.0)
        # Doxorubicin MW ~543 -- above mw_max
        score = r.score(
            "COc1cccc2cc3c(cc12)C(=O)c1c(O)c4c(c(O)c1C3=O)"
            "C[C@@](O)(C(=O)CO)C[C@H]4O[C@H]1C[C@H](N)[C@H](O)[C@H](C)O1"
        )
        assert 0.0 < score < 1.0, f"Above-max MW should give partial score, got {score}"

    def test_ring_count_zero_returns_exactly_zero(self):
        r = RingCountReward(min_rings=1, max_rings=4)
        assert r.score("CCO") == 0.0

    def test_lipinski_fractions_exact(self):
        r = LipinskiSuiteReward()
        valid_vals = {0.0, 0.25, 0.5, 0.75, 1.0}
        for s in VALID:
            val = r.score(s)
            assert val in valid_vals, f"Lipinski score {val} not in {valid_vals}"

    def test_sa_score_ethanol_easier_than_caffeine(self):
        r = SAScoreReward()
        assert r.score("CCO") > r.score("Cn1c(=O)c2c(ncn2C)n(c1=O)C"), (
            "Ethanol should be easier to synthesize than caffeine"
        )


class TestSMILESFormatWeightOverflow:
    def test_default_weights_sum_leq_one(self):
        r = SMILESFormatReward()
        total = r.char_weight + r.balance_weight + r.pattern_weight + r.length_weight
        assert total <= 1.0 + 1e-9, f"Default weight sum {total} exceeds 1.0"

    def test_perfect_smiles_scores_one_with_default_weights(self):
        r = SMILESFormatReward()
        score = r.score("CCO")
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_score_always_in_unit_interval(self):
        r = SMILESFormatReward(
            char_weight=0.25, balance_weight=0.25,
            pattern_weight=0.25, length_weight=0.25,
        )
        for s in VALID + INVALID + ["", "C" * 500]:
            score = r.score(s)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] for '{s[:30]}'"

    def test_overweight_scores_clipped_to_unit_interval(self):
        r = SMILESFormatReward(
            char_weight=0.4, balance_weight=0.4,
            pattern_weight=0.3, length_weight=0.2,
        )
        score = r.score("CCO")
        assert 0.0 <= score <= 1.0


class TestCompositeAggregationSilentErrors:
    def test_weighted_sum_mathematically_correct(self):
        from sci_reward.rewards.base import BaseReward

        class Fixed(BaseReward):
            def __init__(self, v, n="f"):
                self.value = v
                self.name = n

            def score(self, s):
                return self.value

        c = CompositeReward(rewards=[Fixed(0.8, "a"), Fixed(0.4, "b")], weights=[3.0, 1.0])
        expected = 0.75 * 0.8 + 0.25 * 0.4
        result = float(c.batch_score(["CCO"])[0])
        assert abs(result - expected) < 1e-5

    def test_product_mode_numerically_correct(self):
        from sci_reward.rewards.base import BaseReward

        class Fixed(BaseReward):
            def __init__(self, v, n="f"):
                self.value = v
                self.name = n

            def score(self, s):
                return self.value

        c = CompositeReward(rewards=[Fixed(0.8), Fixed(0.6)], mode="product")
        result = float(c.batch_score(["CCO"])[0])
        assert abs(result - 0.48) < 1e-5

    def test_normalize_columns_no_nan_on_constant_column(self):
        m = jnp.array([[0.5, 0.3], [0.5, 0.7], [0.5, 0.5]])
        n = _normalize_columns(m)
        assert not bool(jnp.any(jnp.isnan(n))), "Constant column must not produce NaN"

    def test_normalize_columns_range_exactly_0_to_1(self):
        m = jnp.array([[0.1, 0.5], [0.9, 0.2], [0.5, 0.8]])
        n = _normalize_columns(m)
        np.testing.assert_allclose(float(jnp.min(n[:, 0])), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(jnp.max(n[:, 0])), 1.0, atol=1e-5)

    def test_pareto_top_molecule_scores_one(self):
        m = jnp.array([[1.0, 1.0], [0.5, 0.5], [0.3, 0.7]], dtype=jnp.float32)
        w = jnp.array([0.5, 0.5])
        result = _pareto_scalarize(m, w)
        assert float(result[0]) == pytest.approx(1.0, abs=1e-5), (
            "Dominant molecule must not be penalized"
        )

    def test_pareto_dominated_molecule_penalized(self):
        m = jnp.array([[1.0, 1.0], [0.5, 0.5], [0.3, 0.3]], dtype=jnp.float32)
        w = jnp.array([0.5, 0.5])
        result = _pareto_scalarize(m, w)
        naive = jnp.dot(m, w)
        assert float(result[2]) < float(naive[2]) + 1e-5

    def test_pareto_all_equal_no_penalty(self):
        m = jnp.array([[0.5, 0.5]] * 4, dtype=jnp.float32)
        w = jnp.array([0.5, 0.5])
        result = _pareto_scalarize(m, w)
        np.testing.assert_allclose(np.array(result), [0.5] * 4, atol=1e-5)

    def test_gate_invalid_molecule_is_exactly_zero(self):
        validity = ValiditySMILES()
        c = CompositeReward(rewards=[QEDReward()], gate=validity)
        scores = c.batch_score(["invalid!!", "CCO"])
        assert float(scores[0]) == 0.0
        assert float(scores[1]) > 0.0


class TestCalibratorSilentErrors:
    def test_transform_maps_to_unit_interval(self):
        from sci_reward.rewards.base import BaseReward

        class LenReward(BaseReward):
            name = "len"

            def score(self, s):
                return min(len(s) / 10.0, 1.0)

        cal = RewardCalibrator(LenReward(), n_steps=50, learning_rate=1e-2)
        smiles = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"]
        labels = [0, 0, 0, 1, 1, 1]
        cal.fit(smiles, labels, verbose=False)
        out = cal.transform(jnp.array([0.01, 0.5, 0.99]))
        assert float(jnp.min(out)) >= 0.0
        assert float(jnp.max(out)) <= 1.0

    def test_temperature_positive_after_fit(self):
        from sci_reward.rewards.base import BaseReward

        class LenReward(BaseReward):
            name = "len"

            def score(self, s):
                return min(len(s) / 10.0, 1.0)

        cal = RewardCalibrator(LenReward(), n_steps=50)
        cal.fit(["C", "CC", "CCC", "CCCC"], [0, 0, 1, 1], verbose=False)
        assert cal.temperature > 0.0

    def test_running_stats_welford_matches_numpy(self):
        rng = np.random.default_rng(0)
        vals = rng.random(100).tolist()
        s = RunningStats()
        s.update(vals)
        assert abs(s.mean - float(np.mean(vals))) < 1e-8
        assert abs(s.variance - float(np.var(vals, ddof=1))) < 1e-6

    def test_variance_estimator_scores_in_unit_interval(self):
        reward = BioactivityReward(
            hidden_dim=32, n_layers=2, dropout_rate=0.2
        ).initialize(jax.random.PRNGKey(42))
        estimator = RewardVarianceEstimator(reward, n_samples=5, rng_seed=0)
        mean, var = estimator.estimate(VALID)
        assert all(0.0 <= float(m) <= 1.0 for m in mean)
        assert all(float(v) >= 0.0 for v in var)
