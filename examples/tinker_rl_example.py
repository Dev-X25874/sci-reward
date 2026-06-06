"""End-to-end example: composite reward + bioactivity training + Tinker job spec."""

from __future__ import annotations

import jax
import numpy as np

from sci_reward.rewards.bioactivity import BioactivityReward, BioactivityRewardTrainer
from sci_reward.rewards.chemical import LipinskiSuiteReward, QEDReward, SAScoreReward, ValiditySMILES
from sci_reward.rewards.composite import CompositeReward
from sci_reward.tinker_integration.callback import SciRewardCallback, build_chemistry_job
from sci_reward.training.calibration import calibrate_composite

DRUG_LIKE = [
    "CC(=O)Oc1ccccc1C(=O)O",        # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",   # ibuprofen
    "Cn1c(=O)c2c(ncn2C)n(c1=O)C",   # caffeine
    "Oc1ccc(cc1)C(O)=O",             # 4-hydroxybenzoic acid
    "Nc1ccc(cc1)S(=O)(=O)N",         # sulfanilamide
]
SIMPLE = ["CCO", "CC", "CCC", "CO", "C=C"]
TEST_SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "Cn1c(=O)c2c(ncn2C)n(c1=O)C", "invalid!!", "CCO"]


# --- 1. Rule-based composite ---
rule_composite = CompositeReward(
    rewards=[ValiditySMILES(), QEDReward(), SAScoreReward(), LipinskiSuiteReward()],
    weights=[0.3, 0.3, 0.2, 0.2],
    gate=ValiditySMILES(),
)
print("Rule-based scores:")
for smi in TEST_SMILES:
    b = rule_composite.explain(smi)
    print(f"  {smi[:35]:35s} composite={b['composite']:.3f}")

# --- 2. Train bioactivity head ---
print("\nTraining bioactivity reward head...")
pairs = [(d, s) for d, s in zip(DRUG_LIKE * 5, SIMPLE * 5)]
bioact = BioactivityReward(hidden_dim=128, n_layers=3).initialize(jax.random.PRNGKey(42))
BioactivityRewardTrainer(bioact, learning_rate=3e-4).train(pairs, n_epochs=5, batch_size=8)

# --- 3. Full composite ---
full_composite = CompositeReward(
    rewards=[ValiditySMILES(), QEDReward(), SAScoreReward(), bioact],
    weights=[0.25, 0.25, 0.25, 0.25],
    gate=ValiditySMILES(),
)
print("\nFull composite scores (with learned bioactivity):")
for smi in TEST_SMILES:
    b = full_composite.explain(smi)
    print(f"  {smi[:35]:35s} composite={b['composite']:.3f}  bioactivity={b['bioactivity']:.3f}")

# --- 4. Calibrate ---
calib_data = list(zip(DRUG_LIKE + SIMPLE, [1] * 5 + [0] * 5))
calibrators = calibrate_composite(full_composite, calib_data, verbose=True)

# --- 5. Tinker callback ---
callback = SciRewardCallback(reward_fn=full_composite, output_format="smiles", normalize=True, log_stats=True)

completions = [
    "SMILES: CC(=O)Oc1ccccc1C(=O)O",
    "The molecule is: CCO",
    "SMILES: invalid!!",
    "Answer: Cn1c(=O)c2c(ncn2C)n(c1=O)C",
]
rewards = callback(prompts=[""] * 4, completions=completions)
print("\nCallback rewards:")
for comp, r in zip(completions, rewards):
    print(f"  {comp[:50]:50s} -> {r:.4f}")

# --- 6. Job spec ---
job = build_chemistry_job(model="meta-llama/Llama-3-70B", reward_callback=callback)
print("\nTinker job spec:")
for k, v in job.items():
    if k != "reward_callback":
        print(f"  {k}: {v}")
