# sci-reward

JAX-native reward modeling toolkit for RL fine-tuning of LLMs on scientific tasks. Provides composable, `jit`-able reward functions for chemistry and molecular reasoning, designed to plug into [Tinker](https://thinkingmachines.ai/tinker/) RL training loops.

---

## Why this exists

Tinker handles distributed RL fine-tuning infrastructure. Reward modeling — the hardest part — is left to the user. For scientific domains this is non-trivial:

- Chemical validity functions (RDKit) are not JAX-traceable
- Rewards are multi-objective: validity, drug-likeness, and synthesizability simultaneously
- Naive weighted sums invite reward hacking — models learn to game individual scores without learning real chemistry
- Batched RL rollouts require `jit`-able reward computation

`sci-reward` handles this with a clean two-layer design: rule-based rewards (RDKit) plus a learned Flax reward head trained via Bradley-Terry preference ranking.

---

## Architecture

```
sci_reward/
├── rewards/
│   ├── base.py          # BaseReward ABC with explicit JAX boundary contract
│   ├── chemical.py      # RDKit rewards: validity, QED, SA score, LogP, Lipinski
│   ├── format.py        # SMILES/IUPAC syntax rewards (no RDKit dependency)
│   ├── bioactivity.py   # Flax MLP reward head + Bradley-Terry trainer
│   └── composite.py     # Multi-objective aggregation: weighted, product, min, Pareto
├── training/
│   ├── calibration.py   # Temperature scaling, Welford running stats, MC dropout variance
│   └── reward_trainer.py
├── tinker_integration/
│   └── callback.py      # Drop-in SciRewardCallback for Tinker RL jobs
├── benchmarks/
│   └── iupac_bench.py   # IUPAC-to-formula evaluation harness
└── examples/
    └── tinker_rl_example.py
```

---

## Install

```bash
git clone https://github.com/Dev-X25874/sci-reward
cd sci-reward
pip install -e ".[dev]"
```

**Requirements:** JAX ≥ 0.4.25, Flax ≥ 0.8.0, RDKit ≥ 2024.03, Optax ≥ 0.2.2

---

## Quick start

```python
from sci_reward.rewards.composite import CompositeReward
from sci_reward.rewards.chemical import ValiditySMILES, QEDReward, SAScoreReward

reward_fn = CompositeReward(
    rewards=[ValiditySMILES(), QEDReward(), SAScoreReward()],
    weights=[0.4, 0.35, 0.25],
    gate=ValiditySMILES(),
)

scores = reward_fn.batch_score(["CCO", "c1ccccc1", "invalid!!"])
# DeviceArray([..., ..., 0.0], dtype=float32)
```

### Tinker integration

```python
from sci_reward.tinker_integration.callback import SciRewardCallback, build_chemistry_job

callback = SciRewardCallback(reward_fn=reward_fn, output_format="smiles")
job = build_chemistry_job(model="meta-llama/Llama-3-70B", reward_callback=callback)
```

### Training a learned reward head

```python
from sci_reward.rewards.bioactivity import BioactivityReward, BioactivityRewardTrainer

reward = BioactivityReward().initialize()
trainer = BioactivityRewardTrainer(reward)
trainer.train(pairs)  # list of (smiles_chosen, smiles_rejected) tuples
```

---

## Aggregation modes

| Mode | Description |
|------|-------------|
| `weighted` | Weighted sum (default) |
| `product` | Geometric mean — all objectives must be non-zero |
| `min` | Bottleneck — score is limited by the weakest objective |
| `pareto` | Pareto-aware scalarization — penalizes dominated molecules across the batch |

---

## Tests

```bash
pytest tests/ -v
```

## Benchmark harness

```bash
python benchmarks/iupac_bench.py --simulate
```

---

## License

Apache 2.0
