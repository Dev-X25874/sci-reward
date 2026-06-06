"""Learned bioactivity reward head (Flax MLP + Bradley-Terry trainer).

Architecture: Morgan fingerprint (radius=2, 2048-bit) -> residual MLP -> scalar logit.
Training objective: Bradley-Terry pairwise ranking loss on (chosen, rejected) SMILES pairs,
identical in form to the RLHF reward model objective.

JAX boundary: fingerprint computation (RDKit, numpy) happens before the JAX array is
constructed. The model forward pass is fully jit-able.

Dropout correctness: each BT forward pass (chosen and rejected) uses an independently
split RNG key so dropout masks are statistically independent, giving unbiased gradient
estimates throughout training.

Reproducibility: batch permutation inside train() uses per-epoch default_rng(epoch) so
training is fully deterministic and never touches the global numpy random state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from sci_reward.rewards.base import BaseReward


FINGERPRINT_DIM = 2048
FINGERPRINT_RADIUS = 2


# ---------------------------------------------------------------------------
# Fingerprint utilities
# ---------------------------------------------------------------------------

def smiles_to_fingerprint(smiles: str) -> np.ndarray | None:
    """Morgan fingerprint as float32 (FINGERPRINT_DIM,), or None if SMILES is invalid."""
    if not smiles or not smiles.strip():
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_DIM
        )
        return np.array(fp, dtype=np.float32)
    except ImportError:
        raise ImportError("RDKit is required for bioactivity reward.")


def batch_fingerprints(smiles_list: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Morgan fingerprints for a list of SMILES strings.

    Returns:
        fps   : (N, FINGERPRINT_DIM) float32 — zero-filled for invalid SMILES
        valid : (N,) bool — True where the SMILES parsed successfully
    """
    fps = np.zeros((len(smiles_list), FINGERPRINT_DIM), dtype=np.float32)
    valid = np.zeros(len(smiles_list), dtype=bool)
    for i, s in enumerate(smiles_list):
        fp = smiles_to_fingerprint(s)
        if fp is not None:
            fps[i] = fp
            valid[i] = True
    return fps, valid


# ---------------------------------------------------------------------------
# Flax model
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    features: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: chex.Array, training: bool = False) -> chex.Array:
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.features)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(self.features)(x)
        if residual.shape[-1] != self.features:
            residual = nn.Dense(self.features, use_bias=False)(residual)
        return x + residual


class BioactivityModel(nn.Module):
    """Residual MLP: (N, FINGERPRINT_DIM) -> (N,) scalar logits."""

    hidden_dim: int = 512
    n_layers: int = 3
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: chex.Array, training: bool = False) -> chex.Array:
        x = nn.gelu(nn.Dense(self.hidden_dim)(x))
        for _ in range(self.n_layers):
            x = ResidualBlock(
                features=self.hidden_dim, dropout_rate=self.dropout_rate
            )(x, training=training)
        x = nn.Dense(1)(nn.LayerNorm()(x))
        return x.squeeze(-1)


# ---------------------------------------------------------------------------
# Reward wrapper
# ---------------------------------------------------------------------------

class BioactivityReward(BaseReward):
    """
    Learned bioactivity reward backed by a Flax residual MLP.

    Requires .initialize() or .from_pretrained() before scoring;
    raises RuntimeError with a clear message otherwise.

    score()       -> Python float in [0, 1]
    batch_score() -> JAX DeviceArray, dtype float32, shape (N,)

    Invalid SMILES are zeroed out via the valid-mask before any JAX
    computation — they never reach the model.
    """

    name = "bioactivity"

    def __init__(
        self,
        hidden_dim: int = 512,
        n_layers: int = 3,
        dropout_rate: float = 0.1,
        params=None,
    ):
        self.model = BioactivityModel(
            hidden_dim=hidden_dim, n_layers=n_layers, dropout_rate=dropout_rate
        )
        self.params = params
        self._apply_jit = jax.jit(self._apply)

    def _apply(self, params, fps: chex.Array) -> chex.Array:
        return jax.nn.sigmoid(self.model.apply({"params": params}, fps, training=False))

    def initialize(self, rng_key=None) -> "BioactivityReward":
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        variables = self.model.init(rng_key, jnp.zeros((1, FINGERPRINT_DIM)), training=False)
        self.params = variables["params"]
        return self

    def _require_params(self) -> None:
        if self.params is None:
            raise RuntimeError(
                "BioactivityReward has no params. "
                "Call .initialize() or .from_pretrained() first."
            )

    def score(self, smiles: str) -> float:
        self._require_params()
        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            return 0.0
        return float(self._apply_jit(self.params, jnp.array(fp[None]))[0])

    def batch_score(self, smiles_list: Sequence[str]) -> chex.Array:
        self._require_params()
        fps, valid = batch_fingerprints(smiles_list)
        scores = self._apply_jit(self.params, jnp.array(fps))
        return scores * jnp.array(valid, dtype=jnp.float32)

    def save(self, path: str | Path) -> None:
        """Serialize params to disk. Raises RuntimeError if not yet initialized."""
        self._require_params()
        import flax.serialization
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.params))

    @classmethod
    def from_pretrained(
        cls, path: str | Path, hidden_dim: int = 512, n_layers: int = 3
    ) -> "BioactivityReward":
        import flax.serialization
        reward = cls(hidden_dim=hidden_dim, n_layers=n_layers).initialize()
        with open(path, "rb") as f:
            reward.params = flax.serialization.from_bytes(reward.params, f.read())
        return reward


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BioactivityRewardTrainer:
    """
    Bradley-Terry preference trainer for BioactivityReward.

    Loss: -mean(log σ(r_chosen - r_rejected))

    Dropout correctness
    -------------------
    _bt_loss splits the incoming dropout_rng into two independent keys
    (rng_c, rng_r) so the chosen and rejected forward passes apply
    uncorrelated dropout masks. Using the same key for both passes would
    correlate the masks and bias the gradient estimate.

    Reproducibility
    ---------------
    Epoch-level batch permutation uses np.random.default_rng(epoch), which
    is a local generator and never touches the global numpy random state.
    Training is therefore fully deterministic given the same JAX seed.
    """

    def __init__(
        self,
        reward: BioactivityReward,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        if reward.params is None:
            reward.initialize()
        self.reward = reward
        self.state = train_state.TrainState.create(
            apply_fn=reward.model.apply,
            params=reward.params,
            tx=optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        )
        self._rng = jax.random.PRNGKey(0)
        self._train_step = jax.jit(self._step)

    @staticmethod
    def _bt_loss(
        params, model, fps_chosen, fps_rejected, dropout_rng
    ) -> chex.Array:
        # Split into two independent keys so chosen and rejected
        # dropout masks are uncorrelated, giving unbiased gradients.
        rng_c, rng_r = jax.random.split(dropout_rng)
        r_c = model.apply(
            {"params": params}, fps_chosen, training=True, rngs={"dropout": rng_c}
        )
        r_r = model.apply(
            {"params": params}, fps_rejected, training=True, rngs={"dropout": rng_r}
        )
        return -jnp.mean(jax.nn.log_sigmoid(r_c - r_r))

    def _step(self, state, fps_chosen, fps_rejected, dropout_rng):
        loss, grads = jax.value_and_grad(self._bt_loss)(
            state.params, self.reward.model, fps_chosen, fps_rejected, dropout_rng
        )
        return state.apply_gradients(grads=grads), loss

    def train(
        self,
        pairs: list[tuple[str, str]],
        n_epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list[float]:
        """Train on (smiles_chosen, smiles_rejected) pairs. Returns per-epoch mean losses."""
        c_fps, _ = batch_fingerprints([p[0] for p in pairs])
        r_fps, _ = batch_fingerprints([p[1] for p in pairs])
        chosen, rejected = jnp.array(c_fps), jnp.array(r_fps)
        N = len(pairs)
        losses = []

        for epoch in range(n_epochs):
            # Use a local generator seeded by epoch — never mutates global numpy rng.
            perm = np.random.default_rng(epoch).permutation(N)
            batch_losses = []
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                self._rng, dropout_rng = jax.random.split(self._rng)
                self.state, loss = self._train_step(
                    self.state, chosen[idx], rejected[idx], dropout_rng
                )
                batch_losses.append(float(loss))
            epoch_loss = float(np.mean(batch_losses))
            losses.append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{n_epochs} | loss={epoch_loss:.4f}")

        self.reward.params = self.state.params
        return losses
