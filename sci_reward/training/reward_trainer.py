"""Production reward trainer: dataset loading, validation, LR scheduling, checkpointing.

PreferenceDataset
    Holds (smiles_chosen, smiles_rejected) pairs with precomputed fingerprints.
    Supports CSV, JSON, and in-memory construction. iter_batches() uses a local
    seeded generator so batch ordering is deterministic and does not mutate the
    global numpy random state.

_bt_loss_and_acc
    Bradley-Terry loss plus ranking accuracy. The incoming dropout_rng is split
    into two independent keys before the chosen and rejected forward passes so
    dropout masks are uncorrelated, giving unbiased gradient estimates.

RewardTrainer
    Production training loop with cosine LR schedule, warmup, gradient clipping,
    per-epoch validation metrics, early stopping, and best-checkpoint saving.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from sci_reward.rewards.bioactivity import (
    BioactivityReward,
    FINGERPRINT_DIM,
    batch_fingerprints,
)


class PreferenceDataset:
    """
    Dataset of (smiles_chosen, smiles_rejected) preference pairs.

    Fingerprints are precomputed at construction time. Accepts CSV
    (columns: smiles_chosen, smiles_rejected), JSON (list of
    {"chosen": ..., "rejected": ...}), or a list of tuples directly.

    iter_batches() uses np.random.default_rng(seed) — a local generator —
    so it is fully deterministic and never touches the global numpy rng.
    """

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs
        self._c_fps, self._c_valid = batch_fingerprints([p[0] for p in pairs])
        self._r_fps, self._r_valid = batch_fingerprints([p[1] for p in pairs])

    def __len__(self) -> int:
        return len(self.pairs)

    @classmethod
    def from_csv(cls, path: str | Path, sep: str = ",") -> "PreferenceDataset":
        import csv
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=sep)
            return cls(
                [(row["smiles_chosen"], row["smiles_rejected"]) for row in reader]
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "PreferenceDataset":
        with open(path) as f:
            return cls([(d["chosen"], d["rejected"]) for d in json.load(f)])

    def train_val_split(
        self, val_fraction: float = 0.1, seed: int = 42
    ) -> tuple["PreferenceDataset", "PreferenceDataset"]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.pairs))
        n_val = max(1, int(len(self.pairs) * val_fraction))
        return (
            PreferenceDataset([self.pairs[i] for i in idx[n_val:]]),
            PreferenceDataset([self.pairs[i] for i in idx[:n_val]]),
        )

    def get_batch(self, indices: np.ndarray) -> tuple[chex.Array, chex.Array]:
        return jnp.array(self._c_fps[indices]), jnp.array(self._r_fps[indices])

    def iter_batches(
        self, batch_size: int, shuffle: bool = True, seed: int = 0
    ) -> Iterator[tuple[chex.Array, chex.Array]]:
        N = len(self.pairs)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(N) if shuffle else np.arange(N)
        for start in range(0, N, batch_size):
            yield self.get_batch(indices[start : start + batch_size])


def _bt_loss_and_acc(
    params, model, fps_chosen, fps_rejected, dropout_rng
) -> tuple[chex.Array, chex.Array]:
    """Bradley-Terry loss + pairwise ranking accuracy.

    dropout_rng is split into two independent keys so the chosen and rejected
    forward passes use uncorrelated dropout masks, giving unbiased gradients.
    """
    rng_c, rng_r = jax.random.split(dropout_rng)
    diff = (
        model.apply(
            {"params": params}, fps_chosen, training=True, rngs={"dropout": rng_c}
        )
        - model.apply(
            {"params": params}, fps_rejected, training=True, rngs={"dropout": rng_r}
        )
    )
    return (
        -jnp.mean(jax.nn.log_sigmoid(diff)),
        jnp.mean((diff > 0).astype(jnp.float32)),
    )


class RewardTrainer:
    """
    Production training loop for BioactivityReward.

    Features cosine LR schedule with warmup, per-epoch val metrics,
    gradient clipping, early stopping, and best-checkpoint saving.
    """

    def __init__(
        self,
        reward: BioactivityReward,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 100,
        batch_size: int = 64,
        n_epochs: int = 20,
        checkpoint_dir: str | Path | None = None,
        patience: int = 5,
    ):
        if reward.params is None:
            reward.initialize()
        self.reward = reward
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.patience = patience

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=n_epochs * 500,
            end_value=learning_rate * 0.01,
        )
        self.state = train_state.TrainState.create(
            apply_fn=reward.model.apply,
            params=reward.params,
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
            ),
        )
        self._rng = jax.random.PRNGKey(0)
        self._train_step = jax.jit(self._train_step_fn)
        self._eval_step = jax.jit(self._eval_step_fn)

    def _train_step_fn(self, state, fps_c, fps_r, dropout_rng):
        (loss, acc), grads = jax.value_and_grad(_bt_loss_and_acc, has_aux=True)(
            state.params, self.reward.model, fps_c, fps_r, dropout_rng
        )
        return state.apply_gradients(grads=grads), loss, acc

    def _eval_step_fn(self, params, fps_c, fps_r):
        diff = (
            self.reward.model.apply({"params": params}, fps_c, training=False)
            - self.reward.model.apply({"params": params}, fps_r, training=False)
        )
        return (
            -jnp.mean(jax.nn.log_sigmoid(diff)),
            jnp.mean((diff > 0).astype(jnp.float32)),
        )

    def train(
        self,
        train_dataset: PreferenceDataset,
        val_dataset: PreferenceDataset | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train and return loss/accuracy history dict."""
        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
        }
        best_val_loss = float("inf")
        no_improve = 0

        for epoch in range(self.n_epochs):
            t0 = time.time()
            b_losses, b_accs = [], []
            for fps_c, fps_r in train_dataset.iter_batches(
                self.batch_size, shuffle=True, seed=epoch
            ):
                self._rng, dropout_rng = jax.random.split(self._rng)
                self.state, loss, acc = self._train_step(
                    self.state, fps_c, fps_r, dropout_rng
                )
                b_losses.append(float(loss))
                b_accs.append(float(acc))

            history["train_loss"].append(float(np.mean(b_losses)))
            history["train_acc"].append(float(np.mean(b_accs)))

            val_loss, val_acc = float("nan"), float("nan")
            if val_dataset is not None:
                vl, va = [], []
                for fps_c, fps_r in val_dataset.iter_batches(
                    self.batch_size, shuffle=False
                ):
                    loss, acc = self._eval_step(self.state.params, fps_c, fps_r)
                    vl.append(float(loss))
                    va.append(float(acc))
                val_loss, val_acc = float(np.mean(vl)), float(np.mean(va))
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                    if self.checkpoint_dir is not None:
                        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        self.reward.params = self.state.params
                        self.reward.save(self.checkpoint_dir / "best.flax")
                else:
                    no_improve += 1

            if verbose:
                print(
                    f"Epoch {epoch + 1:3d}/{self.n_epochs} | "
                    f"train loss={history['train_loss'][-1]:.4f} "
                    f"acc={history['train_acc'][-1]:.3f} | "
                    f"val loss={val_loss:.4f} acc={val_acc:.3f} | "
                    f"{time.time() - t0:.1f}s"
                )

            if val_dataset is not None and no_improve >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}.")
                break

        self.reward.params = self.state.params
        return history
