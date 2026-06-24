"""Tinker RL reward callback and job-spec builder.

SciRewardCallback
    Drop-in reward callback for the Tinker RLHF trainer.  Accepts a batch of
    (prompt, completion) pairs, extracts SMILES or IUPAC names from the
    completion text, scores them via a CompositeReward, and optionally
    normalizes the scores to zero mean / unit variance using a running
    Welford accumulator.

    Invalid-reward semantics
    ------------------------
    invalid_reward (default 0.0) is applied AFTER optional normalization so
    that molecules that failed extraction always receive exactly
    invalid_reward and are never accidentally rescaled to a non-zero value.

    Specifically:
        1. Extract SMILES/IUPAC from each completion.
        2. Score all extracted strings via CompositeReward.batch_score().
        3. If normalize=True: update RunningStats with scores from molecules
           that were both successfully extracted AND received a non-zero score
           (i.e. passed any gate). Gate-blocked molecules are excluded so their
           forced 0.0 scores do not bias the running normalization statistics.
        4. Overwrite entries where extraction failed with invalid_reward.

extract_smiles / extract_iupac
    Regex-based extractors.  Return None when no match is found — callers
    must not pass None to the reward function.

build_chemistry_job
    Convenience factory that assembles a TinkerJobSpec dict with sensible
    defaults for SMILES-generation chemistry fine-tuning.
"""

from __future__ import annotations

import re
from typing import Callable, TypedDict

import numpy as np

from sci_reward.rewards.composite import CompositeReward
from sci_reward.rewards.chemical import QEDReward, ValiditySMILES, SAScoreReward, LogPReward
from sci_reward.rewards.format import SMILESFormatReward
from sci_reward.training.calibration import RunningStats


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_SMILES_PATTERNS = [
    re.compile(r"SMILES\s*:\s*(\S+)", re.IGNORECASE),
    re.compile(r"```smiles\s*\n([^\n`]+)", re.IGNORECASE),
    re.compile(r"```\s*\n([^\n`]+)", re.IGNORECASE),
]

_IUPAC_PATTERNS = [
    re.compile(r"IUPAC\s+name\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE),
    re.compile(r"compound\s+name\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE),
]


def extract_smiles(text: str) -> str | None:
    """Return the first SMILES string found in text, or None."""
    for pattern in _SMILES_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip() or None
    return None


def extract_iupac(text: str) -> str | None:
    """Return the first IUPAC name found in text, or None."""
    for pattern in _IUPAC_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip() or None
    return None


_EXTRACTORS: dict[str, Callable[[str], str | None]] = {
    "smiles": extract_smiles,
    "iupac": extract_iupac,
}


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class SciRewardCallback:
    """
    Tinker RL reward callback for molecular generation tasks.

    Parameters
    ----------
    reward_fn : CompositeReward
        Reward function applied to extracted molecules.
    output_format : str
        "smiles" or "iupac" — selects the built-in extractor.
        Ignored when extract_fn is provided.
    extract_fn : callable, optional
        Custom extractor (str) -> str | None.  Overrides output_format.
    normalize : bool
        Normalize rewards to zero mean / unit variance using a running
        Welford accumulator over valid molecules seen so far.
    invalid_reward : float
        Score assigned to completions from which no molecule could be
        extracted.  Applied AFTER normalization.
    log_stats : bool
        Print per-batch statistics to stdout.
    """

    def __init__(
        self,
        reward_fn: CompositeReward,
        output_format: str = "smiles",
        extract_fn: Callable[[str], str | None] | None = None,
        normalize: bool = True,
        invalid_reward: float = 0.0,
        log_stats: bool = True,
    ):
        if extract_fn is not None:
            self._extract = extract_fn
        elif output_format in _EXTRACTORS:
            self._extract = _EXTRACTORS[output_format]
        else:
            raise ValueError(
                f"Unknown output_format '{output_format}'. "
                f"Choose from {list(_EXTRACTORS)} or pass extract_fn."
            )

        self.reward_fn = reward_fn
        self.normalize = normalize
        self.invalid_reward = invalid_reward
        self.log_stats = log_stats
        self._stats = RunningStats()
        self._call_count = 0

    def __call__(
        self, prompts: list[str], completions: list[str]
    ) -> list[float]:
        """Score a batch of completions. Returns list[float] of length len(completions)."""
        extracted: list[str] = []
        valid_mask: list[bool] = []
        for completion in completions:
            mol = self._extract(completion)
            if mol:
                extracted.append(mol)
                valid_mask.append(True)
            else:
                extracted.append("")
                valid_mask.append(False)

        raw = np.array(self.reward_fn.batch_score(extracted), dtype=np.float32)
        valid_arr = np.array(valid_mask)

        if self.normalize:
            # Update stats only from molecules that were both successfully extracted
            # AND received a non-zero score (i.e. passed the reward gate).
            # Including gate-blocked molecules (raw=0.0) would bias the running mean
            # toward zero and distort normalization for genuinely low-scoring molecules.
            valid_and_nonzero = valid_arr & (raw > 0.0)
            self._stats.update(raw[valid_and_nonzero])
            rewards = np.array(self._stats.normalize(raw), dtype=np.float32)
            rewards[~valid_arr] = self.invalid_reward
            rewards = rewards.tolist()
        else:
            for i, valid in enumerate(valid_mask):
                if not valid:
                    raw[i] = self.invalid_reward
            rewards = raw.tolist()

        if self.log_stats:
            valid_r = [r for r, v in zip(rewards, valid_mask) if v]
            n_valid = len(valid_r)
            mean_r = float(np.mean(valid_r)) if valid_r else float("nan")
            print(
                f"[SciRewardCallback] step={self._call_count} | "
                f"valid={n_valid}/{len(completions)} | "
                f"mean_reward={mean_r:.4f}"
            )

        self._call_count += 1
        return rewards

    def explain_batch(self, completions: list[str]) -> list[dict]:
        """Return per-molecule component breakdowns for a batch of completions."""
        results = []
        for completion in completions:
            mol = self._extract(completion)
            if mol:
                results.append(self.reward_fn.explain(mol))
            else:
                results.append({"composite": 0.0, "extracted": None})
        return results

    def reset_stats(self) -> None:
        """Reset the running normalization statistics."""
        self._stats.reset()


# ---------------------------------------------------------------------------
# Job spec
# ---------------------------------------------------------------------------

class TinkerJobSpec(TypedDict):
    model: str
    algorithm: str
    reward_callback: SciRewardCallback
    max_steps: int
    lora_rank: int
    lora_alpha: float
    kl_coeff: float
    rollout_batch_size: int
    generation_kwargs: dict


def build_chemistry_job(
    model: str = "mistralai/Mistral-7B-v0.1",
    max_steps: int = 5_000,
    lora_rank: int = 32,
    kl_coeff: float = 0.05,
    rollout_batch_size: int = 64,
    generation_kwargs: dict | None = None,
    qed_weight: float = 0.40,
    sa_weight: float = 0.25,
    logp_weight: float = 0.25,
    format_weight: float = 0.10,
    validity_gate: bool = True,
) -> TinkerJobSpec:
    """
    Assemble a TinkerJobSpec for SMILES-generation chemistry fine-tuning.

    The composite reward combines QED, SA score, LogP proximity, and SMILES
    format quality. ValiditySMILES is used as a hard gate so invalid molecules
    receive reward 0.0 regardless of the other components.

    Default weights sum to 1.0 (qed=0.40, sa=0.25, logp=0.25, format=0.10).
    """
    rewards = [
        QEDReward(),
        SAScoreReward(),
        LogPReward(target=2.5, std=2.0),
        SMILESFormatReward(),
    ]
    weights = [qed_weight, sa_weight, logp_weight, format_weight]
    gate = ValiditySMILES() if validity_gate else None

    composite = CompositeReward(rewards=rewards, weights=weights, gate=gate)
    callback = SciRewardCallback(
        reward_fn=composite,
        output_format="smiles",
        normalize=True,
        invalid_reward=0.0,
        log_stats=True,
    )

    return TinkerJobSpec(
        model=model,
        algorithm="ppo",
        reward_callback=callback,
        max_steps=max_steps,
        lora_rank=lora_rank,
        lora_alpha=float(lora_rank * 2),
        kl_coeff=kl_coeff,
        rollout_batch_size=rollout_batch_size,
        generation_kwargs=generation_kwargs or {
            "max_new_tokens": 128,
            "temperature": 0.9,
            "top_p": 0.95,
            "do_sample": True,
        },
    )
