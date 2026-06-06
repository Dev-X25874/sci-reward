"""RDKit-based chemical property rewards."""

from __future__ import annotations

import os
import sys
import warnings
from typing import Sequence

import numpy as np

from sci_reward.rewards.base import BaseReward


def _rdkit_mol(smiles: str):
    if not smiles or not smiles.strip():
        return None
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        return mol
    except ImportError:
        raise ImportError("RDKit is required for chemical rewards: pip install rdkit")


def _sa_score(mol) -> float:
    """Ertl & Schuffenhauer SA score, normalized to [0, 1] (higher = easier)."""
    try:
        from rdkit.Chem import RDConfig
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer
        raw = sascorer.calculateScore(mol)  # [1, 10], lower = easier to synthesize
        return float(np.clip((10.0 - raw) / 9.0, 0.0, 1.0))
    except Exception as exc:
        warnings.warn(f"SA score failed: {exc}")
        return 0.5


class ValiditySMILES(BaseReward):
    """
    Binary validity reward.

    Returns 1.0 if RDKit can parse and sanitize the SMILES, 0.0 otherwise.
    `sanitize=True` (default) additionally catches valence errors that
    parse syntactically but are chemically impossible.
    """

    name = "validity"

    def __init__(self, sanitize: bool = True):
        self.sanitize = sanitize

    def score(self, smiles: str) -> float:
        if not smiles or not smiles.strip():
            return 0.0
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles.strip(), sanitize=self.sanitize)
            return 1.0 if mol is not None else 0.0
        except Exception:
            return 0.0


class QEDReward(BaseReward):
    """
    Quantitative Estimate of Drug-likeness (Bickerton et al., 2012).

    Score in [0, 1]; combines MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS
    via a desirability function fit to approved drugs.
    """

    name = "qed"

    def score(self, smiles: str) -> float:
        mol = _rdkit_mol(smiles)
        if mol is None:
            return 0.0
        try:
            from rdkit.Chem import QED
            return float(QED.qed(mol))
        except Exception:
            return 0.0


class SAScoreReward(BaseReward):
    """
    Synthetic Accessibility reward (Ertl & Schuffenhauer, 2009).

    Normalized to [0, 1] where 1 = trivially synthesizable. Penalizing
    poor SA prevents reward hacking via exotic but unsynthesizable structures.
    """

    name = "sa_score"

    def score(self, smiles: str) -> float:
        mol = _rdkit_mol(smiles)
        if mol is None:
            return 0.0
        return _sa_score(mol)


class LogPReward(BaseReward):
    """
    Wildman-Crippen LogP reward.

    Gaussian centered at `target` (default 2.5) with std `std` (default 2.0),
    clipped to [0, 1]. Drug-like range is roughly [-0.4, 5.6] (Lipinski).
    """

    name = "logp"

    def __init__(self, target: float = 2.5, std: float = 2.0):
        self.target = target
        self.std = std

    def score(self, smiles: str) -> float:
        mol = _rdkit_mol(smiles)
        if mol is None:
            return 0.0
        try:
            from rdkit.Chem import Descriptors
            logp = Descriptors.MolLogP(mol)
            return float(np.clip(np.exp(-0.5 * ((logp - self.target) / self.std) ** 2), 0.0, 1.0))
        except Exception:
            return 0.0


class MolecularWeightReward(BaseReward):
    """
    Molecular weight reward. Scores 1.0 inside [mw_min, mw_max], with
    linear decay outside the range (soft penalty, not hard cutoff).
    Default window: [160, 500] Da (Lipinski).
    """

    name = "mol_weight"

    def __init__(self, mw_min: float = 160.0, mw_max: float = 500.0):
        self.mw_min = mw_min
        self.mw_max = mw_max

    def score(self, smiles: str) -> float:
        mol = _rdkit_mol(smiles)
        if mol is None:
            return 0.0
        try:
            from rdkit.Chem import Descriptors
            mw = Descriptors.MolWt(mol)
            if self.mw_min <= mw <= self.mw_max:
                return 1.0
            if mw < self.mw_min:
                return float(np.clip(mw / self.mw_min, 0.0, 1.0))
            excess = mw - self.mw_max
            return float(np.clip(1.0 - excess / self.mw_max, 0.0, 1.0))
        except Exception:
            return 0.0


class RingCountReward(BaseReward):
    """
    Ring count reward. Scores 1.0 for [min_rings, max_rings], 0.0 below,
    and linearly decays above. Default window: [1, 4] rings.
    """

    name = "ring_count"

    def __init__(self, min_rings: int = 1, max_rings: int = 4):
        self.min_rings = min_rings
        self.max_rings = max_rings

    def score(self, smiles: str) -> float:
        mol = _rdkit_mol(smiles)
        if mol is None:
            return 0.0
        try:
            from rdkit.Chem import rdMolDescriptors
            n = rdMolDescriptors.CalcNumRings(mol)
            if self.min_rings <= n <= self.max_rings:
                return 1.0
            if n < self.min_rings:
                return 0.0
            return float(np.clip(1.0 - (n - self.max_rings) / self.max_rings, 0.0, 1.0))
        except Exception:
            return 0.0


class LipinskiSuiteReward(BaseReward):
    """
    Lipinski Rule-of-Five: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10.

    Score = fraction of rules passed ∈ {0.0, 0.25, 0.5, 0.75, 1.0}.
    """

    name = "lipinski"

    def score(self, smiles: str) -> float:
        mol = _rdkit_mol(smiles)
        if mol is None:
            return 0.0
        try:
            from rdkit.Chem import Descriptors, rdMolDescriptors
            rules = [
                Descriptors.MolWt(mol) <= 500,
                Descriptors.MolLogP(mol) <= 5,
                rdMolDescriptors.CalcNumHBD(mol) <= 5,
                rdMolDescriptors.CalcNumHBA(mol) <= 10,
            ]
            return float(sum(rules)) / len(rules)
        except Exception:
            return 0.0
