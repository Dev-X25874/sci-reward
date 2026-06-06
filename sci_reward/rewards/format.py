"""Format and syntax rewards for molecular string representations.

Purely heuristic — no RDKit dependency — so these run as fast pre-filters
before more expensive reward heads.

SMILESFormatReward scores four orthogonal dimensions:
  - character cleanliness   (fraction of valid SMILES alphabet)
  - bracket balance         (parentheses and square brackets)
  - absence of hallucination patterns (all-caps runs, long digit sequences …)
  - length plausibility     (inside [min_len, max_len])

IUPACFormatReward scores IUPAC nomenclature conventions:
  - carbon-chain prefix presence (meth, eth, prop, …)
  - functional-group suffix presence (ol, one, oic acid, …)
  - locant formatting (digit followed by comma or dash)
"""

from __future__ import annotations

import re

import numpy as np

from sci_reward.rewards.base import BaseReward


_SMILES_VALID_CHARS = set(
    "BCNOPSFIbcnops"    # atoms
    "=#@/\\.+-"         # bonds and stereo
    "[]()%0123456789"   # brackets, ring closures
    "HhRrXx"            # hydrogen, wildcards
)

_SMILES_BAD_RE = re.compile(
    r"\s{2,}"       # multiple spaces
    r"|[A-Z]{4,}"   # long all-caps runs
    r"|\d{5,}"      # 5+ digit sequences
    r"|[\[\]]{3,}"  # 3+ consecutive brackets
)

_IUPAC_PREFIXES = re.compile(
    r"\b(meth|eth|prop|but|pent|hex|hept|oct|non|dec|undec|dodec"
    r"|cyclo|benz|naph|phen|anthrac|iso|neo|sec|tert|di|tri|tetra"
    r"|penta|hexa|hepta|octa|nona|deca)\b",
    re.IGNORECASE,
)
_IUPAC_SUFFIXES = re.compile(
    r"\b(ane|ene|yne|ol|al|one|oic acid|amine|amide|ether|ester"
    r"|nitrile|thiol|sulfide|oxide|chloride|bromide|fluoride|iodide"
    r"|carboxylate|acetate|benzoate|phosphate|sulfate)\b",
    re.IGNORECASE,
)
_LOCANTS = re.compile(r"\b\d+[,-]")


class SMILESFormatReward(BaseReward):
    """
    SMILES format quality reward — beyond binary validity.

    Scores character cleanliness, bracket balance, absence of hallucination
    patterns, and length. Weights are configurable; balance failures carry
    the highest default penalty.

    Empty strings and whitespace-only strings always return 0.0.
    """

    name = "smiles_format"

    def __init__(
        self,
        min_len: int = 2,
        max_len: int = 300,
        char_weight: float = 0.2,
        balance_weight: float = 0.3,
        pattern_weight: float = 0.2,
        length_weight: float = 0.1,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.char_weight = char_weight
        self.balance_weight = balance_weight
        self.pattern_weight = pattern_weight
        self.length_weight = length_weight

    def _char_score(self, s: str) -> float:
        if not s:
            return 0.0
        return sum(c in _SMILES_VALID_CHARS for c in s) / len(s)

    def _balance_score(self, s: str) -> float:
        depth_r = depth_sq = 0
        for c in s:
            if c == "(":
                depth_r += 1
            elif c == ")":
                depth_r -= 1
                if depth_r < 0:
                    return 0.0
            elif c == "[":
                depth_sq += 1
            elif c == "]":
                depth_sq -= 1
                if depth_sq < 0:
                    return 0.0
        return 1.0 if depth_r == 0 and depth_sq == 0 else 0.0

    def _pattern_score(self, s: str) -> float:
        return 0.0 if _SMILES_BAD_RE.search(s) else 1.0

    def _length_score(self, s: str) -> float:
        n = len(s)
        if self.min_len <= n <= self.max_len:
            return 1.0
        if n < self.min_len:
            return 0.0
        return float(np.clip(1.0 - (n - self.max_len) / self.max_len, 0.0, 1.0))

    def score(self, smiles: str) -> float:
        if not smiles:
            return 0.0
        s = smiles.strip()
        if not s:
            return 0.0
        c  = self._char_score(s)
        b  = self._balance_score(s)
        p  = self._pattern_score(s)
        ln = self._length_score(s)

        weighted = (
            c  * self.char_weight
            + b  * self.balance_weight
            + p  * self.pattern_weight
            + ln * self.length_weight
        )
        base_weight = 1.0 - (
            self.char_weight + self.balance_weight
            + self.pattern_weight + self.length_weight
        )
        return float(np.clip(weighted + base_weight * min(c, b, p, ln), 0.0, 1.0))


class IUPACFormatReward(BaseReward):
    """
    IUPAC name format reward.

    Scores prefix presence (carbon chain descriptors), suffix presence
    (functional group), and locant formatting. Optionally performs a
    round-trip via pubchempy to verify chemical correctness (slower,
    requires network access).
    """

    name = "iupac_format"

    def __init__(
        self,
        check_roundtrip: bool = False,
        prefix_weight: float = 0.3,
        suffix_weight: float = 0.4,
        locant_weight: float = 0.3,
    ):
        self.check_roundtrip = check_roundtrip
        self.prefix_weight = prefix_weight
        self.suffix_weight = suffix_weight
        self.locant_weight = locant_weight

    def _prefix_score(self, name: str) -> float:
        return min(1.0, len(_IUPAC_PREFIXES.findall(name)) / 2.0)

    def _suffix_score(self, name: str) -> float:
        return 1.0 if _IUPAC_SUFFIXES.search(name) else 0.0

    def _locant_score(self, name: str) -> float:
        if len(name.split("-")) <= 1:
            return 1.0
        return 1.0 if _LOCANTS.search(name) else 0.5

    def _roundtrip_score(self, name: str) -> float:
        try:
            import pubchempy as pcp
            compounds = pcp.get_compounds(name, "name")
            if not compounds:
                return 0.0
            from rdkit import Chem
            return 1.0 if Chem.MolFromSmiles(compounds[0].isomeric_smiles) else 0.0
        except Exception:
            return 0.5  # network/parse failure — don't penalize

    def score(self, iupac_name: str) -> float:
        if not iupac_name or not iupac_name.strip():
            return 0.0
        name = iupac_name.strip().lower()
        base = (
            self._prefix_score(name) * self.prefix_weight
            + self._suffix_score(name) * self.suffix_weight
            + self._locant_score(name) * self.locant_weight
        )
        if self.check_roundtrip and self._roundtrip_score(iupac_name) < 0.5:
            return float(np.clip(base * 0.3, 0.0, 1.0))
        return float(np.clip(base, 0.0, 1.0))
