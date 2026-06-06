"""IUPAC-to-formula accuracy evaluation harness.

Evaluates a model's ability to convert IUPAC names to molecular formulas,
expressed as SMILES -> formula via RDKit. Includes a simulate mode that
runs without a live model endpoint for CI and local testing.

Usage:
    python benchmarks/iupac_bench.py --simulate
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from sci_reward.rewards.chemical import QEDReward, ValiditySMILES
from sci_reward.rewards.composite import CompositeReward
from sci_reward.rewards.chemical import SAScoreReward


BENCHMARK_DATA = [
    {"iupac": "ethanol",              "formula": "C2H6O",    "smiles": "CCO"},
    {"iupac": "propan-2-ol",          "formula": "C3H8O",    "smiles": "CC(C)O"},
    {"iupac": "benzene",              "formula": "C6H6",     "smiles": "c1ccccc1"},
    {"iupac": "2-methylpropan-1-ol",  "formula": "C4H10O",   "smiles": "CC(C)CO"},
    {"iupac": "acetic acid",          "formula": "C2H4O2",   "smiles": "CC(=O)O"},
    {"iupac": "acetone",              "formula": "C3H6O",    "smiles": "CC(C)=O"},
    {"iupac": "cyclohexane",          "formula": "C6H12",    "smiles": "C1CCCCC1"},
    {"iupac": "aniline",              "formula": "C6H7N",    "smiles": "Nc1ccccc1"},
    {"iupac": "toluene",              "formula": "C7H8",     "smiles": "Cc1ccccc1"},
    {"iupac": "phenol",               "formula": "C6H6O",    "smiles": "Oc1ccccc1"},
    {"iupac": "naphthalene",          "formula": "C10H8",    "smiles": "c1ccc2ccccc2c1"},
    {"iupac": "glucose",              "formula": "C6H12O6",  "smiles": "OCC1OC(O)C(O)C(O)C1O"},
    {"iupac": "aspirin",              "formula": "C9H8O4",   "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    {"iupac": "caffeine",             "formula": "C8H10N4O2","smiles": "Cn1c(=O)c2c(ncn2C)n(c1=O)C"},
    {"iupac": "ibuprofen",            "formula": "C13H18O2", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"},
]


def parse_formula(formula: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        result[element] = result.get(element, 0) + (int(count) if count else 1)
    return result


def formula_from_smiles(smiles: str) -> str | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        return rdMolDescriptors.CalcMolFormula(mol) if mol else None
    except Exception:
        return None


def element_match_score(pred: str, target: str) -> float:
    try:
        p, t = parse_formula(pred), parse_formula(target)
        elements = set(p) | set(t)
        return sum(p.get(e, 0) == t.get(e, 0) for e in elements) / len(elements)
    except Exception:
        return 0.0


class IUPACBenchmark:
    def __init__(self, data: list[dict] | None = None):
        self.data = data or BENCHMARK_DATA

    def evaluate(self, predictions: list[str]) -> dict[str, float]:
        """
        Evaluate predicted SMILES against benchmark ground truth.

        Args:
            predictions: One predicted SMILES per benchmark entry.

        Returns:
            Dict with exact_acc, partial_acc, validity_rate, mean_qed.
        """
        assert len(predictions) == len(self.data)
        validity = ValiditySMILES()
        qed = QEDReward()
        exact, partial, valid_flags, qeds = [], [], [], []

        for pred, item in zip(predictions, self.data):
            is_valid = validity.score(pred) > 0.5
            valid_flags.append(is_valid)
            if is_valid:
                pred_formula = formula_from_smiles(pred)
                exact.append(pred_formula is not None and pred_formula.upper() == item["formula"].upper())
                partial.append(element_match_score(pred_formula or "", item["formula"]))
                qeds.append(qed.score(pred))
            else:
                exact.append(False)
                partial.append(0.0)
                qeds.append(0.0)

        return {
            "exact_acc": float(np.mean(exact)),
            "partial_acc": float(np.mean(partial)),
            "validity_rate": float(np.mean(valid_flags)),
            "mean_qed": float(np.mean(qeds)),
            "n_samples": len(self.data),
        }

    def reference_scores(self) -> dict[str, float]:
        """Composite reward scores on ground-truth SMILES (upper bound reference)."""
        composite = CompositeReward(
            rewards=[ValiditySMILES(), QEDReward(), SAScoreReward()],
            weights=[0.4, 0.35, 0.25],
            gate=ValiditySMILES(),
        )
        scores = np.array(composite.batch_score([d["smiles"] for d in self.data]))
        return {"mean": float(scores.mean()), "min": float(scores.min()), "max": float(scores.max())}

    @staticmethod
    def print_results(results: dict[str, float], label: str = ""):
        tag = f" [{label}]" if label else ""
        print(f"=== IUPAC Benchmark{tag} ===")
        print(f"  Exact accuracy : {results.get('exact_acc', 0):.1%}")
        print(f"  Partial match  : {results.get('partial_acc', 0):.1%}")
        print(f"  Validity rate  : {results.get('validity_rate', 0):.1%}")
        print(f"  Mean QED       : {results.get('mean_qed', 0):.3f}")
        print(f"  N              : {results.get('n_samples', 0)}")
        print()


def _simulate_predictions(data: list[dict], accuracy: float) -> list[str]:
    rng = np.random.default_rng(42)
    wrong = ["CCN", "CCCC", "c1ccccc1N", "CC(=O)N", "CCCCO"]
    return [item["smiles"] if rng.random() < accuracy else rng.choice(wrong) for item in data]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true", default=True)
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()

    data = BENCHMARK_DATA
    if args.data_path:
        with open(args.data_path) as f:
            data = json.load(f)

    bench = IUPACBenchmark(data=data)

    print("Reference scores (ground-truth SMILES):")
    print(json.dumps(bench.reference_scores(), indent=2), "\n")

    if args.simulate:
        bench.print_results(bench.evaluate(_simulate_predictions(data, 0.15)), label="baseline ~15%")
        bench.print_results(bench.evaluate(_simulate_predictions(data, 0.50)), label="post-RL ~50%")


if __name__ == "__main__":
    main()
