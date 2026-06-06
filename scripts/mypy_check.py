"""mypy_check.py – invoke mypy on the sci_reward package and exit non-zero on errors.

Used by GitHub Actions as a standalone static type check step so mypy output
appears in its own CI job rather than mixed with pytest output.
"""

import subprocess
import sys

result = subprocess.run(
    [
        sys.executable, "-m", "mypy",
        "sci_reward",
        "--ignore-missing-imports",
        "--no-strict-optional",
        "--allow-untyped-defs",
        "--allow-untyped-calls",
        "--no-warn-return-any",
        "--no-error-summary",
        "--disable-error-code=attr-defined",
        "--disable-error-code=arg-type",
    ],
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)
