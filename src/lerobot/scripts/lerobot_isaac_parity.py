#!/usr/bin/env python
"""Compare native ISAAC inference/training parity artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.policies.perceptron_isaac.parity import compare_isaac_parity_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = compare_isaac_parity_artifacts(
        args.reference,
        args.candidate,
        raise_on_failure=False,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.write_text(rendered)
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
