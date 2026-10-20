from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.data import load_gpace_data
from gpace.splits import final_external_sets, make_outer_split


def _sequence_length(value: str) -> int | None:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("sequence length must be non-negative")
    return parsed or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GPACE source tables and joins")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--sequence-length",
        type=_sequence_length,
        default=100,
        help="Daily environmental days per environment; use 0 for the complete season",
    )
    args = parser.parse_args()
    data = load_gpace_data(args.data_dir, sequence_length=args.sequence_length)
    print(json.dumps(data.audit, indent=2, ensure_ascii=False))
    for split_number in range(1, 11):
        split = make_outer_split(data, split_number)
        print(
            f"split={split_number} train={len(split.train)} cv0={len(split.cv0)} "
            f"cv1={len(split.cv1)} cv00={len(split.cv00)}"
        )
    external = final_external_sets(data)
    print(
        f"external_g_ne={len(external['external_g_ne'])} "
        f"external_ng_ne={len(external['external_ng_ne'])}"
    )


if __name__ == "__main__":
    main()
