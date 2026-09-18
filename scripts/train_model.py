"""Train the rainfall-risk classifier + rainfall-amount regressor.

Usage:
    python scripts/train_model.py --district "Thiruvananthapuram"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_DISTRICT
from src.ml.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--district", default=DEFAULT_DISTRICT)
    parser.add_argument("--state", default=None)
    args = parser.parse_args()

    results = train(district=args.district, state=args.state)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
