#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run a configured experiment protocol.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--name", help="Optional experiment name; timestamp is always appended.")
    args = parser.parse_args()
    config = load_config(args.config)
    mode = config.get("experiment_mode", "static")

    if mode == "static":
        from scripts.run_evaluation import main as run_static

        sys.argv = [sys.argv[0], "--config", args.config]
        if args.name:
            sys.argv.extend(["--name", args.name])
        run_static()
        return

    if mode == "online_growth":
        from scripts.run_online_growth import main as run_growth

        sys.argv = [sys.argv[0], "--config", args.config]
        if args.name:
            sys.argv.extend(["--name", args.name])
        run_growth()
        return

    raise ValueError(
        f"Unsupported experiment_mode '{mode}'. "
        "Expected 'static' or 'online_growth'."
    )


if __name__ == "__main__":
    main()