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
    parser.add_argument("--mode", choices=("static", "online_growth", "full_chat"),
                        help="Override experiment_mode for this run without changing the config.")
    parser.add_argument("--max-questions", type=int,
                        help="Only used by full_chat; defaults to full_chat.max_questions.")
    args = parser.parse_args()
    config = load_config(args.config)
    mode = args.mode or config.get("experiment_mode", "static")

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

    if mode == "full_chat":
        from scripts.run_full_chat import main as run_full_chat

        sys.argv = [sys.argv[0], "--config", args.config]
        if args.name:
            sys.argv.extend(["--name", args.name])
        if args.max_questions is not None:
            sys.argv.extend(["--max-questions", str(args.max_questions)])
        run_full_chat()
        return

    raise ValueError(
        f"Unsupported experiment_mode '{mode}'. "
        "Expected 'static', 'online_growth', or 'full_chat'."
    )


if __name__ == "__main__":
    main()