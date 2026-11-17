"""Evaluate a trained DSFNet checkpoint."""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-c", "-cfg", "--config", required=True,
                        help="Configuration path relative to the repository root.")
    parser.add_argument("-ckpt", "--checkpoint", required=True,
                        help="Path to the checkpoint produced by training.")
    parser.add_argument("-g", "--gpus", default="0", help="Visible GPU IDs.")
    parser.add_argument("-d", "--device_type", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("-b", "--batch_size", type=int, default=None,
                        help="Override the evaluation batch size.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    import basicts

    basicts.launch_evaluation(cfg=args.config, ckpt_path=args.checkpoint,
                              device_type=args.device_type, gpus=args.gpus,
                              batch_size=args.batch_size)


if __name__ == "__main__":
    main()
