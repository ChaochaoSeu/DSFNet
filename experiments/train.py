"""Train DSFNet using a dataset-specific configuration."""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--cfg", default="baselines/DSFNet_NJRC/NJRC.py",
                        help="Configuration path relative to the repository root.")
    parser.add_argument("-g", "--gpus", default="0", help="Visible GPU IDs, for example 0 or 0,1.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    import torch
    import basicts

    torch.set_num_threads(2)
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)


if __name__ == "__main__":
    main()
