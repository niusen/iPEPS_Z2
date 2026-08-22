import argparse
import os
import sys

import numpy
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IPEPS_REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, IPEPS_REPO)

from analyze_captured_raw import main as analyze_main


if __name__ == "__main__":
    # This wrapper intentionally exists so the same analysis can be launched from
    # a different entry point while preserving all command-line arguments.
    analyze_main()
