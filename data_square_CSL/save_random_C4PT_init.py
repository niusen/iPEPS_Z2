import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ansatz.square_iPEPS import IPEPS_SQUARE_C4_PT, random_square_iPEPS, save_square_iPEPS


def main():
    print("pid= " + str(os.getpid()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-prefix", default="ipepsz2_random_C4PT_init_D3_seed0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config_kwargs = {
        "backend": "torch",
        "default_dtype": "complex128",
        "default_device": args.device,
        "Lx": 1,
        "Ly": 1,
        "checkerboard_spin_transform": "sigmay",
    }

    A_set = random_square_iPEPS(args.D, 2, config_kwargs)
    state = IPEPS_SQUARE_C4_PT(A_set, config_kwargs, impose_c4=True, impose_pt=True)
    state.normalize()
    save_square_iPEPS(state.A_set, args.out_prefix, config_kwargs)

    print("saved_random_C4PT_init:", args.out_prefix + ".json")
    print("D:", args.D)
    print("seed:", args.seed)


if __name__ == "__main__":
    main()
