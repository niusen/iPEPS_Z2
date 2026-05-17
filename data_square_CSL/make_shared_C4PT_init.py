import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\My Documents\Code\python_codes\iPEPS_Z2")

from ansatz.square_iPEPS import dense_tensor_from_array, save_square_iPEPS


def make_c4v_a1(A):
    A = 0.5 * (A + A.permute(0, 1, 4, 3, 2))
    A = 0.5 * (A + A.permute(0, 3, 2, 1, 4))
    A = 0.5 * (A + A.permute(0, 4, 1, 2, 3))
    A = 0.5 * (A + A.permute(0, 2, 3, 4, 1))
    return A


def make_c4v_a2(A):
    A = 0.5 * (A - A.permute(0, 1, 4, 3, 2))
    A = 0.5 * (A - A.permute(0, 4, 3, 2, 1))
    A = 0.5 * (A + A.permute(0, 4, 1, 2, 3))
    A = 0.5 * (A + A.permute(0, 3, 4, 1, 2))
    return A


def write_juraj_c4v_state(A, path):
    A_cpu = A.detach().cpu().numpy()
    entries = []
    for inds in numpy.ndindex(A_cpu.shape):
        val = A_cpu[inds]
        entries.append(
            " ".join(str(i) for i in inds)
            + " "
            + repr(float(numpy.real(val)))
            + " "
            + repr(float(numpy.imag(val)))
        )
    data = {
        "lX": 1,
        "lY": 1,
        "siteIds": ["A0"],
        "map": [{"siteId": "A0", "x": 0, "y": 0}],
        "sites": [
            {
                "dtype": "complex128",
                "dims": list(A_cpu.shape),
                "numEntries": len(entries),
                "entries": entries,
                "siteId": "A0",
            }
        ],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--square-prefix", default="data_square_CSL/shared_init_D3_seed0")
    parser.add_argument(
        "--juraj-state",
        default=r"D:\My Documents\Code\python_codes\Juraj\tn-torch_dev\test_csl\shared_init_D3_seed0_state.json",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    raw_re = torch.randn((2, args.D, args.D, args.D, args.D), dtype=torch.float64)
    raw_im = torch.randn((2, args.D, args.D, args.D, args.D), dtype=torch.float64)
    A_juraj = make_c4v_a1(raw_re) + 1.0j * make_c4v_a2(raw_im)
    A_juraj = A_juraj / torch.linalg.norm(A_juraj)

    write_juraj_c4v_state(A_juraj, args.juraj_state)

    config_kwargs = {
        "backend": "torch",
        "default_dtype": "complex128",
        "default_device": "cpu",
        "Lx": 1,
        "Ly": 1,
        "checkerboard_spin_transform": "sigmay",
    }
    # Juraj/peps-torch stores [p,u,l,d,r]. iPEPS_Z2 square code uses [l,d,r,u,p].
    A_square = numpy.transpose(A_juraj.detach().cpu().numpy(), (2, 3, 4, 1, 0))
    A_set = OrderedDict({"1,1": dense_tensor_from_array(A_square, config_kwargs, dual=[0, 1, 0, 1, 0])})
    save_square_iPEPS(A_set, args.square_prefix, config_kwargs)

    print("square_prefix:", args.square_prefix)
    print("juraj_state:", args.juraj_state)


if __name__ == "__main__":
    main()
