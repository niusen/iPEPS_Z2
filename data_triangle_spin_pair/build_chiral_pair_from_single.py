"""Build a d=4, D^2 chiral x conjugate-chiral iPESS from a d=2, D state."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
source_code_dir = "/home/sniu/python_code/iPESS_triangle_spin"
sys.path.insert(0, source_code_dir)

import torch

from ansatz.bosonic_triangle_iPESS import (
    IPESS_TRIANGLE_DENSE,
    bosonic_triangle_iPESS_bond_dimension,
    chiral_pair_from_single_bosonic_triangle_iPESS,
    load_bosonic_triangle_iPESS,
    save_bosonic_triangle_iPESS,
)
from model.bosonic_spin_pair_ob_iPESS import (
    TriangleSpinChiralPairModel,
    fixed_d4_to_two_spin_unitary,
)


# Prefixes without .json; relative paths are resolved beside this script.
single_state_file = "triangle_spin_J1_1_Jchi_0p4_D2_chi40_Lx1_Ly1"
output_state_file = None  # None -> <single name>_chiral_pair_D<single_D^2>
device = "cpu"
default_dtype = "complex128"


def _prefix(name):
    return name if os.path.isabs(name) else os.path.join(HERE, name)


def _detect_cell(prefix):
    with open(prefix + ".json") as stream:
        payload = json.load(stream)
    keys = set(payload["B_set"])
    if not keys or keys != set(payload["T_set"]):
        raise ValueError("B_set and T_set must contain the same non-empty cell")
    coords = {tuple(int(value) for value in key.split(",")) for key in keys}
    Lx = max(coord[0] for coord in coords)
    Ly = max(coord[1] for coord in coords)
    expected = {(cx, cy) for cx in range(1, Lx + 1) for cy in range(1, Ly + 1)}
    if coords != expected:
        raise ValueError("input JSON has an incomplete rectangular unit cell")
    return Lx, Ly


def main():
    input_prefix = _prefix(single_state_file)
    Lx, Ly = _detect_cell(input_prefix)
    config_kwargs = {
        "backend": "torch",
        "default_dtype": default_dtype,
        "default_device": device,
        "Lx": Lx,
        "Ly": Ly,
    }
    B_set, T_set = load_bosonic_triangle_iPESS(input_prefix, config_kwargs)
    single = IPESS_TRIANGLE_DENSE(B_set, T_set, config_kwargs)
    if single.T_set["1,1"].get_shape()[1] != 2:
        raise ValueError("input single-layer state must have physical dimension d=2")
    single_D = bosonic_triangle_iPESS_bond_dimension(single)

    # U is constructed once and passed to both state construction and model
    # operator construction. The fingerprint is logged for reproducibility.
    dtype = torch.complex64 if default_dtype == "complex64" else torch.complex128
    split_unitary = fixed_d4_to_two_spin_unitary(dtype=dtype, device=device)
    pair_model = TriangleSpinChiralPairModel(config_kwargs, split_unitary)
    pair = chiral_pair_from_single_bosonic_triangle_iPESS(single, pair_model.split_unitary)
    pair_D = bosonic_triangle_iPESS_bond_dimension(pair)
    if pair_D != single_D**2 or pair.T_set["1,1"].get_shape()[1] != 4:
        raise RuntimeError("constructed chiral-pair state has inconsistent dimensions")

    if output_state_file is None:
        output_prefix = input_prefix + "_chiral_pair_D" + str(pair_D)
    else:
        output_prefix = _prefix(output_state_file)
    save_bosonic_triangle_iPESS(
        pair.B_set, pair.T_set, output_prefix, config_kwargs
    )
    print("pid= " + str(os.getpid()))
    print("source state= " + repr(input_prefix + ".json"))
    print("output state= " + repr(output_prefix + ".json"))
    print("unit cell= " + str((Lx, Ly)))
    print("single physical dimension=2, D=" + str(single_D))
    print("pair physical dimension=4, D=" + str(pair_D))
    print("physical split convention: p=2*a+b")
    print("physical split unitary fingerprint:", pair_model.unitary_fingerprint())


if __name__ == "__main__":
    main()
