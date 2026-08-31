"""Compute observables of a saved d=2 Z2 fermionic triangular spin iPESS.

Edit the hard-coded settings below. This program performs no optimization and
builds no autograd graph.
"""

import json
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
HERE = os.path.dirname(os.path.abspath(__file__))

# Absolute source-code directory on the server.
source_code_dir = "/home/sniu/python_code/iPESS_triangle_spin"
sys.path.insert(0, source_code_dir)

import torch

from ansatz.fermionic_spin_triangle_iPESS import (
    fermionic_spin_bond_dimension,
    load_fermionic_spin_triangle_iPESS,
    make_fermionic_spin_state,
)
from ansatz.triangle_iPESS import Cell_to_device
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS
from ctmrg.Fermionic_CTMRG_unitcell_iPESS import Fermionic_CTMRG_cell_iPESS
from model.fermionic_spin_ob_iPESS import evaluate_fermionic_triangle_spin_energy


########################
# Hamiltonian parameters
########################
J1 = 1.0
Jchi = 0.4
parameters = {"J1": J1, "Jchi": Jchi}


########################
# State and CTMRG parameters
########################
# Leave these as None to infer the complete rectangular unit cell from JSON.
# Setting either value explicitly turns it into a consistency check.
Lx = None
Ly = None
expected_D = None
chi = 80
CTM_ite_nums = 100
CTM_trun_tol = 1.0e-8
CTM_conv_tol = 1.0e-8
doublelayer_on_cpu = False
use_sub_checkpoint = False

# Prefix without .json. Relative paths are resolved beside this file.
state_file = "triangle_spin_fermionic_J1_1_Jchi_0p4_D8_chi80_Lx2_Ly2"


########################
# Runtime parameters
########################
device = "cuda:1"
default_dtype = "complex128"
n_cpu = 10


def _state_prefix():
    return state_file if os.path.isabs(state_file) else os.path.join(HERE, state_file)


def detect_unit_cell(prefix):
    """Return Lx,Ly from JSON keys and reject incomplete/mismatched cells."""
    with open(prefix + ".json") as stream:
        payload = json.load(stream)
    B_keys = set(payload["B_set"])
    T_keys = set(payload["T_set"])
    if not B_keys or B_keys != T_keys:
        raise ValueError("B_set and T_set must contain the same non-empty cell")
    try:
        coordinates = {
            tuple(int(value) for value in key.split(",")) for key in B_keys
        }
    except (TypeError, ValueError) as error:
        raise ValueError("invalid unit-cell key in state JSON") from error
    if any(len(coordinate) != 2 for coordinate in coordinates):
        raise ValueError("unit-cell keys must have the form 'cx,cy'")
    detected_Lx = max(coordinate[0] for coordinate in coordinates)
    detected_Ly = max(coordinate[1] for coordinate in coordinates)
    expected_keys = {
        (cx, cy)
        for cx in range(1, detected_Lx + 1)
        for cy in range(1, detected_Ly + 1)
    }
    if coordinates != expected_keys:
        raise ValueError(
            "state JSON does not contain a complete rectangular unit cell: "
            + str(sorted(coordinates))
        )
    return detected_Lx, detected_Ly


def show_settings(title, settings):
    print(title + ":")
    for name, value in settings.items():
        print("  " + name + " = " + repr(value))


def print_observables(energy, observables):
    print("E= " + str(energy.item()))
    print("scalar chirality (up): " + str(observables["chi_up"].tolist()))
    print(
        "scalar chirality (down, common orientation): "
        + str(observables["chi_down"].tolist())
    )
    print("S.S x: " + str(observables["SS_x"].tolist()))
    print("S.S y: " + str(observables["SS_y"].tolist()))
    print("S.S diagonal: " + str(observables["SS_diagonal"].tolist()))
    print("magnetization components:")
    print(observables["Sx"].tolist())
    print(observables["Sy"].tolist())
    print(observables["Sz"].tolist())
    print("total magnetization:")
    magnetization = torch.sqrt(
        observables["Sx"] ** 2
        + observables["Sy"] ** 2
        + observables["Sz"] ** 2
    )
    print(magnetization.tolist())


def main():
    prefix = _state_prefix()
    detected_Lx, detected_Ly = detect_unit_cell(prefix)
    if Lx is not None and Lx != detected_Lx:
        raise ValueError(
            "configured Lx=" + str(Lx)
            + ", but JSON contains Lx=" + str(detected_Lx)
        )
    if Ly is not None and Ly != detected_Ly:
        raise ValueError(
            "configured Ly=" + str(Ly)
            + ", but JSON contains Ly=" + str(detected_Ly)
        )
    cell_Lx = detected_Lx
    cell_Ly = detected_Ly
    print("pid= " + str(os.getpid()))
    print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    show_settings("Hamiltonian parameters", parameters)
    show_settings(
        "State and CTMRG parameters",
        {
            "source_code_dir": source_code_dir,
            "state_file": prefix + ".json",
            "Lx": cell_Lx,
            "Ly": cell_Ly,
            "expected_D": expected_D,
            "physical_dim": 2,
            "chi": chi,
            "CTM_ite_nums": CTM_ite_nums,
            "CTM_trun_tol": CTM_trun_tol,
            "CTM_conv_tol": CTM_conv_tol,
            "doublelayer_on_cpu": doublelayer_on_cpu,
            "use_sub_checkpoint": use_sub_checkpoint,
            "device": device,
            "default_dtype": default_dtype,
            "n_cpu": n_cpu,
        },
    )
    torch.set_num_threads(n_cpu)

    config_kwargs = {
        "backend": "torch",
        "default_dtype": default_dtype,
        "default_device": device,
        "Lx": cell_Lx,
        "Ly": cell_Ly,
    }
    global_args = GLOBALARGS()
    global_args.Lx, global_args.Ly, global_args.device = cell_Lx, cell_Ly, device

    B_set, T_set = load_fermionic_spin_triangle_iPESS(prefix, config_kwargs)
    state = make_fermionic_spin_state(B_set, T_set, config_kwargs)
    actual_D = fermionic_spin_bond_dimension(state)
    if expected_D is not None and actual_D != expected_D:
        raise ValueError(
            "loaded state has D=" + str(actual_D)
            + ", expected D=" + str(expected_D)
        )
    print("loaded state bond dimension: D=" + str(actual_D))
    state.require_grad(False)
    state.to_device(device)
    state.normalize()

    ctm_args = CTMARGS()
    ctm_args.CTM_ite_info = True
    ctm_args.chi = chi
    ctm_args.CTM_ite_nums = CTM_ite_nums
    ctm_args.CTM_trun_tol = CTM_trun_tol
    ctm_args.CTM_conv_tol = CTM_conv_tol
    ctm_args.doublelayer_on_cpu = doublelayer_on_cpu
    ctm_args.use_sub_checkpoint = use_sub_checkpoint
    print(ctm_args)

    with torch.no_grad():
        CTM_cell, double_B_set, double_T_set, ite_num, ite_err = (
            Fermionic_CTMRG_cell_iPESS(
                state.B_set,
                state.T_set,
                INITCTMARGS(),
                None,
                ctm_args,
                global_args,
            )
        )
        if doublelayer_on_cpu and device != double_B_set["1,1"].device:
            double_B_set = Cell_to_device(double_B_set, device, global_args)
            double_T_set = Cell_to_device(double_T_set, device, global_args)
        energy, observables = evaluate_fermionic_triangle_spin_energy(
            parameters,
            state.B_set,
            state.T_set,
            double_B_set,
            double_T_set,
            CTM_cell,
            config_kwargs,
            global_args,
            return_observables=True,
        )

    print("CTM ite_num=" + str(ite_num) + ", ite_err=" + str(ite_err))
    print_observables(energy, observables)


if __name__ == "__main__":
    main()
