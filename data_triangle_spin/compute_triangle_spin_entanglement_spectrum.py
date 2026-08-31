"""Momentum-resolved entanglement spectrum of a saved triangular spin iPESS.

The EH matvec is ported from tn-torch_dev.  Its streamed-chi implementation is
enabled by default.  Edit the hard-coded settings in this file, then submit
with ``run_ES_2x1.sh``.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
HERE = os.path.dirname(os.path.abspath(__file__))

# Absolute source-code directory on the server.
source_code_dir = "/home/sniu/python_code/iPESS_triangle_spin"
sys.path.insert(0, source_code_dir)

import numpy as np
import torch
from scipy.io import savemat

from ansatz.bosonic_triangle_iPESS import (
    IPESS_TRIANGLE_DENSE,
    bosonic_triangle_iPESS_bond_dimension,
    load_bosonic_triangle_iPESS,
)
from ansatz.triangle_iPESS import Cell_to_device
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS
from ctmrg.CTMRG_unitcell_iPESS import CTMRG_cell_iPESS
from entanglement.generic_eh_momentum_stream import (
    apply_expEH_once,
    get_EH_spec_Ttensor_momentum,
)
from entanglement.triangle_ipess_adapter import (
    build_dense_ipeps_and_env,
    conversion_double_layer_relative_error,
)
from model.bosonic_spin_ob_iPESS import evaluate_triangle_spin_energy


########################
# Hamiltonian parameters used for the pre-EH energy check
########################
J1 = 1.0
Jchi = 0.4
parameters = {"J1": J1, "Jchi": Jchi}


########################
# State and CTMRG parameters
########################
# Prefix without .json. Relative paths are resolved beside this file.
state_file = "triangle_spin_J1_1_Jchi_0p4_D8_chi80_Lx2_Ly1"
chi = 40
CTM_ite_nums = 100
CTM_trun_tol = 1.0e-8
CTM_conv_tol = 1.0e-8
doublelayer_on_cpu = False
use_sub_checkpoint = False


########################
# Entanglement-spectrum parameters
########################
EH_L_values = (4, 6, 8)
EH_n = 80
coord = (0, 0)                 # zero-based tn-torch_dev coordinate
direction = (1, 0)             # original code supports x transfer only
return_momentum = True
EH_full_diag_threshold = 4096
stream_chi = True              # O(chi*D**L), not O(chi**2*D**L)
print_each_matvec = True       # flush one progress line after every EH matvec

# Optional small-size algebra check of streamed vs dense chi trace.
compare_dense_matvec = False
compare_dense_max_dim = 256

write_mat = True
mat_file = None                # None builds one name per L from state_file


########################
# Runtime parameters
########################
device = "cuda:1"
default_dtype = "complex128"
n_cpu = 10
seed = 1234


def _state_prefix():
    return state_file if os.path.isabs(state_file) else os.path.join(HERE, state_file)


def _detect_unit_cell(prefix):
    with open(prefix + ".json") as stream:
        payload = json.load(stream)
    keys = set(payload["B_set"])
    if not keys or keys != set(payload["T_set"]):
        raise ValueError("B_set and T_set must contain the same non-empty cell")
    coordinates = {tuple(int(v) for v in key.split(",")) for key in keys}
    Lx = max(c[0] for c in coordinates)
    Ly = max(c[1] for c in coordinates)
    expected = {(cx, cy) for cx in range(1, Lx + 1) for cy in range(1, Ly + 1)}
    if coordinates != expected:
        raise ValueError("state JSON has an incomplete cell: " + str(sorted(coordinates)))
    return Lx, Ly


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _energies_from_spec(spec):
    return -torch.log(torch.clamp(torch.linalg.norm(spec, dim=1), min=1.0e-300))


def _fmt_complex_pair(row):
    z = complex(float(row[0]), float(row[1]))
    return f"{z.real:.16e}{z.imag:+.16e}j"


def _print_spectrum(spec, energies, k_phases=None):
    for i in range(spec.size(0)):
        if k_phases is None:
            print(f"{i}, {_fmt_complex_pair(spec[i])}, {energies[i].item():.16e}")
        else:
            phase_i = torch.complex(k_phases[i, 0], k_phases[i, 1])
            angle = torch.angle(phase_i).item()
            print(
                f"{i}, {_fmt_complex_pair(spec[i])}, {energies[i].item():.16e}, "
                f"k_phase={_fmt_complex_pair(k_phases[i])}, k_angle={angle:.16e}"
            )


def _mat_file_for_L(prefix, L):
    if mat_file is None:
        return str(
            Path(prefix).with_name(
                Path(prefix).name + f"_generic_EH_L{L}_chi{chi}.mat"
            )
        )
    path = Path(mat_file)
    suffix = path.suffix if path.suffix else ".mat"
    stem = path.stem if path.suffix else path.name
    return str(path.with_name(f"{stem}_L{L}{suffix}"))


def _print_energy_check(energy, observables):
    print("PRE_EH_J1_JCHI_ENERGY= " + str(energy.item()))
    print("PRE_EH_chi_up= " + str(observables["chi_up"].tolist()))
    print("PRE_EH_chi_down= " + str(observables["chi_down"].tolist()))
    print("PRE_EH_SS_x= " + str(observables["SS_x"].tolist()))
    print("PRE_EH_SS_y= " + str(observables["SS_y"].tolist()))
    print("PRE_EH_SS_diagonal= " + str(observables["SS_diagonal"].tolist()))


def main():
    prefix = _state_prefix()
    Lx, Ly = _detect_unit_cell(prefix)
    print("pid= " + str(os.getpid()))
    print("source_code_dir= " + repr(source_code_dir))
    print("state_file= " + repr(prefix + ".json"))
    print("J1= " + repr(J1) + ", Jchi= " + repr(Jchi))
    print("Lx= " + str(Lx) + ", Ly= " + str(Ly))
    print("chi= " + str(chi))
    print("EH_L_values= " + repr(EH_L_values))
    print("EH_n= " + str(EH_n))
    print("coord= " + repr(coord) + ", direction= " + repr(direction))
    print("stream_chi= " + str(stream_chi))
    print("EH_full_diag_threshold= " + str(EH_full_diag_threshold))
    print("device= " + repr(device) + ", n_cpu= " + str(n_cpu))

    torch.set_num_threads(n_cpu)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config_kwargs = {
        "backend": "torch",
        "default_dtype": default_dtype,
        "default_device": device,
        "Lx": Lx,
        "Ly": Ly,
    }
    global_args = GLOBALARGS()
    global_args.Lx, global_args.Ly, global_args.device = Lx, Ly, device

    B_set, T_set = load_bosonic_triangle_iPESS(prefix, config_kwargs)
    ipess_state = IPESS_TRIANGLE_DENSE(B_set, T_set, config_kwargs)
    ipess_state.require_grad(False)
    ipess_state.to_device(device)
    ipess_state.normalize()
    bond_dim = bosonic_triangle_iPESS_bond_dimension(ipess_state)
    print("bond_dim= " + str(bond_dim))

    ctm_args = CTMARGS()
    ctm_args.CTM_ite_info = True
    ctm_args.chi = chi
    ctm_args.CTM_ite_nums = CTM_ite_nums
    ctm_args.CTM_trun_tol = CTM_trun_tol
    ctm_args.CTM_conv_tol = CTM_conv_tol
    ctm_args.doublelayer_on_cpu = doublelayer_on_cpu
    ctm_args.use_sub_checkpoint = use_sub_checkpoint

    started = time.time()
    with torch.no_grad():
        CTM_cell, double_B_set, double_T_set, ite_num, ite_err = CTMRG_cell_iPESS(
            ipess_state.B_set,
            ipess_state.T_set,
            INITCTMARGS(),
            None,
            ctm_args,
            global_args,
        )
        if doublelayer_on_cpu and device != double_B_set["1,1"].device:
            double_B_set = Cell_to_device(double_B_set, device, global_args)
            double_T_set = Cell_to_device(double_T_set, device, global_args)
        energy, observables = evaluate_triangle_spin_energy(
            parameters,
            ipess_state.B_set,
            ipess_state.T_set,
            double_B_set,
            double_T_set,
            CTM_cell,
            config_kwargs,
            global_args,
            return_observables=True,
        )
    ctm_time = time.time() - started
    print(
        "CTM ite_num=" + str(ite_num) + ", ite_err=" + str(ite_err)
        + ", converged=" + str(ite_err < CTM_conv_tol)
        + ", time=" + str(ctm_time)
    )
    _print_energy_check(energy, observables)

    A_set, state, env = build_dense_ipeps_and_env(
        ipess_state.B_set, ipess_state.T_set, CTM_cell, global_args
    )
    conversion_errors = conversion_double_layer_relative_error(
        ipess_state.B_set, ipess_state.T_set, A_set
    )
    print("IPESS_TO_IPEPS_DOUBLE_LAYER_RELERR= " + str(conversion_errors))
    print("EH_actual_env_chi= " + str(env.chi))

    base_mat_data = {
        "instate": np.asarray(prefix + ".json"),
        "J1": np.asarray(J1),
        "Jchi": np.asarray(Jchi),
        "energy": _to_numpy(energy),
        "chi": np.asarray(chi),
        "actual_env_chi": np.asarray(env.chi),
        "bond_dim": np.asarray(bond_dim),
        "coord": np.asarray(coord),
        "direction": np.asarray(direction),
        "EH_n": np.asarray(EH_n),
        "EH_full_diag_threshold": np.asarray(EH_full_diag_threshold),
        "stream_chi": np.asarray(int(stream_chi)),
        "ctm_sweeps": np.asarray(ite_num),
        "ctm_conv_crit": np.asarray(ite_err),
        "ctm_converged": np.asarray(int(ite_err < CTM_conv_tol)),
        "ctm_time": np.asarray(ctm_time),
    }

    print(
        "EH_generic, chi, actual_env_chi, bond_dim, coord, direction, "
        "ctm_sweeps, ctm_conv_crit, ctm_converged, ctm_time, stream_chi"
    )
    print(
        f"EH_generic, {chi}, {env.chi}, {bond_dim}, {coord}, {direction}, "
        f"{ite_num}, {ite_err}, {ite_err < CTM_conv_tol}, {ctm_time}, {stream_chi}"
    )

    for L in EH_L_values:
        dim = bond_dim ** L
        n_eff = min(EH_n, max(dim - 2, 1))
        if n_eff < 1 or dim <= 2:
            print(f"\nEH L={L}: skipped because boundary dimension D^L={dim} is too small")
            continue
        print(f"\nEH L={L} dim={dim} n={n_eff}")

        if compare_dense_matvec and dim <= compare_dense_max_dim:
            rng = np.random.default_rng(seed + L)
            vector = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            streamed = apply_expEH_once(
                vector, L, coord, direction, state, env, stream_chi=True
            )
            dense = apply_expEH_once(
                vector, L, coord, direction, state, env, stream_chi=False
            )
            denominator = max(float(torch.linalg.norm(dense)), 1.0e-300)
            relative_error = float(torch.linalg.norm(streamed - dense)) / denominator
            print("EH_STREAM_DENSE_MATVEC_RELERR= " + str(relative_error))

        started_eh = time.time()
        result = get_EH_spec_Ttensor_momentum(
            n_eff,
            L,
            coord,
            direction,
            state,
            env,
            return_momentum=return_momentum,
            full_diag_threshold=EH_full_diag_threshold,
            stream_chi=stream_chi,
            print_each_matvec=print_each_matvec,
        )
        eh_time = time.time() - started_eh
        if result is None:
            print(f"EH L={L}: skipped because EH dimension is not larger than n")
            continue
        if return_momentum:
            spec, k_phases, k_phases_inverse = result
        else:
            spec = result
            k_phases = None
            k_phases_inverse = None
        energies = _energies_from_spec(spec)
        _print_spectrum(spec, energies, k_phases)
        print(f"EH_TIME L={L} {eh_time}")

        if write_mat:
            mat_data = dict(base_mat_data)
            mat_data["EH_L"] = np.asarray(L)
            mat_data["EH_dim"] = np.asarray(dim)
            mat_data["EH_n_eff"] = np.asarray(n_eff)
            mat_data["EH_time"] = np.asarray(eh_time)
            mat_data["EH_spec"] = _to_numpy(spec)
            mat_data["EH_energy"] = _to_numpy(energies)
            if k_phases is not None:
                mat_data["EH_k_phase"] = _to_numpy(k_phases)
                mat_data["EH_k_angle"] = _to_numpy(
                    torch.angle(torch.complex(k_phases[:, 0], k_phases[:, 1]))
                )
                mat_data["EH_k_phase_inverse"] = _to_numpy(k_phases_inverse)
                mat_data["EH_k_angle_inverse"] = _to_numpy(
                    torch.angle(
                        torch.complex(
                            k_phases_inverse[:, 0], k_phases_inverse[:, 1]
                        )
                    )
                )
            output_file = _mat_file_for_L(prefix, L)
            savemat(output_file, mat_data)
            print("MATLAB output written to " + output_file)


if __name__ == "__main__":
    main()
