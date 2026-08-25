"""Z2-symmetric fermionic triangular iPESS with a two-state spin leg.

The virtual legs and fermionic tensor ordering are inherited from the old
Hofstadter-Hubbard iPESS.  The physical leg, however, contains only the odd
Z2 sector ``(|up>, |down>)``.  Thus its total dimension is two and the
Gutzwiller constraint is exact, rather than imposed by a penalty.
"""

import json
from collections import OrderedDict

import numpy
import yastn

from ansatz.triangle_iPESS import IPESS_TRIANGLE, load_triangle_iPESS


FORMAT = "fermionic_spin_triangle_iPESS_d2_v1"


def _without_driver_options(config_kwargs):
    ignored = {"Lx", "Ly", "save_file_prefix"}
    return {key: value for key, value in config_kwargs.items() if key not in ignored}


def fermionic_z2_config(config_kwargs):
    return yastn.make_config(
        sym="Z2", fermionic=True, **_without_driver_options(config_kwargs)
    )


def _charge_to_int(charge):
    return int(charge[0]) if isinstance(charge, tuple) else int(charge)


def _encode_tensor(tensor):
    legs = tensor.get_legs()
    blocks = []
    for charges in tensor.get_blocks_charge():
        array = numpy.asarray(tensor[charges].detach().to("cpu"))
        flat = numpy.reshape(array, -1, order="F")
        blocks.append(
            {
                "charges": [_charge_to_int(charge) for charge in charges],
                "shape": list(array.shape),
                "real": numpy.real(flat).tolist(),
                "imag": numpy.imag(flat).tolist(),
            }
        )
    return {
        "signatures": list(tensor.s),
        "legs": [
            {
                "charges": [_charge_to_int(charge) for charge in leg.t],
                "dimensions": list(leg.D),
            }
            for leg in legs
        ],
        "blocks": blocks,
    }


def _decode_tensor(data, config):
    legs = [
        yastn.Leg(
            config,
            s=signature,
            t=tuple(leg_data["charges"]),
            D=tuple(leg_data["dimensions"]),
        )
        for signature, leg_data in zip(data["signatures"], data["legs"])
    ]
    tensor = yastn.zeros(config=config, legs=legs)
    for block in data["blocks"]:
        flat = numpy.asarray(block["real"]) + 1j * numpy.asarray(block["imag"])
        array = numpy.reshape(flat, tuple(block["shape"]), order="F")
        tensor.set_block(
            ts=tuple(block["charges"]), Ds=tuple(block["shape"]), val=array
        )
    return tensor


def save_fermionic_spin_triangle_iPESS(B_set, T_set, filenm, config_kwargs):
    """Save a d=2 fermionic-spin iPESS without assuming two sectors per leg."""
    payload = {"format": FORMAT, "B_set": {}, "T_set": {}}
    for cx in range(1, config_kwargs["Lx"] + 1):
        for cy in range(1, config_kwargs["Ly"] + 1):
            key = str(cx) + "," + str(cy)
            _validate_spin_physical_leg(T_set[key], key)
            payload["B_set"][key] = _encode_tensor(B_set[key])
            payload["T_set"][key] = _encode_tensor(T_set[key])
    with open(filenm + ".json", "w") as stream:
        json.dump(payload, stream)


def load_fermionic_spin_triangle_iPESS(filenm, config_kwargs):
    """Load the charge-aware d=2 fermionic-spin JSON format."""
    with open(filenm + ".json") as stream:
        payload = json.load(stream)
    if payload.get("format") != FORMAT:
        raise ValueError(
            "expected " + FORMAT + "; convert the old d=4 state first"
        )
    config = fermionic_z2_config(config_kwargs)
    B_set, T_set = OrderedDict(), OrderedDict()
    for cx in range(1, config_kwargs["Lx"] + 1):
        for cy in range(1, config_kwargs["Ly"] + 1):
            key = str(cx) + "," + str(cy)
            B_set[key] = _decode_tensor(payload["B_set"][key], config)
            T_set[key] = _decode_tensor(payload["T_set"][key], config)
            _validate_spin_physical_leg(T_set[key], key)
    return B_set, T_set


def _validate_spin_physical_leg(tensor, key=""):
    if tensor.get_rank() != 4:
        raise ValueError("site tensor " + key + " must have rank four")
    physical_leg = tensor.get_legs(axes=1)
    charges = tuple(_charge_to_int(charge) for charge in physical_leg.t)
    if charges != (1,) or tuple(physical_leg.D) != (2,):
        raise ValueError(
            "site tensor " + key
            + " must have d=2 odd physical leg t=(1,), D=(2,); found t="
            + str(charges) + ", D=" + str(tuple(physical_leg.D))
        )


def gutzwiller_project_d4_sets(B_set, T_set, config_kwargs):
    """Return an exact d=2 copy of a d=4 spinful-fermion iPESS.

    ``T=(M,s,R,D)`` and physical-axis charge one is copied in full.  In the
    old basis this block is ``(|up>, |down>)``.  Empty and doubly occupied
    entries (physical charge zero) are discarded.  B tensors and every
    virtual Z2 sector are unchanged.
    """
    config = fermionic_z2_config(config_kwargs)
    projected_B, projected_T = OrderedDict(), OrderedDict()
    for key in B_set:
        old_T = T_set[key]
        if old_T.get_rank() != 4:
            raise ValueError("site tensor " + key + " must have rank four")
        old_physical = old_T.get_legs(axes=1)
        charge_dims = {
            _charge_to_int(charge): dim
            for charge, dim in zip(old_physical.t, old_physical.D)
        }
        if charge_dims.get(1) != 2:
            raise ValueError(
                "old site tensor " + key
                + " must have a two-dimensional odd physical sector; found "
                + str(charge_dims)
            )

        old_legs = old_T.get_legs()
        new_physical = yastn.Leg(config, s=old_T.s[1], t=(1,), D=(2,))
        new_T = yastn.zeros(
            config=config,
            legs=[old_legs[0], new_physical, old_legs[2], old_legs[3]],
        )
        for charges in old_T.get_blocks_charge():
            charge_tuple = tuple(_charge_to_int(charge) for charge in charges)
            if charge_tuple[1] == 1:
                block = old_T[charges]
                new_T.set_block(
                    ts=charge_tuple, Ds=tuple(block.shape), val=block
                )
        projected_B[key] = B_set[key].copy()
        projected_T[key] = new_T
        _validate_spin_physical_leg(new_T, key)
    return projected_B, projected_T


def convert_gutzwiller_d4_to_d2(input_prefix, output_prefix, config_kwargs):
    """Load an old d=4 JSON, project it, and save an independent d=2 JSON.

    This is a genuine Gutzwiller projection, not a lossless change of storage:
    the even physical sector is discarded, so the many-body norm and spin
    expectation values generally differ from those of the unprojected d=4
    state.  No normalization factor is inserted here; normalized tensor-
    network expectation values account for the norm of the projected state.
    """
    B_set, T_set = load_triangle_iPESS(input_prefix, config_kwargs)
    B_spin, T_spin = gutzwiller_project_d4_sets(
        B_set, T_set, config_kwargs
    )
    save_fermionic_spin_triangle_iPESS(
        B_spin, T_spin, output_prefix, config_kwargs
    )
    return B_spin, T_spin


def fermionic_spin_bond_dimension(state):
    """Return and validate the common total virtual bond dimension."""
    dimensions = set()
    for key in state.B_set:
        B, T = state.B_set[key], state.T_set[key]
        _validate_spin_physical_leg(T, key)
        dimensions.update(sum(leg.D) for leg in B.get_legs())
        dimensions.update(sum(T.get_legs(axes=axis).D) for axis in (0, 2, 3))
    if len(dimensions) != 1:
        raise ValueError("virtual legs do not share one D: " + str(dimensions))
    return dimensions.pop()


def add_fermionic_spin_noise(state, noise):
    """Add relative random noise without enlarging the d=2 physical space."""
    if noise == 0:
        return state
    for key in state.B_set:
        for tensor_set in (state.B_set, state.T_set):
            tensor = tensor_set[key]
            random_tensor = yastn.rand(
                config=tensor.config, legs=tensor.get_legs()
            )
            scale = yastn.linalg.norm(tensor) / yastn.linalg.norm(random_tensor)
            tensor_set[key] = tensor + noise * scale * random_tensor
    state.normalize()
    return state


def make_fermionic_spin_state(B_set, T_set, global_args):
    """Construct the standard state container after validating d=2 legs."""
    for key, tensor in T_set.items():
        _validate_spin_physical_leg(tensor, key)
    return IPESS_TRIANGLE(B_set, T_set, global_args)
