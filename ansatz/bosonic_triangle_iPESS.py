"""Dense (no symmetry, bosonic) triangular-lattice iPESS states.

The geometry follows ``triangle_iPESS.py`` exactly:
``B=(L,U,M)`` is the simplex tensor and ``T=(M,s,R,D)`` is the site tensor.
Only the tensor statistics and the physical space differ from the fermionic
Hofstadter-Hubbard implementation.
"""

import json
from collections import OrderedDict

import numpy
import yastn

from ansatz.triangle_iPESS import IPESS_TRIANGLE


def _without_lattice(config_kwargs):
    local = {"Lx", "Ly", "save_file_prefix"}
    return {key: value for key, value in config_kwargs.items() if key not in local}


def dense_config(config_kwargs):
    return yastn.make_config(sym="dense", fermionic=False, **_without_lattice(config_kwargs))


def _cell_key(cx, cy):
    return str(cx) + "," + str(cy)


class IPESS_TRIANGLE_DENSE(IPESS_TRIANGLE):
    """Triangular iPESS whose YASTN tensors have no symmetry sectors."""

    def copy(self, preserve_grad=False):
        B_set = OrderedDict()
        T_set = OrderedDict()
        for key in self.B_set:
            if preserve_grad:
                B_set[key] = self.B_set[key].clone()
                T_set[key] = self.T_set[key].clone()
            else:
                B_set[key] = self.B_set[key].copy()
                T_set[key] = self.T_set[key].copy()
        return IPESS_TRIANGLE_DENSE(B_set, T_set, self.global_args)


def dense_tensor_from_array(array, config_kwargs, signatures):
    array = numpy.asarray(array)
    if len(signatures) != array.ndim:
        raise ValueError("One signature is required for every tensor leg")
    config = dense_config(config_kwargs)
    legs = [
        yastn.Leg(config, s=signatures[axis], D=(array.shape[axis],))
        for axis in range(array.ndim)
    ]
    tensor = yastn.zeros(config=config, legs=legs)
    tensor.set_block(ts=(), val=array)
    return tensor


def random_bosonic_triangle_iPESS(D, config_kwargs, physical_dim=2, seed=None):
    """Create a complex random spin iPESS with unit cell ``Lx x Ly``."""
    if seed is not None:
        numpy.random.seed(seed)
    config = dense_config(config_kwargs)
    B_legs = [yastn.Leg(config, s=s, D=(D,)) for s in (1, 1, -1)]
    T_dims = (D, physical_dim, D, D)
    T_legs = [
        yastn.Leg(config, s=s, D=(dim,))
        for s, dim in zip((1, 1, -1, -1), T_dims)
    ]
    B_set = OrderedDict()
    T_set = OrderedDict()
    for cx in range(1, config_kwargs["Lx"] + 1):
        for cy in range(1, config_kwargs["Ly"] + 1):
            key = _cell_key(cx, cy)
            B_set[key] = yastn.rand(config=config, legs=B_legs)
            T_set[key] = yastn.rand(config=config, legs=T_legs)
    state = IPESS_TRIANGLE_DENSE(B_set, T_set, config_kwargs)
    state.normalize()
    return state


def save_bosonic_triangle_iPESS(B_set, T_set, filenm, config_kwargs):
    def encode(tensor):
        dense = numpy.asarray(tensor.to_dense().to("cpu"))
        flat = numpy.reshape(dense, -1, order="F")
        return {
            "T_real": numpy.real(flat).tolist(),
            "T_imag": numpy.imag(flat).tolist(),
            "dims": list(dense.shape),
            "signatures": list(tensor.s),
        }

    payload = {"format": "dense_bosonic_triangle_iPESS_v1", "B_set": {}, "T_set": {}}
    for cx in range(1, config_kwargs["Lx"] + 1):
        for cy in range(1, config_kwargs["Ly"] + 1):
            key = _cell_key(cx, cy)
            payload["B_set"][key] = encode(B_set[key])
            payload["T_set"][key] = encode(T_set[key])
    with open(filenm + ".json", "w") as stream:
        json.dump(payload, stream)


def load_bosonic_triangle_iPESS(filenm, config_kwargs):
    with open(filenm + ".json") as stream:
        payload = json.load(stream)

    def decode(data, default_signatures):
        flat = numpy.asarray(data["T_real"]) + 1j * numpy.asarray(data["T_imag"])
        array = numpy.reshape(flat, tuple(data["dims"]), order="F")
        return dense_tensor_from_array(
            array, config_kwargs, data.get("signatures", default_signatures)
        )

    B_set = OrderedDict()
    T_set = OrderedDict()
    for cx in range(1, config_kwargs["Lx"] + 1):
        for cy in range(1, config_kwargs["Ly"] + 1):
            key = _cell_key(cx, cy)
            B_set[key] = decode(payload["B_set"][key], (1, 1, -1))
            T_set[key] = decode(payload["T_set"][key], (1, 1, -1, -1))
    return B_set, T_set


def bosonic_triangle_iPESS_bond_dimension(state):
    """Return the common virtual-bond dimension of a dense triangular iPESS."""
    if not state.B_set or set(state.B_set) != set(state.T_set):
        raise ValueError("B_set and T_set must contain the same non-empty unit cell")

    dimensions = set()
    for key in state.B_set:
        B = state.B_set[key]
        T = state.T_set[key]
        if B.config.fermionic or T.config.fermionic:
            raise ValueError("bond-dimension expansion is only for bosonic tensors")

        B_shape = tuple(B.get_shape())
        T_shape = tuple(T.get_shape())
        if len(B_shape) != 3 or len(T_shape) != 4:
            raise ValueError(
                "expected B=(L,U,M) and T=(M,s,R,D) at cell " + str(key)
            )
        dimensions.update(B_shape + (T_shape[0], T_shape[2], T_shape[3]))

    if len(dimensions) != 1:
        raise ValueError(
            "all iPESS virtual legs must have one common dimension; found "
            + str(sorted(dimensions))
        )
    return dimensions.pop()


def expand_bosonic_triangle_iPESS(state, target_D):
    """Embed a dense triangular iPESS into a larger virtual-bond space.

    The original tensor is copied into the leading block of every virtual leg.
    Newly added entries are zero. Apply ``add_bosonic_noise`` afterwards when
    noise is desired; this gives the same noise semantics whether or not the
    state was expanded.
    """
    current_D = bosonic_triangle_iPESS_bond_dimension(state)
    if not isinstance(target_D, (int, numpy.integer)) or target_D <= 0:
        raise ValueError("target_D must be a positive integer")
    target_D = int(target_D)
    if target_D < current_D:
        raise ValueError(
            "target_D=" + str(target_D)
            + " is smaller than the loaded bond dimension " + str(current_D)
        )
    if target_D == current_D:
        return state.copy()

    def expand_tensor(tensor, virtual_axes):
        array = numpy.asarray(tensor.to_dense().to("cpu"))
        new_shape = list(array.shape)
        for axis in virtual_axes:
            new_shape[axis] = target_D

        expanded = numpy.zeros(tuple(new_shape), dtype=array.dtype)
        old_block = tuple(slice(0, size) for size in array.shape)
        expanded[old_block] = array

        return dense_tensor_from_array(expanded, state.global_args, tensor.s)

    B_set = OrderedDict()
    T_set = OrderedDict()
    for key in state.B_set:
        B_set[key] = expand_tensor(state.B_set[key], (0, 1, 2))
        T_set[key] = expand_tensor(state.T_set[key], (0, 2, 3))
    return IPESS_TRIANGLE_DENSE(B_set, T_set, state.global_args)


def add_bosonic_noise(state, noise):
    if noise == 0:
        return state
    for key in state.B_set:
        for tensors, name in ((state.B_set, "B"), (state.T_set, "T")):
            tensor = tensors[key]
            perturbation = yastn.rand(config=tensor.config, legs=tensor.get_legs())
            scale = yastn.linalg.norm(tensor) / yastn.linalg.norm(perturbation)
            tensors[key] = tensor + noise * scale * perturbation
    state.normalize()
    return state
