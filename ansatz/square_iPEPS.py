import math
import json
import os
from collections import OrderedDict

import numpy
import yastn


def _get_lattice_size(global_args):
    if isinstance(global_args, dict):
        return global_args["Lx"], global_args["Ly"]
    return global_args.Lx, global_args.Ly


def _config_without_lattice(config_kwargs):
    return {key: val for key, val in config_kwargs.items() if key not in ("Lx", "Ly")}


def dense_config(config_kwargs):
    return yastn.make_config(sym="dense", fermionic=False, **_config_without_lattice(config_kwargs))


def _cell_key(cx, cy):
    return str(cx) + "," + str(cy)


class IPEPS_SQUARE():
    def __init__(self, A_set, global_args):
        self.A_set = A_set
        self.global_args = global_args
        self.Lx, self.Ly = _get_lattice_size(global_args)

    def require_grad(self, require):
        for cx in range(1, self.Lx + 1):
            for cy in range(1, self.Ly + 1):
                self.A_set[_cell_key(cx, cy)].requires_grad_(requires_grad=require)

    def to_device(self, device_):
        for cx in range(1, self.Lx + 1):
            for cy in range(1, self.Ly + 1):
                key = _cell_key(cx, cy)
                self.A_set[key] = self.A_set[key].to(device_)

    def normalize(self):
        for cx in range(1, self.Lx + 1):
            for cy in range(1, self.Ly + 1):
                key = _cell_key(cx, cy)
                self.A_set[key] = self.A_set[key] / yastn.linalg.norm(self.A_set[key])

    def norm(self):
        norm_sq = 0
        for cx in range(1, self.Lx + 1):
            for cy in range(1, self.Ly + 1):
                norm_sq += (yastn.linalg.norm(self.A_set[_cell_key(cx, cy)]).item()) ** 2
        return math.sqrt(norm_sq)

    def build_double_layer_iPEPS(self):
        A_double_set = OrderedDict()
        for cx in range(1, self.Lx + 1):
            for cy in range(1, self.Ly + 1):
                key = _cell_key(cx, cy)
                A_double_set[key] = build_double_layer(Ap=self.A_set[key].conj(), A=self.A_set[key])
        return A_double_set

    def copy(self, preserve_grad=False):
        A_set_new = OrderedDict()
        for key in self.A_set:
            if preserve_grad:
                A_set_new[key] = self.A_set[key].clone()
            else:
                A_set_new[key] = self.A_set[key].copy()
        return IPEPS_SQUARE(A_set_new, self.global_args)


def dense_tensor_from_array(T, config_kwargs, dual=None):
    T = numpy.asarray(T)
    rank = len(T.shape)
    if rank != 5:
        raise ValueError("Square iPEPS site tensor should have rank 5, got rank " + str(rank))

    if dual is None:
        dual = [0] * rank
    if len(dual) != rank:
        raise ValueError("dual should have length " + str(rank) + ", got " + str(len(dual)))

    config_dense = dense_config(config_kwargs)
    legs = [
        yastn.Leg(config_dense, s=2 * (0.5 - dual[axis]), D=(T.shape[axis],))
        for axis in range(rank)
    ]
    tt = yastn.zeros(config=config_dense, legs=legs)
    tt.set_block(ts=(), val=T)
    return tt


def load_square_iPEPS(filenm, config_kwargs):
    with open(filenm + ".json") as f:
        data = json.load(f)

    def convert_to_yastn(tensor_dict):
        T_real = numpy.array(tensor_dict["T_real"], order="F")
        T_imag = numpy.array(tensor_dict["T_imag"], order="F")
        T = T_real + 1j * T_imag
        assert len(numpy.shape(T)) == 1

        dims = tensor_dict.get("dims")
        if dims is None:
            even_dims = numpy.array(tensor_dict.get("even_dims", []))
            odd_dims = numpy.array(tensor_dict.get("odd_dims", []))
            dims = (even_dims + odd_dims).astype(int).tolist()
        if len(dims) != 5:
            raise ValueError("Square iPEPS tensor should have 5 legs, got dims=" + str(dims))

        T = numpy.reshape(T, tuple(dims), order="F")
        dual = tensor_dict.get("dual", [0] * len(dims))
        return dense_tensor_from_array(T, config_kwargs, dual=dual)

    Lx, Ly = _get_lattice_size(config_kwargs)
    A_set_data = data["A_set"]
    A_set = OrderedDict()
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = _cell_key(cx, cy)
            A_set[key] = convert_to_yastn(A_set_data[key])
    return A_set


def save_square_iPEPS(Am_set, filenm, config_kwargs):
    def yastn_to_dict(TT):
        sigs = TT.s
        dual = (0.5 - numpy.array(sigs) / 2).astype(int).tolist()
        dims = list(TT.get_shape())
        TT_dense = numpy.array(TT.to_dense().to("cpu"))
        TT_dense = numpy.reshape(TT_dense, -1, order="F")
        T_real = numpy.real(TT_dense)
        T_imag = numpy.imag(TT_dense)
        return {"T_real": T_real.tolist(), "T_imag": T_imag.tolist(), "dims": dims, "dual": dual}

    Lx, Ly = _get_lattice_size(config_kwargs)
    A_set = OrderedDict()
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = _cell_key(cx, cy)
            A_set[key] = yastn_to_dict(Am_set[key])

    dirname = os.path.dirname(filenm)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filenm + ".json", "w") as f:
        json.dump({"A_set": A_set}, f)


def random_square_iPEPS(D, d, config_kwargs, dual=None):
    Lx, Ly = _get_lattice_size(config_kwargs)
    if dual is None:
        dual = [0, 1, 0, 1, 0]
    config_dense = dense_config(config_kwargs)
    dims = [D, D, D, D, d]
    legs = [yastn.Leg(config_dense, s=2 * (0.5 - dual[axis]), D=(dims[axis],)) for axis in range(5)]

    A_set = OrderedDict()
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            A_set[_cell_key(cx, cy)] = yastn.rand(config=config_dense, legs=legs)
    return A_set


def build_double_layer(Ap, A):
    # A has legs L, D, R, U, physical. The double layer keeps physical legs contracted.
    return yastn.tensordot(Ap, A, axes=([4], [4])).fuse_legs(axes=((0, 4), (1, 5), (2, 6), (3, 7)))


def CTM_to_device(CTM_set, Device, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = _cell_key(cx, cy)
            CTM_set["Cset"][key]["C1"] = CTM_set["Cset"][key]["C1"].to(Device)
            CTM_set["Cset"][key]["C2"] = CTM_set["Cset"][key]["C2"].to(Device)
            CTM_set["Cset"][key]["C3"] = CTM_set["Cset"][key]["C3"].to(Device)
            CTM_set["Cset"][key]["C4"] = CTM_set["Cset"][key]["C4"].to(Device)
            CTM_set["Tset"][key]["T1"] = CTM_set["Tset"][key]["T1"].to(Device)
            CTM_set["Tset"][key]["T2"] = CTM_set["Tset"][key]["T2"].to(Device)
            CTM_set["Tset"][key]["T3"] = CTM_set["Tset"][key]["T3"].to(Device)
            CTM_set["Tset"][key]["T4"] = CTM_set["Tset"][key]["T4"].to(Device)
    return CTM_set


def CTM_detach(CTM_set, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = _cell_key(cx, cy)
            CTM_set["Cset"][key]["C1"] = CTM_set["Cset"][key]["C1"].detach()
            CTM_set["Cset"][key]["C2"] = CTM_set["Cset"][key]["C2"].detach()
            CTM_set["Cset"][key]["C3"] = CTM_set["Cset"][key]["C3"].detach()
            CTM_set["Cset"][key]["C4"] = CTM_set["Cset"][key]["C4"].detach()
            CTM_set["Tset"][key]["T1"] = CTM_set["Tset"][key]["T1"].detach()
            CTM_set["Tset"][key]["T2"] = CTM_set["Tset"][key]["T2"].detach()
            CTM_set["Tset"][key]["T3"] = CTM_set["Tset"][key]["T3"].detach()
            CTM_set["Tset"][key]["T4"] = CTM_set["Tset"][key]["T4"].detach()
    return CTM_set


def Cset_detach(C_set, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = _cell_key(cx, cy)
            C_set[key]["C1"] = C_set[key]["C1"].detach()
            C_set[key]["C2"] = C_set[key]["C2"].detach()
            C_set[key]["C3"] = C_set[key]["C3"].detach()
            C_set[key]["C4"] = C_set[key]["C4"].detach()
    return C_set


def Tset_detach(T_set, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = _cell_key(cx, cy)
            T_set[key]["T1"] = T_set[key]["T1"].detach()
            T_set[key]["T2"] = T_set[key]["T2"].detach()
            T_set[key]["T3"] = T_set[key]["T3"].detach()
            T_set[key]["T4"] = T_set[key]["T4"].detach()
    return T_set


def Cset_requires_grad_(C_set, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = _cell_key(cx, cy)
            C_set[key]["C1"].requires_grad_(requires_grad=True)
            C_set[key]["C2"].requires_grad_(requires_grad=True)
            C_set[key]["C3"].requires_grad_(requires_grad=True)
            C_set[key]["C4"].requires_grad_(requires_grad=True)
    return C_set


def Tset_requires_grad_(T_set, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = _cell_key(cx, cy)
            T_set[key]["T1"].requires_grad_(requires_grad=True)
            T_set[key]["T2"].requires_grad_(requires_grad=True)
            T_set[key]["T3"].requires_grad_(requires_grad=True)
            T_set[key]["T4"].requires_grad_(requires_grad=True)
    return T_set


def CTM_copy(CTM_set, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    Cset_new = OrderedDict()
    Tset_new = OrderedDict()
    CTM_set_new = OrderedDict()
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = _cell_key(cx, cy)
            Cs = OrderedDict()
            Ts = OrderedDict()
            Cs["C1"] = CTM_set["Cset"][key]["C1"].clone()
            Cs["C2"] = CTM_set["Cset"][key]["C2"].clone()
            Cs["C3"] = CTM_set["Cset"][key]["C3"].clone()
            Cs["C4"] = CTM_set["Cset"][key]["C4"].clone()
            Ts["T1"] = CTM_set["Tset"][key]["T1"].clone()
            Ts["T2"] = CTM_set["Tset"][key]["T2"].clone()
            Ts["T3"] = CTM_set["Tset"][key]["T3"].clone()
            Ts["T4"] = CTM_set["Tset"][key]["T4"].clone()
            Cset_new[key] = Cs
            Tset_new[key] = Ts
    CTM_set_new["Cset"] = Cset_new
    CTM_set_new["Tset"] = Tset_new
    return CTM_set_new


def Cell_to_device(A_set, Device, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = _cell_key(cx, cy)
            A_set[key] = A_set[key].to(Device)
    return A_set


def Cell_detach(A_set, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = _cell_key(cx, cy)
            A_set[key] = A_set[key].detach()
    return A_set


def add_noise(state, noise, config_kwargs):
    config_dense = dense_config(config_kwargs)
    Lx, Ly = _get_lattice_size(config_kwargs)
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = _cell_key(cx, cy)
            tt = state.A_set[key]
            tt_new = yastn.rand(config=config_dense, legs=tt.get_legs())
            norm0 = yastn.linalg.norm(tt)
            norm1 = yastn.linalg.norm(tt_new)
            state.A_set[key] = tt + tt_new / norm1 * norm0 * noise
    return state
