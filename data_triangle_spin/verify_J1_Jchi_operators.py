"""Verify dense d=2 spin operators against the old d=4 single-occupancy sector."""

import os
import sys

import numpy
import yastn


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from model.Spin_ob_iPESS import Operators_spinful_Z2
from model.bosonic_spin_ob_iPESS import spin_half_operators_dense


def reconstruct_SS(Sa, Sb):
    return yastn.ncon([Sa, Sb], [[1, -1, -2], [1, -3, -4]])


def reconstruct_chirality(S1, S2, S3):
    return yastn.ncon(
        [S1, S2, S3], [[-1, -4, 1], [1, -2, -5, 2], [2, -3, -6]]
    )


def main():
    config = {
        "backend": "torch",
        "default_dtype": "complex128",
        "default_device": "cpu",
    }
    old = Operators_spinful_Z2(config)
    new = spin_half_operators_dense(config)
    old_SS = reconstruct_SS(old[0], old[1]).to_dense().detach().numpy()
    new_SS = reconstruct_SS(new[0], new[1]).to_dense().detach().numpy()
    old_chi = reconstruct_chirality(old[3], old[4], old[5]).to_dense().detach().numpy()
    new_chi = reconstruct_chirality(new[3], new[4], new[5]).to_dense().detach().numpy()

    single = numpy.array([2, 3])  # |up>, |down> in the old reordered d=4 basis
    old_SS = old_SS[numpy.ix_(single, single, single, single)]
    old_chi = old_chi[numpy.ix_(single, single, single, single, single, single)]
    SS_error = numpy.linalg.norm(old_SS - new_SS)
    chi_error = numpy.linalg.norm(old_chi - new_chi)

    assert new[0].get_shape()[0] == 3
    assert new[3].get_shape()[-1] == 3
    assert new[4].get_shape()[0] == new[4].get_shape()[-1] == 3
    assert new[5].get_shape()[0] == 3
    assert SS_error < 1.0e-13
    assert chi_error < 1.0e-13
    print("S.S projected error:", SS_error)
    print("chirality projected error:", chi_error)
    print("operator virtual bonds: SS=3, chirality=3x3")


if __name__ == "__main__":
    main()
