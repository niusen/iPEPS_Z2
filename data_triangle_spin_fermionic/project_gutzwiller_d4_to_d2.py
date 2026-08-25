"""One-time Gutzwiller projection of an old d=4 Z2 fermionic iPESS to d=2.

Edit the settings below and run this file once.  It does not optimize the
state.  The output JSON contains only the odd physical sector
``(|up>, |down>)`` and is the input of
``optimize_triangle_spin_fermionic_iPESS.py``.

Because empty and doubly occupied amplitudes are removed, this changes the
wavefunction norm and generally changes observables relative to the original
unprojected state.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Absolute source-code directory on the server.
source_code_dir = "/home/sniu/python_code/iPESS_triangle_spin"
sys.path.insert(0, source_code_dir)

from ansatz.fermionic_spin_triangle_iPESS import (
    convert_gutzwiller_d4_to_d2,
)


########################
# Conversion settings
########################
Lx = 2
Ly = 2
device = "cpu"
default_dtype = "complex128"

# Prefixes do not include the .json suffix. Relative paths are resolved beside
# this conversion file; absolute paths can also be used.
input_d4_state_file = "Z2_D8_chi80_3.45158"
output_d2_state_file = "Z2_spin_Gutzwiller_D8_Lx2_Ly2"


def _prefix(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def main():
    print("pid= " + str(os.getpid()))
    print("Gutzwiller conversion settings:")
    for name, value in {
        "source_code_dir": source_code_dir,
        "Lx": Lx,
        "Ly": Ly,
        "device": device,
        "default_dtype": default_dtype,
        "input_d4_state_file": _prefix(input_d4_state_file) + ".json",
        "output_d2_state_file": _prefix(output_d2_state_file) + ".json",
    }.items():
        print("  " + name + " = " + repr(value))

    config_kwargs = {
        "backend": "torch",
        "default_dtype": default_dtype,
        "default_device": device,
        "Lx": Lx,
        "Ly": Ly,
    }
    convert_gutzwiller_d4_to_d2(
        _prefix(input_d4_state_file),
        _prefix(output_d2_state_file),
        config_kwargs,
    )
    print("saved projected d=2 state:", _prefix(output_d2_state_file) + ".json")


if __name__ == "__main__":
    main()
